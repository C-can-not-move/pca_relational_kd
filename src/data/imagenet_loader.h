// src/data/imagenet_loader.h —— 直接从原始 JPEG 读，不做任何中间二进制缓存
//
// 设计上的四条：
//   ① 索引与迭代分开。索引只在构造时建一次、完全确定；每 epoch 只换 order。
//   ② ImageBatch 里带 indices —— 出问题能答"第 7 张是谁"，对拍时也能对上号。
//   ③ train / val 同一个类，靠 Opts 区分。变换不同，索引、队列、线程那套完全相同。
//   ④ drop_last 只对 train 生效，体现在 num_batches() 里；val 不丢尾巴，末批可短。
//
// 命名：本类的输出叫 ImageBatch，和 kd/loss.h 里那个 train::Batch 不是一回事 ——
// 后者是【损失函数的输入】（logits / feat / target），前者是【原始图像 + 标签】。
//
// C++20
// 要求 GCC >= 11 / MSVC >= 19.29。GCC 9 编不过。
#pragma once

// ---- 第三方 ----
#include <torch/torch.h>

// ---- 标准库：基础类型与容器 ----
#include <cstdint>
#include <string>
#include <vector>

// ---- 标准库：文件与流 ----
#include <filesystem>
#include <fstream>

// ---- 标准库：并发（C++20）----
// 需要 GCC >= 11 / MSVC >= 19.29。CMakeLists 里已经按这个拦过了。
#include <atomic>
#include <barrier>      // start_bar_ / done_bar_
#include <exception>    // std::exception_ptr
#include <memory>       // std::unique_ptr
#include <mutex>        // err_mu_
#include <thread>       // std::jthread

// ---- 标准库：杂项 ----
#include <algorithm>    // std::sort / std::min（build_index 与 next 里用）
#include <cctype>       // std::tolower（扩展名白名单）
#include <chrono>       // steady_clock
#include <cstring>
#include <numeric>      // std::iota（reset_order）
#include <stdexcept>    // std::runtime_error

namespace prkd::data
{
    // ------------------------------------------------------------------------
    // LoaderOpts —— 这一次装载怎么读
    //
    // 种子只有一处：cfg::RunConfig.seed，构造时传进来。
    // ------------------------------------------------------------------------
    struct LoaderOpts
    {
        int64_t batch_size = 64;
        int64_t workers = 0;        // 0 = 同步单线程；>0 = 起这么多个读图线程
        int64_t short_side = 256;   // Resize(短边) —— 只对 train=false 生效；256 = V1 口径，V2 用 232
        int64_t crop = 224;         // CenterCrop / RRC 的输出边长

        /*
        true  = RandomResizedCrop(crop) + RandomHorizontalFlip，且每 epoch 洗牌
        false = Resize 短边 short_side + CenterCrop(crop)，且不洗牌（顺序固定）
        */
        bool train = true;

    };

    // ------------------------------------------------------------------------
    // ImageBatch —— 一次取出来的东西
    // ------------------------------------------------------------------------
    struct ImageBatch
    {
        torch::Tensor images;          // (N, 3, crop, crop) float32，输出前已 /255 并减均值除方差
        torch::Tensor labels;          // (N,) int64
        std::vector<int64_t> indices;  // order 里的原始样本下标。调试、对拍、复现都靠它
    };

    // ------------------------------------------------------------------------
    // DecodeDiag —— 这张图解码得干不干净。由 load_one_into 带出来。
    //
    // 为什么不直接在 load_one_into 里累加：它是 const，碰不到 stats_；
    // 而且每张图加一次锁不划算 —— 按批收齐、末尾只加一次锁。
    // 线程版起来之后，这个结构会挂在 worker 的结果项上一起回收集线程，不必另开通道。
    // ------------------------------------------------------------------------
    struct DecodeDiag
    {
        bool complete = true;   // false ⇒ libjpeg 提前停住，没写到的像素是预清零的 0（黑）
        int  warnings = 0;      // >0 ⇒ 解码过程中报过警告（截断、多余字节、坏 Huffman 码…）
    };

    // ------------------------------------------------------------------------
    // LoaderStats —— 吞吐统计。靠这个判断数据管线是不是瓶颈
    // ------------------------------------------------------------------------
    struct LoaderStats
    {
        int64_t images = 0;
        double  seconds = 0.0;

        // 下面两项是"只计数、不跳过"的产物 —— 不动 order_ / has_next() / num_batches()。
        // 两者【互斥】：incomplete 是更严重的那一档，warned 指"有警告但读完了"。
        // 所以干净张数 = images - incomplete_images - warned_images。
        int64_t incomplete_images = 0;   // !diag.complete
        int64_t warned_images = 0;       // diag.complete && diag.warnings > 0

        double img_per_sec() const
        {
            return seconds > 0.0 ? static_cast<double>(images) / seconds : 0.0;
        }
    };

    // ------------------------------------------------------------------------
    // ImageNetLoader
    // ------------------------------------------------------------------------
    class ImageNetLoader
    {
    public:
        // 构造里就建索引。以下任一情况抛 std::runtime_error：
        //   · root 不存在
        //   · ImageFolder 模式下没有类目录
        //   · 平铺模式下文件数与标签表行数不一致
        ImageNetLoader(std::filesystem::path root, const LoaderOpts& opt, uint64_t seed);

        // stop + join。中途 break 出循环也安全
        ~ImageNetLoader();

        // 不可拷贝；可移动。移动构造必须带 noexcept —— 否则放进 std::vector 时
        // 扩容会退回拷贝，而拷贝已删除，报错会出现在 vector 内部，很难读。
        ImageNetLoader(const ImageNetLoader&) = delete;
        ImageNetLoader& operator=(const ImageNetLoader&) = delete;
        ImageNetLoader(ImageNetLoader&&) noexcept;
        ImageNetLoader& operator=(ImageNetLoader&&) noexcept;

        // ---- 每个 epoch 开头调一次 ----
        // train=true ：按 seed_order(seed, epoch) 重新洗牌
        // train=false：不洗，光标归零
        // 同时重置统计
        void start_epoch(int64_t epoch);

        // ---- 取数据 ----
        bool       has_next() const;
        ImageBatch next();              // 调之前 has_next() 必须为真

        // ---- 查询 ----
        int64_t size() const;                    // 样本总数
        int64_t num_batches() const;             // train: size()/bs 整除（drop_last）｜val: 向上取整，末批可短
        int64_t num_classes() const;
        int64_t epoch() const;
        bool    train() const;
        const std::vector<std::string>& classes() const;
        const LoaderOpts& options() const;
        LoaderStats stats() const;

        void worker_loop();              // 每个 worker 线程的主循环

        void build_index_imagefolder();
        void reset_order();
        void start_workers();
        void stop_workers();

        // 读图 + 解码 + 变换，结果写进 out 的第 slot 行。
        // 返回这张图的解码诊断；统计在 next() 里按批集中累加。
        DecodeDiag load_one_into(ImageBatch& out, int64_t slot, int64_t index) const;

        // ---- 配置与索引 ----
        std::filesystem::path    root_;
        LoaderOpts               opt_;
        uint64_t                 seed_ = 0;
        int64_t                  epoch_ = 0;

        std::vector<std::string> classes_;   // 类别名，按标签序
        std::vector<std::string> paths_;     // 与 labels_ 一一对应
        std::vector<int64_t>     labels_;
        std::vector<int64_t>     order_;     // 本轮访问顺序
        int64_t                  cursor_ = 0;

        // ---- 并发：一批内动态领取 ----
        //
        // 为什么不是队列：一批的任务集合在开头就完全确定（就是这 n 个下标），
        // 动态领取就退化成"抢一个原子计数器"。每张一次 fetch_add（~30 ns），
        // 比 mutex + deque + notify（0.5–2 µs）便宜十几倍。
        //
        // 为什么不是静态切分：ImageNet 的图从 3 万到 1 千万像素，单张耗时差一个
        // 数量级以上。静态切分等于"等自己区间里最慢的那张"，会白等近一倍。
        //
        // 两级同步用 std::barrier（可重复使用），而不是每批重建 latch ——
        // 重建就得管"worker 可能正卡在旧的那个上面"的生命周期问题。
        std::vector<std::jthread>        workers_;
        std::unique_ptr<std::barrier<>>  start_bar_;    // nw + 1：放行本批
        std::unique_ptr<std::barrier<>>  done_bar_;     // nw + 1：收齐本批
        std::atomic<int64_t>             ticket_{ 0 };  // 本批抢到第几张
        std::atomic<bool>                stop_{ false };

        // 每批发布给 worker 的东西，只在 next() 期间有效（可见性由 barrier 建立）
        ImageBatch*  cur_out_ = nullptr;
        int64_t      cur_cursor_ = 0;      // 本批第 0 张对应 order_ 的哪个位置
        int64_t      cur_n_ = 0;

        std::vector<DecodeDiag>  diags_;   // 每批 n 个，用 slot 直接索引
        std::exception_ptr       err_;     // 第一个 worker 异常，由 next() 重抛
        std::mutex               err_mu_;

        // ---- 统计 ----
        mutable std::mutex stats_mu_;
        LoaderStats        stats_;
    };
}  // namespace prkd::data