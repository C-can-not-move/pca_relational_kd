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
// 并发设计总思路：单图粒度。线程可以同时扑在同一批的不同格子上，也可以分散在三四批上。
//
// C++20
// 要求 GCC >= 11 / MSVC >= 19.29。GCC 9 编不过。
#pragma once

// ---- 第三方 ----
#include <torch/torch.h>

// ---- 标准库：基础类型与容器 ----
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

// ---- 标准库：文件与流 ----
#include <filesystem>
#include <fstream>

// ---- 标准库：并发（C++20）----
// 需要 GCC >= 11 / MSVC >= 19.29。CMakeLists 里已经按这个拦过了。
#include <atomic>
#include <condition_variable>   // q_cv_ / ready_cv_
#include <exception>    // std::exception_ptr
#include <mutex>        // q_mu_ / err_mu_ / bad_mu_ / stats_mu_
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
        int64_t batch_size = 64;    // 这个决定了训练时的batch长度
        int64_t workers = 0;        // 0 = 同步单线程；>0 = 起这么多个读图线程
        int64_t prefetch_depth = 2; // 池深：同时几个批在飞。1 = 退化成同步等齐
        int64_t short_side = 256;   // Resize(短边) —— 只对 train=false 生效；256 = V1 口径，V2 用 232
        int64_t crop = 224;         // CenterCrop / RRC 的输出边长

        /*
        true  = RandomResizedCrop(crop) + RandomHorizontalFlip，且每 epoch 洗牌
        false = Resize 短边 short_side + CenterCrop(crop)，且不洗牌（顺序固定）
        */
        bool is_train = true;

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
    // BadImage —— 解码不干净的图。只记前 N 条，用来定位；总数另计。
    // ------------------------------------------------------------------------
    struct BadImage
    {
        std::string path;
        std::string why;
    };

    // 记录多少条坏图路径。128 万张里几千条也就几百 KB，但没必要全记 ——
    // 2000 条足够看出"是哪一类问题"。真实总数由 bad_count() 另给。
    inline constexpr int64_t kMaxBadPaths = 2000;

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

        // 不可拷贝；可移动。移动构造必须带 noexcept —— 否则放进 std::vector 时扩容会退回拷贝，而拷贝已删除，报错会出现在 vector 内部，很难读。
        ImageNetLoader(const ImageNetLoader&) = delete;
        ImageNetLoader& operator=(const ImageNetLoader&) = delete;
        ImageNetLoader(ImageNetLoader&&) noexcept;
        ImageNetLoader& operator=(ImageNetLoader&&) noexcept;

        // ---- 每个 epoch 开头调一次 ----
        // LoaderOpts.is_train=true ：按 seed_order(seed, epoch) 重新洗牌
        // LoaderOpts.is_train=false：不洗，光标归零
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
        bool    is_train() const;
        const std::vector<std::string>& classes() const;
        const LoaderOpts& options() const;
        LoaderStats stats() const;

        // ---- 断点 ----
        // position() 返回【下一批】在 order_ 里的起点 = 已交付的样本数。
        // ★ 不能用 cursor_：池化之后它领先交付进度好几批，拿它当断点会静默跳过数据。
        int64_t position() const { return delivered_pos_; }
        void    seek(int64_t pos);

        // 解码不干净的前 N 条（N=20）。要全量清单得离线扫。
        const std::vector<BadImage>& bad_images() const { return bad_; }
        int64_t bad_count() const { return bad_count_; }

        void worker_loop();              // 每个 worker 线程的主循环
        void publish_more();             // 有空槽就发布新批（不等完成）
        void drain_inline();             // 同步路径：主线程把 q_ 里的任务全干完

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

        // ---- 索引：扁平 + 起点表 ----
        //
        // 批 b 的范围 = [batch_pos_[b], batch_pos_[b+1])。
        // 不用 vector<vector<>>：那样 2000 多个批就是 2000 多次小分配，
        // 而且每次访问多一跳指针。扁平一次分配、连续。
        // 洗牌仍然作用在扁平数组上，所以 (seed, epoch) 的可复现性一行不变。
        std::vector<int64_t> batch_pos_;    // 长度 = num_batches + 1
        std::vector<bool>    batch_done_;   // 长度 = num_batches
		std::vector<int64_t> batch_slot_;   // 批号 → 池槽；-1 = 还没发布，或者已经被取走 记录的是对应批在 pool_ 里的槽号。类似于滑动窗口

        // ---- 进度：三个游标 ----
        //   cursor_         生产进度：下一个要发布的区间起点
        //   next_batch_     交付进度（批号）
        //   delivered_pos_  交付进度（样本位置）← 断点用这个
        //
        // 不变量：delivered_pos_ == batch_pos_[next_batch_]
        int64_t next_batch_ = 0;
        int64_t delivered_pos_ = 0;

        // ---- 池 ----
        struct PoolSlot
        {
            ImageBatch  batch;
            std::vector<DecodeDiag> diags;
            int64_t     n = 0;                    // 这批的实际张数（val 末批可短）
            int64_t     batch_idx = 0;            // 批号 —— 报齐时写 batch_done_[batch_idx]
            int64_t     order_pos = 0;            // 在 order_ 里的起点
            std::atomic<int64_t> remaining{ 0 };  // 还剩几张没做完；减到 0 ⇒ 这批齐了
        };

        struct Task                               // 一条取图任务 = 一张图
        {
            int64_t slot = 0;                     // 填进哪个池槽
            int64_t k = 0;                        // 填该槽的第几行
            int64_t index = 0;                    // 样本下标（order_ 里的值）
        };

        // 一张图：worker 和同步路径共用这一份逻辑。
        // ★ 必须声明在 struct Task 之后 —— 前面只写了 worker_loop 那一组，
        //   那时 Task 还不存在，编译器会把参数当成 int。
        void run_one_task(const Task& t);

        std::vector<std::jthread>  workers_;
        std::vector<PoolSlot>      pool_;
        std::deque<int64_t>        free_slots_;
        std::deque<Task>           q_;

        mutable std::mutex         q_mu_;             // 保护 q_ / free_slots_ / batch_* / ready_cv_
        std::condition_variable    q_cv_;             // worker 等任务
        std::condition_variable    ready_cv_;         // 主线程等"该交付的那批"
        std::atomic<bool>          stop_{ false };
        std::atomic<bool>          has_err_{ false }; // 供 wait 谓词无锁读；err_ 本身另加锁

        std::exception_ptr         err_;              // 第一个 worker 异常
        std::mutex                 err_mu_;

        // ---- 解码不干净的图（只记前 20 条）----
        std::vector<BadImage>      bad_;
        int64_t                    bad_count_ = 0;
        std::mutex                 bad_mu_;

        // ---- 统计 ----
        mutable std::mutex stats_mu_;
        LoaderStats        stats_;
    };
}  // namespace prkd::data