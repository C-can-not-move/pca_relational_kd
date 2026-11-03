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
// ⚠ C++20：本头用到 std::latch / std::counting_semaphore / std::jthread（含 std::stop_token）。
//   要求 GCC >= 11 / MSVC >= 19.29。GCC 9 编不过。
#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <latch>
#include <memory>
#include <mutex>
#include <numeric>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
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
        int64_t queue_depth = 4;    // 队列里最多压几个 batch。workers>0 才有意义
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
    // LoaderStats —— 吞吐统计。靠这个判断数据管线是不是瓶颈
    // ------------------------------------------------------------------------
    struct LoaderStats
    {
        int64_t images = 0;
        double  seconds = 0.0;

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

    private:
        struct Task                 // 一条取图任务
        {
            int64_t index = 0;      // 样本下标
            int64_t slot = 0;       // 填进 slots_ 的哪一格
        };

        void build_index_imagefolder();
        void reset_order();
        void start_workers();
        void stop_workers();

        void load_one_into(ImageBatch& out, int64_t slot, int64_t index) const;

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

        // ---- 并发：生产者（worker 线程）→ 消费者（next） ----
        std::vector<std::jthread>   workers_;
        std::deque<Task>            q_;
        mutable std::mutex          q_mu_;
        std::condition_variable     q_not_empty_;
        std::condition_variable     q_not_full_;
        std::counting_semaphore<>   empty_slots_;      // 构造时初始化成 queue_depth
        std::vector<torch::Tensor>  slots_;            // 当前 batch 的写入口
        std::unique_ptr<std::latch> done_;             // 每个 batch 一个，BS 归零
        std::atomic<bool>           stop_{ false };

        // ---- 统计 ----
        mutable std::mutex stats_mu_;
        LoaderStats        stats_;
    };
}  // namespace prkd::data