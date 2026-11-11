// src/data/loader_core.h —— 数据装载的"机器"：索引、洗牌、批边界、预取池、并发交付。
//
// 与具体数据集无关。数据集之间的差别（目录约定、选哪张、怎么变换）全部由
// SampleSource 的两个虚函数承担。
//
// 设计四条（沿用 ImageNetLoader 已验证的语义）：
//   ① 索引与迭代分开。索引在构造时建一次、完全确定；每 epoch 只换 order_。
//   ② 断点的口径是【交付进度】而不是生产进度 —— 池化之后 cursor_ 领先好几批，
//      拿它当断点会静默跳过数据。
//   ③ 批是预分配的（池里固定几个槽），空槽才发布新批，发布不等完成。
//   ④ 每个样本的随机种子是 (seed, epoch, index) 的纯函数 —— 与线程、调用顺序无关。
//
// C++20 / MSVC 19.29+ / GCC 11+
#pragma once

#include <torch/torch.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "data/decode.h"

namespace prkd::data
{
    // ------------------------------------------------------------------------
    // 共享类型
    //
    // ⚠ 迁移期：这些在 imagenet_loader.h 里也各有一份。现在两者不可能被同一个
    //   .cpp 同时 include（loader_core 还不被任何地方引用），所以不冲突。
    //   等 ImageNetLoader 改成继承 LoaderCore 之后，把 imagenet_loader.h 里那份删掉，
    //   并让那边 #include "data/loader_core.h"。
    // ------------------------------------------------------------------------
#ifndef PRKD_LOADER_SHARED_TYPES
#define PRKD_LOADER_SHARED_TYPES

    // 这张图解码得干不干净。由 SampleSource::load_into 带出来。
    struct DecodeDiag
    {
        bool complete = true;   // false ⇒ 解码器提前停住，没写到的像素是预清零的 0（黑）
        int  warnings = 0;      // >0 ⇒ 解码过程中报过警告（截断、多余字节、坏码…）
    };

    // 解码不干净的图。只记前 N 条，用来定位；总数另计。
    struct BadImage
    {
        std::string path;
        std::string why;
    };

    inline constexpr int64_t kMaxBadPaths = 2000;

    // 吞吐统计。靠这个判断数据管线是不是瓶颈。
    struct LoaderStats
    {
        int64_t images = 0;
        double  seconds = 0.0;

        // 两者互斥：incomplete 是更严重的那一档，warned 指"有警告但读完了"。
        // 干净张数 = images - incomplete_images - warned_images。
        int64_t incomplete_images = 0;
        int64_t warned_images = 0;

        double img_per_sec() const
        {
            return seconds > 0.0 ? static_cast<double>(images) / seconds : 0.0;
        }
    };
#endif  // PRKD_LOADER_SHARED_TYPES

    // ------------------------------------------------------------------------
    // SampleMeta —— 每个样本跟着批走的附带信息
    //
    // ImageNet 只填 label；ALOI 还要角度（分析"主轴抓的是类间还是视角"时按它分组）。
    // ------------------------------------------------------------------------
    struct SampleMeta
    {
        int64_t label = 0;
        int64_t angle = -1;     // -1 = 该数据集没有这个属性
    };

    // ------------------------------------------------------------------------
    // 机器参数。与"怎么读一张图"无关，只决定形状与并发。
    // ------------------------------------------------------------------------
    struct CoreOpts
    {
        int64_t batch_size     = 64;
        int64_t workers        = 0;     // 0 = 同步单线程
        int64_t prefetch_depth = 2;     // 池深：同时几个批在飞；workers=0 时退化成 1

        int64_t out_h = 224;            // 批张量尺寸 —— 池里要预分配，所以必须在这里
        int64_t out_w = 224;

        bool shuffle   = true;          // 每 epoch 重排 order_
        bool drop_last = true;          // 截尾使批整除（训练必须；评测不能）
    };

    // ------------------------------------------------------------------------
    // LoaderCommonOpts —— 调用方契约的共用基底（不是机器的一部分）
    //
    // 机器只认 CoreOpts。这个结构体放在这里，只为了让 core() 与 CoreOpts 的定义
    // 同处一处 —— CoreOpts 加字段时，这里必须跟着改，而编译期不会替你发现
    // （聚合初始化，字段顺序错了照样能编过）。
    //
    // 两个数据集共用：ImageNet 加 short_side / is_train，ALOI 加视角范围。
    // ------------------------------------------------------------------------
    struct LoaderCommonOpts
    {
        int64_t batch_size     = 64;
        int64_t workers        = 0;
        int64_t prefetch_depth = 2;
        int64_t crop           = 224;

        bool shuffle   = true;      // 与训练/评测解耦：显式填，不靠 is_train 派生
        bool drop_last = true;

        // 唯一的映射点。校验放在这里：它在基类构造前求值，
        // 所以"crop <= 0"会在被当成 out_h 用之前就被拦住。
        CoreOpts core() const
        {
            TORCH_CHECK(batch_size > 0, "loader: batch_size 必须 > 0");
            TORCH_CHECK(crop > 0, "loader: crop 必须 > 0");
            if (workers < 0)
            {
                throw std::runtime_error("loader: workers 不能为负");
            }
            return CoreOpts{ batch_size, workers, prefetch_depth,
                             crop, crop, shuffle, drop_last };
        }
    };

    // ------------------------------------------------------------------------
    // ImageBatch —— 一次交付的东西
    //
    // 张量是池槽里预分配的那个，交付时整体 move 出去；槽立刻还池。
    // ------------------------------------------------------------------------
    struct ImageBatch
    {
        torch::Tensor           images;     // (N, 3, out_h, out_w) float32，已归一化
        torch::Tensor           labels;     // (N,) int64
        std::vector<int64_t>    indices;    // order_ 里的原始样本下标
        std::vector<SampleMeta> metas;      // 与 indices 一一对应

        // 实际张数。val 末批可短；train 因 drop_last 恒等于 batch_size。
        int64_t n = 0;
    };

    // ------------------------------------------------------------------------
    // SampleSource —— 一个数据集的"约定"
    //
    // 两个纯虚函数，生命周期完全不同：
    //   build_index   构造时一次
    //   load_into     热路径，每张一次
    //
    // ★ 实现不能引用派生 loader 的成员：基类析构时派生成员已销毁，而线程可能还在跑。
    //   需要的配置（root、是否训练、输出尺寸…）在 Source 自己的构造里拷一份。
    // ------------------------------------------------------------------------
    class SampleSource
    {
    public:
        virtual ~SampleSource() = default;

        // 把目录约定翻译成扁平表。classes 决定 num_classes()。
        // paths 与 metas 必须等长；三者都在 LoaderCore 构造里被 move 进去。
        virtual void build_index(std::vector<std::string>& paths,
                                 std::vector<SampleMeta>&  metas,
                                 std::vector<std::string>& classes) = 0;

        // 读第 index 张，写进 out.images 的第 slot 行，填 out.labels[slot]。
        //
        // path 由 LoaderCore 传入（paths_[index]）—— Source 不必自己再存一份路径表。
        // epoch 传进来是为了让"每样本种子 = f(seed, epoch, index)"成立，Source 自己不必存状态。
        virtual DecodeDiag load_into(ImageBatch& out, int64_t slot,
                                     int64_t index, const std::string& path,
                                     int64_t epoch) const = 0;
    };

    // ------------------------------------------------------------------------
    // 两个给 Source 用的工具（放这里是因为它们与"机器"同源，换数据集也要用）
    // ------------------------------------------------------------------------

    // 把文件读进调用方提供的缓冲。返回原始文件大小（不是解码后尺寸）。
    // 缓冲只增不减：长到最大那张之后不再分配、不再清零。
    size_t read_file_into(const std::string& path, std::vector<uint8_t>& buf);

    // splitmix64 finalizer + (seed, epoch, index) 派生出的每样本种子。
    uint64_t sample_seed(uint64_t seed, int64_t epoch, int64_t index);

    // ------------------------------------------------------------------------
    // LoaderCore —— 机器
    // ------------------------------------------------------------------------
    class LoaderCore
    {
    public:
        LoaderCore(CoreOpts opt, uint64_t seed, std::unique_ptr<SampleSource> src);
        virtual ~LoaderCore();

        // 不可拷贝；可移动。移动构造必须带 noexcept，否则放进 vector 扩容会退回拷贝。
        LoaderCore(const LoaderCore&) = delete;
        LoaderCore& operator=(const LoaderCore&) = delete;
        LoaderCore(LoaderCore&&) noexcept;
        LoaderCore& operator=(LoaderCore&&) noexcept;

        // 每个 epoch 开头调一次：按 (seed, epoch) 重排 order_，清池，重置统计。
        void start_epoch(int64_t epoch);

        bool       has_next() const;
        ImageBatch next();                  // 调之前 has_next() 必须为真

        int64_t size() const;
        int64_t num_batches() const;
        int64_t num_classes() const;
        int64_t epoch() const;
        const std::vector<std::string>& classes() const;
        LoaderStats stats() const;

        // 断点：下一批在 order_ 里的起点 = 已交付的样本数。
        // ★ 不能用 cursor_：池化之后它领先交付进度好几批。
        int64_t position() const { return delivered_pos_; }
        void    seek(int64_t pos);

        const std::vector<BadImage>& bad_images() const { return bad_; }
        int64_t bad_count() const { return bad_count_; }

        // 机器实际生效的那份参数。改名 core_options() 是为了不与
        // ImageNetLoader::options()（返回 LoaderOpts）同名 —— 同名不同返回类型
        // 会让基类那个永远摸不到，而且不报错。
        const CoreOpts& core_options() const { return opt_; }

    private:
        struct PoolSlot
        {
            ImageBatch  batch;
            std::vector<DecodeDiag> diags;
            int64_t     n = 0;
            int64_t     batch_idx = 0;
            int64_t     order_pos = 0;
            std::atomic<int64_t> remaining{ 0 };
        };

        struct Task
        {
            int64_t slot = 0;      // 填进哪个池槽
            int64_t k = 0;         // 该槽的第几行
            int64_t index = 0;     // 样本下标（order_ 里的值）
        };

        void reset_order();
        void start_workers();
        void stop_workers();
        void worker_loop();
        void publish_more();
        void drain_inline();
        void run_one_task(const Task& t);

        CoreOpts                      opt_;
        uint64_t                      seed_ = 0;
        int64_t                       epoch_ = 0;
        std::unique_ptr<SampleSource> source_;

        std::vector<std::string> classes_;
        std::vector<std::string> paths_;
        std::vector<SampleMeta>  metas_;
        std::vector<int64_t>     order_;

        // 扁平批表：批 b 的范围 = [batch_pos_[b], batch_pos_[b+1])。
        // 不用 vector<vector<>>：那样几千个批就是几千次小分配，每次访问多一跳指针。
        std::vector<int64_t> batch_pos_;
        std::vector<bool>    batch_done_;
        std::vector<int64_t> batch_slot_;

        // 三个进度：cursor_ 生产、next_batch_ 交付(批)、delivered_pos_ 交付(样本)。
        // 不变量：delivered_pos_ == batch_pos_[next_batch_]
        int64_t next_batch_ = 0;
        int64_t cursor_ = 0;
        int64_t delivered_pos_ = 0;

        std::vector<std::jthread>  workers_;
        std::vector<PoolSlot>      pool_;
        std::deque<int64_t>        free_slots_;
        std::deque<Task>           q_;

        mutable std::mutex         q_mu_;
        std::condition_variable    q_cv_;
        std::condition_variable    ready_cv_;
        std::atomic<bool>          stop_{ false };
        std::atomic<bool>          has_err_{ false };

        std::exception_ptr         err_;
        std::mutex                 err_mu_;

        std::vector<BadImage>      bad_;
        int64_t                    bad_count_ = 0;
        std::mutex                 bad_mu_;

        mutable std::mutex         stats_mu_;
        LoaderStats                stats_;
    };
}  // namespace prkd::data
