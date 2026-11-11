// src/data/loader_core.cpp —— 机器本体。
//
// 逻辑是从 imagenet_loader.cpp 逐段搬来的（那份已经在 ImageNet 上验过），
// 只做了三类替换：
//   opt_.crop            → opt_.out_h / opt_.out_w
//   opt_.is_train        → opt_.shuffle / opt_.drop_last
//   load_one_into(...)   → source_->load_into(...)
// 报错信息里的 ImageNetLoader: 改成了 LoaderCore:。
//
// ★ 在 ImageNetLoader 改成继承它之前，这两份逻辑是重复的。
//   改这里记得对照 imagenet_loader.cpp；等那边切过来后，旧副本整段删掉。
#include "data/loader_core.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>

namespace prkd::data
{
    namespace
    {
        // splitmix64 的 finalizer：把顺序变化的种子打散成互不相关的种子。
        // mt19937_64(s) 会把 s 直接放进 state[0]，所以 s 只差 1 的种子之间存在可测的相关性。
        inline uint64_t mix(uint64_t x)
        {
            x += 0x9E3779B97F4A7C15ull;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
            return x ^ (x >> 31);
        }

        // 一个种子，多条派生流。不要从共享流里依次取 —— 取用顺序会依赖线程调度。
        inline uint64_t seed_order(uint64_t s, int64_t epoch)
        {
            return mix(mix(s) ^ static_cast<uint64_t>(epoch));
        }
    }  // namespace

    uint64_t sample_seed(uint64_t seed, int64_t epoch, int64_t index)
    {
        return mix(mix(mix(seed) ^ static_cast<uint64_t>(epoch))
                   ^ static_cast<uint64_t>(index));
    }

    size_t read_file_into(const std::string& path, std::vector<uint8_t>& buf)
    {
        // 先 stat，不 open：打不开就早退，不占文件句柄
        std::error_code ec;
        const auto sz = std::filesystem::file_size(path, ec);
        if (ec) throw std::runtime_error("loader: 打不开 " + path);
        if (sz == 0) throw std::runtime_error("loader: 空文件 " + path);

        const size_t want = static_cast<size_t>(sz);
        if (buf.size() < want)
        {
            buf.resize(want);          // 只在长大时零填一次
        }

        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("loader: 打不开 " + path);

        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(want));

        // 缓冲是复用的：读短了的话，buf 里留着【上一张图】的数据。
        // gcount 是唯一能分辨"这一张没读全"和"读全了"的信号。
        if (static_cast<size_t>(f.gcount()) != want)
        {
            throw std::runtime_error("loader: 读取不完整 " + path);
        }
        return want;
    }

    LoaderCore::LoaderCore(CoreOpts opt, uint64_t seed, std::unique_ptr<SampleSource> src)
        : opt_(opt)
        , seed_(seed)
        , source_(std::move(src))
    {
        if (!source_)
        {
            throw std::runtime_error("LoaderCore: SampleSource 为空");
        }
        TORCH_CHECK(opt_.batch_size > 0, "LoaderCore: batch_size 必须 > 0");
        TORCH_CHECK(opt_.out_h > 0 && opt_.out_w > 0, "LoaderCore: out_h/out_w 必须 > 0");
        if (opt_.workers < 0)
        {
            throw std::runtime_error("LoaderCore: workers 不能为负");
        }

        source_->build_index(paths_, metas_, classes_);

        if (paths_.empty())
        {
            throw std::runtime_error("LoaderCore: 索引为空（数据目录里没找到任何样本）");
        }
        if (metas_.size() != paths_.size())
        {
            throw std::runtime_error(
                "LoaderCore: SampleSource 给的 paths 与 metas 长度不一致 "
                + std::to_string(paths_.size()) + " vs " + std::to_string(metas_.size()));
        }

        // drop_last 在「样本数 < batch_size」时会把 order_ 截成 0 项，
        // 于是训练循环一个 batch 都读不到 —— 这是配置错，在构造时说明。
        if (opt_.drop_last && static_cast<int64_t>(paths_.size()) < opt_.batch_size)
        {
            throw std::runtime_error(
                "LoaderCore: drop_last 会得到 0 个 batch（样本 "
                + std::to_string(paths_.size()) + " < batch_size "
                + std::to_string(opt_.batch_size) + "）");
        }

        start_epoch(0);

        // 生效值一行。
        // shuffle / drop_last 与 transform 分支已经解耦，能配出"合法但反直觉"的组合
        // （比如评测却 drop_last=true，白丢末尾不到一批的样本）。
        // 不拦，只让它出现在启动输出里 —— 事后猜比启动时看一眼贵得多。
        std::printf("[loader] batch=%lld out=%lldx%lld workers=%lld "
                    "shuffle=%d drop_last=%d | samples=%lld classes=%lld batches=%lld\n",
                    (long long)opt_.batch_size,
                    (long long)opt_.out_h, (long long)opt_.out_w,
                    (long long)opt_.workers,
                    static_cast<int>(opt_.shuffle), static_cast<int>(opt_.drop_last),
                    (long long)paths_.size(),
                    (long long)classes_.size(),
                    (long long)(batch_pos_.size() - 1));
        std::fflush(stdout);

        // worker 线程常驻：起一次，析构时停。每批靠任务队列 + remaining 计数，不用 barrier。
        start_workers();
    }

    LoaderCore::~LoaderCore()
    {
        stop_workers();
    }

    // 有线程在飞时不能搬：workers_ 里的 lambda 捕获的是旧 this 的指针。
    LoaderCore::LoaderCore(LoaderCore&& o) noexcept
        : opt_(o.opt_)
        , seed_(o.seed_)
        , epoch_(o.epoch_)
        , source_(std::move(o.source_))
        , classes_(std::move(o.classes_))
        , paths_(std::move(o.paths_))
        , metas_(std::move(o.metas_))
        , order_(std::move(o.order_))
        , batch_pos_(std::move(o.batch_pos_))
        , batch_done_(std::move(o.batch_done_))
        , batch_slot_(std::move(o.batch_slot_))
        , next_batch_(o.next_batch_)
        , cursor_(o.cursor_)
        , delivered_pos_(o.delivered_pos_)
        , pool_(std::move(o.pool_))
        , free_slots_(std::move(o.free_slots_))
        , q_(std::move(o.q_))
        , bad_(std::move(o.bad_))
        , bad_count_(o.bad_count_)
    {
        // 被搬走的那份不能再停线程（已经没线程了，且 pool_ 也被偷了）。
        o.stop_.store(true, std::memory_order_relaxed);
        o.workers_.clear();          // 若调用方违反约定，这里会 join 掉正在跑的线程
        o.pool_.clear();
        o.free_slots_.clear();
        o.q_.clear();
        o.source_.reset();
    }

    LoaderCore& LoaderCore::operator=(LoaderCore&& o) noexcept
    {
        if (this == &o) return *this;
        stop_workers();
        opt_           = o.opt_;
        seed_          = o.seed_;
        epoch_         = o.epoch_;
        source_        = std::move(o.source_);
        classes_       = std::move(o.classes_);
        paths_         = std::move(o.paths_);
        metas_         = std::move(o.metas_);
        order_         = std::move(o.order_);
        batch_pos_     = std::move(o.batch_pos_);
        batch_done_    = std::move(o.batch_done_);
        batch_slot_    = std::move(o.batch_slot_);
        next_batch_    = o.next_batch_;
        cursor_        = o.cursor_;
        delivered_pos_ = o.delivered_pos_;
        pool_          = std::move(o.pool_);
        free_slots_    = std::move(o.free_slots_);
        q_             = std::move(o.q_);
        bad_           = std::move(o.bad_);
        bad_count_     = o.bad_count_;
        o.stop_.store(true, std::memory_order_relaxed);
        o.workers_.clear();
        o.pool_.clear();
        o.free_slots_.clear();
        o.q_.clear();
        o.source_.reset();
        return *this;
    }

    void LoaderCore::reset_order()
    {
        order_.resize(paths_.size());
        std::iota(order_.begin(), order_.end(), static_cast<int64_t>(0));

        const int64_t bs = opt_.batch_size;

        if (opt_.shuffle)
        {
            std::mt19937_64 rng(seed_order(seed_, epoch_));
            std::shuffle(order_.begin(), order_.end(), rng);
        }

        if (opt_.drop_last)
        {
            // 关系类损失（RKD-D/A）在 n 太小时会算出 NaN，所以训练必须整除。
            const int64_t nb = static_cast<int64_t>(order_.size()) / bs;
            order_.resize(static_cast<size_t>(nb * bs));
        }

        const int64_t n  = static_cast<int64_t>(order_.size());
        const int64_t nb = opt_.drop_last ? (n / bs) : ((n + bs - 1) / bs);

        // 每个批的起点。末项 = order_.size()，所以批 b 的范围是 [pos[b], pos[b+1])。
        // ★ 末批必须用 min 夹住 —— 不丢尾巴时 nb*bs 会超过 n。
        batch_pos_.resize(static_cast<size_t>(nb) + 1);
        for (int64_t b = 0; b <= nb; ++b)
        {
            batch_pos_[static_cast<size_t>(b)] = std::min(b * bs, n);
        }

        batch_done_.assign(static_cast<size_t>(nb), false);
        batch_slot_.assign(static_cast<size_t>(nb), -1);
    }

    void LoaderCore::start_epoch(int64_t epoch)
    {
        epoch_ = epoch;
        reset_order();

        // 清池：把在飞的批全部作废。
        // 正常流程下 epoch 结束时池里已经没有未交付的批，所以这里通常是空操作；
        // 提前 break / 异常时才有东西丢。那些批不浪费 —— 派生种子保证重算逐位相同。
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            q_.clear();
            free_slots_.clear();
            for (int64_t i = 0; i < static_cast<int64_t>(pool_.size()); ++i)
            {
                free_slots_.push_back(i);
            }
            std::fill(batch_done_.begin(), batch_done_.end(), false);
            std::fill(batch_slot_.begin(), batch_slot_.end(), -1);
        }

        cursor_        = 0;
        next_batch_    = 0;
        delivered_pos_ = 0;

        {
            std::lock_guard<std::mutex> lk(err_mu_);
            err_ = nullptr;
        }
        has_err_.store(false, std::memory_order_relaxed);

        {
            std::lock_guard<std::mutex> lk(stats_mu_);
            stats_ = LoaderStats{};
        }

        // bad_ 不清 —— 那是"这份数据长什么样"的体检结果，跨 epoch 累计更有意义。
    }

    // 池【永远存在】—— 同步路径（workers = 0）也要用它，只是池深退化成 1、
    // 并且没有 worker 线程，由主线程自己把队列干完。
    void LoaderCore::start_workers()
    {
        const int64_t depth = (opt_.workers > 0)
            ? std::max<int64_t>(1, opt_.prefetch_depth)
            : 1;

        // PoolSlot 里有 std::atomic，既不可拷贝也不可移动 ——
        // 所以 assign / resize / emplace_back 都不行（它们要求 MoveInsertable）。
        // vector(n) 只需要 DefaultInsertable；整体 move 赋值只偷 buffer，不搬元素。
        pool_ = std::vector<PoolSlot>(static_cast<size_t>(depth));

        {
            std::lock_guard<std::mutex> lk(q_mu_);
            q_.clear();
            free_slots_.clear();
            for (int64_t i = 0; i < depth; ++i) free_slots_.push_back(i);
        }

        stop_.store(false, std::memory_order_relaxed);
        has_err_.store(false, std::memory_order_relaxed);

        if (opt_.workers <= 0) return;      // 同步路径：只有池，没有线程

        workers_.reserve(static_cast<size_t>(opt_.workers));
        for (int64_t i = 0; i < opt_.workers; ++i)
        {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    // 只能在【没有批在飞】的时候调：构造末尾、start_epoch 之前、或析构里。
    void LoaderCore::stop_workers()
    {
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            stop_.store(true, std::memory_order_relaxed);
            q_.clear();                     // 丢弃未领取的任务
        }
        q_cv_.notify_all();
        ready_cv_.notify_all();

        workers_.clear();                   // jthread 析构 = request_stop + join

        {
            std::lock_guard<std::mutex> lk(q_mu_);
            free_slots_.clear();
        }
        pool_.clear();
    }

    void LoaderCore::worker_loop()
    {
        for (;;)
        {
            Task t;
            {
                std::unique_lock<std::mutex> lk(q_mu_);
                q_cv_.wait(lk, [&] {
                    return stop_.load(std::memory_order_relaxed) || !q_.empty();
                });
                if (q_.empty()) return;     // stop_ 且队列已空
                t = q_.front();
                q_.pop_front();
            }
            run_one_task(t);
        }
    }

    // 一张图：worker 和同步路径共用这一份逻辑。
    void LoaderCore::run_one_task(const Task& t)
    {
        PoolSlot& ps = pool_[static_cast<size_t>(t.slot)];

        try
        {
            const std::string& path = paths_[static_cast<size_t>(t.index)];
            const DecodeDiag d = source_->load_into(ps.batch, t.k, t.index, path, epoch_);
            ps.diags[static_cast<size_t>(t.k)] = d;

            // ---- 有损：不抛，计数 + 记路径（只记前 N 条）----
            if (!d.complete || d.warnings > 0)
            {
                std::lock_guard<std::mutex> lk(bad_mu_);
                ++bad_count_;
                if (bad_.size() < static_cast<size_t>(kMaxBadPaths))
                {
                    bad_.push_back(BadImage{
                        paths_[static_cast<size_t>(t.index)],
                        d.complete ? ("解码警告 x" + std::to_string(d.warnings))
                                   : std::string("数据不全（缺口是 0 黑）") });
                }
            }
        }
        catch (...)
        {
            // ---- 致命：异常绝不能逃出 jthread（会 std::terminate）----
            {
                std::lock_guard<std::mutex> lk(err_mu_);
                if (!err_) err_ = std::current_exception();
            }
            has_err_.store(true, std::memory_order_relaxed);
        }

        // 最后一个做完的人负责报齐。
        // ★ 异常路径上也必须减 —— 否则 remaining 永远到不了 0，主线程死锁。
        // ★ acq_rel：给"前面的写"和"读到 0 的那个人"之间建立 happens-before。
        if (ps.remaining.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            batch_done_[static_cast<size_t>(ps.batch_idx)] = true;
            ready_cv_.notify_one();
        }
    }

    // 有空槽就发布新批 —— 发布【不等】完成，这是池化能做重叠的原因。
    void LoaderCore::publish_more()
    {
        std::lock_guard<std::mutex> lk(q_mu_);

        while (!free_slots_.empty() &&
               cursor_ < static_cast<int64_t>(order_.size()))
        {
            // cursor_ 一定等于某个批的起点，所以批号可以推出来，不必另存计数器
            const int64_t b = static_cast<int64_t>(
                std::lower_bound(batch_pos_.begin(), batch_pos_.end(), cursor_)
                - batch_pos_.begin());

            const int64_t n = batch_pos_[static_cast<size_t>(b) + 1]
                            - batch_pos_[static_cast<size_t>(b)];

            const int64_t slot = free_slots_.front();
            free_slots_.pop_front();

            PoolSlot& ps = pool_[static_cast<size_t>(slot)];
            ps.batch.images  = torch::empty({ n, 3, opt_.out_h, opt_.out_w }, torch::kFloat32);
            ps.batch.labels  = torch::empty({ n }, torch::kInt64);
            ps.batch.indices.resize(static_cast<size_t>(n));
            ps.batch.metas.resize(static_cast<size_t>(n));
            ps.batch.n       = n;
            ps.diags.assign(static_cast<size_t>(n), DecodeDiag{});
            ps.n         = n;
            ps.batch_idx = b;
            ps.order_pos = cursor_;
            ps.remaining.store(n, std::memory_order_relaxed);

            batch_slot_[static_cast<size_t>(b)] = slot;
            batch_done_[static_cast<size_t>(b)] = false;

            for (int64_t k = 0; k < n; ++k)
            {
                const int64_t idx = order_[static_cast<size_t>(cursor_ + k)];
                const SampleMeta& m = metas_[static_cast<size_t>(idx)];
                ps.batch.indices[static_cast<size_t>(k)] = idx;
                ps.batch.metas[static_cast<size_t>(k)]   = m;
                ps.batch.labels[k] = m.label;
                q_.push_back(Task{ slot, k, idx });
            }
            cursor_ += n;
        }

        q_cv_.notify_all();
    }

    // 同步路径：主线程自己把 q_ 里的任务全干完。
    void LoaderCore::drain_inline()
    {
        for (;;)
        {
            Task t;
            {
                std::lock_guard<std::mutex> lk(q_mu_);
                if (q_.empty()) return;
                t = q_.front();
                q_.pop_front();
            }
            run_one_task(t);
        }
    }

    bool LoaderCore::has_next() const
    {
        // ★ 必须是【交付】口径，不是 cursor_。
        //   池化之后 cursor_ 领先交付进度好几批，用它当判据会提早结束 epoch。
        return delivered_pos_ < static_cast<int64_t>(order_.size());
    }

    ImageBatch LoaderCore::next()
    {
        if (!has_next())
        {
            throw std::logic_error("LoaderCore::next(): 调之前 has_next() 必须为真");
        }

        const auto t0 = std::chrono::steady_clock::now();
        const int64_t b = next_batch_;      // 这一轮要交付的批号

        // 发布【不等】完成 —— 这是池化的全部意义
        publish_more();

        // 同步路径：主线程自己把这一批干完
        if (opt_.workers <= 0) drain_inline();

        // 等【该交付的那一批】完成。别的批先完成也没用 —— next_batch_ 保证顺序。
        {
            std::unique_lock<std::mutex> lk(q_mu_);
            ready_cv_.wait(lk, [&] {
                return batch_done_[static_cast<size_t>(b)]
                    || has_err_.load(std::memory_order_relaxed);
            });
        }

        // 致命错误：整个 loader 停，半成品【不交付】
        if (has_err_.load(std::memory_order_relaxed))
        {
            std::lock_guard<std::mutex> lk(err_mu_);
            if (err_) std::rethrow_exception(err_);
        }

        // 取走 + 立刻还槽。
        // ★ 忘了 free_slots_.push_back 的话，池会越用越少，几百批后死锁在"没有空槽"。
        int64_t slot = -1;
        int64_t n = 0;
        ImageBatch out;
        std::vector<DecodeDiag> diag;
        {
            std::lock_guard<std::mutex> lk(q_mu_);
            slot = batch_slot_[static_cast<size_t>(b)];
            batch_slot_[static_cast<size_t>(b)] = -1;
            batch_done_[static_cast<size_t>(b)] = false;
            n    = pool_[static_cast<size_t>(slot)].n;
            out  = std::move(pool_[static_cast<size_t>(slot)].batch);
            diag = std::move(pool_[static_cast<size_t>(slot)].diags);
            free_slots_.push_back(slot);
        }

        ++next_batch_;
        delivered_pos_ = batch_pos_[static_cast<size_t>(next_batch_)];

        publish_more();                     // 腾出的槽立刻补发下一批

        // ---- 诊断收口 + 统计 ----
        int64_t n_incomplete = 0;
        int64_t n_warned = 0;
        for (int64_t k = 0; k < n; ++k)
        {
            const DecodeDiag& d = diag[static_cast<size_t>(k)];
            if (!d.complete)         ++n_incomplete;   // 更严重的那一档
            else if (d.warnings > 0) ++n_warned;       // 有警告但读完了
        }

        const double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        {
            std::lock_guard<std::mutex> lk(stats_mu_);
            stats_.images += n;
            stats_.seconds += dt;
            stats_.incomplete_images += n_incomplete;
            stats_.warned_images += n_warned;
        }
        return out;
    }

    void LoaderCore::seek(int64_t pos)
    {
        if (pos < 0 || pos > static_cast<int64_t>(order_.size()))
        {
            throw std::runtime_error("LoaderCore::seek(): 位置越界");
        }

        // 必须恰好落在某个批边界上。
        // 这道闸门正好挡住「改过 batch_size 却拿旧断点恢复」——
        // 没有它的话，恢复会静默错位到某一批的中间。
        if (!std::binary_search(batch_pos_.begin(), batch_pos_.end(), pos))
        {
            throw std::runtime_error(
                "LoaderCore::seek(): 位置 " + std::to_string(pos) +
                " 不在批边界上 —— 是不是改过 batch_size？");
        }

        if (delivered_pos_ != 0 || cursor_ != 0 || !q_.empty())
        {
            throw std::logic_error(
                "LoaderCore::seek(): 只能在 start_epoch() 之后、取数据之前调");
        }

        next_batch_ = static_cast<int64_t>(
            std::lower_bound(batch_pos_.begin(), batch_pos_.end(), pos) - batch_pos_.begin());
        cursor_        = pos;
        delivered_pos_ = pos;
    }

    int64_t LoaderCore::size() const
    {
        return static_cast<int64_t>(paths_.size());
    }

    int64_t LoaderCore::num_batches() const
    {
        // 唯一来源是 batch_pos_ —— 再单独算一遍 n/bs 总有一天会和它对不上
        return static_cast<int64_t>(batch_pos_.size()) - 1;
    }

    int64_t LoaderCore::num_classes() const
    {
        return static_cast<int64_t>(classes_.size());
    }

    int64_t LoaderCore::epoch() const { return epoch_; }

    const std::vector<std::string>& LoaderCore::classes() const { return classes_; }

    LoaderStats LoaderCore::stats() const
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        return stats_;
    }
}  // namespace prkd::data
