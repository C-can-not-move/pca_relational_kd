// src/data/imagenet_loader.cpp —— 同步版（workers = 0）
//
// workers = 0：主线程自己读，行为和以前完全一样。
// workers > 0：一批的任务由 workers 个常驻线程动态领取，
//              每批两级同步（start_bar_ 放行 / done_bar_ 收齐）。
#include "data/imagenet_loader.h"

// 项目内的跨模块依赖：loader 要调解码和变换（两个头各自已自给自足）
#include "data/decode.h"
#include "data/transform.h"

namespace fs = std::filesystem;

namespace prkd::data
{
    namespace
    {
        // splitmix64 的 finalizer：把顺序变化的种子打散成互不相关的种子。
        // mt19937_64(s) 会把 s 直接放进 state[0]，所以 s 只差 1 的种子之间
        // 早期输出存在可测的相关性 —— 而我们是按 index 逐个生成 10 万条独立流。
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

        inline uint64_t seed_aug(uint64_t s, int64_t epoch, int64_t index)
        {
            return mix(mix(mix(s) ^ static_cast<uint64_t>(epoch)) ^ static_cast<uint64_t>(index));
        }

		// 把文件读进【调用方提供的缓冲】。返回的是原始文件大小，不是解码后的图像尺寸。
        //
        // 不返回 vector：那样每张图一次分配，而且 vector 的 size 构造会清零这 n 个字节，
        // 紧接着 read 又全部覆盖 —— 平均 108 KB × 128 万张 = 每 epoch 138 GB 的无用内存写。
        // 缓冲只增不减：长到最大那张图的尺寸之后，不再分配、不再清零。
        size_t read_into(const std::string& path, std::vector<uint8_t>& buf)
        {
            // 先 stat，不 open：打不开就早退，不占文件句柄
            std::error_code ec;
            const auto sz = std::filesystem::file_size(path, ec);
            if (ec) throw std::runtime_error("ImageNetLoader: 打不开 " + path);
            if (sz == 0) throw std::runtime_error("ImageNetLoader: 空文件 " + path);

            const size_t want = static_cast<size_t>(sz);
            if (buf.size() < want)
            {
                buf.resize(want);          // 只在长大时零填一次
            }

            std::ifstream f(path, std::ios::binary);
            if (!f) throw std::runtime_error("ImageNetLoader: 打不开 " + path);

            // read 按原始字节读取，不遇到换行就停
            f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(want));

            // 缓冲是复用的：读短了的话，buf 里留着【上一张图】的数据。
            // gcount 是唯一能分辨"这一张没读全"和"读全了"的信号。
            if (static_cast<size_t>(f.gcount()) != want)
            {
                throw std::runtime_error("ImageNetLoader: 读取不完整 " + path);
            }
            return want;
        }
    }  // namespace

    ImageNetLoader::ImageNetLoader(fs::path root, const LoaderOpts& opt, uint64_t seed)
        : root_(std::move(root))
        , opt_(opt)
        , seed_(seed)
    {
        TORCH_CHECK(opt_.batch_size > 0, "ImageNetLoader: batch_size 必须 > 0");
        TORCH_CHECK(opt_.crop > 0, "ImageNetLoader: crop 必须 > 0");
        TORCH_CHECK(opt_.short_side > 0, "ImageNetLoader: short_side 必须 > 0");
        if (opt_.workers < 0)
        {
            throw std::runtime_error("ImageNetLoader: workers 不能为负");
        }

        build_index_imagefolder();

        // drop_last 在「样本数 < batch_size」时会把 order_ 截成 0 项，
        // 于是训练循环一个 batch 都读不到 —— 这是配置错。
        // 在构造时说明，别等到循环打印出 num_batches = 0 才发现。
        if (opt_.is_train && static_cast<int64_t>(paths_.size()) < opt_.batch_size)
        {
            throw std::runtime_error(
                "ImageNetLoader: 训练模式下 drop_last 会得到 0 个 batch（样本 "
                + std::to_string(paths_.size()) + " < batch_size "
                + std::to_string(opt_.batch_size) + "）。把 batch_size 调小，或改用 --val。");
        }

        start_epoch(0);
        // worker 线程常驻：起一次，析构时停。每批靠任务队列 + remaining 计数，不用 barrier。
        start_workers();
    }

    ImageNetLoader::~ImageNetLoader()
    {
        stop_workers();
    }

    // ------------------------------------------------------------------------
    // 索引：类目录字典序 → 标签 0..N-1，类内文件名字典序
    //
    // ★ readdir 的顺序标准未规定（ext4 是哈希序、NTFS 接近字典序），
    //   所以两处都必须显式排序，否则同一份数据在两台机器上类别号不同。
    //   不排的后果：权重照样加载、loss 照样降、验证准确率掉到接近随机，而且不报错。
    // ------------------------------------------------------------------------
    void ImageNetLoader::build_index_imagefolder()
    {
        if (!fs::exists(root_) || !fs::is_directory(root_))
        {
            throw std::runtime_error("ImageNetLoader: 目录不存在 " + root_.string());
        }

        std::vector<fs::path> dirs;
        for (const auto& e : fs::directory_iterator(root_))
        {
            if (e.is_directory()) dirs.push_back(e.path());
        }
        if (dirs.empty())
        {
            throw std::runtime_error("ImageNetLoader: 没有类目录 " + root_.string());
        }

        // 按 filename 的字符串排，不按 path 排（path 的比较受分隔符和转义影响）
        const auto by_name = [](const fs::path& a, const fs::path& b) {
            return a.filename().string() < b.filename().string();
        };

        // 只收图片扩展名。没有这一层的话，Tiny 每个类目录里的 n0xxxxx_boxes.txt
        // 会被当成一张图索引进来，然后在解码时炸 —— 而且报错位置离原因很远。
        // 白名单与 torchvision 的 IMG_EXTENSIONS 保持一致。
        const auto is_image = [](const fs::path& p) {
            std::string ext = p.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return ext == ".jpeg" || ext == ".jpg" || ext == ".png" ||
                   ext == ".bmp" || ext == ".ppm" || ext == ".pgm";
        };
        std::sort(dirs.begin(), dirs.end(), by_name);

        classes_.clear();
        paths_.clear();
        labels_.clear();
        classes_.reserve(dirs.size());

        for (const auto& d : dirs)
        {
            const int64_t label = static_cast<int64_t>(classes_.size());
            classes_.push_back(d.filename().string());

            std::vector<fs::path> files;
            for (const auto& e : fs::directory_iterator(d))
            {
                if (e.is_regular_file() && is_image(e.path())) files.push_back(e.path());
            }
            std::sort(files.begin(), files.end(), by_name);

            // 类目录里一张图都没有 ⇒ 布局不对（最常见：还多着一层 images/）
            if (files.empty())
            {
                throw std::runtime_error(
                    "ImageNetLoader: 类目录里没有图片 " + d.string()
                    + "（若是 Tiny-ImageNet，检查是否还有一层 images/ 没整理掉）");
            }

            for (const auto& f : files)
            {
                paths_.push_back(f.string());
                labels_.push_back(label);
            }
        }

        if (paths_.empty())
        {
            throw std::runtime_error("ImageNetLoader: 一个文件都没找到 " + root_.string());
        }
    }

    void ImageNetLoader::reset_order()
    {
        order_.resize(paths_.size());
        std::iota(order_.begin(), order_.end(), static_cast<int64_t>(0));

        const int64_t bs = opt_.batch_size;

        if (opt_.is_train)
        {
            std::mt19937_64 rng(seed_order(seed_, epoch_));
            std::shuffle(order_.begin(), order_.end(), rng);

            // drop_last：把尾巴截掉，让它根本不存在。
            // 关系类损失（RKD-D/A）在 n 太小时会算出 NaN，所以训练必须整除。
            const int64_t nb = static_cast<int64_t>(order_.size()) / bs;
            order_.resize(static_cast<size_t>(nb * bs));
        }

        const int64_t n  = static_cast<int64_t>(order_.size());
        const int64_t nb = opt_.is_train ? (n / bs) : ((n + bs - 1) / bs);

        // 每个批的起点。末项 = order_.size()，所以批 b 的范围是 [pos[b], pos[b+1])。
        // ★ 末批必须用 min 夹住 —— val 的 nb*bs 会超过 n。
        batch_pos_.resize(static_cast<size_t>(nb) + 1);
        for (int64_t b = 0; b <= nb; ++b)
        {
            batch_pos_[static_cast<size_t>(b)] = std::min(b * bs, n);
        }

        batch_done_.assign(static_cast<size_t>(nb), false);
        batch_slot_.assign(static_cast<size_t>(nb), -1);
    }
    void ImageNetLoader::start_epoch(int64_t epoch)
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
    void ImageNetLoader::start_workers()
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
    void ImageNetLoader::stop_workers()
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

    void ImageNetLoader::worker_loop()
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
    void ImageNetLoader::run_one_task(const Task& t)
    {
        PoolSlot& ps = pool_[static_cast<size_t>(t.slot)];

        try
        {
            const DecodeDiag d = load_one_into(ps.batch, t.k, t.index);
            ps.diags[static_cast<size_t>(t.k)] = d;

            // ---- 有损：不抛，计数 + 记路径（只记前 20 条）----
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
    void ImageNetLoader::publish_more()
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
            ps.batch.images = torch::empty({ n, 3, opt_.crop, opt_.crop }, torch::kFloat32);
            ps.batch.labels = torch::empty({ n }, torch::kInt64);
            ps.batch.indices.resize(static_cast<size_t>(n));
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
                ps.batch.indices[static_cast<size_t>(k)] = idx;
                ps.batch.labels[k] = labels_[static_cast<size_t>(idx)];
                q_.push_back(Task{ slot, k, idx });
            }
            cursor_ += n;
        }

        q_cv_.notify_all();
    }

    // 同步路径：主线程自己把 q_ 里的任务全干完。
    void ImageNetLoader::drain_inline()
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
    DecodeDiag ImageNetLoader::load_one_into(ImageBatch& out, int64_t slot, int64_t index) const
    {
        // 每线程一份缓冲：大小只增不减，避免每张图一次分配 + 一次清零。
        // ★ 必须传 len 而不是 tls_buf.size() —— size() 是历史最大值，
        //   多出来的那段是上一张图的残留，解码器读到也不报错。
        thread_local std::vector<uint8_t> tls_buf;
        const size_t len = read_into(paths_[static_cast<size_t>(index)], tls_buf);
        const auto bmp = decode_rgb8(tls_buf.data(), len);

        // 诊断在解码这一步就齐了；致命错误已经抛出去了，能走到这里的都是"解出来了"。
        DecodeDiag diag;
        diag.complete = bmp.decode_complete;
        diag.warnings = bmp.decode_warnings;

        auto x = to_chw01(bmp.data.data(), bmp.h, bmp.w);

        if (opt_.is_train)
        {
            // 每样本一个独立 RNG：与线程、调用顺序、epoch 都无关 ⇒ 可复现。
            // ★ 两个 random_* 必须共用这个 rng 且按此顺序调用，对应 torchvision 的 Compose。
            std::mt19937_64 rng(seed_aug(seed_, epoch_, index));
            x = random_resized_crop(x, opt_.crop, rng);
            x = random_horizontal_flip(x, rng);
        }
        else
        {
            x = resize(x, opt_.short_side);
            x = center_crop(x, opt_.crop, opt_.crop);
        }

        // copy_ 而不是 operator=：images.select(0,slot) 是视图，
        // 用 = 只会给临时对象重新绑定，不会写回父张量 —— 结果是一个全零 batch。
        // imagenet_norm() 每次调用都新建两个 std::vector<float> —— 放在这里就是每张图
        // 两次堆分配，而它其实是个常量。static 局部变量的初始化是线程安全的（C++11 起）。
        static const NormStats kNorm = imagenet_norm();

        out.images.select(0, slot).copy_(normalize(x, kNorm));
        return diag;
    }

    bool ImageNetLoader::has_next() const
    {
        // ★ 必须是【交付】口径，不是 cursor_。
        //   池化之后 cursor_ 领先交付进度好几批，用它当判据会提早结束 epoch。
        return delivered_pos_ < static_cast<int64_t>(order_.size());
    }

    void ImageNetLoader::seek(int64_t pos)
    {
        if (pos < 0 || pos > static_cast<int64_t>(order_.size()))
        {
            throw std::runtime_error("ImageNetLoader::seek(): 位置越界");
        }

        // 必须恰好落在某个批边界上。
        // 这道闸门正好挡住「改过 batch_size 却拿旧断点恢复」——
        // 没有它的话，恢复会静默错位到某一批的中间。
        if (!std::binary_search(batch_pos_.begin(), batch_pos_.end(), pos))
        {
            throw std::runtime_error(
                "ImageNetLoader::seek(): 位置 " + std::to_string(pos) +
                " 不在批边界上 —— 是不是改过 batch_size？");
        }

        if (delivered_pos_ != 0 || cursor_ != 0 || !q_.empty())
        {
            throw std::logic_error(
                "ImageNetLoader::seek(): 只能在 start_epoch() 之后、取数据之前调");
        }

        next_batch_ = static_cast<int64_t>(
            std::lower_bound(batch_pos_.begin(), batch_pos_.end(), pos) - batch_pos_.begin());
        cursor_        = pos;
        delivered_pos_ = pos;
    }
    ImageBatch ImageNetLoader::next()
    {
        if (!has_next())
        {
            throw std::logic_error("ImageNetLoader::next(): 调之前 has_next() 必须为真");
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
    int64_t ImageNetLoader::size() const
    {
        return static_cast<int64_t>(paths_.size());
    }

    int64_t ImageNetLoader::num_batches() const
    {
        // 唯一来源是 batch_pos_ —— 再单独算一遍 n/bs 总有一天会和它对不上
        return static_cast<int64_t>(batch_pos_.size()) - 1;
    }
    int64_t ImageNetLoader::num_classes() const
    {
        return static_cast<int64_t>(classes_.size());
    }

    int64_t ImageNetLoader::epoch()  const { return epoch_; }
    bool    ImageNetLoader::is_train()  const { return opt_.is_train; }

    const std::vector<std::string>& ImageNetLoader::classes() const { return classes_; }
    const LoaderOpts&               ImageNetLoader::options() const { return opt_; }

    LoaderStats ImageNetLoader::stats() const
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        return stats_;
    }
}  // namespace prkd::data