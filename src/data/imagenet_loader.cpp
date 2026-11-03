// src/data/imagenet_loader.cpp —— 同步版（workers = 0）
//
// 这一版把索引、解码、变换、拼 batch 全部验完；线程那套（槽位/latch/队列）
// 仍然声明在头里，但构造函数直接拒绝 workers > 0。
// 先让正确性成立，再谈吞吐。
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

        // 把文件读进【调用方提供的缓冲】，返回实际长度。
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
        // std::counting_semaphore 没有默认构造，必须在这里给初值。
        // 单位是【任务数】= 批数 × 每批样本数；同步版用不到它，但仍然要初始化。
        , empty_slots_(std::max<int64_t>(1, opt.queue_depth * opt.batch_size))
    {
        TORCH_CHECK(opt_.batch_size > 0, "ImageNetLoader: batch_size 必须 > 0");
        TORCH_CHECK(opt_.crop > 0, "ImageNetLoader: crop 必须 > 0");
        TORCH_CHECK(opt_.short_side > 0, "ImageNetLoader: short_side 必须 > 0");
        if (opt_.workers > 0)
        {
            throw std::runtime_error("ImageNetLoader: workers > 0 尚未实现（本版只做同步）");
        }

        build_index_imagefolder();

        // drop_last 在「样本数 < batch_size」时会把 order_ 截成 0 项，
        // 于是训练循环一个 batch 都读不到 —— 这是配置错，不是"刚好这么设计"。
        // 在构造时就说清楚，别等到循环打印出 num_batches = 0 才发现。
        if (opt_.train && static_cast<int64_t>(paths_.size()) < opt_.batch_size)
        {
            throw std::runtime_error(
                "ImageNetLoader: 训练模式下 drop_last 会得到 0 个 batch（样本 "
                + std::to_string(paths_.size()) + " < batch_size "
                + std::to_string(opt_.batch_size) + "）。把 batch_size 调小，或改用 --val。");
        }

        start_epoch(0);
    }

    ImageNetLoader::~ImageNetLoader() = default;

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

        if (!opt_.train)
        {
            return;                       // 验证集不洗，顺序固定
        }

        std::mt19937_64 rng(seed_order(seed_, epoch_));
        std::shuffle(order_.begin(), order_.end(), rng);

        // drop_last：把尾巴截掉，让它根本不存在。
        // 关系类损失（RKD-D/A）在 n 太小时会算出 NaN，所以训练必须整除。
        const int64_t bs = opt_.batch_size;
        const int64_t nb = static_cast<int64_t>(order_.size()) / bs;
        order_.resize(static_cast<size_t>(nb * bs));
    }

    void ImageNetLoader::start_epoch(int64_t epoch)
    {
        epoch_ = epoch;
        reset_order();
        cursor_ = 0;
        {
            std::lock_guard<std::mutex> lk(stats_mu_);
            stats_ = LoaderStats{};
        }
    }

    void ImageNetLoader::start_workers() {}   // 同步版：无操作
    void ImageNetLoader::stop_workers() {}    // 同步版：无操作

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

        if (opt_.train)
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
        return cursor_ < static_cast<int64_t>(order_.size());
    }

    ImageBatch ImageNetLoader::next()
    {
        if (!has_next())
        {
            throw std::logic_error("ImageNetLoader::next(): 调之前 has_next() 必须为真");
        }

        const auto t0 = std::chrono::steady_clock::now();

        const int64_t n = std::min<int64_t>(opt_.batch_size,
            static_cast<int64_t>(order_.size()) - cursor_);

        ImageBatch out;
        out.images = torch::empty({ n, 3, opt_.crop, opt_.crop }, torch::kFloat32);
        out.labels = torch::empty({ n }, torch::kInt64);
        out.indices.resize(static_cast<size_t>(n));

        // 按批收集，末尾只加一次锁 —— 每张图加锁的代价比读一张图高得多。
        int64_t n_incomplete = 0;
        int64_t n_warned = 0;

        for (int64_t k = 0; k < n; ++k)
        {
            const int64_t idx = order_[static_cast<size_t>(cursor_ + k)];
            out.indices[static_cast<size_t>(k)] = idx;
            out.labels[k] = labels_[static_cast<size_t>(idx)];

            const DecodeDiag d = load_one_into(out, k, idx);
            if (!d.complete)
            {
                ++n_incomplete;          // 更严重的那一档
            }
            else if (d.warnings > 0)
            {
                ++n_warned;              // 有警告但读完了
            }
        }
        cursor_ += n;

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
        const int64_t n = static_cast<int64_t>(order_.size());
        if (opt_.train)
        {
            return n / opt_.batch_size;                       // drop_last，整除
        }
        return (n + opt_.batch_size - 1) / opt_.batch_size;   // 向上取整，末批可短
    }

    int64_t ImageNetLoader::num_classes() const
    {
        return static_cast<int64_t>(classes_.size());
    }

    int64_t ImageNetLoader::epoch()  const { return epoch_; }
    bool    ImageNetLoader::train()  const { return opt_.train; }

    const std::vector<std::string>& ImageNetLoader::classes() const { return classes_; }
    const LoaderOpts&               ImageNetLoader::options() const { return opt_; }

    LoaderStats ImageNetLoader::stats() const
    {
        std::lock_guard<std::mutex> lk(stats_mu_);
        return stats_;
    }
}  // namespace prkd::data