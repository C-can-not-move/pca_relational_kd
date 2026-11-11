// src/data/imagenet_loader.cpp —— 只剩"策略"：目录约定 + 一张图怎么变。
//
// 机器（索引迭代、批边界、预取池、并发交付、断点）已全部移到 loader_core.cpp。
// 这里的两段逻辑是从改造前的 ImageNetLoader 原文搬来的，逐行未改语义：
//   build_index_imagefolder  →  ImageNetSource::build_index
//   load_one_into            →  ImageNetSource::load_into
#include "data/imagenet_loader.h"

// 项目内的跨模块依赖：Source 要调解码和变换
#include "data/decode.h"
#include "data/transform.h"

#include <algorithm>
#include <cctype>
#include <random>
#include <stdexcept>

namespace fs = std::filesystem;

namespace prkd::data
{
    namespace
    {
        // LoaderOpts → CoreOpts。
        //
        // 校验放在这里而不是派生类构造函数体里：to_core() 在基类构造【之前】求值，
        // 于是"crop <= 0"会在基类拿它当 out_h 之前就被拦住，报错信息也更准。
        //
        // 唯一有语义的映射：is_train 同时决定"洗牌"和"丢尾巴"——
        // 这两件事在 ImageNet 上历来绑定，分开了反而容易配错。
        CoreOpts to_core(const LoaderOpts& o)
        {
            TORCH_CHECK(o.batch_size > 0, "ImageNetLoader: batch_size 必须 > 0");
            TORCH_CHECK(o.crop > 0, "ImageNetLoader: crop 必须 > 0");
            TORCH_CHECK(o.short_side > 0, "ImageNetLoader: short_side 必须 > 0");
            if (o.workers < 0)
            {
                throw std::runtime_error("ImageNetLoader: workers 不能为负");
            }

            CoreOpts c;
            c.batch_size     = o.batch_size;
            c.workers        = o.workers;
            c.prefetch_depth = o.prefetch_depth;
            c.out_h          = o.crop;
            c.out_w          = o.crop;
            c.shuffle        = o.is_train;
            c.drop_last      = o.is_train;
            return c;
        }

        // --------------------------------------------------------------------
        // ImageNetSource —— 一阶目录 + 一类一个子目录
        // --------------------------------------------------------------------
        class ImageNetSource final : public SampleSource
        {
        public:
            ImageNetSource(fs::path root, const LoaderOpts& opt, uint64_t seed)
                : root_(std::move(root))
                , opt_(opt)
                , seed_(seed)
            {}

            void build_index(std::vector<std::string>& paths,
                             std::vector<SampleMeta>&  metas,
                             std::vector<std::string>& classes) override;

            DecodeDiag load_into(ImageBatch& out, int64_t slot,
                                 int64_t index, const std::string& path,
                                 int64_t epoch) const override;

        private:
            fs::path   root_;
            LoaderOpts opt_;    // is_train / short_side / crop 都在这
            uint64_t   seed_;   // 每样本种子 = sample_seed(seed, epoch, index)
        };

        // --------------------------------------------------------------------
        // 索引：类目录字典序 → 标签 0..N-1，类内文件名字典序
        //
        // ★ readdir 的顺序标准未规定（ext4 是哈希序、NTFS 接近字典序），
        //   所以两处都必须显式排序，否则同一份数据在两台机器上类别号不同。
        //   不排的后果：权重照样加载、loss 照样降、验证准确率掉到接近随机，而且不报错。
        // --------------------------------------------------------------------
        void ImageNetSource::build_index(std::vector<std::string>& paths,
                                         std::vector<SampleMeta>&  metas,
                                         std::vector<std::string>& classes)
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
            const auto is_image = [](const fs::path& p) {
                std::string ext = p.extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                    [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
                return ext == ".jpeg" || ext == ".jpg" || ext == ".png" ||
                       ext == ".bmp" || ext == ".ppm" || ext == ".pgm";
            };
            std::sort(dirs.begin(), dirs.end(), by_name);

            classes.clear();
            paths.clear();
            metas.clear();
            classes.reserve(dirs.size());

            for (const auto& d : dirs)
            {
                const int64_t label = static_cast<int64_t>(classes.size());
                classes.push_back(d.filename().string());

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
                    paths.push_back(f.string());
                    metas.push_back(SampleMeta{ label, -1 });   // ImageNet 没有视角属性
                }
            }

            if (paths.empty())
            {
                throw std::runtime_error("ImageNetLoader: 一个文件都没找到 " + root_.string());
            }
        }

        // --------------------------------------------------------------------
        // 读一张：解码 → 变换 → 写进批的第 slot 行
        // --------------------------------------------------------------------
        DecodeDiag ImageNetSource::load_into(ImageBatch& out, int64_t slot,
                                             int64_t index, const std::string& path,
                                             int64_t epoch) const
        {
            // 每线程一份缓冲：大小只增不减，避免每张图一次分配 + 一次清零。
            // ★ 必须传 len 而不是 tls_buf.size() —— size() 是历史最大值，
            //   多出来的那段是上一张图的残留，解码器读到也不报错。
            thread_local std::vector<uint8_t> tls_buf;
            const size_t len = read_file_into(path, tls_buf);
            const auto   bmp = decode_rgb8(tls_buf.data(), len);

            // 诊断在解码这一步就齐了；致命错误已经抛出去了，能走到这里的都是"解出来了"。
            DecodeDiag diag;
            diag.complete = bmp.decode_complete;
            diag.warnings = bmp.decode_warnings;

            auto x = to_chw01(bmp.data.data(), bmp.h, bmp.w);

            if (opt_.is_train)
            {
                // 每样本一个独立 RNG：与线程、调用顺序、epoch 都无关 ⇒ 可复现。
                // ★ 两个 random_* 必须共用这个 rng 且按此顺序调用，对应 torchvision 的 Compose。
                std::mt19937_64 rng(sample_seed(seed_, epoch, index));
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
    }  // namespace

    ImageNetLoader::ImageNetLoader(fs::path root, const LoaderOpts& opt, uint64_t seed)
        : LoaderCore(to_core(opt), seed,
                     std::make_unique<ImageNetSource>(std::move(root), opt, seed))
        , opt_(opt)
    {
        // 校验已在 to_core() 里做过（那时还没拿 crop 当 out_h 用）。
    }

    ImageNetLoader::~ImageNetLoader() = default;
}  // namespace prkd::data
