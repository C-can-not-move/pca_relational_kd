#include "data/aloi_loader.h"

#include "data/decode.h"      // decode_rgb8：JPEG 与 PNG 都走同一个入口
#include "data/transform.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace prkd::data
{
    namespace
    {
        namespace fs = std::filesystem;

        // --------------------------------------------------------------------
        // AloiSource —— 一阶目录：目录名是对象 id，文件名是 <id>_r<angle>.png
        //
        // 两条必须遵守的约定：
        //   ① 目录名必须全是数字 —— 这条顺便排掉 _r0（它是所有 r0 的副本，
        //      不排掉就会变成一个 1000 样本的假类别，分布立刻失衡）
        //   ② 视角按【数值】排，不按文件名排 —— 文件名的字典序是
        //      0, 10, 100, 105, …，一旦按它排，angle 就不单调了，
        //      而 angle 后面要拿去做 PCA 的对照量。
        // --------------------------------------------------------------------
        class AloiSource final : public SampleSource
        {
        public:
            AloiSource(fs::path root, const AloiOpts& opt)
                : root_(std::move(root))
            {
                TORCH_CHECK(opt.view_begin >= 0 && opt.view_end <= 360,
                            "AloiLoader: 视角范围必须在 [0,360]，收到 [",
                            opt.view_begin, ",", opt.view_end, ")");
                TORCH_CHECK(opt.view_end > opt.view_begin,
                            "AloiLoader: view_end 必须 > view_begin");

                // 基类构造函数会先调 build_index，那时派生类还没构造 ——
                // 所以 Source 必须自己留一份配置，不能引用 AloiLoader::opt_。
                opt_.view_begin     = opt.view_begin;
                opt_.view_end       = opt.view_end;
                opt_.batch_size     = opt.batch_size;
                opt_.workers        = opt.workers;
                opt_.prefetch_depth = opt.prefetch_depth;
                opt_.crop           = opt.crop;
                opt_.shuffle        = opt.shuffle;
                opt_.drop_last      = opt.drop_last;
            }

            // 注意：不是 const 成员 —— 基类的签名就是非 const（它要写 push_back 进来）。
            // 多写一个 const 不算"签名差不多"，而是另一个函数，override 会直接报错。
            void build_index(std::vector<std::string>& paths,
                             std::vector<SampleMeta>& metas,
                             std::vector<std::string>& classes) override
            {
                if (!fs::exists(root_))
                {
                    throw std::runtime_error("AloiLoader: 目录不存在 " + root_.string());
                }

                // ---- ① 数字目录，按数值升序 ----
                std::vector<std::pair<int64_t, fs::path>> objs;
                for (const auto& e : fs::directory_iterator(root_))
                {
                    if (!e.is_directory()) continue;

                    const std::string name = e.path().filename().string();
                    if (name.empty()) continue;
                    if (name.find_first_not_of("0123456789") != std::string::npos) continue;

                    objs.emplace_back(std::stoll(name), e.path());
                }
                std::sort(objs.begin(), objs.end(),
                          [](const auto& a, const auto& b) { return a.first < b.first; });

                if (objs.empty())
                {
                    throw std::runtime_error("AloiLoader: 没找到任何数字目录 " + root_.string());
                }

                classes.reserve(objs.size());
                for (const auto& obj : objs) classes.push_back(std::to_string(obj.first));

                // ---- ② 每对象：解析角度 → 过滤视角范围 → 按角度升序 ----
                for (size_t label = 0; label < objs.size(); ++label)
                {
                    const int64_t   id      = objs[label].first;
                    const fs::path& dir     = objs[label].second;
                    const std::string id_str = std::to_string(id);

                    std::vector<std::pair<int64_t, std::string>> items;

                    for (const auto& f : fs::directory_iterator(dir))
                    {
                        if (!f.is_regular_file()) continue;
                        if (f.path().extension() != ".png") continue;

                        const std::string stem = f.path().stem().string();   // 例如 "1_r0"
                        const auto us = stem.rfind('_');
                        if (us == std::string::npos) continue;
                        if (us + 2 > stem.size()) continue;

                        // 目录 id 与文件名 id 不一致 → 不猜，跳过
                        if (stem.substr(0, us) != id_str) continue;
                        if (stem[us + 1] != 'r') continue;

                        int64_t angle = 0;
                        try { angle = std::stoll(stem.substr(us + 2)); }
                        catch (const std::exception&) { continue; }

                        if (angle < opt_.view_begin || angle >= opt_.view_end) continue;

                        items.emplace_back(angle, f.path().string());
                    }

                    std::sort(items.begin(), items.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });

                    for (const auto& item : items)
                    {
                        paths.push_back(item.second);
                        metas.push_back(SampleMeta{ static_cast<int64_t>(label), item.first });
                    }
                }

                if (paths.empty())
                {
                    throw std::runtime_error(
                        "AloiLoader: 视角范围 [" + std::to_string(opt_.view_begin) + ","
                        + std::to_string(opt_.view_end) + ") 内没有任何样本");
                }
            }

            DecodeDiag load_into(ImageBatch& out, int64_t slot, int64_t index,
                                 const std::string& path, int64_t epoch) const override
            {
                (void)index;
                (void)epoch;        // 无增强 ⇒ 不需要每样本种子，也不受线程数影响

                thread_local std::vector<uint8_t> tls_buf;
                const size_t len = read_file_into(path, tls_buf);
                const auto   bmp = decode_rgb8(tls_buf.data(), len);   // PNG 在 decode 里分流

                auto x = to_chw01(bmp.data.data(), bmp.h, bmp.w);

                // 384×288（4:3）→ 224×224 直接拉伸。
                // 不裁边：ALOI 物体居中，裁掉两侧就裁掉了轮廓信息；
                // 拉伸的 14% 横向压缩对全部 72,000 张一致，不影响 PCA 的方向判定。
                x = resize_to(x, opt_.crop, opt_.crop);

                static const NormStats kNorm = imagenet_norm();
                out.images.select(0, slot).copy_(normalize(x, kNorm));

                DecodeDiag d;
                d.complete = bmp.decode_complete;
                d.warnings = bmp.decode_warnings;
                return d;
            }

        private:
            fs::path root_;
            AloiOpts opt_{};
        };

    }  // namespace

    AloiLoader::AloiLoader(fs::path root, const AloiOpts& opt, uint64_t seed)
        : LoaderCore(opt.core(), seed,
                     std::make_unique<AloiSource>(std::move(root), opt))
        , opt_(opt)
    {
        // 校验已经在 AloiSource 的构造函数里做过（那时还没建索引）。
    }

    AloiLoader::~AloiLoader() = default;
    AloiLoader::AloiLoader(AloiLoader&&) noexcept = default;
    AloiLoader& AloiLoader::operator=(AloiLoader&&) noexcept = default;

}  // namespace prkd::data
