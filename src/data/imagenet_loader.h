// src/data/imagenet_loader.h —— ImageNet/Tiny-ImageNet 的目录约定
//
// 只用两类东西：
//   LoaderOpts     这一次装载怎么读（对外接口，调用点用到的就是它）
//   ImageNetLoader 薄壳：把 LoaderOpts 翻成 CoreOpts，把"策略"交给 ImageNetSource
//
// 机器（索引迭代、批边界、预取池、并发交付、断点）全部在 loader_core.h 里。
// 本类不再声明 size()/next()/seek()/stats() 等 —— 它们从 LoaderCore 继承，
// 所以 pca_relational_kd.cpp 与 loader_check.cpp 的调用一行不用改。
//
// C++20
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <string>

#include "data/loader_core.h"   // 共享类型 + SampleSource + LoaderCore

namespace prkd::data
{
    // ------------------------------------------------------------------------
    // LoaderOpts —— ImageNet 的这一次装载怎么读
    //
    // 共用部分（batch/workers/prefetch/crop/shuffle/drop_last）从 LoaderCommonOpts 继承，
    // core() 也只在那边写一次。这里只加 ImageNet 特有的两项：
    //   short_side  评测时 Resize 的短边
    //   is_train    只决定变换分支（增强 vs 中心裁剪），不再绑 shuffle/drop_last
    // ------------------------------------------------------------------------
    struct LoaderOpts : LoaderCommonOpts
    {
        int64_t short_side = 256;   // Resize(短边) —— 只对 is_train=false 生效
        bool    is_train   = true;  // true = RRC + flip；false = resize + center_crop
    };

    // ------------------------------------------------------------------------
    // ImageNetLoader
    // ------------------------------------------------------------------------
    class ImageNetLoader : public LoaderCore
    {
    public:
        // 构造签名与改造前完全一致 —— 调用点零改动。
        // 以下任一情况抛 std::runtime_error：
        //   · root 不存在 / 没有类目录 / 类目录里没有图片
        //   · SampleSource 给的 paths 与 metas 长度不一致
        //   · is_train 且样本数 < batch_size（drop_last 会得到 0 个 batch）
        ImageNetLoader(std::filesystem::path root, const LoaderOpts& opt, uint64_t seed);

        ~ImageNetLoader() override;

        // 基类已删除拷贝；这里保留移动（有线程在飞时不该移动，见 loader_core.cpp 注释）
        ImageNetLoader(const ImageNetLoader&) = delete;
        ImageNetLoader& operator=(const ImageNetLoader&) = delete;
        ImageNetLoader(ImageNetLoader&&) noexcept = default;
        ImageNetLoader& operator=(ImageNetLoader&&) noexcept = default;

        // ---- 只有本类特有的两个查询 ----
        // options() 返回的是 LoaderOpts（不是 CoreOpts）—— loader_check.cpp 依赖它。
        const LoaderOpts& options() const { return opt_; }
        bool              is_train() const { return opt_.is_train; }

    private:
        // 逻辑配置的一份副本，给 options() 用。
        // ★ ImageNetSource 内部另存了一份（基类构造时派生成员还不存在）。
        LoaderOpts opt_;
    };
}  // namespace prkd::data
