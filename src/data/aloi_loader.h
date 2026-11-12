// src/data/aloi_loader.h —— ALOI：1000 个物体 × 72 个视角，PNG，384×288。
//
// 这个 loader 只服务一件事：把整批特征喂给 Pca，看前几根主成分抓的是
// 【物体身份】还是【视角】。所以：
//   · 不做任何增强 —— 增强会把"视角"和"随机裁剪"混在一起，PCA 分不开
//   · is_train 不参与分支 —— 这里没有训练分支可选
//   · 变换是 to_chw01 → resize_to(crop) → imagenet_norm，三步，无第四步
#pragma once

#include "data/loader_core.h"

#include <cstdint>
#include <filesystem>
#include <string>

namespace prkd::data
{
    // ------------------------------------------------------------------------
    // AloiOpts —— 共用部分继承 LoaderCommonOpts，这里只加视角范围
    //
    // 左闭右开：view_begin <= angle < view_end
    //   半圈训练/测试：{0, 180} 与 {180, 360}
    //   全量看 PCA  ：{0, 360}
    //
    // 视角范围就是 ALOI 的"split" —— 它没有 /train 与 /val 目录，
    // 所以划分只能靠这两个数，而不是目录名。
    // ------------------------------------------------------------------------
    struct AloiOpts : LoaderCommonOpts
    {
        int64_t view_begin = 0;     // 含
        int64_t view_end   = 360;   // 不含
    };

    class AloiLoader : public LoaderCore
    {
    public:
        AloiLoader(std::filesystem::path root, const AloiOpts& opt, uint64_t seed);
        ~AloiLoader() override;

        AloiLoader(AloiLoader&&) noexcept;
        AloiLoader& operator=(AloiLoader&&) noexcept;
        AloiLoader(const AloiLoader&) = delete;
        AloiLoader& operator=(const AloiLoader&) = delete;

        // 与 ImageNetLoader 同名同用：调用方问 loader 要"它实际在用的那套"。
        const AloiOpts& options() const { return opt_; }

    private:
        AloiOpts opt_;
    };

}  // namespace prkd::data
