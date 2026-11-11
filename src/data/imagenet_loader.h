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
    // LoaderOpts —— 这一次装载怎么读
    //
    // 前四项直接映射到 CoreOpts；后三项只影响 ImageNetSource 内部的变换分支。
    // ------------------------------------------------------------------------
    struct LoaderOpts
    {
        int64_t batch_size = 64;
        int64_t workers = 0;        // 0 = 同步单线程；>0 = 起这么多个读图线程
        int64_t prefetch_depth = 2; // 池深：同时几个批在飞。1 = 退化成同步等齐

        int64_t short_side = 256;   // Resize(短边) —— 只对 is_train=false 生效
        int64_t crop = 224;         // CenterCrop / RRC 的输出边长（也是 out_h/out_w）

        /*
        true  = RandomResizedCrop(crop) + RandomHorizontalFlip，且每 epoch 洗牌、drop_last
        false = Resize 短边 short_side + CenterCrop(crop)，且不洗牌（顺序固定）、不丢尾巴
        */
        bool is_train = true;
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
