#include "data/transform.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace prkd::data
{
    namespace
    {
        // torchvision RandomResizedCrop 的默认值。要改就改这里，不放进函数签名 ——
        // 目前没有人会改它们。
        constexpr double kScaleMin = 0.08;
        constexpr double kScaleMax = 1.00;
        constexpr double kRatioMin = 3.0 / 4.0;
        constexpr double kRatioMax = 4.0 / 3.0;

        // 复刻 Python 的 round()：银行家舍入。
        // std::nearbyint 用当前浮点舍入模式，默认 FE_TONEAREST 就是 half-to-even；
        // std::round 是 half-away-from-zero，两者在 .5 上差 1，会让裁剪平移一列。
        inline int64_t py_round(double v)
        {
            return static_cast<int64_t>(std::nearbyint(v));
        }

        // (C,) -> (1,C,1,1)，用于逐通道广播
        torch::Tensor channel_tensor(const std::vector<float>& v, const torch::TensorOptions& opt)
        {
            const int64_t c = static_cast<int64_t>(v.size());
            auto t = torch::empty({ 1, c, 1, 1 }, opt);
            for (int64_t i = 0; i < c; ++i)
            {
                t.select(1, i).fill_(v[static_cast<size_t>(i)]);
            }
            return t;
        }

        // (C,H,W) -> (C,th,tw)，bilinear + antialias
        torch::Tensor bilinear(const torch::Tensor& chw, int64_t th, int64_t tw)
        {
            return torch::nn::functional::interpolate(
                chw.unsqueeze(0),
                torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{ th, tw })
                .mode(torch::kBilinear)
                .align_corners(false)
                .antialias(true)).squeeze(0);
        }

        // Resize(短边→target)：返回 {new_h, new_w}
        // 长边 = target * 长 / 短，取 int()（向零截断）—— 不是四舍五入
        std::pair<int64_t, int64_t> resized_hw(int64_t h, int64_t w, int64_t target)
        {
            const bool w_le_h = (w <= h);
            const int64_t sh = w_le_h ? w : h;
            const int64_t lo = w_le_h ? h : w;
            const int64_t nl = static_cast<int64_t>(
                static_cast<double>(target) * static_cast<double>(lo) / static_cast<double>(sh));
            return w_le_h ? std::pair<int64_t, int64_t>{ nl, target }
                          : std::pair<int64_t, int64_t>{ target, nl };
        }

        struct CropBox { int64_t i = 0, j = 0, h = 0, w = 0; };   // i=top, j=left

        // torchvision RandomResizedCrop.get_params 的抄本。
        // 最多重试 10 次：抽到的 area / aspect 放不进原图就重抽。
        // 10 次都不成 → 兜底中央裁（那两处 torchvision 用 //2，向下取整）。
        //
        // ★ 随机源由调用方传入：不能用 torch 的全局 RNG，那个的消耗顺序依赖线程调度。
        CropBox random_crop_box(int64_t H, int64_t W, std::mt19937_64& rng)
        {
            const double area = static_cast<double>(H) * static_cast<double>(W);
            const double log_lo = std::log(kRatioMin);
            const double log_hi = std::log(kRatioMax);
            std::uniform_real_distribution<double> uni(0.0, 1.0);

            for (int attempt = 0; attempt < 10; ++attempt)
            {
                const double target_area = area * (kScaleMin + (kScaleMax - kScaleMin) * uni(rng));
                const double aspect = std::exp(log_lo + (log_hi - log_lo) * uni(rng));

                const int64_t w = py_round(std::sqrt(target_area * aspect));
                const int64_t h = py_round(std::sqrt(target_area / aspect));

                if (w > 0 && w <= W && h > 0 && h <= H)
                {
                    std::uniform_int_distribution<int64_t> di(0, H - h);
                    std::uniform_int_distribution<int64_t> dj(0, W - w);
                    CropBox box;
                    box.i = di(rng);
                    box.j = dj(rng);
                    box.h = h;
                    box.w = w;
                    return box;
                }
            }

            const double in_ratio = static_cast<double>(W) / static_cast<double>(H);
            int64_t h = 0, w = 0;
            if (in_ratio < kRatioMin)
            {
                w = W;
                h = py_round(static_cast<double>(w) / kRatioMin);
            }
            else if (in_ratio > kRatioMax)
            {
                h = H;
                w = py_round(static_cast<double>(h) * kRatioMax);
            }
            else
            {
                w = W;
                h = H;
            }
            CropBox box;
            box.i = (H - h) / 2;
            box.j = (W - w) / 2;
            box.h = h;
            box.w = w;
            return box;
        }

        void check_chw(const torch::Tensor& x, const char* who)
        {
            TORCH_CHECK(x.dim() == 3, who, ": 期望 (C,H,W)，实际 ", x.dim(), " 维 ", x.sizes());
            TORCH_CHECK(x.is_floating_point(), who, ": 期望浮点张量，实际 ", x.scalar_type());
        }
    }  // namespace

    NormStats imagenet_norm()
    {
        return NormStats{
            { 0.485f, 0.456f, 0.406f },
            { 0.229f, 0.224f, 0.225f },
        };
    }

    torch::Tensor to_chw01(const uint8_t* hwc, int64_t h, int64_t w)
    {
        TORCH_CHECK(hwc != nullptr, "to_chw01: 像素指针为空");
        TORCH_CHECK(h > 0 && w > 0, "to_chw01: 尺寸非法 ", h, "x", w);

        // ★ from_blob 不持有内存，必须 clone —— 否则读的是已释放的缓冲区，
        //   而且大多数时候还能算对，只在偶尔崩。
        return torch::from_blob(const_cast<uint8_t*>(hwc), { h, w, 3 }, torch::kByte)
            .clone()
            .permute({ 2, 0, 1 })
            .to(torch::kFloat32)
            .div_(255.0);
    }

    torch::Tensor resize(const torch::Tensor& chw, int64_t short_side)
    {
        check_chw(chw, "resize");
        TORCH_CHECK(short_side > 0, "resize: short_side 必须 > 0");

        const auto nhw = resized_hw(chw.size(-2), chw.size(-1), short_side);
        return bilinear(chw, nhw.first, nhw.second);
    }

    torch::Tensor center_crop(const torch::Tensor& chw, int64_t out_h, int64_t out_w)
    {
        check_chw(chw, "center_crop");
        TORCH_CHECK(out_h > 0 && out_w > 0, "center_crop: 输出尺寸非法");

        const int64_t h = chw.size(-2);
        const int64_t w = chw.size(-1);
        TORCH_CHECK(out_h <= h && out_w <= w,
            "center_crop: 裁 ", out_h, "x", out_w, " 大于图像 ", h, "x", w,
            "（ImageNet 上不该发生：短边已先缩到 >= crop）");

        const int64_t top = py_round((static_cast<double>(h) - out_h) / 2.0);
        const int64_t left = py_round((static_cast<double>(w) - out_w) / 2.0);
        return chw.narrow(-2, top, out_h).narrow(-1, left, out_w);
    }

    torch::Tensor random_resized_crop(const torch::Tensor& chw, int64_t out_size,
                                      std::mt19937_64& rng)
    {
        check_chw(chw, "random_resized_crop");
        TORCH_CHECK(out_size > 0, "random_resized_crop: out_size 必须 > 0");

        const auto box = random_crop_box(chw.size(-2), chw.size(-1), rng);
        const auto cropped = chw.narrow(-2, box.i, box.h).narrow(-1, box.j, box.w);
        return bilinear(cropped, out_size, out_size);
    }

    torch::Tensor random_horizontal_flip(const torch::Tensor& chw, std::mt19937_64& rng,
                                         double p)
    {
        check_chw(chw, "random_horizontal_flip");
        std::uniform_real_distribution<double> coin(0.0, 1.0);
        return coin(rng) < p ? torch::flip(chw, { -1 }) : chw;
    }

    torch::Tensor normalize(const torch::Tensor& chw, const NormStats& stats)
    {
        check_chw(chw, "normalize");
        TORCH_CHECK(stats.mean.size() == stats.std.size(),
            "normalize: mean/std 长度不一致");
        TORCH_CHECK(static_cast<int64_t>(stats.mean.size()) == chw.size(0),
            "normalize: 通道数不一致，stats 有 ", stats.mean.size(),
            " 个，张量有 ", chw.size(0), " 个");

        const auto mean = channel_tensor(stats.mean, chw.options());
        const auto std = channel_tensor(stats.std, chw.options());
        return chw.sub(mean).div(std);
    }

    torch::Tensor denormalize(const torch::Tensor& chw, const NormStats& stats)
    {
        check_chw(chw, "denormalize");
        TORCH_CHECK(stats.mean.size() == stats.std.size(),
            "denormalize: mean/std 长度不一致");
        TORCH_CHECK(static_cast<int64_t>(stats.mean.size()) == chw.size(0),
            "denormalize: 通道数不一致");

        const auto mean = channel_tensor(stats.mean, chw.options());
        const auto std = channel_tensor(stats.std, chw.options());
        return chw.mul(std).add(mean);
    }
}  // namespace prkd::data