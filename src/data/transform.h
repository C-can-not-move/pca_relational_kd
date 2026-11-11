// src/data/transform.h —— 图像变换原语。一个名字一件事，与 torchvision 同名。
//
// 工作对象是 CHW 的 float32 张量（不是像素指针）。只有 to_chw01 收 HWC 的 uint8。
// 一旦拆成原语，它们就该在张量层 —— 否则每个函数都要重复判断 HWC 还是 CHW。
//
// 口径：对齐 torchvision 的【tensor 路径】
//   interpolate(mode=bilinear, align_corners=false, antialias=true)
// ★ antialias 不能省。SVDKD 那份没有它，那正是 PIL 口径与 ATen 口径差 4/255 的来源。
//
// 两条取整（都复刻 Python 的语义；写错不报错，只让图平移一列）：
//   · resize 的长边：int()，向零截断
//   · center_crop 的偏移：round()，银行家舍入（.5 向偶数）
//
// ⚠ 组合顺序由调用方保证，尤其是 random_resized_crop 与 random_horizontal_flip
//   必须共用同一个 rng 且按此顺序调用 —— 那对应 torchvision 的 Compose 顺序。
#pragma once

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace prkd::data
{
    // ------------------------------------------------------------------------
    // 归一化参数
    //
    // 用 std::vector 不写死 std::array<float,3>：将来换模型、通道数变化时不用改签名。
    // 
    // ------------------------------------------------------------------------
    struct NormStats
    {
        std::vector<float> mean;
        std::vector<float> std;
    };

    // Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    NormStats imagenet_norm();

    // HWC uint8 → CHW float32 / 255（ToTensor 的两半：转置 + 缩放）
    torch::Tensor to_chw01(const uint8_t* hwc, int64_t h, int64_t w);

    // Resize(short_side)：短边对齐，长边按 int() 截断
    torch::Tensor resize(const torch::Tensor& chw, int64_t short_side);

    // Resize to (out_h, out_w)：直接拉伸到目标尺寸，不保持长宽比。
    //
    // 与 resize 的区别：resize 只定短边、长边跟着比例走；这里两边都定死。
    // 用在 ALOI（384×288 → 224×224），横向压掉约 25%。
    // 形变可接受：所有样本同一个变换，PCA 的主轴判断不受影响；
    // 真正要避免的是"只对一部分样本做不同处理"。
    torch::Tensor resize_to(const torch::Tensor& chw, int64_t out_h, int64_t out_w);

    // CenterCrop(out_h, out_w)：纯切片，偏移用银行家舍入
    torch::Tensor center_crop(const torch::Tensor& chw, int64_t out_h, int64_t out_w);

    // RandomResizedCrop(out_size, scale=(0.08,1.0), ratio=(3/4,4/3))
    // 框在【原图】上取 —— 这是它和「先缩到 256 再裁」的根本区别
    torch::Tensor random_resized_crop(const torch::Tensor& chw, int64_t out_size,std::mt19937_64& rng);

    // RandomHorizontalFlip(p)
    torch::Tensor random_horizontal_flip(const torch::Tensor& chw, std::mt19937_64& rng,double p = 0.5);

    // Normalize：(x - mean) / std，逐通道广播
    torch::Tensor normalize(const torch::Tensor& chw, const NormStats& stats);

    // 反向：x * std + mean。给可视化和调试用，不给数据管线用
    torch::Tensor denormalize(const torch::Tensor& chw, const NormStats& stats);
}