// src/train/trainer.h —— 最简训练循环。一次一个 epoch，损失写死在函数里。
//
// 与 DKD_C++ 那版的差别只有一处：那边数据是一整块张量、自己洗牌切片；
// 这边有 loader，batch 由 loader 交付，所以循环里没有 randperm、没有 index_select。
//
// 刻意不做的：
//   · 不抽象 Loss 接口 —— 现在只有交叉熵，写死比多一层基类清楚
//   · 不做 lr scheduler —— 要衰减时直接改 optimizer 的 param_group
//   · 不存 best —— 存不存、存哪儿由调用方决定
#pragma once

#include "data/loader_core.h"
#include "model/resnet.h"

#include <torch/torch.h>
#include <torch/optim.h>

#include <cstdint>
#include <string>

namespace prkd::train
{
    struct TrainConfig
    {
        int64_t log_every = 20;      // 每多少个 batch 打一行；0 = 不打
    };

    // 一个 epoch 的结果。（复数，与 LoaderStats 一致：它是一组统计，不是一条）
    struct EpochStats
    {
        double  loss = 0.0;          // 批平均 loss
        int64_t correct = 0;
        int64_t total = 0;           // 实际参与计算的样本数（末批可能短）
        int64_t batches = 0;
        double  seconds = 0.0;       // 含取数据 + 前向 + 反向

        double acc() const
        {
            return total > 0 ? static_cast<double>(correct) / static_cast<double>(total) : 0.0;
        }
        double img_per_sec() const
        {
            return seconds > 0 ? static_cast<double>(total) / seconds : 0.0;
        }
    };

    // 跑一个 epoch 的【训练】。
    //
    // 前向 → cross_entropy → backward → step，四步，中间不插别的东西。
    // 损失写死在这里：这个项目当前只做分类，交叉熵够用。
    //
    // 设备：数据在 loader 里是 CPU float32，每批 .to(device) 搬一次。
    //       模型必须已经在 device 上，否则第一句就报"设备不一致"。
    EpochStats train_one_epoch(model::ResNet& model,
                              data::LoaderCore& ld,
                              torch::optim::Optimizer& optimizer,
                              const torch::Device& device,
                              const TrainConfig& cfg);

    // 跑一个 epoch 的【评估】：不建图、不改权重、不做增强（增强由 loader 的开关决定）。
    EpochStats evaluate(model::ResNet& model,
                       data::LoaderCore& ld,
                       const torch::Device& device,
                       const TrainConfig& cfg);

}  // namespace prkd::train
