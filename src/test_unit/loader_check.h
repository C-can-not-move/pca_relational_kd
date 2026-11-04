// src/test_unit/loader_check.h —— 数据管线的验收检查。
//
// 不引入任何测试框架：每条检查失败就 throw，main 只负责按顺序调用。
// 刻意不返回 bool —— 那样每个调用点都可能忘了写 if，
// 检查就静默失效了，比没有检查更坑。
//
// 为什么单独成文件：这些断言以后要在改 loader / transform / decode 之后反复跑。
// 混在 main 里，早晚会被当成一次性代码删掉。
#pragma once

#include <cstdint>
#include <string>

#include "data/imagenet_loader.h"

namespace prkd::test_unit
{
    // 索引规模：类数 / 样本数 / batch_size / num_batches 与期望一致。
    void check_index(const data::ImageNetLoader& ld);

    // 单批的形状、dtype、标签范围、有限性。
    // 每批都调 —— 代价是几次元数据比较，相对读一张图可以忽略。
    void check_batch(const data::ImageBatch& b, const data::ImageNetLoader& ld, int64_t k);

    // workers=0 与 workers=N 产出的前 n_batches 批是否【逐元素相同】。
    //
    // 只比 indices 不够：copy_ 写错 slot 时下标照样对，图像却是错的，
    // 而训练会照常跑、loss 照常降 —— 静默错误。
    //
    // 自己建两个 loader，不碰任何外部实例的 cursor。
    void check_workers_equivalence(const std::string& data_dir,
                                   const data::LoaderOpts& opt,
                                   uint64_t seed,
                                   int64_t n_batches);

    // 跑满一个 epoch：每批 check_batch，前 3 批打印，最后报吞吐。
    void run_one_epoch(data::ImageNetLoader& ld);

    // 前 5 个 indices / labels + 这一批的 min/max。只用于人眼核对顺序可复现。
    void print_batch_head(const data::ImageBatch& b, int64_t k);

    // 解码体检：干净 / 有警告 / 数据不全，三者互斥。
    void print_decode_health(const data::LoaderStats& st);
}
