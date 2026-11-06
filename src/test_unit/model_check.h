// src/test_unit/model_check.h —— 网络结构的验收检查。
//
// 与 loader_check 同一套约定：不引入测试框架，失败就 throw，main 按顺序调用。
// 刻意不返回 bool —— 那样每个调用点都可能忘了写 if，检查就静默失效。
//
// 为什么单独成文件：前向/反向的正确性要在改 resnet 之后反复跑。
// 混在 main 里，早晚会被当成一次性代码删掉。
#pragma once

#include <cstdint>

#include "model/resnet.h"

namespace prkd::test_unit
{
    // 参数量。结构写错一处，这个数立刻不对 —— 它是整条链上最便宜也最有效的指纹。
    int64_t count_parameters(const model::ResNet& net);

    // 前向验收：
    //   ① 参数量 == expect_params（写错结构时最先暴露）
    //   ② 输出形状 (batch, num_classes)
    //   ③ 输出全部有限（NaN/Inf 说明 BN 或初始化炸了）
    //
    // ★ 按值传 holder，与 DKD 的 train_classifier 一致。
    //   ModuleHolder 没有 const 版 operator->，所以 const ResNet& 调不动 forward。
    //   拷贝只是 shared_ptr 引用计数 +1，不复制模型。
    void check_forward(
        model::ResNet net,
        int64_t batch,
        int64_t num_classes,
        int64_t expect_params);

    // 反向验收：
    //   ① 每个 requires_grad 的参数都拿到了 defined 的 grad
    //   ② 所有梯度有限
    //   ③ 不是全零 —— 全零说明计算图和 loss 断了，但训练会照常「跑」
    //
    // ★ 按值传的 net 和调用方【指向同一个 Impl】，所以 .grad 会落到原对象上。
    //   函数末尾 zero_grad() 收尾，不把测试用的梯度留给调用方。
    void check_backward(
        model::ResNet net,
        int64_t batch,
        int64_t num_classes);

    // 把上面两项都跑一遍，打印每步结果。
    void run_model_checks();
}
