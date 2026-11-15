// src/save/param_index.h
//
// 「参数与 buffer 的有序表」——整个项目里只有这一处定义顺序。
//
// 为什么需要它：
//   权重文件按这个顺序写，优化器状态按"第几个参数"编号，批次指纹按 epoch 派生；
//   三处必须用同一份顺序，否则「键名对上了、顺序换了」这种错误极难发现。
//
// 顺序契约（改它等于改文件格式）：
//   named_parameters() 全部，然后 named_buffers() 全部。
//   以 BN 为例 running_mean / running_var / num_batches_tracked 在 buffer 里，
//   只存参数的话，模型在 eval() 下是废的 —— 而且加载时不会报错。
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "model/resnet.h"
#include "save/tensor_table.h"

namespace prkd::save
{
	// 按契约顺序列出所有键名（不含张量，开销极小）
	std::vector<std::string> names(const prkd::model::ResNet& net);

	// 按契约顺序导出一份可落盘的表（已 detach、CPU、连续）
	TensorMap dump(const prkd::model::ResNet& net);

	// 把表搬回模型。每个参数与 buffer 都必须有对应键，且字节数一致。
	//   用【字节数】对账，不用形状 —— 零维与 [1] 的 numel 都是 1、字节数都是 8，
	//   这样 num_batches_tracked 那个 0 维坑自动被兜住。
	// 返回值 = 实际搬进模型的张量个数（本项目 h_resnet18 = 122）。
	//   ⚠ 这个数由结构决定，不抛就必然对 —— 它不能证明文件里是什么。
	std::size_t restore(prkd::model::ResNet& net, const TensorMap& items);
}
