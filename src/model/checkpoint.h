// src/model/checkpoint.h
//
// 这一层只是【对接】：把 ResNet 的权重 + buffer 与 save::TensorMap 相连。
//
// 文件格式、8 字节对齐、原子写、dtype 白名单 —— 全部在 save/tensor_table.{h,cpp}，
//   「顺序」在 save/param_index.{h,cpp}。这里不重复定义任何一条，免得两处说法分叉。
//
// 为什么不用 libtorch 的 torch::serialize::OutputArchive / InputArchive：
//   那一对读写的 是【TorchScript 归档】（内部走 jit::Module，必须带 constants.pkl），
//   而 Python 的 torch.save(state_dict) 写的是 data.pkl + data/N，从不写 constants.pkl，
//   于是它报 "failed locating file constants.pkl ... your checkpoint file is corrupted"。
//   更详细的原因写在 save/tensor_table.h 顶部。
//
// 契约（跨语言兼容的那部分，改动它等于改文件格式）：
//   ① 存/取都必须带上 named_buffers()
//      以 BN 为例 running_mean / running_var / num_batches_tracked 在 buffer 里；
//      只存参数的话，模型在 eval() 下是废的 —— 而且加载时不会报错。
//   ② 顺序 = named_parameters() 全部，然后 named_buffers() 全部
//      由 save/param_index 统一提供，优化器键名也用同一份顺序。
//   ③ 键名就是 PyTorch 的原名（如 "conv1.weight"）
//      ⇒ Python 侧 export_teacher_flat.py 导出的 .txt/.bin 可以直接读。
#pragma once

#include <string>

#include "model/resnet.h"

namespace prkd::model
{
	// 写 <base>.txt（键表）+ <base>.bin（裸字节）。
	// 原子性：bin 先 rename、txt 后 ⇒ 有 .txt 就等于这一代完整；中途被杀只会留下 .tmp。
	void save_checkpoint(const ResNet& model, const std::string& base);

	// 模型的每个参数和 buffer 都必须在表里；缺键 / 字节数不符 / 偏移越界 / 偏移未 8 字节对齐都会抛。
	//
	// 返回值 = 实际搬进模型的张量个数（本项目 h_resnet18 = 122；h_resnet152 未测）。
	//   ⚠ 这个数由结构决定，不抛就必然对 —— 它不能证明文件里是什么。
	//   要看文件：键表的行数，函数里会打出来。
	//
	// 两份来源都能吃：
	//   ① C++ 自己 save_checkpoint 写的
	//   ② Python 侧 export_teacher_flat.py 导出的（跨语言那条，第一次跑就是它的验证）
	size_t load_checkpoint(ResNet& model, const std::string& base);
}
