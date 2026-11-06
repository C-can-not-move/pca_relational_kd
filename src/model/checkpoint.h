// src/model/checkpoint.h
//
// 从 DKD_C++ 搬来，只改了命名空间。契约与格式完全一致：
//   同一个 tool/export_teacher_flat.py 产出的 .txt + .bin 两边都能读。
#pragma once

#include <string>

#include "model/resnet.h"

namespace prkd::model
{
	// 一份 checkpoint = 两个文件，共用同一个 base：
	//   <base>.txt   键表，一行一个张量：  name dtype shape off nbytes
	//   <base>.bin   裸字节，所有张量按 8 字节对齐首尾相接
	//
	// 为什么不直接用 libtorch 的 torch::serialize::OutputArchive / InputArchive：
	//   那一对是【TorchScript 归档】的读写器 —— InputArchive 的私有成员就是
	//   jit::Module module_ + hierarchy_prefix_，load_from() 直接转给 torch::jit::load()，
	//   而 TorchScript 归档必须带 constants.pkl。
	//   Python 的 torch.save(state_dict) 写的是 data.pkl + data/N，从来没写过 constants.pkl，
	//   于是它报 "failed locating file constants.pkl ... your checkpoint file is corrupted"。
	//   文件没坏，是读档器不对口；而 Python 侧没有 API 能写出 TorchScript 式的权重表。
	//
	// 所以这里自己定契约：文本键表 + 裸字节。torch 换版本不影响它，
	//   而且 .txt 用记事本打开就知道里面装了什么。
	//
	// 存/取都必须带上 named_buffers()：
	//   以 BN 为例，running_mean / running_var / num_batches_tracked 都在 buffer 里。
	//   只存参数的话，模型在 eval() 下是废的 —— 而且加载时不会报错。
	void save_checkpoint(const ResNet& model, const std::string& base);

	// 模型的每个参数和 buffer 都必须在键表里；缺键 / 字节数不符 / 偏移越界都会抛。
	//
	// 返回值 = 实际搬进模型的张量个数（resnet8x4 = 62，resnet32x4 = 206）。
	//   ⚠ 这个数由结构决定，不抛就必然对 —— 它不能证明文件里是什么。
	//   要看文件：键表的行数，函数里会打出来。
	//
	// 两份来源都能吃：
	//   ① C++ 自己 save_checkpoint 写的
	//   ② Python 侧 export_teacher_flat.py 导出的（跨语言那条，第一次跑就是它的验证）
	size_t load_checkpoint(ResNet& model, const std::string& base);
}
