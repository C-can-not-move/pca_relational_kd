// src/save/tensor_table.h
//
// 「任意名字 → 张量」表的落盘格式。整个项目只有这一处定义格式。
//
//   <base>.txt  文本键表，一行一个张量：  name dtype shape off nbytes
//               shape 用逗号分隔；零维张量写 "0"
//   <base>.bin  裸字节，所有张量按 8 字节对齐首尾相接
//
// 为什么不用 torch::serialize::OutputArchive：
//   那一对是 TorchScript 归档的读写器，必须带 constants.pkl；
//   Python 的 torch.save(state_dict) 从不写它。
//   所以这里自己定契约：文本键表 + 裸字节，换 torch 版本也不影响，记事本能打开。
//
// 原子写：先写 .tmp，验 ofstream 状态后 rename（bin 先、txt 后）。
//   ⇒ 有 .txt 就等于这一份完整；中途被杀只会留下 .tmp。
//
// 谁在用：
//   权重      model/checkpoint.cpp（ResNet 专用，附带跨语言兼容的契约）
//   优化器    train/optim_slots.*（键名 p%04d.<field>）
//   任何别的  直接调 save/load
#pragma once

#include <string>
#include <utility>
#include <vector>

#include <torch/torch.h>

namespace prkd::save
{
	// 名字必须唯一；张量可以是任意 device / 非连续，save 里会转 CPU 连续。
	// 空张量（numel()==0）不允许 —— 键表里没法表达"零字节"。
	using TensorMap = std::vector<std::pair<std::string, torch::Tensor>>;

	// extra_header 会以 "# …" 形式追加在标准表头之后（读的时候被跳过），
	// 用来记"这张表里的 p0000 是哪个参数"之类的顺序信息。
	void save(const TensorMap& items, const std::string& base,
		const std::vector<std::string>& extra_header = {});

	// 读回来全部在 CPU、已 detach、已 clone（调用方拥有内存）。
	// 返回顺序 = 文件里的行顺序。
	TensorMap load(const std::string& base);

	// 只读键表，不读 .bin。用于"加载前先核对有哪些键"。
	std::vector<std::string> keys(const std::string& base);
}
