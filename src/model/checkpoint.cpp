// src/model/checkpoint.cpp
//
// 这一层只做一件事：把 ResNet 的权重 + buffer 与 save::TensorMap 之间对接。
// 文件格式、对齐、原子写都在 save/tensor_table.* 里，这里不重复实现。
#include "model/checkpoint.h"

#include <torch/torch.h>

#include <iostream>
#include <string>

#include "save/param_index.h"
#include "save/tensor_table.h"

namespace prkd::model
{
	void save_checkpoint(const ResNet& model, const std::string& base)
	{
		const auto tensors = save::dump(model);   // 顺序由 save/param_index 定义
		save::save(tensors, base);
		std::cout << "[ckpt] " << tensors.size() << " tensors -> " << base << ".{txt,bin}\n";
	}

	size_t load_checkpoint(ResNet& model, const std::string& base)
	{
		const auto tensors = save::load(base);    // 已在 CPU、已 clone
		const size_t n = save::restore(model, tensors);
		std::cout << "[ckpt] " << n << " tensors <- " << base << ".{txt,bin}"
			<< "   (键表 " << tensors.size() << " 行)\n";
		return n;
	}

}  // namespace prkd::model
