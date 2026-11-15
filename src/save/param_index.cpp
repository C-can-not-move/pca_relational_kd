// src/save/param_index.cpp
#include "save/param_index.h"

#include <torch/torch.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace prkd::save
{
	std::vector<std::string> names(const prkd::model::ResNet& net)
	{
		std::vector<std::string> out;
		out.reserve(256);

		for (auto& p : net->named_parameters()) out.push_back(p.key());
		for (auto& b : net->named_buffers())    out.push_back(b.key());

		return out;
	}

	TensorMap dump(const prkd::model::ResNet& net)
	{
		TensorMap out;
		out.reserve(256);

		// .to(kCPU) 可能造出临时张量；emplace_back 按值收下，保证活到写完
		for (auto& p : net->named_parameters())
			out.emplace_back(p.key(), p.value().detach().to(torch::kCPU).contiguous());
		for (auto& b : net->named_buffers())
			out.emplace_back(b.key(), b.value().detach().to(torch::kCPU).contiguous());

		return out;
	}

	std::size_t restore(prkd::model::ResNet& net, const TensorMap& items)
	{
		std::unordered_map<std::string, const torch::Tensor*> table;
		table.reserve(items.size() * 2);
		for (const auto& kv : items) table.emplace(kv.first, &kv.second);

		// copy_ 进叶子张量必须关 autograd，否则报
		//   "a leaf Variable that requires grad is being used in an in-place operation"
		torch::NoGradGuard no_grad;

		std::size_t n = 0;
		auto pull = [&](const std::string& key, torch::Tensor& dst)
			{
				const auto it = table.find(key);
				TORCH_CHECK(it != table.end(), "表里没有 ", key);

				const auto& src = *it->second;
				const long long want = static_cast<long long>(dst.numel() * dst.element_size());
				const long long have = static_cast<long long>(src.numel() * src.element_size());
				TORCH_CHECK(have == want, key, " 字节数不符: 文件 ", have,
					" 期望 ", want, "（模型那边是 ", dst.sizes(), " ",
					torch::toString(dst.scalar_type()), "）");

				dst.copy_(src.reshape(dst.sizes()).to(dst.device()));
				++n;
			};

		for (auto& p : net->named_parameters()) pull(p.key(), p.value());
		for (auto& b : net->named_buffers())    pull(b.key(), b.value());

		return n;
	}

}  // namespace prkd::save
