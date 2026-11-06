// src/model/checkpoint.cpp
#include "model/checkpoint.h"

#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <filesystem>

namespace prkd::model
{
	namespace
	{
		// 键表的一行 = 一个张量在 .bin 里的位置和解释方式
		struct Entry
		{
			std::vector<int64_t> sizes;                    // 空 = 零维张量
			torch::ScalarType    dtype = torch::kFloat32;
			long long            off = 0;                  // 相对 .bin 开头的字节偏移
			long long            nbytes = 0;
		};

		torch::ScalarType dtype_of(const std::string& s)
		{
			if (s == "f32") return torch::kFloat32;
			if (s == "f64") return torch::kFloat64;
			if (s == "i64") return torch::kInt64;
			if (s == "u8")  return torch::kUInt8;
			TORCH_CHECK(false, "键表里出现不认识的 dtype: ", s);
			return torch::kFloat32;                        // 给编译器看的，TORCH_CHECK 已经抛了
		}

		const char* dtype_name(torch::ScalarType t)
		{
			if (t == torch::kFloat32) return "f32";
			if (t == torch::kFloat64) return "f64";
			if (t == torch::kInt64)   return "i64";
			if (t == torch::kUInt8)   return "u8";
			TORCH_CHECK(false, "不支持的 dtype: ", torch::toString(t));
			return "";
		}

		// 参数 + buffer 摊平成一张有序表；顺序就是写的顺序，也是读的顺序。
		std::vector<std::pair<std::string, torch::Tensor>> flatten(const ResNet& model)
		{
			std::vector<std::pair<std::string, torch::Tensor>> out;

			// .to(kCPU) 可能造出临时张量，emplace_back 按值收下，保证活到写完
			for (auto& p : model->named_parameters())
				out.emplace_back(p.key(), p.value().detach().to(torch::kCPU).contiguous());
			for (auto& b : model->named_buffers())
				out.emplace_back(b.key(), b.value().detach().to(torch::kCPU).contiguous());

			return out;
		}

		std::unordered_map<std::string, Entry> read_table(const std::string& txt)
		{
			std::ifstream f(txt);
			TORCH_CHECK(f, "打不开键表: ", txt);

			std::unordered_map<std::string, Entry> table;
			std::string line;
			while (std::getline(f, line))
			{
				if (line.empty() || line[0] == '#') continue;

				std::istringstream ls(line);
				std::string key, dt, shape;
				long long off = 0, nbytes = 0;
				ls >> key >> dt >> shape >> off >> nbytes;
				TORCH_CHECK(bool(ls), "键表这一行看不懂: ", line);

				Entry e;
				e.dtype = dtype_of(dt);
				e.off = off;
				e.nbytes = nbytes;
				if (shape != "0")                          // "0" = 零维，sizes 留空
				{
					std::stringstream ss(shape);
					std::string d;
					while (std::getline(ss, d, ',')) e.sizes.push_back(std::stoll(d));
				}

				const std::string k = key;
				table.emplace(k, std::move(e));
			}

			TORCH_CHECK(!table.empty(), "键表是空的: ", txt);
			return table;
		}
	}  // namespace

	void save_checkpoint(const ResNet& model, const std::string& base)
	{
		const auto tensors = flatten(model);

		// ofstream 不会建目录 —— ./run 不存在就直接失败。这一步交给保存方，别让调用者各建一次。
		const auto parent = std::filesystem::path(base + ".txt").parent_path();
		if (!parent.empty()) std::filesystem::create_directories(parent);

		std::ofstream txt(base + ".txt", std::ios::binary);
		std::ofstream bin(base + ".bin", std::ios::binary);
		TORCH_CHECK(txt && bin, "写不了 ", base, ".txt / .bin");

		txt << "# name dtype shape off nbytes\n";

		long long cur = 0;
		for (const auto& kv : tensors)
		{
			const auto& key = kv.first;
			const auto& t = kv.second;

			// 8 字节对齐：下次读进来 from_blob 的指针是齐的。
			// ⚠ 必须写 (8 - cur % 8) % 8 —— C++ 的 % 对负数向零取整，
			//   写成 (-cur) % 8 会得到负数，pad 永远不生效。
			const long long pad = (8 - cur % 8) % 8;
			for (long long i = 0; i < pad; ++i) { bin.put('\0'); ++cur; }

			std::string shape;
			for (auto d : t.sizes()) { if (!shape.empty()) shape += ','; shape += std::to_string(d); }
			if (shape.empty()) shape = "0";

			const long long nbytes = t.numel() * t.element_size();
			txt << key << ' ' << dtype_name(t.scalar_type()) << ' ' << shape
				<< ' ' << cur << ' ' << nbytes << '\n';

			bin.write(static_cast<const char*>(t.data_ptr()), nbytes);
			cur += nbytes;
		}

		bin.flush();
		std::cout << "[ckpt] saved " << tensors.size() << " tensors -> "
			<< base << ".{txt,bin}   (" << cur << " B)\n";
	}

	size_t load_checkpoint(ResNet& model, const std::string& base)
	{
		const auto table = read_table(base + ".txt");

		std::ifstream bf(base + ".bin", std::ios::binary);
		TORCH_CHECK(bf, "打不开权重: ", base, ".bin");
		std::vector<char> blob((std::istreambuf_iterator<char>(bf)),
			std::istreambuf_iterator<char>());
		TORCH_CHECK(!blob.empty(), "权重文件是空的: ", base, ".bin");

		// copy_ 进叶子张量必须关 autograd，否则报
		//   "a leaf Variable that requires grad is being used in an in-place operation"
		torch::NoGradGuard no_grad;

		size_t n = 0;
		auto pull = [&](const std::string& key, torch::Tensor& dst)
			{
				const auto it = table.find(key);
				TORCH_CHECK(it != table.end(), "键表里没有 ", key);
				const auto& e = it->second;

				// 用【字节数】对账，不用形状 —— 零维和 [1] 的 numel 都是 1、字节数都是 8，
				// 这样 num_batches_tracked 那个 0 维坑自动被兜住。
				const long long want = static_cast<long long>(dst.numel() * dst.element_size());
				TORCH_CHECK(e.nbytes == want, key, " 字节数不符: 文件 ", e.nbytes,
					" 期望 ", want, "（模型那边是 ", dst.sizes(), " ",
					torch::toString(dst.scalar_type()), "）");
				TORCH_CHECK(e.off >= 0 && e.off + e.nbytes <= static_cast<long long>(blob.size()),
					key, " 越界: ", e.off, '+', e.nbytes, " > ", blob.size());

				// from_blob 不复制，只是包一层视图 —— 真正的搬运在下面的 copy_
				const auto src = torch::from_blob(blob.data() + e.off, e.sizes,
					torch::TensorOptions().dtype(e.dtype));

				dst.copy_(src.reshape(dst.sizes()));
				++n;
			};

		for (auto& p : model->named_parameters()) pull(p.key(), p.value());
		for (auto& b : model->named_buffers())    pull(b.key(), b.value());

		std::cout << "[ckpt] loaded " << n << " tensors <- " << base << ".{txt,bin}"
			<< "   (键表 " << table.size() << " 行)\n";
		return n;
	}

}  // namespace prkd::model
