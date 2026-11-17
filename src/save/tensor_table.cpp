// src/save/tensor_table.cpp
#include "save/tensor_table.h"

#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace prkd::save
{
	namespace
	{
		struct Entry
		{
			std::vector<int64_t> sizes;
			torch::ScalarType    dtype = torch::kFloat32;
			long long            off = 0;
			long long            nbytes = 0;
		};

		// 设计上只支持这四种：覆盖权重(f32/f64)、整数 buffer(i64)、掩码(u8)。
		// 需要 bf16 / fp16 时在这里加一行，同时 load 那边也会跟着支持。
		torch::ScalarType dtype_of(const std::string& s)
		{
			if (s == "f32") return torch::kFloat32;
			if (s == "f64") return torch::kFloat64;
			if (s == "i64") return torch::kInt64;
			if (s == "u8")  return torch::kUInt8;
			TORCH_CHECK(false, "键表里出现不认识的 dtype: ", s);
			return torch::kFloat32;
		}

		const char* dtype_name(torch::ScalarType t)
		{
			if (t == torch::kFloat32) return "f32";
			if (t == torch::kFloat64) return "f64";
			if (t == torch::kInt64)   return "i64";
			if (t == torch::kUInt8)   return "u8";
			TORCH_CHECK(false, "不支持的 dtype: ", torch::toString(t),
				"（要支持它，请改 save/tensor_table.cpp 的 dtype_name / dtype_of）");
			return "";
		}

		std::vector<std::pair<std::string, Entry>> read_table(const std::string& txt)
		{
			std::ifstream f(txt);
			TORCH_CHECK(f, "打不开键表: ", txt);

			std::vector<std::pair<std::string, Entry>> rows;
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
				if (shape != "0")
				{
					std::stringstream ss(shape);
					std::string d;
					while (std::getline(ss, d, ',')) e.sizes.push_back(std::stoll(d));
				}
				rows.emplace_back(key, std::move(e));
			}
			TORCH_CHECK(!rows.empty(), "键表是空的: ", txt);
			return rows;
		}
	}  // namespace

	void save(const TensorMap& items, const std::string& base, const std::vector<std::string>& extra_header)
	{
		TORCH_CHECK(!items.empty(), "save: 表是空的 -> ", base);

		const auto parent = std::filesystem::path(base + ".txt").parent_path();
		if (!parent.empty()) std::filesystem::create_directories(parent);

		const std::string tmp_txt = base + ".txt.tmp";
		const std::string tmp_bin = base + ".bin.tmp";

		std::ofstream txt(tmp_txt, std::ios::binary);
		std::ofstream bin(tmp_bin, std::ios::binary);
		TORCH_CHECK(txt && bin, "写不了 ", tmp_txt, " / ", tmp_bin);

		txt << "# name dtype shape off nbytes\n";
		// 顺序留证：读的人一眼能看出 p0000 对应哪个参数。
		// 以 # 开头，read_table 会跳过，不影响解析。
		for (const auto& line : extra_header) txt << "# " << line << '\n';

		long long cur = 0;
		for (const auto& kv : items)
		{
			const auto t = kv.second.detach().to(torch::kCPU).contiguous();
			const long long nbytes = static_cast<long long>(t.numel() * t.element_size());
			TORCH_CHECK(nbytes > 0, "空张量不允许: ", kv.first);

			// 8 字节对齐：load 时 from_blob 拿到的指针是齐的。
			// ⚠ 必须写 (8 - cur % 8) % 8 —— C++ 的 % 对负数向零取整，
			//   写成 (-cur) % 8 会得到负数，pad 永远不生效。
			const long long pad = (8 - cur % 8) % 8;
			for (long long i = 0; i < pad; ++i) { bin.put('\0'); ++cur; }

			std::string shape;
			for (auto d : t.sizes()) { if (!shape.empty()) shape += ','; shape += std::to_string(d); }
			if (shape.empty()) shape = "0";

			txt << kv.first << ' ' << dtype_name(t.scalar_type()) << ' ' << shape
				<< ' ' << cur << ' ' << nbytes << '\n';

			bin.write(static_cast<const char*>(t.data_ptr()), nbytes);
			cur += nbytes;
		}

		// flush 之后必须验状态：ofstream 默认不抛，磁盘满时只置 failbit
		bin.flush(); txt.flush();
		TORCH_CHECK(bool(txt) && bool(bin), "写 ", base, " 时磁盘出错（满了？）");
		txt.close(); bin.close();

		std::error_code ec;
		std::filesystem::rename(tmp_bin, base + ".bin", ec);
		TORCH_CHECK(!ec, "rename 失败: ", tmp_bin, " -> ", base, ".bin (", ec.message(), ")");
		std::filesystem::rename(tmp_txt, base + ".txt", ec);
		TORCH_CHECK(!ec, "rename 失败: ", tmp_txt, " -> ", base, ".txt (", ec.message(), ")");

		std::cout << "[io] saved " << items.size() << " tensors -> " << base
			<< ".{txt,bin}   (" << cur << " B)\n";
	}

	TensorMap load(const std::string& base)
	{
		const auto rows = read_table(base + ".txt");

		std::ifstream bf(base + ".bin", std::ios::binary);
		TORCH_CHECK(bf, "打不开 ", base, ".bin");
		std::vector<char> blob((std::istreambuf_iterator<char>(bf)), std::istreambuf_iterator<char>());
		TORCH_CHECK(!blob.empty(), "文件是空的: ", base, ".bin");

		TensorMap out;
		out.reserve(rows.size());
		for (const auto& r : rows)
		{
			const auto& e = r.second;
			TORCH_CHECK(e.off >= 0 && e.off + e.nbytes <= static_cast<long long>(blob.size()),
				r.first, " 越界: ", e.off, '+', e.nbytes, " > ", blob.size());
			TORCH_CHECK(e.off % 8 == 0, r.first, " 偏移未 8 字节对齐: ", e.off);

			// from_blob 只包一层视图，blob 出了作用域就悬空 ⇒ 必须 clone
			auto view = torch::from_blob(blob.data() + e.off, e.sizes,
				torch::TensorOptions().dtype(e.dtype));
			out.emplace_back(r.first, view.clone());
		}

		std::cout << "[io] loaded " << out.size() << " tensors <- " << base << ".{txt,bin}\n";
		return out;
	}

	std::vector<std::string> keys(const std::string& base)
	{
		std::vector<std::string> out;
		for (const auto& r : read_table(base + ".txt")) out.push_back(r.first);
		return out;
	}

}  // namespace prkd::save
