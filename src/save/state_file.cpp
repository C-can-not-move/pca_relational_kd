// src/save/state_file.cpp
#include "save/state_file.h"

#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>

namespace prkd::save
{
	namespace
	{
		constexpr const char* kFormat = "pca-relational-kd/state-1";

		std::string read_all(const std::string& path)
		{
			std::ifstream f(path, std::ios::binary);
			TORCH_CHECK(f, "打不开 ", path);
			return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
		}

		void write_atomic(const std::string& path, const std::string& content)
		{
			const std::string tmp = path + ".tmp";
			{
				std::ofstream f(tmp, std::ios::binary);
				TORCH_CHECK(f, "写不了 ", tmp);
				f.write(content.data(), static_cast<std::streamsize>(content.size()));
				f.flush();
				TORCH_CHECK(bool(f), "写 ", tmp, " 时磁盘出错（满了？）");
			}
			std::error_code ec;
			std::filesystem::rename(tmp, path, ec);
			TORCH_CHECK(!ec, "rename 失败: ", tmp, " -> ", path, " (", ec.message(), ")");
		}

		void put(std::ostringstream& os, const char* k, int64_t v) { os << k << '=' << v << '\n'; }
		void put(std::ostringstream& os, const char* k, int v)     { os << k << '=' << v << '\n'; }
		void put(std::ostringstream& os, const char* k, double v)  { os << k << '=' << v << '\n'; }
		void put(std::ostringstream& os, const char* k, const std::string& v) { os << k << '=' << v << '\n'; }
	}  // namespace

	// ---------------- 成员：写 ----------------

	void SlotState::save_to(const std::string& path) const
	{
		std::ostringstream os;
		os << "# pca_relational_kd · 一个存档槽的状态。字段说明见进度页第五节。\n";
		os << "format=" << kFormat << '\n';

		os << "# 进度（断点口径 = 已交付样本数）\n";
		put(os, "epoch", epoch);
		put(os, "samples_delivered", samples_delivered);
		put(os, "total_steps", total_steps);

		os << "# 复现与对账\n";
		put(os, "seed", seed);
		put(os, "dataset_size", dataset_size);
		put(os, "batch_size", batch_size);
		put(os, "drop_last", drop_last);
		put(os, "accum", accum);

		os << "# 留证（批次顺序不存：由 seed 与 epoch 派生）\n";
		put(os, "model_format", model_format);
		put(os, "config_sha1", config_sha1);
		put(os, "flags", flags);

		os << "# 收尾状态\n";
		put(os, "finished", finished);
		put(os, "best_metric", best_metric);
		put(os, "best_loss", best_loss);
		put(os, "best_epoch", best_epoch);
		put(os, "saved_at", saved_at);

		const auto parent = std::filesystem::path(path).parent_path();
		if (!parent.empty()) std::filesystem::create_directories(parent);
		write_atomic(path, os.str());
	}

	// ---------------- 成员：读 ----------------

	SlotState SlotState::load_from(const std::string& path)
	{
		const std::string text = read_all(path);

		std::unordered_map<std::string, std::string> kv;
		std::istringstream is(text);
		std::string line;
		while (std::getline(is, line))
		{
			if (line.empty() || line[0] == '#') continue;
			const auto eq = line.find('=');
			if (eq == std::string::npos) continue;
			kv.emplace(line.substr(0, eq), line.substr(eq + 1));
		}

		auto need = [&](const char* k) -> const std::string&
			{
				const auto it = kv.find(k);
				TORCH_CHECK(it != kv.end(), "state.txt 缺字段: ", k, "  (", path, ")");
				return it->second;
			};

		TORCH_CHECK(need("format") == kFormat,
			"state.txt 的 format 不认识: ", need("format"), "（期望 ", kFormat, "）");

		SlotState st;
		st.epoch             = std::stoll(need("epoch"));
		st.samples_delivered = std::stoll(need("samples_delivered"));
		st.total_steps       = std::stoll(need("total_steps"));
		st.seed              = std::stoll(need("seed"));
		st.dataset_size      = std::stoll(need("dataset_size"));
		st.batch_size        = std::stoll(need("batch_size"));
		st.drop_last         = std::stoi(need("drop_last"));
		st.accum             = std::stoll(need("accum"));
		st.model_format      = need("model_format");
		st.config_sha1       = need("config_sha1");
		st.flags             = need("flags");
		st.finished     = std::stoi(need("finished"));
		st.best_metric  = std::stod(need("best_metric"));
		// best_loss 是后来加的字段：旧存档没有它，按 0 处理（不抛，兼容老槽）
		{
			const auto it = kv.find("best_loss");
			st.best_loss = (it == kv.end()) ? 0.0 : std::stod(it->second);
		}
		st.best_epoch   = std::stoll(need("best_epoch"));
		st.saved_at     = need("saved_at");
		return st;
	}

}  // namespace prkd::save
