// src/save/slot_dir.cpp
#include "save/slot_dir.h"

#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include "model/checkpoint.h"
#include "save/param_index.h"
#include "save/sha1.h"
#include "save/state_file.h"
#include "save/tensor_table.h"

namespace prkd::save
{
	namespace
	{
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

		// archive 没有 .tmp 机制，所以自己走一遍：先写 .tmp，再 rename
		void write_archive_atomic(const std::string& path, const prkd::model::ResNet& net)
		{
			const std::string tmp = path + ".tmp";
			{
				torch::serialize::OutputArchive ar;
				net->save(ar);          // nn::Module::save：内部遍历 state_dict
				ar.save_to(tmp);
			}
			std::error_code ec;
			std::filesystem::rename(tmp, path, ec);
			TORCH_CHECK(!ec, "rename 失败: ", tmp, " -> ", path, " (", ec.message(), ")");
		}

		void write_optimizer_atomic(const std::string& path, const torch::optim::Optimizer& opt)
		{
			const std::string tmp = path + ".tmp";
			{
				torch::serialize::OutputArchive ar;
				opt.save(ar);
				ar.save_to(tmp);
			}
			std::error_code ec;
			std::filesystem::rename(tmp, path, ec);
			TORCH_CHECK(!ec, "rename 失败: ", tmp, " -> ", path, " (", ec.message(), ")");
		}
	}  // namespace

	const char* to_string(ModelFormat f)
	{
		switch (f)
		{
		case ModelFormat::Archive: return "archive";
		case ModelFormat::Both:    return "both";
		}
		return "archive";
	}

	void save_slot(const std::string& dir,
		const prkd::model::ResNet& net,
		torch::optim::Optimizer* opt,
		SlotState st,
		const std::string& config_path,
		ModelFormat fmt)
	{
		std::filesystem::create_directories(dir);

		// ① 模型：官方 archive 总写；Both 时再落一份文本表
		write_archive_atomic(dir + "/model.pt", net);
		if (fmt == ModelFormat::Both)
			prkd::model::save_checkpoint(net, dir + "/model");

		// ② 优化器（官方 archive；超参一并写进去）
		if (opt) write_optimizer_atomic(dir + "/opt.pt", *opt);

		// ③ 配置原文 + 哈希
		if (!config_path.empty())
		{
			const std::string cfg = read_all(config_path);
			write_atomic(dir + "/config.ini", cfg);
			st.config_sha1 = sha1_hex(cfg);
			std::cout << "[slot] config.ini <- " << config_path
				<< "   sha1 " << st.config_sha1.substr(0, 12) << "…\n";
		}

		// ④ 状态 —— 最后写；它在 = 这个槽完整
		st.model_format = to_string(fmt);
		st.saved_at = now_iso8601();
		st.save_to(dir + "/state.txt");

		std::cout << "[slot] saved -> " << dir
			<< "   epoch=" << st.epoch << " samples=" << st.samples_delivered
			<< " total_steps=" << st.total_steps
			<< "  model=" << st.model_format
			<< (opt ? "  +opt" : "  no-opt") << '\n';
	}

	SlotState load_slot_state(const std::string& dir)
	{
		const SlotState st = SlotState::load_from(dir + "/state.txt");

		std::error_code ec;
		if (std::filesystem::exists(dir + "/config.ini", ec))
		{
			const auto cur = sha1_hex(read_all(dir + "/config.ini"));
			if (!st.config_sha1.empty() && cur != st.config_sha1)
			{
				std::cout << "[slot] ⚠ config.ini 与 state.txt 记录的哈希不符："
					<< "存档 " << st.config_sha1.substr(0, 12) << "… / 现在 "
					<< cur.substr(0, 12) << "…（是否接受由调用方决定）\n";
			}
		}
		return st;
	}

	std::size_t load_model(const std::string& dir, prkd::model::ResNet& net)
	{
		const SlotState st = load_slot_state(dir);

		// archive 与 both 都优先用 model.pt：更快，且 archive 是权威副本。
		// both 多出来的 model.txt/bin 是给归档与跨语言用的，需要时调用方自己走模型接口。
		bool use_archive = (st.model_format.empty() || st.model_format == "archive"
			|| st.model_format == "both");

		if (use_archive)
		{
			const std::string path = dir + "/model.pt";
			std::error_code ec;
			if (std::filesystem::exists(path, ec))
			{
				torch::serialize::InputArchive ar;
				ar.load_from(path);
				net->load(ar);

				// nn::Module::load 不返回个数，按 state_dict 的规模报一个
				std::size_t n = 0;
				for (auto& p : net->named_parameters()) { (void)p; ++n; }
				for (auto& b : net->named_buffers())    { (void)b; ++n; }
				std::cout << "[slot] model <- " << path << "   (" << n << " tensors via archive)\n";
				return n;
			}
			TORCH_CHECK(st.model_format != "archive",
				"state 记的是 archive，但找不到 ", path);
		}

		// 只在 state 说不清格式、且 model.pt 不存在时，才退到文本表
		std::cout << "[slot] model_format=" << st.model_format
			<< "，没有 model.pt，改用文本表\n";
		return prkd::model::load_checkpoint(net, dir + "/model");
	}

	void load_optimizer_state(const std::string& dir, torch::optim::Optimizer& opt)
	{
		const std::string path = dir + "/opt.pt";
		std::error_code ec;
		TORCH_CHECK(std::filesystem::exists(path, ec), "没有优化器存档: ", path);

		torch::serialize::InputArchive ar;
		ar.load_from(path);
		opt.load(ar);                 // ⚠ 会把 param_groups 的超参也覆盖成存档里的值

		std::cout << "[slot] optimizer <- " << path
			<< "   （注意：lr 等超参已被存档覆盖）\n";
	}

}  // namespace prkd::save
