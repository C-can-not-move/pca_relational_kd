// src/task/task_eval152.cpp —— 任务 4：评估 resnet152 的预训练权重
//
// 权重前缀取 prkd.ini 的 run.ckpt。
// 结构必须与导出权重时一致，否则 save::restore 会在字节数校验处直接抛 ——
// 那是刻意的，宁可大声失败，也不静默错位。

#include "task/task.h"

#include <torch/torch.h>

#include <chrono>   // steady_clock（评估计时）
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>

#include "data/imagenet_loader.h"
#include "model/checkpoint.h"
#include "save/param_index.h"
#include "save/slot_dir.h"
#include "save/state_file.h"
#include "train/trainer.h"

// 任务 4：只评估 resnet152 的预训练权重。
//
// 结构与导出权重时必须一致，否则 save::restore 会在字节数校验处直接抛。


namespace prkd::task
{
	// ---- 本文件内部用的小工具（只在这个 cpp 里可见）----
	namespace
	{
		// ==================== 从配置到组件：只做翻译，不含算法 ====================

		// cfg::LoaderSettings → data::LoaderOpts
		prkd::data::LoaderOpts make_loader_opts(const prkd::cfg::Settings& st, bool is_train)
		{
			prkd::data::LoaderOpts opt;
			opt.batch_size     = st.loader.batch;
			opt.workers        = st.loader.workers;
			opt.prefetch_depth = st.loader.prefetch;
			opt.short_side     = st.loader.short_side;
			opt.crop           = st.loader.crop;

			// 三个开关互相独立：训练 = 增强 + 洗牌 + 截尾；评估 = 全关
			opt.is_train  = is_train;
			opt.shuffle   = is_train;
			opt.drop_last = is_train;
			return opt;
		}

		// ImageNet 的目录约定：split 是目录名。唯一知道 "/train" / "/val" 的地方。
		std::string split_dir(const prkd::cfg::Settings& st, bool is_train)
		{
			return st.run.data_dir + (is_train ? "/train" : "/val");
		}

		std::unique_ptr<prkd::data::ImageNetLoader>
		make_imagenet_loader(const prkd::cfg::Settings& st, bool is_train, uint64_t seed)
		{
			return std::make_unique<prkd::data::ImageNetLoader>(
				split_dir(st, is_train), make_loader_opts(st, is_train), seed);
		}

		// 导出"可交付"的预训练权重：<name>_acc0.4123.{txt,bin}
		//
		// 磁盘上只保留一份：写入前先删掉同目录里同名前缀的旧文件。
		// 名字里的 acc 是给人看的，但版本不累积 —— 累积会让目录变成一堆分不清先后的文件。
		std::string export_pretrained(const prkd::model::ResNet& model,
			const std::string& dir,
			const std::string& name_prefix,
			double acc)
		{
			std::filesystem::create_directories(dir);

			const std::string prefix = name_prefix + "_acc";
			std::error_code ec;
			for (const auto& e : std::filesystem::directory_iterator(dir, ec))
			{
				if (!e.is_regular_file()) continue;
				const std::string fn = e.path().filename().string();
				if (fn.rfind(prefix, 0) != 0) continue;
				const std::string ext = e.path().extension().string();
				if (ext != ".txt" && ext != ".bin") continue;
				std::filesystem::remove(e.path(), ec);
			}

			char name[160];
			std::snprintf(name, sizeof(name), "%s_acc%.4f", name_prefix.c_str(), acc);

			const std::string base = dir + "/" + name;
			prkd::model::save_checkpoint(model, base);
			std::printf("[export] 预训练模型 <- %s.{txt,bin}（旧的那份已删除）\n", name);
			return base;
		}
	}  // namespace

	int run_eval152(const prkd::cfg::Settings& st)
	{
		const uint64_t seed = static_cast<uint64_t>(st.run.seed);
		const std::string weights_base = st.run.ckpt;   // 权重前缀取自配置 run.ckpt
		const std::string model_name   = "resnet152";

		torch::Device device = torch::kCPU;
		if (st.run.device_index >= 0 && torch::cuda::is_available())
			device = torch::Device(torch::kCUDA, st.run.device_index);
		std::printf("[dev] %s\n", device.str().c_str());

		// ---- 只建 val 一个 loader ----
		auto val_ld = make_imagenet_loader(st, /*is_train=*/false, seed);

		const int64_t num_classes = val_ld->num_classes();
		std::printf("[eval] classes=%lld  val_images=%lld  batches=%lld\n",
			(long long)num_classes,
			(long long)val_ld->size(),
			(long long)val_ld->num_batches());

		// ---- 模型：结构必须与导出权重时一致 ----
		prkd::model::ResNet model = prkd::model::h_resnet152(num_classes);
		model->to(device);

		// ---- 读权重（文本键表：<base>.txt 索引 + <base>.bin 裸字节）----
		const prkd::save::TensorMap items = prkd::save::load(weights_base);
		const std::size_t n = prkd::save::restore(model, items);

		// 122 = h_resnet18；h_resnet152 是 632（50 个 bottleneck × 12 + 首尾）。
		// 这里只打印个数，不写死判断 —— 结构对不对由 restore 的字节数校验兜。
		std::printf("[eval] 权重 %s  张量数=%zu\n", weights_base.c_str(), n);

		model->eval();

		// ---- 评估循环 ----
		int64_t correct = 0;
		int64_t total = 0;
		double  total_sec = 0.0;

		torch::NoGradGuard no_grad;
		while (val_ld->has_next())
		{
			const auto t0 = std::chrono::steady_clock::now();

			auto batch = val_ld->next();
			auto x = batch.images.to(device, /*non_blocking=*/true);
			auto y = batch.labels.to(device);

			auto logits = model->forward(x);

			// 只算 top-1：libtorch 的 topk 返回 std::tuple，没有 .indices，这里也用不上
			auto pred = logits.argmax(/*dim=*/1);
			correct += pred.eq(y).sum().item<int64_t>();
			total   += y.size(0);

			const double dt = std::chrono::duration<double>(
				std::chrono::steady_clock::now() - t0).count();
			total_sec += dt;

			std::printf("[eval] %lld / %lld   acc %.4f   (%.0f img/s)\n",
				(long long)total, (long long)val_ld->size(),
				static_cast<double>(correct) / static_cast<double>(total),
				dt > 0 ? static_cast<double>(y.size(0)) / dt : 0.0);
			std::fflush(stdout);
		}

		const double acc = total > 0 ? static_cast<double>(correct) / total : 0.0;

		std::printf("\n[eval] %s  样本 %lld\n", model_name.c_str(), (long long)total);
		std::printf("[eval] acc = %.4f   用时 %.1f s（%.0f img/s）\n",
			acc, total_sec,
			total_sec > 0 ? static_cast<double>(total) / total_sec : 0.0);
		std::printf("OK\n");
		return 0;
	}
}  // namespace prkd::task
