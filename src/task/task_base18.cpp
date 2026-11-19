// src/task/task_base18.cpp —— 任务 1：训练 resnet18 基准
//
// 这个文件是自足的：loader、模型、优化器、存档、断点、导出全在里面。
// 不通过工厂或共享函数间接拿模型 —— 换 resnet152 请改 task_pre152.cpp。
//
// 输出：
//   <out_dir>/train/resnet18_seed<seed>/auto/     每轮覆盖：权重 + 优化器 + 状态
//   <out_dir>/train/resnet18_seed<seed>/best/     训练后段每次刷新覆盖：只有权重
//   <out_dir>/pretrained/resnet18_acc<acc>        收尾导一次（文本键表格式）

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

// 任务 1：训练 resnet18 基准。
//
// 这个文件是自足的：数据、循环、存档、断点、导出都在这里面，
// 不依赖别的任务，也不通过工厂间接拿模型。


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

	int run_base18(const prkd::cfg::Settings& st, const std::string& cfg_path)
	{
		const uint64_t seed = static_cast<uint64_t>(st.run.seed);
		const std::string model_name = "resnet18";   // 本任务固定

		// ---- 设备 ----
		torch::Device device = torch::kCPU;
		if (st.run.device_index >= 0 && torch::cuda::is_available())
			device = torch::Device(torch::kCUDA, st.run.device_index);
		std::printf("[dev] %s\n", device.str().c_str());

		// ---- 两个 loader ----
		auto train_ld = make_imagenet_loader(st, /*is_train=*/true,  seed);
		auto val_ld   = make_imagenet_loader(st, /*is_train=*/false, seed);

		const int64_t num_classes = train_ld->num_classes();
		std::printf("\n[train] classes=%lld  train_batches=%lld  val_batches=%lld\n",
			(long long)num_classes,
			(long long)train_ld->num_batches(),
			(long long)val_ld->num_batches());

		// ---- 模型（由工厂决定深度）----
		prkd::model::ResNet model = prkd::model::h_resnet18(num_classes);
		model->to(device);

		if (st.debug.selftest)
		{
			int64_t nparam = 0;
			for (auto& p : model->named_parameters())
			{
				nparam += p.value().numel();
			}
			std::printf("[model] %s  params=%.2f M\n", model_name.c_str(),static_cast<double>(nparam) / 1e6);
		}

		// ---- 优化器 ----
		// 三个超参见 config.h 的 TrainSettings 注释；这里只负责塞进去。
		torch::optim::SGD optimizer(
			model->parameters(),
			torch::optim::SGDOptions(st.train.lr).momentum(st.train.momentum).weight_decay(st.train.weight_decay));

		prkd::train::TrainConfig tcfg;
		tcfg.log_every = st.train.log_every;   // 只保留一个开关：train.log_every

		std::printf("[train] batch=%lld workers=%lld lr=%.4f momentum=%.2f wd=%.1e epochs=%lld\n",(long long)st.loader.batch, (long long)st.loader.workers,st.train.lr, st.train.momentum, st.train.weight_decay,(long long)st.train.epochs);

		// ---- 存档目录 ----
		//   <out_dir>/train/<model>_seed<seed>/auto/   每轮覆盖：权重 + 优化器 + 状态
		//   <out_dir>/train/<model>_seed<seed>/best/   训练后段每次刷新覆盖：只有权重
		//   <out_dir>/pretrained/                      收尾导一次：文本键表格式
		const std::string run_dir = st.run.out_dir + "/train/" + model_name + "_seed" + std::to_string(st.run.seed);
		const std::string auto_dir     = run_dir + "/auto";
		const std::string best_dir     = run_dir + "/best";
		const std::string pretrain_dir = st.run.out_dir + "/pretrained";
		std::filesystem::create_directories(auto_dir);
		std::filesystem::create_directories(best_dir);
		std::printf("[ckpt] run_dir = %s\n", run_dir.c_str());

		// ---- 算子层的确定性：只回读，不在这里改写 ----
		// 设置发生在 main（进程级全局量，见 pca_relational_kd.cpp）：
		//   torch::set_num_threads / manual_seed / setBenchmarkCuDNN(false)
		// 这里把【运行时真实状态】读回来写进存档，供以后对比两次 run 的差异来源。
		const bool bench     = at::globalContext().benchmarkCuDNN();
		const bool det_cudnn = at::globalContext().deterministicCuDNN();
		const bool det_algo  = at::globalContext().deterministicAlgorithms();
		const int  threads   = static_cast<int>(torch::get_num_threads());

		const std::string flags =
			std::string("benchmark=")  + (bench ? "1" : "0")
			+ " determ_cudnn="         + (det_cudnn ? "1" : "0")
			+ " determ_algo="          + (det_algo ? "1" : "0")
			+ " threads="              + std::to_string(threads);

		std::printf("[op] %s\n", flags.c_str());
		if (threads != st.debug.intra)
			std::printf("[op] ⚠ 线程数与配置不符：config=%d, runtime=%d\n", st.debug.intra, threads);

		// ---- 状态变量 ----
		int64_t total_steps = 0;      // 跨 epoch 累计的优化步
		int64_t last_epoch  = 0;      // 最后一轮跑到哪（收尾标 finished 用）

		double  best_acc   = -1.0;
		double  best_loss  = 0.0;
		int64_t best_epoch = -1;

		// best 的"内存版"：权重快照 + 当时的 SlotState。
		prkd::save::TensorMap best_snapshot;
		prkd::save::SlotState best_state;
		bool best_ready   = false;
		bool best_on_disk = false;

		// best 从训练的哪个位置开始写盘（比例 → 轮次）
		const int64_t best_write_from_epoch =
			std::max<int64_t>(1, static_cast<int64_t>(
				std::ceil(static_cast<double>(st.train.epochs) * st.train.best_write_from)));

		// 每个槽都要写一遍的公共字段
		auto make_state = [&](int64_t epoch)
		{
			prkd::save::SlotState s;
			s.epoch             = epoch;
			s.samples_delivered = train_ld->position();
			s.total_steps       = total_steps;
			s.seed              = st.run.seed;
			s.dataset_size      = train_ld->size();
			s.batch_size        = st.loader.batch;
			s.drop_last         = st.loader.drop_last ? 1 : 0;
			s.accum             = 1;
			s.flags             = flags;
			return s;
		};

		// ---- 断点检测：按种子找目录 ----
		int64_t start_epoch = 1;
		bool resumed = false;

		const std::string auto_state = auto_dir + "/state.txt";
		if (std::filesystem::exists(auto_state))
		{
			const auto s = prkd::save::load_slot_state(auto_dir);

			std::printf("[ckpt] 发现断点 %s\n", auto_dir.c_str());
			std::printf("[ckpt]   epoch=%lld  samples_delivered=%lld  total_steps=%lld  model=%s\n",(long long)s.epoch, (long long)s.samples_delivered,(long long)s.total_steps, s.model_format.c_str());
			std::printf("[ckpt]   saved_at=%s\n", s.saved_at.c_str());

			TORCH_CHECK(s.seed == st.run.seed,"断点的 seed 是 ", s.seed, "，配置里是 ", st.run.seed);
			TORCH_CHECK(s.dataset_size == train_ld->size(),"断点的数据集有 ", s.dataset_size, " 张，现在 ", train_ld->size()," 张 —— run.data_dir 可能换了，续跑没有意义");
			if (s.batch_size != st.loader.batch)
				std::printf("[ckpt] ⚠ batch 变了：断点 %lld，现在 %lld —— ""学习率与步数语义都会变，建议换 batch 后从新种子起\n",(long long)s.batch_size, (long long)st.loader.batch);

			prkd::save::load_model(auto_dir, model);
			if (std::filesystem::exists(auto_dir + "/opt.pt"))
			{
				prkd::save::load_optimizer_state(auto_dir, optimizer);

				// opt.load() 会把 param_groups 的 options 一起恢复 ——
				// lr / momentum / weight_decay 全变回存档那一刻的值。
				// 目前只对 lr 提供"按配置覆盖"（set_lr 是虚函数，其余要 cast）。
				// ⇒ 续跑时改 momentum / weight_decay 不会生效，这是有意的取舍。
				if (st.train.resume_lr_from_config)
				{
					for (auto& g : optimizer.param_groups())
					{
						const double old_lr = g.options().get_lr();
						g.options().set_lr(st.train.lr);
						if (old_lr != st.train.lr)
							std::printf("[ckpt]   lr %.6g -> %.6g（按配置覆盖）\n", old_lr, st.train.lr);
					}
				}
				else
				{
					for (const auto& g : optimizer.param_groups())
						std::printf("[ckpt]   lr %.6g（保留 opt.pt 里的值）\n", g.options().get_lr());
				}
			}
			else
			{
				std::printf("[ckpt] ⚠ 没有 opt.pt —— 只恢复权重，优化器从头开始（收敛会抖）\n");
			}

			total_steps = s.total_steps;
			start_epoch = s.epoch + 1;

			// best 单独一个槽：指标与权重都要接过来
			if (std::filesystem::exists(best_dir + "/state.txt"))
			{
				const auto b = prkd::save::load_slot_state(best_dir);
				best_acc   = b.best_metric;
				best_loss  = b.best_loss;
				best_epoch = b.best_epoch;
				std::printf("[ckpt]   best epoch=%lld acc=%.4f loss=%.4f（沿用）\n",
					(long long)best_epoch, best_acc, best_loss);

				if (std::filesystem::exists(best_dir + "/model.pt"))
				{
					// 把 best 的权重拷进内存快照，再把 auto 的权重装回去 —— 不改动当前训练状态
					const prkd::save::TensorMap auto_snapshot = prkd::save::dump(model);
					prkd::save::load_model(best_dir, model);
					best_snapshot = prkd::save::dump(model);
					prkd::save::restore(model, auto_snapshot);

					best_state   = b;
					best_ready   = true;
					best_on_disk = true;
					std::printf("[ckpt]   best 权重已接进内存快照（收尾会从它导出）\n");
				}
			}

			resumed = true;
			if (start_epoch > st.train.epochs)
			{
				std::printf("[ckpt] 断点已经跑到 epoch %lld（本次上限 %lld），没有要继续的了\n",
					(long long)s.epoch, (long long)st.train.epochs);
				return 0;
			}
			std::printf("[ckpt] 从 epoch %lld 接着跑（总上限 %lld）\n",
				(long long)start_epoch, (long long)st.train.epochs);
		}
		else
		{
			std::printf("[ckpt] 没有断点（%s 不存在），从 epoch 1 开始\n", auto_state.c_str());
		}
		(void)resumed;

		// ---- 训练循环 ----
		for (int64_t epoch = start_epoch; epoch <= st.train.epochs; ++epoch)
		{
			std::printf("\n===== epoch %lld / %lld =====\n",(long long)epoch, (long long)st.train.epochs);

			train_ld->start_epoch(epoch);
			const auto tr = prkd::train::train_one_epoch(model, *train_ld, optimizer, device, tcfg);
			total_steps += tr.batches;
			last_epoch = epoch;

			std::printf("[epoch %lld] train loss %.4f acc %.4f (%.0f img/s, %.1f s)\n",
				(long long)epoch, tr.loss, tr.acc(), tr.img_per_sec(), tr.seconds);

			// ① 先存 auto（含优化器）—— 评估崩了也不丢这一轮
			prkd::save::save_slot(auto_dir, model, &optimizer, make_state(epoch),
				cfg_path, prkd::save::ModelFormat::Archive);

			// ② 再评估
			val_ld->start_epoch(epoch);
			const auto va = prkd::train::evaluate(model, *val_ld, device, tcfg);

			std::printf("[epoch %lld] val   loss %.4f acc %.4f (%.0f img/s, %.1f s)\n",
				(long long)epoch, va.loss, va.acc(), va.img_per_sec(), va.seconds);

			// ③ 更好才更新 best：acc 高者胜；相同看 loss；再相同保留更早的
			const bool better = (best_epoch < 0)|| 
								(va.acc() > best_acc)||
								(va.acc() == best_acc && va.loss < best_loss);

			if (better)
			{
				best_acc   = va.acc();
				best_loss  = va.loss;
				best_epoch = epoch;

				// best 只留在内存里；到训练后段才每次刷新写盘（见 best_write_from）
				best_snapshot = prkd::save::dump(model);
				best_state    = make_state(epoch);
				best_state.best_metric = best_acc;
				best_state.best_loss   = best_loss;
				best_state.best_epoch  = best_epoch;
				best_ready    = true;

				if (epoch >= best_write_from_epoch)
				{
					prkd::save::save_slot(best_dir, model, nullptr, best_state, cfg_path,prkd::save::ModelFormat::Archive);
					best_on_disk = true;
					std::printf("[ckpt] best <- epoch %lld  acc %.4f（已落盘）\n",(long long)epoch, best_acc);
				}
				else
				{
					std::printf("[ckpt] best <- epoch %lld  acc %.4f（仅内存，%lld 轮后才写盘）\n",(long long)epoch, best_acc, (long long)best_write_from_epoch);
				}
			}
			// ④ 补刷 auto/state.txt —— 让 best 的三项也进断点
			//    make_state() 只填进度；best 是 ③ 之后才知道的，所以在这里补写一次。
			//    只写 state.txt：权重与优化器已在 ① 落盘，这里不重复写 45 MB。
			//    没有这一步的话，续跑时只能退回 best/ 那份（可能已是旧值）。
			{
				auto st_now = make_state(epoch);
				st_now.best_metric = best_acc;
				st_now.best_loss   = best_loss;
				st_now.best_epoch  = best_epoch;
				st_now.save_to(auto_dir + "/state.txt");
			}
			std::fflush(stdout);
		}

		// ---- 收尾：best 落一次盘 + 预训练导一次 ----
		std::printf("\n[done] best epoch=%lld acc=%.4f\n",(long long)best_epoch, best_acc);

		if (best_ready)
		{
			prkd::save::restore(model, best_snapshot);

			if (!best_on_disk)
			{
				prkd::save::save_slot(best_dir, model, nullptr, best_state, cfg_path, prkd::save::ModelFormat::Archive);
			}
			const std::string exported = export_pretrained(model, pretrain_dir, model_name, best_acc);
			std::printf("[done] 预训练权重: %s.{txt,bin}\n", exported.c_str());
		}

		// finished=1 的含义：这一段训练是【正常结束】的，不是被 Ctrl+C 掉的。
		// 它只是留证，不是启动开关 —— 故意如此：
		//   跑完了再启动（epochs 不变）      ⇒ start_epoch > epochs，自然打印"已经跑完"
		//   跑完了再启动（epochs 改成更大）  ⇒ 应该接着往下跑
		// 两者都是 finished=1，只有"epoch 与 epochs 的关系"能分开。
		{
			auto fin = make_state(last_epoch);
			fin.finished    = 1;
			fin.best_metric = best_acc;
			fin.best_loss   = best_loss;
			fin.best_epoch  = best_epoch;
			fin.save_to(auto_dir + "/state.txt");
			std::printf("[done] auto/state.txt 已标 finished=1（epoch %lld）\n",(long long)last_epoch);
		}

		std::printf("OK\n");
		return 0;
	}
}  // namespace prkd::task
