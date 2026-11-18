// src/task/task_kd.cpp —— 任务 5：关系蒸馏（PCA 版）
//
// 一句话：教师的特征先过一遍 PCA（取前 keep 个轴），学生的也过一遍自己的 PCA，
//        然后比【批内成对距离矩阵】。类别空间不参与 —— 教师懂 1000 类也无所谓。
//
// 为什么这样能工作（也是它相对 logit 蒸馏的好处）：
//   教师的 fc 从没被调用过。我们只用 forward_features：
//       教师  resnet152 → (B, 2048)  →  PCA → (B, keep)
//       学生  resnet18  → (B,  512)  →  PCA → (B, keep)
//   两边的距离矩阵都是 (B, B)，维度天然对齐。所以
//       · 不需要 wnid → 1000 类索引的映射
//       · 不需要给教师换 200 类的头
//       · 不需要教师见过这 200 个类
//
// PCA 怎么拟合（一次性，在两趟之间）：
//   用训练集的前 kd.pca_samples 张（固定顺序，确定性的）各累加一次统计量，
//   师生各拟合一份。分批累加统计量而不是"分批各自 PCA 再合并"，理由见 util/pca.h。
//
// 损失：
//   loss = CE(student_logits, y) + lambda * huber(Ds/scale, Dt/scale)
//   其中 scale = Dt.mean()（教师的平均距离）；学生用【教师】的那个标量归一化，
//   否则"形状像、尺度全错"会被当成正确。
//
// 刻意不做（这一版）：
//   · 不用教师的 logits（那是 KD 的另一条路，需要类别映射）
//   · 不做白化（keep 只选列；白化会把尺度信息也抹掉，而距离损失正需要它）
//   · 不做断点续跑的 PCA 复用：续跑时重新拟合那 kd.pca_samples 张，开销几十秒
#include "task/task.h"

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "data/imagenet_loader.h"
#include "model/checkpoint.h"
#include "model/resnet.h"
#include "save/param_index.h"
#include "save/slot_dir.h"
#include "save/state_file.h"
#include "train/trainer.h"
#include "kd/pcar_loss.h"
#include "util/pca.h"

namespace prkd::task
{
	// ---- 本文件内部用的小工具（只在这个 cpp 里可见）----
	namespace
	{
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

		// 注：批内距离矩阵不再在这里算 —— 它属于损失的一部分，住在
		//     src/kd/pcar_loss.cpp 的 pairwise_dist()（含 0 距离的保护）。
		//     之前这里留过一份同名实现，从未被调用，已删除。
		// ---- PCA 拟合：第一趟交给这个函数 ----
		//
		// 抽成函数而不是内联，是为了让"新跑"与"续跑"走【同一条路】：
		//   · 新跑（无断点）：学生是随机初值 → 轴反映"初始化时的方差"
		//   · 续跑（有断点）：学生是第 N 轮的权重 → 轴反映"续跑那一刻的方差"
		// 两者都不是"训练全程最优"，但都【确定性可复现】：
		//   同一 seed + 同一 pca_samples + 同一份学生权重 ⇒ 同一组轴。
		//
		// ⚠ 已知代价：续跑会得到一组新轴，rkd 的数值在断点那一刻会有一次跳变。
		//    这个跳变来自实现（换了坐标系），不是模型本身。若不愿接受，
		//    就得把 μ/V 存进存档 —— 那是另一套复杂度，暂时不做。
		void fit_pca(prkd::kd::PcarLoss& pcar,
			prkd::model::ResNet& teacher,
			prkd::model::ResNet& student,
			prkd::data::ImageNetLoader& ld,
			const torch::Device& device,
			int64_t want)
		{
			torch::NoGradGuard g;   // 只取统计量，不建图

			const auto t0 = std::chrono::steady_clock::now();

			int64_t seen = 0;
			int64_t step = (want >= 10) ? (want / 10) : want;   // 每 10% 打一行
			if (step < 1) step = 1;

			while (seen < want && ld.has_next())
			{
				auto batch = ld.next();
				auto x = batch.images.to(device, /*non_blocking=*/true);

				pcar.add_teacher(teacher->forward_features(x));
				pcar.add_student(student->forward_features(x));

				const int64_t before = seen;
				seen += x.size(0);

				if (seen / step != before / step)
				{
					const double e = std::chrono::duration<double>(
						std::chrono::steady_clock::now() - t0).count();
					std::printf("[pcar]   拟合进度 %lld / %lld 张（%.0f img/s，已用 %.0f s）\n",
						(long long)seen, (long long)want,
						e > 0 ? static_cast<double>(seen) / e : 0.0, e);
					std::fflush(stdout);
				}
			}

			pcar.finish_fit(device);

			const double secs = std::chrono::duration<double>(
				std::chrono::steady_clock::now() - t0).count();
			std::printf("[pcar] 拟合耗时 %.1f s（%lld 张，%.0f img/s）\n",
				secs, (long long)seen,
				secs > 0 ? static_cast<double>(seen) / secs : 0.0);
		}
	}  // namespace

	int run_kd(const prkd::cfg::Settings& st, const std::string& cfg_path)
	{
		const uint64_t seed = static_cast<uint64_t>(st.run.seed);

		torch::Device device = torch::kCPU;
		if (st.run.device_index >= 0 && torch::cuda::is_available())
			device = torch::Device(torch::kCUDA, st.run.device_index);
		std::printf("[dev] %s\n", device.str().c_str());

		if (st.run.ckpt.empty())
			throw std::runtime_error("task=kd 需要 run.ckpt 指向教师权重（如 run/resnet152_1000）");

		// ---------------- 数据 ----------------
		auto train_ld = make_imagenet_loader(st, /*is_train=*/true,  seed);
		auto val_ld   = make_imagenet_loader(st, /*is_train=*/false, seed);

		const int64_t num_classes = train_ld->num_classes();
		std::printf("[kd] classes=%lld  train=%lld  val=%lld  batch=%lld\n",
			(long long)num_classes, (long long)train_ld->size(),
			(long long)val_ld->size(), (long long)st.loader.batch);

		// ---------------- 学生 / 教师 ----------------
		prkd::model::ResNet student = prkd::model::h_resnet18(num_classes);
		student->to(device);

		prkd::model::ResNet teacher = prkd::model::h_resnet152(1000);   // ImageNet-1k，头保持 1000
		{
			const auto items = prkd::save::load(st.run.ckpt);
			const std::size_t n = prkd::save::restore(teacher, items);
			std::printf("[kd] 教师 <- %s  张量数=%zu\n", st.run.ckpt.c_str(), n);
		}
		teacher->to(device);
		teacher->eval();

		// ---------------- 第一趟：构造师生的 PCA ----------------
		//
		// 只吃前 kd.pca_samples 张（按 loader 顺序，确定性）。
		// 师生必须喂【同一批图】，否则两个协方差不描述同一份数据。
		prkd::kd::PcarConfig pc;
		pc.keep   = st.kd.keep;
		pc.lambda = st.kd.lambda;
		pc.delta  = 1.0;
		pc.metric = st.kd.metric;          // 矩阵度量（huber / angle），forward 里 switch

		constexpr int64_t kStudentDim = 512;    // resnet18
		constexpr int64_t kTeacherDim = 2048;   // resnet152
		prkd::kd::PcarLoss pcar(kStudentDim, kTeacherDim, pc);

		// loader 一次交付一整批，所以样本数必须圆整到批的倍数。
		// 不圆整的话，"配置默认4096 张" 实际会吃到 4096+（向上取整那一批），
		// 日志与配置就对不上。这里向下圆整：配置多少就吃多少，且是整批。
		const int64_t bs      = st.loader.batch;
		const int64_t want_raw = std::min<int64_t>(st.kd.pca_samples, train_ld->size());
		const int64_t want    = (bs > 0) ? (want_raw / bs) * bs : 0;
		TORCH_CHECK(want > 0,"kd.pca_samples 太小：需要至少一个整批（loader.batch = ", bs, "）");
		std::printf("[kd] 拟合 PCA：前 %lld 张（%lld 批 × %lld）\n",(long long)want, (long long)(want / bs), (long long)bs);

		// ---------------- 存档目录 ----------------
		//   与 base18/pre152 完全同一套目录语义，只是模型名不同（kd18），
		//   所以三套结果在 run/train/ 下平行共存，互不覆盖。
		//     <out_dir>/train/kd18_seed<seed>/auto/    每轮覆盖：权重 + 优化器 + 状态
		//     <out_dir>/train/kd18_seed<seed>/best/    后段每次刷新覆盖：只有权重
		//     <out_dir>/pretrained/kd18_acc<acc>       收尾导一次（文本键表格式）
		const std::string model_name = "kd18";
		const std::string run_dir = st.run.out_dir + "/train/" + model_name + "_seed" + std::to_string(st.run.seed);
		const std::string auto_dir    = run_dir + "/auto";
		const std::string best_dir    = run_dir + "/best";
		const std::string pretrain_dir = st.run.out_dir + "/pretrained";
		std::filesystem::create_directories(auto_dir);

		// ---- 循环用的计数器：必须声明在 make_state 之前，lambda 里要捕获 ----
		double  best_acc    = -1.0;
		double  best_loss   = 0.0;
		int64_t best_epoch  = -1;
		int64_t total_steps = 0;
		int64_t start_epoch = 1;

		// 状态快照：与 make_state 等价，但只在本文件用，所以写成 lambda
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
				s.flags             = "kd";
				return s;
			};

		// ---------------- 第二趟：训练 ----------------
		torch::optim::SGD optimizer(student->parameters(),
			torch::optim::SGDOptions(st.train.lr)
			.momentum(st.train.momentum)
			.weight_decay(st.train.weight_decay));

		prkd::train::TrainConfig tcfg;
		tcfg.log_every = st.debug.log_every;

		prkd::save::TensorMap best_snapshot;
		prkd::save::SlotState best_state;
		bool best_ready   = false;
		bool best_on_disk = false;

		// best 从训练的哪个位置开始写盘（比例 → 轮数）
		const int64_t best_write_from_epoch =
			static_cast<int64_t>(std::ceil(st.train.best_write_from * st.train.epochs));

		// ---------------- 断点 ----------------
		//   只恢复【学生 + 优化器】。教师的 PCA 与教师本身都不变，
		//   每次启动重拟合一遍那 kd.pca_samples 张（确定性顺序，几十秒）。
		const std::string auto_state = auto_dir + "/state.txt";
		if (std::filesystem::exists(auto_state))
		{
			const auto s = prkd::save::SlotState::load_from(auto_state);

			std::printf("[ckpt] 发现断点 %s\n", auto_dir.c_str());
			std::printf("[ckpt]   epoch=%lld  samples_delivered=%lld  total_steps=%lld  model=%s\n",
				(long long)s.epoch, (long long)s.samples_delivered,
				(long long)s.total_steps, s.model_format.c_str());

			TORCH_CHECK(s.seed == st.run.seed,
				"断点的 seed 是 ", s.seed, "，配置里是 ", st.run.seed);

			prkd::save::load_model(auto_dir, student);
			prkd::save::load_optimizer_state(auto_dir, optimizer);
			if (st.train.resume_lr_from_config)
			{
				for (auto& g : optimizer.param_groups())
					g.options().set_lr(st.train.lr);
				std::printf("[ckpt]   lr 按配置覆盖为 %.5f\n", st.train.lr);
			}

			start_epoch = s.epoch + 1;
			total_steps = s.total_steps;

			// best 的元信息：优先用 auto/state.txt（每轮补刷过），否则退回 best/
			if (s.best_epoch >= 0)
			{
				best_acc   = s.best_metric;
				best_loss  = s.best_loss;
				best_epoch = s.best_epoch;
				std::printf("[ckpt]   best epoch=%lld acc=%.4f loss=%.4f（沿用）\n",
					(long long)best_epoch, best_acc, best_loss);
			}

			if (std::filesystem::exists(best_dir + "/state.txt"))
			{
				prkd::save::load_model(best_dir, student);
				best_snapshot = prkd::save::dump(student);
				prkd::save::load_model(auto_dir, student);       // 恢复回最新权重
				best_state = prkd::save::SlotState::load_from(best_dir + "/state.txt");
				best_ready   = true;
				best_on_disk = true;
				std::printf("[ckpt]   best 权重已接进内存快照（收尾会从它导出）\n");
			}

			if (start_epoch > st.train.epochs)
			{
				std::printf("[ckpt] 断点已经跑到 epoch %lld（本次上限 %lld），没有要继续的了\n",(long long)s.epoch, (long long)st.train.epochs);
				// 退出前显式销毁，别把顺序留给静态析构
				train_ld.reset();
				val_ld.reset();
				student = nullptr;      // TORCH_MODULE 就是 shared_ptr
				teacher = nullptr;
				return 0;
			}
			std::printf("[ckpt] 从 epoch %lld 接着跑（总上限 %lld）\n",
				(long long)start_epoch, (long long)st.train.epochs);
		}
		else
		{
			std::printf("[ckpt] 没有断点（%s 不存在），从 epoch 1 开始\n", auto_state.c_str());
		}

		// ---------------- 拟合 PCA（新跑与续跑都只做一次）----------------
		//
		// 位置很关键：必须在【断点恢复之后】。
		//   续跑时学生的权重已经是第 N 轮的，用这个学生去拟合，
		//   得到的轴才与"接下来要训练的那个学生"处在同一片特征空间里。
		//   放在恢复之前的话，会用随机初值拟合，等于给续跑换了一套坐标系。
		// ---- 坐标系：live = 每次重拟合；saved = 存一次、之后一直读回 ----
		//   判据在 config 的 kd.pca_source。
		//   saved 模式下若存档还不存在，就照常拟合一次，然后立刻写盘。
		//
		// ★ 写盘放在【两个拟合分支之外】：不管是"用当前学生拟合"还是
		//   "用 pca_ckpt 的固定快照拟合"，只要这次真的拟合了、且是 saved，
		//   就应该落盘。之前它写在 else 里，pca_ckpt + saved 那条路根本没写。
		const std::string pca_base = auto_dir + "/pca";
		bool fitted_now = false;

		if (st.kd.pca_source == "saved" && std::filesystem::exists(pca_base + ".txt"))
		{
			pcar.load_coords(prkd::save::load(pca_base), device, st.kd.keep);
			std::printf("[pcar] 用存档里的坐标系（不重拟合）\n");
		}
		else if (!st.kd.pca_ckpt.empty())
		{
			// ---- 用另一份权重（固定快照）来算学生侧的轴 ----
			//   这份模型只为拟合服务：eval()、NoGrad，用完就销毁。
			auto items = prkd::save::load(st.kd.pca_ckpt);

			// 探针的类别数【从权重里读】：restore() 是严格模式，缺一个参数就抛，
			// 所以不能把 fc 删掉，只能让探针的 fc 和权重对上（200 类 / 1000 类都行）。
			// fc 从头到尾不会被调用 —— 只用 forward_features。
			int64_t probe_classes = num_classes;
			for (const auto& kv : items)
				if (kv.first == "fc.weight") probe_classes = kv.second.size(0);

			prkd::model::ResNet probe = prkd::model::h_resnet18(probe_classes);

			const std::size_t n = prkd::save::restore(probe, items);
			probe->to(device);
			probe->eval();
			std::printf("[pcar] 学生侧坐标轴改用 %s（探针 %lld 类；装入 %zu 个张量）\n",
				st.kd.pca_ckpt.c_str(), (long long)probe_classes, n);

			fit_pca(pcar, teacher, probe, *train_ld, device, want);
			probe = nullptr;   // 用完即弃
			fitted_now = true;
		}
		else
		{
			fit_pca(pcar, teacher, student, *train_ld, device, want);
			fitted_now = true;
		}

		if (fitted_now && st.kd.pca_source == "saved")
		{
			prkd::save::save(pcar.dump_coords(), pca_base);
			std::printf("[pcar] 坐标系已写入 %s.{txt,bin}（之后一直用它）\n", pca_base.c_str());
		}

		for (int64_t epoch = start_epoch; epoch <= st.train.epochs; ++epoch)
		{
			std::printf("\n===== epoch %lld / %lld =====\n",
				(long long)epoch, (long long)st.train.epochs);

			train_ld->start_epoch(epoch);
			student->train();

			const auto t0 = std::chrono::steady_clock::now();
			double sum_ce = 0.0, sum_rkd = 0.0, sum_cos = 0.0;
			int64_t correct = 0, total = 0, batches = 0;

			while (train_ld->has_next())
			{
				auto batch = train_ld->next();
				auto x = batch.images.to(device, /*non_blocking=*/true);
				auto y = batch.labels.to(device);

				// 学生：一次前向，特征喂两个损失
				auto fs     = student->forward_features(x);
				auto logits = student->forward_head(fs);
				const auto ce = torch::nn::functional::cross_entropy(logits, y);

				// 教师：不建图（forward 内部还会 detach 一次，双保险）
				// ★ 返回的已经是 λ·rkd（权重在 PcarLoss 里乘），所以这里直接相加
				torch::Tensor rkd_term;
				{
					torch::NoGradGuard g;
					auto ft = teacher->forward_features(x);
					rkd_term = pcar.forward(fs, ft);
				}

				const auto loss = ce + rkd_term;

				optimizer.zero_grad();
				loss.backward();
				optimizer.step();
				++total_steps;

				sum_ce  += ce.item<double>();
				sum_rkd += pcar.last_rkd();          // 原始（未乘 λ）
				sum_cos += pcar.last_cos();          // 相关系数（两种 mode 都记）
				correct += logits.argmax(1).eq(y).sum().item<int64_t>();
				total   += y.size(0);
				++batches;

				if (tcfg.log_every > 0 && batches % tcfg.log_every == 0)
				{
					std::printf("[kd] batch %lld  ce %.4f  rkd %.4f  cos %.4f  acc %.4f\n",
						(long long)batches, sum_ce / batches, sum_rkd / batches,
						sum_cos / batches, static_cast<double>(correct) / total);
					std::fflush(stdout);
				}
			}

			const double secs = std::chrono::duration<double>(
				std::chrono::steady_clock::now() - t0).count();
			std::printf("[epoch %lld] train ce %.4f  rkd %.4f  cos %.4f  acc %.4f (%.0f img/s, %.1f s)\n",
				(long long)epoch,
				batches ? sum_ce / batches : 0.0,
				batches ? sum_rkd / batches : 0.0,
				batches ? sum_cos / batches : 0.0,
				total ? static_cast<double>(correct) / total : 0.0,
				secs > 0 ? static_cast<double>(total) / secs : 0.0, secs);

			// ① 先存 auto（含优化器）—— 评估崩了也不丢这一轮
			prkd::save::save_slot(auto_dir, student, &optimizer, make_state(epoch),
				cfg_path, prkd::save::ModelFormat::Archive);

			// ② 再评估
			student->eval();
			val_ld->start_epoch(epoch);
			const auto va = prkd::train::evaluate(student, *val_ld, device, tcfg);
			std::printf("[epoch %lld] val   loss %.4f acc %.4f (%.0f img/s, %.1f s)\n",
				(long long)epoch, va.loss, va.acc(), va.img_per_sec(), va.seconds);

			// ③ 更好才更新 best：acc 高者胜；相同看 loss；再相同保留更早的
			const bool better = (best_epoch < 0)
				|| (va.acc() > best_acc)
				|| (va.acc() == best_acc && va.loss < best_loss);

			if (better)
			{
				best_acc   = va.acc();
				best_loss  = va.loss;
				best_epoch = epoch;

				best_snapshot = prkd::save::dump(student);
				best_state    = make_state(epoch);
				best_state.best_metric = best_acc;
				best_state.best_loss   = best_loss;
				best_state.best_epoch  = best_epoch;
				best_ready    = true;

				if (epoch >= best_write_from_epoch)
				{
					prkd::save::save_slot(best_dir, student, nullptr, best_state, cfg_path,
						prkd::save::ModelFormat::Archive);
					best_on_disk = true;
					std::printf("[ckpt] best <- epoch %lld  acc %.4f（已落盘）\n",
						(long long)epoch, best_acc);
				}
				else
				{
					std::printf("[ckpt] best <- epoch %lld  acc %.4f（仅内存，%lld 轮后才写盘）\n",
						(long long)epoch, best_acc, (long long)best_write_from_epoch);
				}
			}

			// ④ 补刷 auto/state.txt —— 让 best 的三项也进断点
			{
				auto st_now = make_state(epoch);
				st_now.best_metric = best_acc;
				st_now.best_loss   = best_loss;
				st_now.best_epoch  = best_epoch;
				st_now.save_to(auto_dir + "/state.txt");
			}
			std::fflush(stdout);
		}

		// ---------------- 收尾 ----------------
		std::printf("\n[done] best epoch=%lld acc=%.4f\n",
			(long long)best_epoch, best_acc);

		if (best_ready)
		{
			prkd::save::restore(student, best_snapshot);

			if (!best_on_disk)
			{
				prkd::save::save_slot(best_dir, student, nullptr, best_state, cfg_path,
					prkd::save::ModelFormat::Archive);
			}

			// 导出：<pretrain_dir>/kd18_acc0.5xxx.{txt,bin}，目录里只留一份
			std::filesystem::create_directories(pretrain_dir);
			// 前缀带上 pca_source：saved / live 各自只清自己那一支的旧档，互不误删。
			// （2026-11-18：之前这里是 model_name + "_acc"，而文件名已经带了 pca_source，
			//   于是既不匹配、也什么都没删，却还打了"旧的那份已删除"——误报。）
			const std::string prefix = model_name + "_" + st.kd.pca_source + "_acc";
			std::error_code ec;
			for (const auto& e : std::filesystem::directory_iterator(pretrain_dir, ec))
			{
				if (!e.is_regular_file()) continue;
				const std::string fn = e.path().filename().string();
				if (fn.rfind(prefix, 0) == 0) std::filesystem::remove(e.path(), ec);
			}
			char name[160];
			// 名字里带上坐标系来源，省得回头分不清"哪一份用了哪把尺子"
			std::snprintf(name, sizeof(name), "%s_%s_acc%.4f",
				model_name.c_str(), st.kd.pca_source.c_str(), best_acc);
			const std::string base = pretrain_dir + "/" + name;
			prkd::model::save_checkpoint(student, base);
			std::printf("[done] 预训练权重: %s.{txt,bin}（同一支的旧导出已清理）\n", base.c_str());
		}

		// finished=1 只是留证，不是启动开关（判据是 epoch 与 epochs 的关系）
		{
			auto fin = make_state(st.train.epochs);
			fin.finished    = 1;
			fin.best_metric = best_acc;
			fin.best_loss   = best_loss;
			fin.best_epoch  = best_epoch;
			fin.save_to(auto_dir + "/state.txt");
			std::printf("[done] auto/state.txt 已标 finished=1（epoch %lld）\n",
				(long long)st.train.epochs);
		}

		// 退出前显式销毁，别把顺序留给静态析构
		train_ld.reset();
		val_ld.reset();
		student = nullptr;      // TORCH_MODULE 就是 shared_ptr
		teacher = nullptr;
		std::printf("OK\n");
		return 0;
	}
}
