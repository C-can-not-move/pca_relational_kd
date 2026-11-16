// pca_relational_kd.cpp —— 入口：ResNet-18 的普通 ImageNet 训练循环
//
//   ① 读配置
//   ② 建 loader（train + val 各一个）
//   ③ 建模型 resnet18(num_classes)、搬上设备
//   ④ SGD + 交叉熵，跑 st.train.epochs 轮，每轮：训练 → 存 auto → 评估 → 更好则覆盖 best
//   ⑤ 启动时按 run.seed 找断点，有就接着跑
//
// 项目内的头文件、libtorch、标准库一律由伞头提供 —— 这里只有一行。
// 好处：改依赖只改一个地方；坏处：看不出这个 .cpp 具体用了哪些东西，
// 靠下面的代码自己说话。

#include "pca_relational_kd.h"

namespace
{
	// ==================== 从配置到组件：只做翻译，不含算法 ====================
	//
	// 这里刻意是三个具名函数，不是 main 里的 lambda：
	//   ① 可以被单独调用/测试，不必先跑起一个 main
	//   ② 名字本身就是文档：make_loader_opts / split_dir / make_imagenet_loader
	//   ③ main 只剩下"调用"，读起来是一条流程，不是一堆字段搬运
	//
	// 依赖方向：本文件 → cfg、data。cfg 与 data 互相不认识，谁都不用改。
	// 将来要支持第二个数据集，就把 make_imagenet_loader 扩成 make_loader(st, split)，
	// 在函数里按 st.run.dataset 分派 —— main 仍然一行不动。

	// cfg::LoaderSettings → data::LoaderOpts
	prkd::data::LoaderOpts make_loader_opts(const prkd::cfg::Settings& st, bool is_train)
	{
		prkd::data::LoaderOpts opt;
		opt.batch_size     = st.loader.batch;
		opt.workers        = st.loader.workers;
		opt.prefetch_depth = st.loader.prefetch;
		opt.short_side     = st.loader.short_side;
		opt.crop           = st.loader.crop;

		// 三个开关互相独立：训练 = 增强 + 洗牌 + 截尾；评估 = 全关。
		// is_train 只决定变换分支，不派生另外两个（配错的组合照样成立，启动打印里看得见）。
		opt.is_train  = is_train;
		opt.shuffle   = is_train;
		opt.drop_last = is_train;
		return opt;
	}

	// ImageNet 的目录约定：split 是目录名。**这是唯一知道 "/train" / "/val" 的地方。**
	std::string split_dir(const prkd::cfg::Settings& st, bool is_train)
	{
		return st.run.data_dir + (is_train ? "/train" : "/val");
	}

	// “有线程的对象不要按值传”。这个封装是为了挡住构造临时对象，在移动复制的风险前就把它变成 unique_ptr。
	std::unique_ptr<prkd::data::ImageNetLoader>
	make_imagenet_loader(const prkd::cfg::Settings& st, bool is_train, uint64_t seed)
	{
		return std::make_unique<prkd::data::ImageNetLoader>(
			split_dir(st, is_train), make_loader_opts(st, is_train), seed);
	}
}  // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
	SetConsoleOutputCP(CP_UTF8);
#endif
	try
	{
		// ---- ① 配置 ----
		const std::string cfg_path = (argc > 1) ? argv[1] : "prkd.ini";

		prkd::cfg::Settings st = prkd::cfg::defaults();
		st.load_file(cfg_path);      // 打不开 / 内容有错 → 直接抛

		std::printf("\n[cfg] 配置文件 = %s\n", cfg_path.c_str());
		st.print();          // 完整快照：建 loader / 模型之前打，后面崩了也有据可查
		std::printf("[cfg] summary: %s\n", st.summary().c_str());

		// 进程级全局量，必须在起任何线程之前设
		torch::set_num_threads(st.debug.intra);
		torch::manual_seed(static_cast<int64_t>(st.run.seed));

		if (st.run.dataset != "imagenet")
			throw std::runtime_error("run.dataset = \"" + st.run.dataset + "\" —— 本入口只做 imagenet");
		if (st.run.data_dir.empty())
			throw std::runtime_error("没有指定 data_dir —— 写进 prkd.ini 的 run.data_dir");

		const uint64_t seed = static_cast<uint64_t>(st.run.seed);

		// ---- 设备 ----
		// ⚠ C++ 侧没有 torch::cuda::set_device（那是 Python 的接口）。
		//   只要张量都显式 .to(device)，op 会自己落到对应卡上，不需要设"当前设备"。
		//   真要设（例如自建 CUDA stream）用 c10::cuda::set_device(idx)。
		torch::Device device = torch::kCPU;
		if (st.run.device_index >= 0 && torch::cuda::is_available())
		{
			device = torch::Device(torch::kCUDA, st.run.device_index);
		}
		std::printf("[dev] %s\n", device.str().c_str());

		// ---- ② 两个 loader：train / val ----
		// 字段搬运与目录约定都在上面的 make_imagenet_loader 里，这里只剩"造两个"。
		auto train_ld = make_imagenet_loader(st, /*is_train=*/true,  seed);
		auto val_ld   = make_imagenet_loader(st, /*is_train=*/false, seed);

		const int64_t num_classes = train_ld->num_classes();// 后面要用
		std::printf("\n[train] classes=%lld  train_batches=%lld  val_batches=%lld\n",(long long)num_classes,(long long)train_ld->num_batches(),(long long)val_ld->num_batches());

		// ---- ③ 模型 ----
		prkd::model::ResNet model = prkd::model::h_resnet18(num_classes);
		model->to(device);

		if (st.debug.selftest)
		{
			int64_t nparam = 0;
			for (auto& p : model->named_parameters()) nparam += p.value().numel();
			std::printf("[model] resnet18  params=%.2f M\n",
				static_cast<double>(nparam) / 1e6);
		}

		// ---- ④ 优化器 ----
		//
		// 三个超参的含义与坑（值都来自 prkd.ini 的 [train] 段）：
		//
		//   lr  步长：θ ← θ − lr·g
		//       取大 → 几步之内 loss 抖动、发散、NaN；取小 → 收敛慢，看着像"没在学"
		//       与 batch 成比例：batch 翻倍，lr 也翻倍（梯度是更多样本的平均，噪声更小）
		//       惯例是 batch 256 配 0.1；batch 64 对 0.025
		//
		//   momentum  动量：v ← m·v + g，然后 θ ← θ − lr·v
		//       方向一致时加速，来回震荡时抵消 —— 给优化器"惯性"
		//       典型 0.9；等效步长约 lr/(1−m)，所以 m=0.9 时实际走的是 lr 的十倍
		//       ⚠ m=0 时 SGD 不再分配 momentum_buffer ⇒ 存档里就没有这部分状态
		//
		//   weight_decay  权重衰减（L2）：g ← g + wd·θ
		//       抑制参数变大、减过拟合；典型 1e-4 ~ 5e-4
		//       ⚠ 在纯 SGD 里等价于 L2 正则；在 Adam 里不等价（所以才有 AdamW 那种解耦写法）
		//       ⚠ 这里 BN 的 γ/β 与 bias 也一起被衰减 —— 现代配方常把它们排除，
		//         这是本项目可以改进的一处（"no weight decay on norm/bias"）
		//
		// ⚠ model->parameters() 的顺序决定了 optimizer 的 param_groups 顺序；
		//   存档那边用的是 param_index（named_parameters + named_buffers）。
		//   标准网络下两者一致，但契约不同源 —— 一旦自己 register_parameter 过，就要重新核对。
		torch::optim::SGD optimizer(
			model->parameters(),
			torch::optim::SGDOptions(st.train.lr).momentum(st.train.momentum).weight_decay(st.train.weight_decay));

		prkd::train::TrainConfig tcfg;
		tcfg.log_every = (st.debug.log_every > 0) ? st.debug.log_every : st.train.log_every;

		// 真实生效的组合打一行：批来自 loader，lr 来自这里，两者配不配一眼能看出。
		std::printf("[train] batch=%lld workers=%lld lr=%.4f momentum=%.2f wd=%.1e epochs=%lld\n",
			(long long)st.loader.batch, 
			(long long)st.loader.workers,
			st.train.lr, 
			st.train.momentum, 
			st.train.weight_decay, 
			(long long)st.train.epochs
		);

		// ---- ⑤ 存档目录 ----
		// 一轮一个 run 目录：out_dir/train/seed<seed>/
		//   auto/   每轮覆盖，权重 + 优化器，续跑用
		//   best/   仅在更好时覆盖，只有权重，报数 / 导出用
		const std::string run_dir = st.run.out_dir + "/train/seed" + std::to_string(st.run.seed);
		std::printf("[ckpt] run_dir = %s\n", run_dir.c_str());
		const std::string auto_dir = run_dir + "/auto";
		const std::string best_dir = run_dir + "/best";
		std::filesystem::create_directories(auto_dir);
		std::filesystem::create_directories(best_dir);

		// 算子层的非确定性：关掉 cuDNN 的算法自动选择，卷积实现就固定了。
		// ⚠ 真实 API 是 at::globalContext().setBenchmarkCuDNN(false)
		//   （没有 setBenchmarkEnabled；也没有 torch::globalContext）。
		// determinism 那一档没开：setDeterministicAlgorithms(true) 会让部分算子变慢或直接报错，
		// 先用"固定 benchmark + 固定种子"。
		at::globalContext().setBenchmarkCuDNN(false);

		// flags 记的是【运行时真实状态】，不是作者以为的状态 —— 全部回读，不手写。
		// 理由：留证的字段一旦会和实际脱钩，就比没有更糟。
		//   benchmarkCuDNN()            cuDNN 是否会实测挑算法
		//   deterministicCuDNN()        cuDNN 是否强制确定性实现
		//   deterministicAlgorithms()   全局确定性开关（会让部分算子变慢或报错）
		//   get_num_threads()           ATen intra-op 线程数，与前面 set_num_threads 对一下
		const bool bench     = at::globalContext().benchmarkCuDNN();
		const bool det_cudnn = at::globalContext().deterministicCuDNN();
		const bool det_algo  = at::globalContext().deterministicAlgorithms();
		const int  threads   = static_cast<int>(torch::get_num_threads());

		const std::string flags =
			std::string("benchmark=")      + (bench ? "1" : "0")
			+ " determ_cudnn="             + (det_cudnn ? "1" : "0")
			+ " determ_algo="              + (det_algo ? "1" : "0")
			+ " threads="                  + std::to_string(threads);

		std::printf("[op] %s\n", flags.c_str());
		if (threads != st.debug.intra)
		{
			// 不等于说明有别的代码改过它（例如某个库在初始化时自己设了）
			std::printf("[op] ⚠ 线程数与配置不符：config=%d, runtime=%d\n", st.debug.intra, threads);
		}

		// 跨 epoch 累计的优化步。必须定义在 make_state 之前 —— lambda 按引用捕获它，
		// 写在下面（训练循环段）就会变成"捕获时还没声明"。
		int64_t total_steps = 0;

		// 每个槽都要写一遍的公共字段。samples_delivered 用 loader 的交付进度，
		// 它是断点的唯一口径（不是生产游标）。
		// ⚠ 必须按引用捕获：total_steps 每轮累加，按值捕获会让存档永远停在第一轮的值。
		// 用的是 lambda 而不是函数：它要捕获 total_steps、flags、train_ld。
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

		// ---- ⑤.5 断点检测 ----
		// 断点按【种子】认：run_dir = out_dir/train/seed<seed>，里面 auto/state.txt 在就说明
		// 这个种子跑过。判据只用 state.txt —— 它是槽的完整性标记（最后写的那个文件）。
		double best_acc = -1.0;
		double best_loss = 0.0;
		int64_t best_epoch = -1;
		int64_t start_epoch = 1;
		bool resumed = false;

		const std::string auto_state = auto_dir + "/state.txt";
		if (std::filesystem::exists(auto_state))
		{
			const auto s = prkd::save::load_slot_state(auto_dir);

			std::printf("[ckpt] 发现断点 %s\n", auto_dir.c_str());
			std::printf("[ckpt]   epoch=%lld  samples_delivered=%lld  total_steps=%lld  model=%s\n",(long long)s.epoch, (long long)s.samples_delivered,(long long)s.total_steps, s.model_format.c_str());
			std::printf("[ckpt]   saved_at=%s\n", s.saved_at.c_str());

			// 配置变过就至少说出来 —— 哈希比对在 load_slot_state 里已经做过一次
			if (!s.config_sha1.empty())
				std::printf("[ckpt]   config_sha1=%s…\n", s.config_sha1.substr(0, 12).c_str());

			// 批/种子/数据集变了，续跑就没有意义 —— 直接拒绝，别悄悄跑出个不一样的结果
			TORCH_CHECK(s.seed == st.run.seed,"断点的 seed 是 ", s.seed, "，配置里是 ", st.run.seed, " —— 不该走到这里（目录已按种子分）");
			TORCH_CHECK(s.dataset_size == train_ld->size(),"断点的数据集有 ", s.dataset_size, " 张，现在 ", train_ld->size()," 张 —— run.data_dir 可能换了，续跑没有意义");
			if (s.drop_last != (st.loader.drop_last ? 1 : 0))
				std::printf("[ckpt] ⚠ drop_last 变了：断点 %d，现在 %d —— 每轮步数会不同\n",s.drop_last, st.loader.drop_last ? 1 : 0);
			if (s.batch_size != st.loader.batch)
				std::printf("[ckpt] ⚠ batch 变了：断点 %lld，现在 %lld —— ""学习率与步数语义都会变，建议换 batch 后从新种子起\n",(long long)s.batch_size, (long long)st.loader.batch);

			// 权重与优化器
			prkd::save::load_model(auto_dir, model);
			if (std::filesystem::exists(auto_dir + "/opt.pt"))
			{
				prkd::save::load_optimizer_state(auto_dir, optimizer);

				// opt.pt 里带着存档那一刻的 lr / momentum / weight_decay。
				// 想在续跑时换 lr，必须显式覆盖 —— 否则改 config 是不生效的。
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
				std::printf("[ckpt] ⚠ 没有 opt.pt —— 只恢复权重，优化器从头开始（收敛会抖）\n");

			total_steps = s.total_steps;
			start_epoch = s.epoch + 1;

			// best 单独一个槽，读它的 state 才知道历史上最好是多少
			if (std::filesystem::exists(best_dir + "/state.txt"))
			{
				const auto b = prkd::save::load_slot_state(best_dir);
				best_acc   = b.best_metric;
				best_epoch = b.best_epoch;
				std::printf("[ckpt]   best epoch=%lld acc=%.4f（沿用，不会因为续跑而清空）\n",
					(long long)best_epoch, best_acc);
			}

			resumed = true;
			if (start_epoch > st.train.epochs)
			{
				std::printf("[ckpt] 断点已经跑到 epoch %lld（本次上限 %lld），没有要继续的了\n",(long long)s.epoch, (long long)st.train.epochs);
				std::printf("OK\n");
				return 0;
			}
			std::printf("[ckpt] 从 epoch %lld 接着跑（总上限 %lld）\n",(long long)start_epoch, (long long)st.train.epochs);
		}
		else
		{
			std::printf("[ckpt] 没有断点（%s 不存在），从 epoch 1 开始\n", auto_state.c_str());
		}

		// samples_delivered 在轮末存档时恒等于一轮的长度；若不是，说明是轮中途存的，
		// 这一版没有那种存档点
		if (resumed)
		{
			// 续跑从下一轮开始，位置归零；seek 只在"轮中途存档"时才需要
			std::printf("[ckpt] 断点在轮末 → 直接进下一轮，不需要 seek\n");
		}

		// ---- ⑥ 训练循环 ----
		// best_* 与 start_epoch 都在 ⑤.5 里声明（续跑时要沿用断点里记下的 best、跳过已跑的轮）。
		for (int64_t epoch = start_epoch; epoch <= st.train.epochs; ++epoch)
		{
			std::printf("\n===== epoch %lld / %lld =====\n", (long long)epoch, (long long)st.train.epochs);

			// 每个 epoch 开头重排：order_ 由 (seed, epoch) 决定
			train_ld->start_epoch(epoch);
			const auto tr = prkd::train::train_one_epoch(model, *train_ld, optimizer, device, tcfg);
			total_steps += tr.batches;
			std::printf("[epoch %lld] train loss %.4f acc %.4f (%.0f img/s, %.1f s)\n",(long long)epoch, tr.loss, tr.acc(), tr.img_per_sec(), tr.seconds);

			// ① 先存 auto（含优化器）—— 评估崩了也不丢这一轮
			prkd::save::save_slot(auto_dir, model, &optimizer, make_state(epoch),cfg_path, prkd::save::ModelFormat::Archive);

			// ② 再评估
			val_ld->start_epoch(epoch);
			const auto va = prkd::train::evaluate(model, *val_ld, device, tcfg);
			std::printf("[epoch %lld] val   loss %.4f acc %.4f (%.0f img/s, %.1f s)\n",(long long)epoch, va.loss, va.acc(), va.img_per_sec(), va.seconds);

			// ③ 更好才覆盖 best：acc 高者胜；acc 相同看 loss；再相同保留更早的（避免无意义重写）
			const bool better = (best_epoch < 0)
				|| (va.acc() > best_acc)
				|| (va.acc() == best_acc && va.loss < best_loss);

			if (better)
			{
				best_acc = va.acc();
				best_loss = va.loss;
				best_epoch = epoch;

				auto bs = make_state(epoch);
				bs.best_metric = best_acc;
				bs.best_epoch  = best_epoch;
				prkd::save::save_slot(best_dir, model, nullptr, bs,cfg_path, prkd::save::ModelFormat::Archive);
				std::printf("[ckpt] best <- epoch %lld  acc %.4f\n",(long long)epoch, best_acc);
			}
			std::fflush(stdout);
		}

		std::printf("\n[done] best epoch=%lld acc=%.4f\n",(long long)best_epoch, best_acc);
		std::printf("OK\n");
		return 0;
	}
	catch (const std::exception& e)
	{
		std::fprintf(stderr, "\n[错误] %s\n", e.what());
		return 1;
	}
}
