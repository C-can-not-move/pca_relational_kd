// src/pca_relational_kd.cpp —— 入口：只做三件事（读配置 / 分发任务 / 统一报错）
//
// 三个任务：
//   base18    训练 resnet18 基准
//   pre152    训练 resnet152（预训练或从头）
//   kd        蒸馏（还没实现，见 task/task_kd.cpp）
//
// 任务名从哪儿来：命令行第二个参数优先，其次看 prkd.ini 的 run.task。
//   pca_relational_kd prkd.ini base18
//   pca_relational_kd prkd.ini pre152
//
// 训练流程不在这里 —— 它在 task/task_base.cpp，只写一遍；
// 三个任务真正的差别只有 模型 / 损失 / 名字。
//
// 这里是唯一设置【进程级全局量】的地方：线程数、种子、算子确定性。
// 它们对整个进程生效，所以不该散落在任务里 —— 任务只负责"读回来看现在是什么状态"。

#include "pca_relational_kd.h"

#include "task/task.h"

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

		// ---- ② 进程级全局量 ----
		// 必须在起任何线程 / 做任何 GPU 计算之前设。三样都是"整个进程一份"：
		//   线程数   影响所有算子的并行归约次序
		//   种子     影响所有随机源
		//   算子层   benchmark = cuDNN 实测挑算法；关掉它，卷积实现才固定
		// ⚠ 真实 API 是 at::globalContext().setBenchmarkCuDNN(false)
		//   （没有 setBenchmarkEnabled；也没有 torch::globalContext）
		torch::set_num_threads(st.debug.intra);
		torch::manual_seed(static_cast<int64_t>(st.run.seed));
		at::globalContext().setBenchmarkCuDNN(false);

		if (st.run.data_dir.empty())
			throw std::runtime_error("没有指定 data_dir —— 写进 prkd.ini 的 run.data_dir");

		// ---- ③ 分发 ----
		const std::string task = (argc > 2) ? argv[2] : st.run.task;
		std::printf("[task] %s（模型名取它自己，存档目录 <out_dir>/train/<name>_seed<seed>/）\n",
			task.c_str());

		// 一个任务 = 一个入口函数（各自一个 cpp）。
		// main 不认识的只有具体实现，它只按名字转发。
		if (task == "base18")
			return prkd::task::run_base18(st, cfg_path);

		if (task == "pre152")
			return prkd::task::run_pre152(st, cfg_path);

		if (task == "kd")
			return prkd::task::run_kd(st, cfg_path);

		if (task == "eval18" || task == "eval152")
		{
			if (st.run.ckpt.empty())
				throw std::runtime_error(
					"task=" + task + " 需要 run.ckpt 指向权重前缀（不含扩展名）");

			std::printf("[task] %s  权重=%s\n", task.c_str(), st.run.ckpt.c_str());

			if (task == "eval18")  return prkd::task::run_eval18(st);
			return prkd::task::run_eval152(st);
		}

		throw std::runtime_error(
			"未知 run.task: \"" + task + "\"（可选：base18 / pre152 / kd）");
	}
	catch (const std::exception& e)
	{
		std::fprintf(stderr, "\n[错误] %s\n", e.what());
		return 1;
	}
}
