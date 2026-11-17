// src/task/task.h
//
// 一个任务 = 一个 cpp 里的一个入口函数。main 只按名字分发。
//
//   task_base18.cpp     run_base18      训练 resnet18 基准
//   task_pre152.cpp     run_pre152      训练 resnet152
//   task_eval18.cpp     run_eval18      评估 resnet18 的权重（只用 val）
//   task_eval152.cpp    run_eval152     评估 resnet152 的权重
//   task_kd.cpp         run_kd          蒸馏（未实现）
//
// 这几份实现**刻意不共享**：每个文件自成一体，改一个不会牵动另一个。
// 代价是四份相似的代码；换来的是"改 152 的超参不会影响 18"。
//
// ⚠ 这里不 include 伞头 pca_relational_kd.h：伞头会 include 本文件，反过来就成环。
#pragma once

#include <cstdint>
#include <string>

#include <torch/torch.h>

#include "config.h"
#include "model/resnet.h"

namespace prkd::task
{
	// 训练（监督，交叉熵），带断点续跑与存档。抛异常表示失败，由 main 统一打印。
	int run_base18(const prkd::cfg::Settings& st, const std::string& cfg_path);
	int run_pre152(const prkd::cfg::Settings& st, const std::string& cfg_path);

	// 只评估：权重前缀取 st.run.ckpt，在 val 上跑一遍，打印 acc。
	int run_eval18(const prkd::cfg::Settings& st);
	int run_eval152(const prkd::cfg::Settings& st);

	// 蒸馏：与训练共用数据/存档/断点/日志，差别只在损失（CE + 关系项）。
	// 现在还没写 —— 教师权重链路通了之后再填。
	int run_kd(const prkd::cfg::Settings& st, const std::string& cfg_path);
}
