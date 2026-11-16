// src/task/task_kd.cpp
//
// 任务 3：蒸馏。现在还是空的 —— 教师权重链路（特征导出 + PCA）通了之后再填。
//
// 为什么先放一个空实现而不是删掉：
//   ① 让 main 的分发表现在就能长齐，加任务时不用改 main
//   ② 失败时说清楚"哪一步还没做"，而不是"未知任务"
//
// 填的时候，与 run_train 的差别只有两处：
//   ① 模型：学生 + 教师（教师要 eval()、不建图、不更新）
//   ② 损失：CE(logits, labels) + λ · 关系项(feat_student, feat_teacher)
// 数据、存档、断点、日志全都复用 run_train 里那一套。

#include "task/task.h"

#include <stdexcept>

namespace prkd::task
{
	int run_kd(const prkd::cfg::Settings& /*st*/, const std::string& /*cfg_path*/)
	{
		throw std::runtime_error(
			"task_kd 还没实现：教师权重链路（特征导出 + PCA 坐标系）尚未接通。\n"
			"  已就位的部分：数据、训练循环、存档/断点、预训练导出（见 task_base.cpp）\n"
			"  待做的部分：教师特征导出、PCA 拟合、关系损失、教师权重的加载路径");
	}
}
