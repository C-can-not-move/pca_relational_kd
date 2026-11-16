// src/task/task.h
//
// 三个任务共用一套骨架：
//
//   任务 1  resnet18 基准        run_train(st, cfg, "resnet18", h_resnet18)
//   任务 2  resnet152 预训练      run_train(st, cfg, "resnet152", h_resnet152)
//   任务 3  蒸馏                 run_kd(...)   ← 与前者共用数据/存档，只换损失
//
// 判据：三个任务真正不同的只有三样 —— 模型、损失、名字。
// 所以 main 只做分发，训练流程只写一遍（task_base.cpp）。
//
// ⚠ 这里不 include 伞头 pca_relational_kd.h：伞头会 include 本文件，反过来就成环。
//   任务层只依赖 config / model / torch，与"哪个入口在调它"无关。
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include <torch/torch.h>

#include "config.h"
#include "model/resnet.h"

namespace prkd::task
{
	// 模型工厂：给定类别数，造一个网络。
	//   用 std::function 而不是模板参数，是因为任务表要在运行期按字符串分发。
	using ModelFactory = std::function<prkd::model::ResNet(int64_t num_classes)>;

	// 训练一个学生模型（监督学习，交叉熵）。
	//   st         已经读好的配置
	//   cfg_path   配置文件路径 —— 会被原样复制进每个存档槽（留证）
	//   model_name "resnet18" / "resnet152" —— 决定存档目录与导出文件名
	//   make_model 模型工厂
	//
	// 返回 0 = 正常结束；抛异常表示失败（由 main 统一打印）。
	int run_train(const prkd::cfg::Settings& st,
		const std::string& cfg_path,
		const std::string& model_name,
		const ModelFactory& make_model);

	// 蒸馏：与 run_train 共用数据、存档、断点、日志，差别只在损失（CE + 关系项）。
	// 现在还没写 —— 教师权重链路通了之后再填。
	int run_kd(const prkd::cfg::Settings& st,
		const std::string& cfg_path);
}
