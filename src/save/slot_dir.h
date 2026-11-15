// src/save/slot_dir.h
//
// 「一个存档槽」= 一个目录。自动存档与手动存档【完全同构】，只差目录：
//
//   run/train/<tag>/auto/                 覆盖写，只有一份（游戏里的 autosave）
//   run/train/<tag>/manual/ep012_s003450/  每次新建一个（游戏里的手动存档）
//
//   <dir>/
//   ├── config.ini      保存当时的配置原文（照抄，不重排）
//   ├── model.pt        权重 + buffer（libtorch 官方 archive）
//   ├── model.txt/bin   同一份权重，文本键表 + 裸字节（仅 Both 时存在）
//   ├── opt.pt          优化器状态（官方 archive）
//   └── state.txt       进度 / 复现所需 / 留证
//
// 为什么分两种模型格式（研究员 2026-11-15 定）：
//   断点存档   Archive   快、短，只给 C++ 自己读，没必要为可读性付双份磁盘
//   最终结果   Both      既留官方 archive，也留文本表 —— 归档、跨语言、记事本能看
//   读 Python 预训练模型  用文本表（export_teacher_flat.py 产出的就是它）
//   ⇒ state.txt 里记 model_format，读的时候按它分派，不靠猜。
//
// 优化器只有官方 archive：
//   每个 OptimizerParamState 都实现了 serialize(archive)，不必为 Adam/SGD 各写分支。
//   ⚠ 超参（lr / betas / eps）也一起存，加载时会被覆盖 —— 改 config 的 lr 不生效。
//
// 完整性由写入顺序保证：模型 → 优化器 → config → state，state.txt 最后写。
//   ⇒ state.txt 在 = 这个槽完整；中途被杀只会留下 .tmp 或孤立的 model/opt。
#pragma once

#include <string>

#include <torch/optim.h>

#include "model/resnet.h"
#include "save/state_file.h"
#include "save/tensor_table.h"

namespace prkd::save
{
	enum class ModelFormat
	{
		Archive,   // 只写 model.pt（断点存档用这个）
		Both,      // model.pt + model.txt/bin（最终结果用这个）
	};

	const char* to_string(ModelFormat f);

	// opt 可以为 nullptr —— 例如只想存一份推理用权重。
	// config_path 只读不写；内容原样复制到 <dir>/config.ini，并把 sha1 记进 state。
	// 传入的 st 里 model_format / config_sha1 / saved_at 会被覆盖成实际值。
	void save_slot(const std::string& dir,
		const prkd::model::ResNet& net,
		torch::optim::Optimizer* opt,
		SlotState st,
		const std::string& config_path,
		ModelFormat fmt = ModelFormat::Archive);

	// 只读 state.txt（并核对 config.ini 的哈希），不碰张量 —— 先判断能不能续跑，再谈加载。
	// 哈希不符只打印警告，不抛：是否接受由调用方决定。
	SlotState load_slot_state(const std::string& dir);

	// 按 state.txt 里的 model_format 分派：archive → model.pt，both → 也用 model.pt（更快）。
	// 返回实际搬进模型的张量个数。
	std::size_t load_model(const std::string& dir, prkd::model::ResNet& net);

	// 读 <dir>/opt.pt 装回优化器。⚠ 超参也会被覆盖成存档里的值。
	void load_optimizer_state(const std::string& dir, torch::optim::Optimizer& opt);
}
