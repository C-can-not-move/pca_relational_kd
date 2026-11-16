// src/save/state_file.h
//
// state.txt —— 一个存档槽的「元信息」。只记人读得懂、且决定从哪里接着跑的东西。
//
// 不记什么（有意为之）：
//   RNG 状态     按确定性契约，种子 + (epoch, step) 就能重算（见进度页 5.1）
//   每轮指标历史  那是 meta.json 的事
//   权重          model.txt / model.bin
#pragma once

#include <cstdint>
#include <string>

namespace prkd::save
{
	struct SlotState
	{
		// —— 进度：决定从哪里接着跑 ——
		// 断点口径 = 已交付【样本数】，与 LoaderCore::position() / seek(pos) 直接对应。
		// 不用"本 epoch 第几批"：批是相对量，换 batch_size 就错位。
		int64_t epoch             = 0;
		int64_t samples_delivered = 0;   // 本 epoch 内已交付的样本数
		int64_t total_steps       = 0;   // 跨 epoch 累计的优化步（调度器用）

		// —— 复现与对账所需 ——
		// 五个字段互相校验：samples_delivered 能被 batch_size 整除（drop_last 时）、
		// dataset_size 与 accum 决定"一个 epoch 多长""一步等于几个批"。
		int64_t seed         = 0;
		int64_t dataset_size = 0;
		int64_t batch_size   = 0;
		int     drop_last    = 0;
		int64_t accum        = 1;   // 梯度累积步数

		// —— 留证 ——
		// 不存批次下标 / 排列 / 指纹：顺序由 config 里的 seed 与 (epoch, index) 派生，
		// 可推导的东西不抄第二份。本 epoch 第一个 batch 只在日志里打一行，不进 state。
		std::string model_format;       // "archive"（model.pt）或 "both"（model.pt + model.txt/bin）
		std::string config_sha1;        // config.ini 的 sha1，判断配置有没有被改过

		// flags：算子层的确定性条件快照。
		//   格式 = 空格分隔的 k=v 对，例如
		//     "benchmark=0 determ_cudnn=0 determ_algo=0 threads=1"
		//   ⚠ 只用空格分隔，不要逗号/分号 —— 这样脚本能直接按空格切，按 = 取值。
		//   ⚠ 值必须由【运行时回读】生成（at::globalContext().benchmarkCuDNN() 等），
		//     不要手写：手写的字段迟早会和实际状态脱钩，而脱钩的留证比没有更糟。
		std::string flags;

		// —— 收尾状态 ——
		int         finished    = 0;    // 1 = 训练正常结束（不是被打断）
		double      best_metric = 0.0;
		double      best_loss   = 0.0;   // best 那一轮的验证 loss —— acc 打平时用它比
		int64_t     best_epoch  = -1;
		std::string saved_at;           // ISO8601，本地时区
	};

	// 原子写（.tmp + rename）
	void      save_state(const SlotState& st, const std::string& path);

	// 没有 format 行 / 缺字段就抛 —— 宁可当场失败，也不要带着错状态续跑
	SlotState load_state(const std::string& path);
}
