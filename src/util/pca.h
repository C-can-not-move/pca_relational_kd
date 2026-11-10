// src/util/pca.h
//
// 最小可用的 PCA：吃进一批批特征，吐出均值 μ 与主方向 V。
//
// 只做五步：
//     ① 累积 S1 = Σx、S2 = Σxxᵀ、n
//     ② μ  = S1 / n
//     ③ Cov = S2 / n − μμᵀ
//     ④ Cov = V Λ Vᵀ（对称矩阵用 eigenh，取降序）
//     ⑤ 投影 z = (x − μ)V   —— 由调用方自己乘，本类不管
//
// 为什么是"累积统计量"而不是"分批做 PCA 再合并"：
//   子空间不能相加。分批各自得到的主方向可能差 30°，而它们张成的空间相同，
//   "平均"会得到一个谁都不是的方向。能分批的是统计量，不是坐标系。
//
// 为什么一遍就够：
//   协方差可以写成 Cov = S2/n − μμᵀ，其中 S2 与 μ 无关，
//   所以两个累加器在同一趟里一起填，中心化放到最后做。
//
// 为什么全程 f64：
//   Cov 那一步是两个量级相近的数相减，f32 会灾难性抵消；
//   而主方向的排序只取决于特征值之间的间隔，最后几位一错，顺序就翻。
#pragma once

#include <cstdint>
#include <utility>

#include <torch/torch.h>

namespace prkd::util
{
	// PCA 的结果:
	// 给出一个新的坐标系，通过数据集的均值和协方差来计算
	// “单个维度不对应语义，组合才对应”
	// PCA 只看二阶矩
	// → 看不到多峰分布、类别簇、任何高阶结构
	// “不做选择”时，PCA 只是把同一组数据换了个坐标系表示，没有压缩。
	// PCA 的约定：
	// ① 降序      V 第 0 列是方差最大的方向
	// ② 配对      eigen[j] ↔ V[:, j]，两者永远同时翻转
	// ③ 恒等式    eigen.sum() == Cov.trace()
	// 用这一条做自检：差超过 1e-9（相对）就是哪一步错了
	struct PcaResult
	{
		torch::Tensor mu;     // (D,)    f64   特征均值（中心）。mu就是新坐标系的原点。
		torch::Tensor V;      // (D, D)  f64   列为主方向，按特征值降序。V 的列就是新坐标系的坐标轴。
		torch::Tensor eigen;  // (D,)    f64   特征值，降序；eigen[j] 对应 V 的第 j 列

		// 原始维数（= V.size(0)）
		int64_t dim() const { return V.size(0); }
	};

	class Pca
	{
	public:
		// dim 必须等于特征向量长度；不一致会在第一次 add 就抛。
		explicit Pca(int64_t dim);

		// 吃进一个批。形状 (N, dim)，N 为batch_size。
		// 内部转 f64 并在 CPU 上累加 —— 非专业计算GPU 的 f64 吞吐只有 f32 的 1/32，在消费卡上用它累积是把时间浪费掉。
		void add(const torch::Tensor& feat);

		// 求mu与全部主方向。
		// const：不动累加器，同一份统计量可以反复调用。
		PcaResult fit() const;

		int64_t dim() const;      // 约定的特征维数
		int64_t count() const;    // 已吃进的样本数

	private:
		int64_t dim_ = 0;         // 约定的特征维数
		int64_t n_ = 0;           // 已吃进的样本数
		torch::Tensor s1_;        // (D,)    f64，Σ x S1 是一阶矩
		torch::Tensor s2_;        // (D, D)  f64，Σ x xᵀ S2 是二阶矩
	};

	// 把特征换成PCA坐标
	// feat     (N, D)  原始特征，D 必须等于 coord.V 的行数
	// keep     <= 0 用全部列；> 0 只取前 keep 列（列已按方差降序）
	// dtype    返回类型，默认 f32（下游损失与网络都在 f32 上算）
	//
	// ★ 这一步是纯线性的，两个后果要记住：
	//     · keep <= 0 时是「正交变换 + 平移」，信息一点不丢
	//     · 若下游用 Pearson 距离（对尺度和平移不敏感），
	//       既不选列也不白化，这一步在数值上等价于不做
	//     · 它的价值全在 keep（选列）和白化上
	torch::Tensor transform(
		const PcaResult& coord,
		const torch::Tensor& feat,
		int64_t keep = -1,
		torch::ScalarType dtype = torch::kFloat32);
}
