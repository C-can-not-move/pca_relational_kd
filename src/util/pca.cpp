// src/util/pca.cpp
#include "util/pca.h"

namespace prkd::util
{
	Pca::Pca(int64_t dim): dim_(dim)
	{
		TORCH_CHECK(dim > 0, "Pca: dim 必须为正，收到 ", dim);

		const auto opt = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
		s1_ = torch::zeros({ dim }, opt);
		s2_ = torch::zeros({ dim, dim }, opt);
	}

	void Pca::add(const torch::Tensor& feat)
	{
		TORCH_CHECK(feat.defined(), "Pca::add: 传入了未定义的张量");
		TORCH_CHECK(feat.dim() == 2, "Pca::add: 特征必须是二维 (N, D)，收到 ", feat.dim(), " 维");
		TORCH_CHECK(feat.size(1) == dim_,"Pca::add: 特征维数不符，期望 ", dim_, "，收到 ", feat.size(1));
		TORCH_CHECK(feat.size(0) > 0, "Pca::add: 批里没有样本");

		// detach：特征来自 forward_features，可能还挂着计算图，这里只做统计。
		const auto x = feat.detach().to(torch::kCPU, torch::kFloat64);

		// ① 累积。S1 是一阶矩，S2 是二阶矩。
		//    xᵀx 是这一步唯一的计算量：每张图 D² 次乘加（D=2048 时 4.19M 次）。
		s1_ += x.sum(0);/* keepdim = false */
		s2_ += torch::matmul(x.transpose(0, 1), x);

		n_ += feat.size(0);
	}

	PcaResult Pca::fit() const
	{
		TORCH_CHECK(n_ > 0, "Pca::fit: 还没有 add 过任何样本");

		PcaResult r;

		// ② 均值
		r.mu = s1_ / static_cast<double>(n_);

		// ③ 协方差
		//    Cov = S2/n − μμᵀ
		//    与"先中心化再算协方差"数学等价，但不需要第二遍数据。
		//    代价是这里做了一次大数相减 —— 这就是全程 f64 的理由。
		//    ★ 累加会引入极小的非对称(IEEE浮点数表现法机制)；eigenh 默认只读下三角，所以先不管。
		//    将来若换分解器或要严格对称，加一行 cov = (cov + cov.t()) * 0.5;
		auto cov = s2_ / static_cast<double>(n_) - r.mu.unsqueeze(1) * r.mu.unsqueeze(0);

		// ④ 特征分解。对称矩阵用 eigenh 而不是 svd：前者是专门的三对角化，2048×2048 上快一倍以上，而且保证得到实特征值。
		auto [eigen, V] = torch::linalg_eigh(cov);

		// eigenh 返回升序，所以两个张量一起翻成降序。
		// ★ 必须同时翻：eigen[j] 与 V 的第 j 列是绑定关系，
		//   只翻一个会让"第 j 大方差"对不上"第 j 个方向"。
		r.eigen = eigen.flip(0).contiguous();//  .contiguous() 是“保证连续”的语义，而不是.clone()的“我要一份副本”。
		r.V = V.flip(1).contiguous();

		return r;
	}

	int64_t Pca::dim() const { return dim_; }
	int64_t Pca::count() const { return n_; }
}
