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

	// 纯线性，两个后果：
	//   · keep <= 0 时是正交变换 + 平移，信息不丢
	//   · 下游若用 Pearson（对尺度/平移不敏感），既不选列也不白化就等于没做
	torch::Tensor transform(
		const PcaResult& coord,
		const torch::Tensor& feat,
		int64_t keep,
		torch::ScalarType dtype)
	{
		TORCH_CHECK(feat.defined(), "transform: 传入了未定义的张量");
		TORCH_CHECK(feat.dim() == 2, "transform: 特征必须是 (N, D)，收到 ", feat.dim(), " 维");
		TORCH_CHECK(coord.V.defined() && coord.V.dim() == 2, "transform: coord 没算过");
		TORCH_CHECK(coord.mu.defined() && coord.mu.dim() == 1, "transform: coord.mu 形状不对");

		const int64_t d = coord.V.size(0);
		TORCH_CHECK(feat.size(1) == d,"transform: 特征维数不符，期望 ", d, "，收到 ", feat.size(1));

		// keep <= 0 用全部列；否则截前 keep 列（列已按方差降序）。
		const int64_t cols = coord.V.size(1);
		int64_t k = (keep > 0) ? keep : cols;
		TORCH_CHECK(k <= cols, "transform: keep=", k, " 超过 V 的列数 ", cols);

		// 与 fit 一致：内部在 CPU 上用 f64 算，避免 f32 在减法与矩阵乘上的累积误差。
		const auto x = feat.detach().to(torch::kCPU, torch::kFloat64);
		const auto v = coord.V.narrow(1, 0, k);

		// (x − μ) 是行向量右乘 V —— 顺序写反会让维度对不上。
		auto z = torch::matmul(x - coord.mu.unsqueeze(0), v);
		return z.to(dtype);
	}
}
