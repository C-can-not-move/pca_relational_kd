// src/kd/pcar_loss.cpp
#include "kd/pcar_loss.h"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace prkd::kd
{
	namespace
	{
		// 批内成对距离 (B, B)。
		//
		// 用 cdist 而不是 ||x||²−2xᵀy+||y||² 展开式：
		// 后者是两个量级相近的大数相减，f32 下会把小距离的精度吃掉；
		// 而 n 只有 batch 大小（16～1024），cdist 的直接差值法完全够快。
		//
		// clamp_min(1e-8)：sqrt 在 0 处的导数是无穷。
		//   批里若出现两张完全相同的图（换数据集时并非不可能），
		//   backward 会得到 NaN 并污染整批的梯度。抬高一丁点即可避开奇点，
		//   对这个尺度（特征距离 O(1)）的影响在 1e-8 量级，可忽略。
		torch::Tensor pairwise_dist(const torch::Tensor& z)
		{
			return torch::cdist(z, z, /*p=*/2.0).clamp_min(1e-8);
		}
	}  // namespace

	PcarLoss::PcarLoss(int64_t student_dim, int64_t teacher_dim, PcarConfig cfg)
		: cfg_(cfg)
		, pca_s_(student_dim)
		, pca_t_(teacher_dim)
	{
	}

	void PcarLoss::add_student(const torch::Tensor& feat)
	{
		pca_s_.add(feat.to(torch::kCPU));
	}

	void PcarLoss::add_teacher(const torch::Tensor& feat)
	{
		pca_t_.add(feat.to(torch::kCPU));
	}

	void PcarLoss::finish_fit(const torch::Device& device)
	{
		TORCH_CHECK(pca_s_.count() > 0 && pca_t_.count() > 0,
			"PcarLoss::finish_fit: 还没累加过特征（add_student / add_teacher）");
		TORCH_CHECK(pca_s_.count() == pca_t_.count(),
			"PcarLoss::finish_fit: 师生的样本数不一致 (", pca_s_.count(),
			" vs ", pca_t_.count(), ") —— 两边必须是同一批图");

		const auto cs = pca_s_.fit();
		const auto ct = pca_t_.fit();

		// 师生取【同样多】的轴：距离矩阵比较要求两边的子空间维数一致。
		keep_ = std::min<int64_t>({ cfg_.keep, cs.dim(), ct.dim() });
		TORCH_CHECK(keep_ > 0, "PcarLoss: keep 必须 > 0（配置 kd.keep）");
		if (keep_ != cfg_.keep)
		{
			std::printf("[pcar] keep 从 %lld 截到 %lld（受特征维数限制）\n",
				(long long)cfg_.keep, (long long)keep_);
		}

		// 只取前 keep 列（列已按方差降序，见 util/pca.h 的约定 ①）
		mu_s_ = cs.mu.to(device);
		V_s_  = cs.V.slice(/*dim=*/1, 0, keep_).to(device);
		mu_t_ = ct.mu.to(device);
		V_t_  = ct.V.slice(/*dim=*/1, 0, keep_).to(device);

		fitted_ = true;

		// 留一条能复核的记录：特征值之和必须等于协方差的迹（pca.h 的约定 ③）
		std::printf("[pcar] 拟合完成  keep=%lld  学生 %lld×%lld / 教师 %lld×%lld\n",
			(long long)keep_,
			(long long)V_s_.size(0), (long long)V_s_.size(1),
			(long long)V_t_.size(0), (long long)V_t_.size(1));
	}

	torch::Tensor PcarLoss::forward(const torch::Tensor& student_feat,
		const torch::Tensor& teacher_feat)
	{
		TORCH_CHECK(fitted_, "PcarLoss::forward: 还没 finish_fit()");
		TORCH_CHECK(student_feat.dim() == 2 && teacher_feat.dim() == 2,
			"PcarLoss::forward: 特征必须是 (B, d)，给的是 ",
			student_feat.sizes(), " 与 ", teacher_feat.sizes());
		TORCH_CHECK(student_feat.size(0) == teacher_feat.size(0),
			"PcarLoss::forward: 批大小必须一致");
		TORCH_CHECK(student_feat.size(0) >= 2,
			"PcarLoss::forward: 批里只有 ", student_feat.size(0),
			" 张图 —— 距离矩阵会退化（mean 接近 0），这就是 drop_last 存在的原因");

		// ---- 投影到各自的 PCA 子空间 ----
		// 学生侧保留梯度（V_s / mu_s 是常数，梯度经 matmul 传回 student_feat）
		const auto zs = torch::matmul(student_feat - mu_s_, V_s_);

		const auto zt = torch::matmul(teacher_feat.detach() - mu_t_, V_t_);

		const auto Ds = pairwise_dist(zs);
		const auto Dt = pairwise_dist(zt);

		// ---- 归一化：两边用【同一个】标量（教师的平均距离）----
		// 各自归一化会让"尺度全错、形状对"的情形损失为零，等于没约束尺度。
		const auto scale = Dt.mean().detach() + 1e-6;

		const auto raw = torch::huber_loss(Ds / scale, Dt / scale,
			cfg_.delta, torch::Reduction::Mean);

		last_scale_ = scale.item<double>();
		last_rkd_   = raw.item<double>();          // 原始值（未乘 λ），日志用这个

		// λ 在这里乘上：调用方拿到的就是"要加进总损失的那个量"。
		// 好处是调用点只有一句 ce + rkd_term，权重与损失住在同一处；
		// 代价是 last_rkd() 与返回值不是同一个数（一个是原始、一个是加权）。
		return cfg_.lambda * raw;
	}

}  // namespace prkd::kd
