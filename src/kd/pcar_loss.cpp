// src/kd/pcar_loss.cpp
#include "kd/pcar_loss.h"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <unordered_map>

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

		// 存档要的是【完整的 PCA】，所以 V 留全：投影时再切前 keep 列。
		mu_s_ = cs.mu.to(device);
		V_s_  = cs.V.to(device);
		mu_t_ = ct.mu.to(device);
		V_t_  = ct.V.to(device);

		Vs_use_ = V_s_.slice(/*dim=*/1, 0, keep_);   // view，不复制
		Vt_use_ = V_t_.slice(/*dim=*/1, 0, keep_);

		eig_s_ = cs.eigen.to(torch::kCPU, torch::kFloat32).contiguous();
		eig_t_ = ct.eigen.to(torch::kCPU, torch::kFloat32).contiguous();

		fitted_ = true;

		const double tot_s = cs.eigen.sum().item<double>();
		const double top_s = cs.eigen.slice(0, 0, keep_).sum().item<double>();
		const double tot_t = ct.eigen.sum().item<double>();
		const double top_t = ct.eigen.slice(0, 0, keep_).sum().item<double>();

		std::printf("[pcar] 拟合完成  keep=%lld  学生 %lld×%lld / 教师 %lld×%lld（存档存完整矩阵）\n",
			(long long)keep_,
			(long long)V_s_.size(0), (long long)V_s_.size(1),
			(long long)V_t_.size(0), (long long)V_t_.size(1));
		std::printf("[pcar] 方差占比  学生 %.1f%%  教师 %.1f%%（各取前 %lld 轴）\n",
			tot_s > 0 ? 100.0 * top_s / tot_s : 0.0,
			tot_t > 0 ? 100.0 * top_t / tot_t : 0.0,
			(long long)keep_);
	}

	torch::Tensor PcarLoss::forward(const torch::Tensor& student_feat,
		const torch::Tensor& teacher_feat)
	{
		TORCH_CHECK(fitted_, "PcarLoss::forward: 还没 finish_fit()");
		TORCH_CHECK(student_feat.dim() == 2 && teacher_feat.dim() == 2,"PcarLoss::forward: 特征必须是 (B, d)，给的是 ",student_feat.sizes(), " 与 ", teacher_feat.sizes());
		TORCH_CHECK(student_feat.size(0) == teacher_feat.size(0),"PcarLoss::forward: 批大小必须一致");
		TORCH_CHECK(student_feat.size(0) >= 2,"PcarLoss::forward: 批里只有 ", student_feat.size(0)," 张图 —— 距离矩阵会退化（mean 接近 0），这就是 drop_last 存在的原因");

		// ---- 投影到各自的 PCA 子空间 ----
		// 学生侧保留梯度（V_s / mu_s 是常数，梯度经 matmul 传回 student_feat）
		const auto zs = torch::matmul(student_feat - mu_s_, Vs_use_);
		const auto zt = torch::matmul(teacher_feat.detach() - mu_t_, Vt_use_);

		const auto Ds = pairwise_dist(zs);
		const auto Dt = pairwise_dist(zt);

		// ---- 归一化：两边用【同一个】标量（教师的平均距离）----
		// 各自归一化会让"尺度全错、形状对"的情形损失为零，等于没约束尺度。
		const auto scale = Dt.mean().detach() + 1e-6;

		const auto raw = torch::huber_loss(Ds / scale, Dt / scale,cfg_.delta, torch::Reduction::Mean);

		last_scale_ = scale.item<double>();
		last_rkd_   = raw.item<double>();          // 原始值（未乘 λ），日志用这个

		// λ 在这里乘上：调用方拿到的就是"要加进总损失的那个量"。
		// 好处是调用点只有一句 ce + rkd_term，权重与损失住在同一处；
		// 代价是 last_rkd() 与返回值不是同一个数（一个是原始、一个是加权）。
		return cfg_.lambda * raw;
	}

	// ---------------- 坐标系的存取 ----------------
	//
	// 存档里放的是【完整的 PCA】：mu (D,) 与 V (D, D)，还有特征值。
	// 不放"前 keep 列" —— 因为 keep 是【怎么用它】，不是【它是什么】。
	//   好处：改 kd.keep 只需重读，不必重跑十几分钟的拟合；
	//         方差占比也能按任意 keep 现算。

	prkd::save::TensorMap PcarLoss::dump_coords() const
	{
		TORCH_CHECK(fitted_, "PcarLoss::dump_coords: 还没 finish_fit()");

		prkd::save::TensorMap out;
		out.reserve(6);
		out.emplace_back("mu_s", mu_s_.detach().to(torch::kCPU, torch::kFloat32).contiguous());
		out.emplace_back("V_s",  V_s_.detach().to(torch::kCPU,  torch::kFloat32).contiguous());
		out.emplace_back("mu_t", mu_t_.detach().to(torch::kCPU, torch::kFloat32).contiguous());
		out.emplace_back("V_t",  V_t_.detach().to(torch::kCPU,  torch::kFloat32).contiguous());
		if (eig_s_.defined()) out.emplace_back("eig_s", eig_s_.clone());
		if (eig_t_.defined()) out.emplace_back("eig_t", eig_t_.clone());
		return out;
	}

	void PcarLoss::load_coords(const prkd::save::TensorMap& items,
		const torch::Device& device, int64_t keep)
	{
		std::unordered_map<std::string, const torch::Tensor*> table;
		table.reserve(items.size() * 2);
		for (const auto& kv : items) table.emplace(kv.first, &kv.second);

		auto get = [&](const char* k) -> const torch::Tensor&
			{
				const auto it = table.find(k);
				TORCH_CHECK(it != table.end(), "PCA 存档里缺 ", k);
				return *it->second;
			};

		mu_s_ = get("mu_s").to(device);
		V_s_  = get("V_s").to(device);
		mu_t_ = get("mu_t").to(device);
		V_t_  = get("V_t").to(device);

		TORCH_CHECK(V_s_.dim() == 2 && V_t_.dim() == 2, "PCA 存档里的 V 必须是二维");

		// 只认新版存档：完整 PCA（V 是方阵）且带特征值。
		// 旧版（只存前 keep 列、没有特征值）一律拒绝 —— 与其半懂不懂地兼容，
		// 不如明确要求"删掉重拟合"，这样 keep 才能真的变成运行时参数。
		TORCH_CHECK(V_s_.size(1) == V_s_.size(0) && V_t_.size(1) == V_t_.size(0),
			"PCA 存档不是完整矩阵（学生 ", V_s_.sizes(), " / 教师 ", V_t_.sizes(),
			"）—— 这是旧版存档，请删掉 auto/pca.{txt,bin} 重新拟合一次");
		TORCH_CHECK(table.find("eig_s") != table.end() && table.find("eig_t") != table.end(),
			"PCA 存档里没有 eig_s / eig_t —— 旧版存档，请删掉 auto/pca.{txt,bin} 重新拟合一次");

		const int64_t d_s = V_s_.size(1);
		const int64_t d_t = V_t_.size(1);

		keep_ = std::min<int64_t>({ keep, d_s, d_t });
		TORCH_CHECK(keep_ > 0, "PcarLoss::load_coords: keep 必须 > 0");
		if (keep_ != keep)
		{
			std::printf("[pcar] keep 从 %lld 截到 %lld（受存档维数限制）\n",
				(long long)keep, (long long)keep_);
		}

		Vs_use_ = V_s_.slice(/*dim=*/1, 0, keep_);
		Vt_use_ = V_t_.slice(/*dim=*/1, 0, keep_);

		eig_s_ = table.at("eig_s")->clone();
		eig_t_ = table.at("eig_t")->clone();

		fitted_ = true;

		std::printf("[pcar] 坐标系从存档读回  keep=%lld  学生 %lld×%lld / 教师 %lld×%lld\n",
			(long long)keep_,
			(long long)V_s_.size(0), (long long)d_s,
			(long long)V_t_.size(0), (long long)d_t);

		{
			const double tot_s = eig_s_.sum().item<double>();
			const double top_s = eig_s_.slice(0, 0, keep_).sum().item<double>();
			const double tot_t = eig_t_.sum().item<double>();
			const double top_t = eig_t_.slice(0, 0, keep_).sum().item<double>();
			std::printf("[pcar] 方差占比  学生 %.1f%%  教师 %.1f%%（各取前 %lld 轴，存档值）\n",
				tot_s > 0 ? 100.0 * top_s / tot_s : 0.0,
				tot_t > 0 ? 100.0 * top_t / tot_t : 0.0,
				(long long)keep_);
		}
	}


}  // namespace prkd::kd
