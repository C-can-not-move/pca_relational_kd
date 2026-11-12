// src/pca_relational_kd.cpp —— 入口。
//
// 当前只接 ALOI。ImageNet 那条分支已暂时移除（loader 代码还在，只是入口不走了）——
// 现在这条链路的唯一目标是：把 ALOI 的特征算出来，看 PCA 的前几根轴抓的是什么。
//
// 四段：
//     ① 读配置
//     ② 建模型 + 装权重
//     ③ 建 ALOI loader
//     ④ 跑一遍
//
// 第 ④ 段目前是「读一遍、体检一遍」，特征收集接上来之后形状不变，
// 只是 while 循环体里多一句 forward_features。
//
// 检查类的东西都退到 debug.selftest 后面：模型结构自检、workers 对拍。
// 权重指纹（check_pretrained）不在这里跑 —— 它属于验证工具，不属于每次启动。
#include "pca_relational_kd.h"

#include <string>

namespace
{
    // 设备：-1 = 强制 CPU，其余按索引取卡。
    torch::Device pick_device(int index)
    {
        if (index < 0) return torch::Device(torch::kCPU);
        TORCH_CHECK(torch::cuda::is_available(),
                    "run.device_index=", index, " 但本机 CUDA 不可用（改成 -1 走 CPU）");
        return torch::Device(torch::kCUDA, index);
    }
}  // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    try
    {
        // ---- ① 配置 ----
        prkd::cfg::Settings st = prkd::cfg::defaults();

        const std::string cfg_path = (argc > 1) ? argv[1] : "prkd.ini";
        prkd::cfg::load_file(cfg_path, st);      // 打不开 / 内容有错 → 直接抛

        std::printf("\n[cfg] 配置文件 = %s\n", cfg_path.c_str());
        prkd::cfg::print(st);

        // 进程级全局量，必须在起任何线程之前设。
        torch::set_num_threads(st.debug.intra);

        if (st.debug.selftest)
        {
            prkd::test_unit::run_model_checks();
        }

        // ---- ② 模型 + 权重 ----
        const torch::Device device = pick_device(st.run.device_index);

        auto net = prkd::model::h_resnet18(1000);
        net->to(device);

        std::printf("\n===== 模型 =====\n");
        std::printf("设备        : %s\n", device.str().c_str());
        if (st.run.ckpt.empty())
        {
            std::printf("权重        : 跳过（run.ckpt 未设置）\n");
        }
        else
        {
            const size_t loaded = prkd::model::load_checkpoint(net, st.run.ckpt);
            std::printf("权重        : %s（%zu 个张量）\n", st.run.ckpt.c_str(), loaded);
        }

        // ---- ③ 数据：只接 ALOI ----
        if (st.run.dataset != "aloi")
        {
            throw std::runtime_error(
                "run.dataset = \"" + st.run.dataset + "\" 目前不支持 —— 只接了 aloi"
                "（ImageNet 分支已暂时移除，loader 代码仍在）");
        }
        if (st.run.data_dir.empty())
        {
            throw std::runtime_error("没有指定 data_dir —— 写进 prkd.ini 的 run.data_dir");
        }

        const uint64_t seed = static_cast<uint64_t>(st.run.seed);

        prkd::data::AloiOpts aopt;
        aopt.batch_size     = st.loader.batch;
        aopt.workers        = st.loader.workers;
        aopt.prefetch_depth = st.loader.prefetch;
        aopt.crop           = st.loader.crop;
        aopt.shuffle        = st.loader.shuffle;
        aopt.drop_last      = st.loader.drop_last;
        aopt.view_begin     = st.aloi.view_begin;
        aopt.view_end       = st.aloi.view_end;

        std::printf("\n===== ALOI  视角 [%lld, %lld) =====\n",
            (long long)aopt.view_begin, (long long)aopt.view_end);

        if (st.debug.selftest && aopt.workers > 0)
        {
            prkd::test_unit::check_workers_equivalence_aloi(
                st.run.data_dir, aopt, seed, /*n_batches=*/3);
        }

        prkd::data::AloiLoader ld(st.run.data_dir, aopt, seed);

        // ---- ④ 跑一遍：收集特征 → PCA → 打印谱与判别比（这一轮不落盘）----
        prkd::test_unit::check_index(ld);

        constexpr int64_t kDim = 512;               // resnet18 的 BasicBlock expansion=1，layer4 输出 512
                                                    // 换回 resnet152 时改成 2048
        const int64_t n_total = ld.size();

        prkd::util::Pca pca(kDim);

        std::vector<float>   feats;                 // (N, kDim) f32，行优先
        std::vector<int64_t> labels(n_total, 0);
        std::vector<int64_t> angles(n_total, -1);
        feats.reserve(static_cast<size_t>(n_total) * kDim);

        net->eval();
        const auto t_feat0 = std::chrono::steady_clock::now();
        int64_t written = 0;
        int64_t n_done  = 0;                        // 已处理批数，只用于进度
        const int64_t nb_total = ld.num_batches();
        const int64_t log_every = st.debug.log_every;   // 0 = 不打
        {
            torch::NoGradGuard no_grad;
            while (ld.has_next())
            {
                const prkd::data::ImageBatch batch = ld.next();
                const auto f = net->forward_features(batch.images.to(device))
                                      .to(torch::kCPU).contiguous();   // (n, kDim) f32

                pca.add(f);

                const auto acc = f.accessor<float, 2>();
                for (int64_t i = 0; i < batch.n; ++i)
                {
                    for (int64_t j = 0; j < kDim; ++j) feats.push_back(acc[i][j]);
                    labels[static_cast<size_t>(written)] = batch.metas[static_cast<size_t>(i)].label;
                    angles[static_cast<size_t>(written)] = batch.metas[static_cast<size_t>(i)].angle;
                    ++written;
                }

                // 进度：每 log_every 批一行。跑满 12 分钟不吭声，会让人以为卡死。
                ++n_done;
                if (log_every > 0 && (n_done % log_every) == 0)
                {
                    const double el = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t_feat0).count();
                    std::printf("  [feat] %lld/%lld 批  %lld 张  %.1f img/s  ETA %.0f s\n",
                        (long long)n_done, (long long)nb_total,
                        (long long)written,
                        el > 0 ? written / el : 0.0,
                        el > 0 ? el * static_cast<double>(nb_total - n_done) / n_done : 0.0);
                    std::fflush(stdout);
                }
            }
        }
        const double sec_feat = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_feat0).count();

        std::printf("\n特征收集    : %lld 张 × %lld 维  (%.1f s, %.1f img/s)\n",
            (long long)written, (long long)kDim, sec_feat, written / sec_feat);

        // ---- PCA ----
        const auto coord = pca.fit();

        const auto total_var = coord.eigen.sum().item<double>();
        std::printf("样本 / 维数 : %lld / %lld\n", (long long)pca.count(), (long long)pca.dim());
        std::printf("特征值之和  : %.6f\n", total_var);

        // ---- 判别比 η ----
        // z = (x − μ)V，再算 eta_j = 1 − 组内方差_j / 总方差_j
        // eta≈1 → 这根轴主要由"物体身份"决定；eta≈0 → 由视角决定
        const auto z = prkd::util::transform(
            coord,
            torch::from_blob(feats.data(), { written, kDim }, torch::kFloat32).clone());

        const auto lab = torch::from_blob(labels.data(), { written }, torch::kInt64).clone();
        const int64_t n_class = ld.num_classes();

        auto class_sum = torch::zeros({ n_class, coord.dim() }, coord.eigen.options());
        class_sum.index_add_(0, lab, z.to(torch::kFloat64));
        auto class_cnt = torch::zeros({ n_class }, coord.eigen.options());
        class_cnt.index_add_(0, lab, torch::ones({ written }, coord.eigen.options()));
        const auto class_mean = class_sum / class_cnt.unsqueeze(1);

        const auto dev  = z.to(torch::kFloat64) - class_mean.index_select(0, lab);
        const auto wvar = (dev * dev).mean(0);                   // (D,) 组内方差
        const auto eta  = 1.0 - wvar / coord.eigen;

        // ---- 打印前 20 根 ----
        const int64_t top = std::min<int64_t>(20, coord.dim());
        std::printf("\n  轴   特征值        占比        η      说明\n");
        const auto ev  = coord.eigen.accessor<double, 1>();
        const auto et  = eta.accessor<double, 1>();
        for (int64_t j = 0; j < top; ++j)
        {
            const char* tag = et[j] > 0.7 ? "类间（物体身份）"
                            : et[j] < 0.3 ? "视角（几乎无用）"
                                          : "混合";
            std::printf("  %2lld  %10.4f  %8.4f%%   %6.3f   %s\n",
                (long long)j, ev[j], 100.0 * ev[j] / total_var, et[j], tag);
        }

        double share_top20 = 0.0;
        for (int64_t j = 0; j < top; ++j) share_top20 += ev[j];
        std::printf("\n  前 %lld 根占总方差 %.2f%%\n", (long long)top, 100.0 * share_top20 / total_var);

        std::printf("OK\n");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
