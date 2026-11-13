// pca_relational_kd.cpp —— 入口
//
// 本文件只做一件事：把 ImageNet 的 loader 跑一遍，做数据体检。
//
//   ① 读配置
//   ② 建 ImageNetLoader（多线程读）
//   ③ 体检：索引 / 逐批形状与标签 / 解码健康 / 坏图清单
//
// ★ 这条路【不创建模型、不加载权重、不碰 GPU】—— 所以没有卡、没有 ckpt 的机器
//   也能跑通数据这一侧。换新环境时先用它确认 loader 没坏，再接后面的事。
//
// ★ 只删了这个文件的逻辑，其他源文件（model / pca / trainer / aloi_loader）
//   一个没动 —— 将来要把 ALOI + PCA 那段接回来，从 git 里取旧版即可。

#include "pca_relational_kd.h"

#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    try
    {
        // ---- ① 配置 ----
        const std::string cfg_path = (argc > 1) ? argv[1] : "prkd.ini";
        
        prkd::cfg::Settings st = prkd::cfg::defaults();
        prkd::cfg::load_file(cfg_path, st);      // 打不开 / 内容有错 → 直接抛

        std::printf("\n[cfg] 配置文件 = %s\n", cfg_path.c_str());
        prkd::cfg::print(st);

        // 进程级全局量，必须在起任何线程之前设。
        torch::set_num_threads(st.debug.intra);

        // ---- ② 只接 ImageNet ----
        if (st.run.dataset != "imagenet")
        {
            throw std::runtime_error(
                "run.dataset = \"" + st.run.dataset + "\" —— 本入口目前只做 imagenet 的数据体检");
        }
        if (st.run.data_dir.empty())
        {
            throw std::runtime_error("没有指定 data_dir —— 写进 prkd.ini 的 run.data_dir");
        }

        const uint64_t seed = static_cast<uint64_t>(st.run.seed);

        prkd::data::LoaderOpts opt;
        opt.batch_size     = st.loader.batch;
        opt.workers        = st.loader.workers;
        opt.prefetch_depth = st.loader.prefetch;
        opt.short_side     = st.loader.short_side;
        opt.crop           = st.loader.crop;
        opt.is_train       = st.loader.is_train;
        opt.shuffle        = st.loader.shuffle;
        opt.drop_last      = st.loader.drop_last;

        // ImageNet 的 split 就是目录名，与变换分支共用 is_train 这一个开关
        const std::string data_dir =st.run.data_dir + (st.loader.is_train ? "/train" : "/val");

        std::printf("\n===== %s =====\n", data_dir.c_str());

        // ---- ③ 体检 ----
        // workers>0 时先做多线程与单线程的对拍：同一批数据必须逐字节一致。
        if (st.debug.selftest && opt.workers > 0)
        {
            prkd::test_unit::check_workers_equivalence(data_dir, opt, seed, /*n_batches=*/3);
        }

        prkd::data::ImageNetLoader ld(data_dir, opt, seed);

        prkd::test_unit::check_index(ld);
        prkd::test_unit::run_one_epoch(ld);
        prkd::test_unit::print_decode_health(ld.stats());

        if (ld.bad_count() > 0)
        {
            const std::string bad_path = st.run.out_dir + "/bad_images.tsv";
            const int64_t written = prkd::test_unit::write_bad_images(ld, bad_path);
            std::printf("坏图存档    : %s（%lld 条里的前 %lld 条）\n",
                bad_path.c_str(), (long long)ld.bad_count(), (long long)written);
        }

        std::printf("OK\n");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "\n[错误] %s\n", e.what());
        return 1;
    }
}
