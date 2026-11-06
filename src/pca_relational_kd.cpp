// src/pca_relational_kd.cpp —— 当前阶段：模型检查入口。
//
// 数据管线的冒烟测试保留在 #if 0 里，切回去时删掉 return 0 并取消注释。
#include "pca_relational_kd.h"

#include "test_unit/loader_check.h"
#include "test_unit/model_check.h"

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    try
    {
        // ====================================================================
        // 模型检查：前向形状 / 参数量 / 反向梯度
        // ====================================================================
        prkd::test_unit::run_model_checks();

        std::printf("\nOK\n");
        return 0;

        // ====================================================================
        // 数据管线的冒烟测试，暂时不跑。恢复时删掉上面的 return 0 并放开这里。
        // ====================================================================
#if 0
        prkd::cfg::Settings st = prkd::cfg::defaults();

        const std::string cfg_path = (argc > 1) ? argv[1] : "prkd.ini";
        prkd::cfg::load_file(cfg_path, st);

        std::printf("\n[cfg] 配置文件 = %s\n", cfg_path.c_str());
        prkd::cfg::print(st);

        if (st.run.data_dir.empty())
        {
            throw std::runtime_error(
                "没有指定 data_dir —— 写进 prkd.ini 的 run.data_dir");
        }

        torch::set_num_threads(st.debug.intra);

        prkd::data::LoaderOpts opt;
        opt.batch_size = st.loader.batch;
        opt.workers    = st.loader.workers;
        opt.prefetch_depth = st.loader.prefetch;
        opt.short_side = st.loader.short_side;
        opt.crop       = st.loader.crop;
        opt.is_train   = st.loader.is_train;

        const std::string data_dir =
            st.run.data_dir + (st.loader.is_train ? "/train" : "/val");
        const uint64_t seed = static_cast<uint64_t>(st.run.seed);

        std::printf("\n===== %s =====\n", data_dir.c_str());

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

        std::printf("\n判据（顺序可复现）：同一条命令跑两次，上面的 indices[0..4] 必须逐位相同。\n");
        std::printf("OK\n");
        return 0;
#endif
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
