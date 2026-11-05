// src/pca_relational_kd.cpp —— 数据管线冒烟测试的入口。
//
// 唯一的命令行参数是【配置文件路径】，其余全部来自 prkd.ini。
// 断言都在 test_unit/loader_check.*，这里只负责：读配置 → 组装 loader → 跑一轮。
#include "pca_relational_kd.h"

#include "test_unit/loader_check.h"

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    try
    {
        // ---- 配置 ----
        prkd::cfg::Settings st = prkd::cfg::defaults();

        const std::string cfg_path = (argc > 1) ? argv[1] : "prkd.ini";
        prkd::cfg::load_file(cfg_path, st);      // 打不开 / 内容有错 → 直接抛

        std::printf("\n[cfg] 配置文件 = %s\n", cfg_path.c_str());
        prkd::cfg::print(st);

        if (st.run.data_dir.empty())
        {
            throw std::runtime_error(
                "没有指定 data_dir —— 写进 prkd.ini 的 run.data_dir");
        }

        // 进程级全局量，必须在起任何线程之前设。
        torch::set_num_threads(st.debug.intra);

        // ---- 组装 ----
        prkd::data::LoaderOpts opt;
        opt.batch_size = st.loader.batch;
        opt.workers    = st.loader.workers;
        opt.short_side = st.loader.short_side;
        opt.crop       = st.loader.crop;
        opt.train      = st.loader.train;

        const std::string data_dir =
            st.run.data_dir + (st.loader.train ? "/train" : "/val");
        const uint64_t seed = static_cast<uint64_t>(st.run.seed);

        std::printf("\n===== %s =====\n", data_dir.c_str());

        // ② 对拍自己建两套 loader，放在最前面 —— 不和下面的实例共用
        if (st.debug.selftest && opt.workers > 0)
        {
            prkd::test_unit::check_workers_equivalence(data_dir, opt, seed, /*n_batches=*/3);
        }

        prkd::data::ImageNetLoader ld(data_dir, opt, seed);

        // ① 索引规模
        prkd::test_unit::check_index(ld);

        // ③ 跑满一个 epoch（内部每批 check_batch、前 3 批打印、最后报吞吐）
        prkd::test_unit::run_one_epoch(ld);

        // ④ 解码体检
        prkd::test_unit::print_decode_health(ld.stats());

        std::printf("\n判据（顺序可复现）：同一条命令跑两次，上面的 indices[0..4] 必须逐位相同。\n");
        std::printf("OK\n");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
