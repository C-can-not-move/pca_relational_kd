// src/pca_relational_kd.cpp —— 当前阶段：模型与权重检查的入口。
//
// 顺序：结构（前向/反向/参数量）→ 权重（加载 + 指纹）。
// 数据管线的冒烟测试留在 #if 0 里，切回去时删掉 return 0 并放开注释。
#include "pca_relational_kd.h"

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

        // 进程级全局量，必须在起任何线程之前设。
        torch::set_num_threads(st.debug.intra);

        // ---- 结构：「这个网络是不是对的」 ----
        prkd::test_unit::run_model_checks();

        // ---- 权重：「搬进来的值是不是对的」 ----
        //
        // 四个位置各验一件事：
        //   conv1.weight          文件开头的偏移
        //   bn1.running_mean      buffer 有没有被正确搬运（只存参数的经典错误）
        //   layer4.2.conv3.weight 中间偏移累积
        //   fc.bias               文件末尾的位置
        //
        // 期望值来自 tool/export_teacher_flat.py 导出时的输出。
        {
            auto net = prkd::model::h_resnet152(1000);

            std::printf("\n===== 预训练权重 =====\n");
            if (st.run.ckpt.empty())
            {
                std::printf("run.ckpt 未设置，跳过权重加载\n");
            }
            else
            {
                prkd::test_unit::check_pretrained(net, st.run.ckpt);
            }
        }

        std::printf("\nOK\n");
        // return 0

        // ====================================================================
        // 数据管线的冒烟测试，暂时不跑。
        // ====================================================================

        if (st.run.data_dir.empty())
        {
            throw std::runtime_error(
                "没有指定 data_dir —— 写进 prkd.ini 的 run.data_dir");
        }

        prkd::data::LoaderOpts opt;
        opt.batch_size = st.loader.batch;
        opt.workers = st.loader.workers;
        opt.prefetch_depth = st.loader.prefetch;
        opt.short_side = st.loader.short_side;
        opt.crop = st.loader.crop;
        opt.is_train = st.loader.is_train;

        // 三个开关各自从配置读，谁也不派生谁。
        // ini 里要把三种模式都写全（见 prkd.ini 的注释）：
        //   训练 train=true shuffle=true drop_last=true
        //   评测 train=false shuffle=false drop_last=false
        opt.shuffle   = st.loader.shuffle;
        opt.drop_last = st.loader.drop_last;

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
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}