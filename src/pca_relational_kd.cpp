// src/pca_relational_kd.cpp —— 数据管线冒烟测试的入口。
//
// 断言全部在 test_unit/loader_check.*，这里只负责解析参数 + 按顺序调用：
//   ① 索引规模   ② 多线程 vs 同步逐元素对拍   ③ 全量跑一遍（逐批检查）   ④ 解码体检
#include "pca_relational_kd.h"

#include "test_unit/loader_check.h"

namespace
{
    // ---- 命令行的选项 ----
    // 用 enum + switch，而不是一长串 if-else：新增选项时编译器会提醒漏掉的 case，
    // 而且 "未知参数" 落在 default 上，不会某天被一个写错的 == 悄悄吃掉。
    enum class Cli
    {
        Workers,
        Batch,
        Intra,
        Val,
        NoSelfTest,
        Help,
        Unknown,
    };

    Cli to_cli(const std::string& s)
    {
        if (s == "--workers") return Cli::Workers;
        if (s == "--batch") return Cli::Batch;
        if (s == "--intra") return Cli::Intra;
        if (s == "--val") return Cli::Val;
        if (s == "--no-selftest") return Cli::NoSelfTest;
        if (s == "-h" || s == "--help") return Cli::Help;
        return Cli::Unknown;
    }

    void usage(const char* prog)
    {
        std::printf("用法: %s [选项]\n\n", prog);

        std::printf("选项:\n");
        std::printf("  --workers N    读图线程数（默认 = 逻辑核数/2，即物理核数；0 = 主线程同步）\n");
        std::printf("  --batch N      batch_size（默认 64）\n");
        std::printf("  --intra N      ATen intra-op 线程数（默认 1）\n");
        std::printf("                 默认 1 是实测结论：loader 里的算子只有 3x224x224，\n");
        std::printf("                 切多份的协调开销抵消收益，还会和 worker 抢物理核。\n");
        std::printf("                 只影响 CPU 算子；GPU 上的推理/训练不受它影响。\n");
        std::printf("  --val          跑 val 分支（不洗牌、末批可短）\n");
        std::printf("  --no-selftest  跳过 workers=0 对拍（扫参数时省两次索引构建）\n");
        std::printf("  -h, --help     这一屏\n\n");

        const std::string p = std::string(prog);
        std::printf("示例:\n");
        std::printf("  %-52s %s\n", p.c_str(),                         "默认：train，物理核数个线程");
        std::printf("  %-52s %s\n", (p + " --val").c_str(),           "val 分支（不洗牌、末批可短）");
        std::printf("  %-52s %s\n", (p + " --workers 0").c_str(),     "同步版，看冷/热读的真实差别");
        std::printf("  %-52s %s\n", (p + " --no-selftest --workers 8").c_str(), "扫线程数曲线（跳过对拍）");
        std::printf("  %-52s %s\n", (p + " --batch 128 --workers 6").c_str(),  "batch 更大时一并调线程数");
        std::printf("  %-52s %s\n", (p + " --intra 8 --workers 2").c_str(),    "CPU 上跑推理时的反配：让 intra 拿大");
    }
    // 取下一个参数并转成整数；不是整数就按参数错误处理。
    // 这里直接 exit(2)，因为参数错是"用法错"，不该走 main 那个 catch 变成 ERROR。
    int64_t next_int(int argc, char** argv, int& i, const char* name)
    {
        if (i + 1 >= argc)
        {
            std::printf("%s 需要一个整数参数\n", name);
            std::exit(2);
        }
        const char* s = argv[++i];
        try
        {
            return std::stoll(s);
        }
        catch (const std::exception&)
        {
            std::printf("%s 的参数不是整数: %s\n", name, s);
            std::exit(2);
        }
    }

    // 逻辑核数的一半 ≈ 物理核数（有超线程时）。实测的吞吐膝点正好落在这儿：
    // 再往上加只是多几个线程抢同一批核。
    int64_t default_workers()
    {
        const unsigned hw = std::thread::hardware_concurrency();
        return hw > 1 ? static_cast<int64_t>(hw / 2) : 1;
    }
}  // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    try
    {
        // 用法示例里只显示 exe 名，不显示整条绝对路径（否则一行长得没法读）
        const std::string prog = std::filesystem::path(argv[0]).filename().string();

        prkd::cfg::RunConfig cfg = prkd::cfg::defaults();

        prkd::data::LoaderOpts opt;
        opt.batch_size = 64;
        opt.workers = default_workers();
        opt.train = true;

        int  intra = 1;
        bool selftest = true;

        for (int i = 1; i < argc; ++i)
        {
            const std::string a = argv[i];
            switch (to_cli(a))
            {
            case Cli::Workers:
                opt.workers = next_int(argc, argv, i, "--workers");
                break;
            case Cli::Batch:
                opt.batch_size = next_int(argc, argv, i, "--batch");
                break;
            case Cli::Intra:
                intra = static_cast<int>(next_int(argc, argv, i, "--intra"));
                break;
            case Cli::Val:
                opt.train = false;
                break;
            case Cli::NoSelfTest:
                selftest = false;
                break;
            case Cli::Help:
                usage(prog.c_str());
                return 0;
            case Cli::Unknown:
            default:
                std::printf("未知参数: %s\n\n", a.c_str());
                usage(prog.c_str());
                return 2;
            }
        }

        if (intra < 1) intra = 1;
        if (opt.workers < 0) opt.workers = 0;

        // 在起任何线程之前定下来。它是进程级全局量 —— 所以只能在这里设，
        // 不能让 loader 自己改（那样主线程的训练前向也会被一起改成单线程）。
        // 另：PyTorch 规定必须在任何张量运算之前调用，晚了会警告甚至不生效。
        torch::set_num_threads(intra);

        const std::string data_dir = cfg.data_dir + (opt.train ? "/train" : "/val");
        const uint64_t seed = static_cast<uint64_t>(cfg.seed);

        std::printf("\n===== %s =====\n", data_dir.c_str());
        std::printf("读图线程    : %lld        intra-op: %d\n",
            (long long)opt.workers, intra);

        // ② 对拍自己建两套 loader，放在最前面 —— 不和下面的实例共用 cursor
        if (selftest && opt.workers > 0)
        {
            prkd::test_unit::check_workers_equivalence(data_dir, opt, seed, /*n_batches=*/3);
        }

        prkd::data::ImageNetLoader ld(data_dir, opt, seed);

        // ① 索引规模
        prkd::test_unit::check_index(ld);

        // ③ 跑满一个 epoch（内部每批 check_batch，前 3 批打印，最后报吞吐）
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
