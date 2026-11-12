// src/test_unit/loader_check.cpp —— 数据管线验收检查的实现。
#include "test_unit/loader_check.h"

#include "data/aloi_loader.h"       // 对拍要能构造 ALOI loader

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace prkd::test_unit
{
    namespace
    {
        // 失败时统一从这里抛，带上 batch 号，方便定位是第几批出的问题。
        [[noreturn]] void fail(int64_t k, const std::string& what)
        {
            throw std::runtime_error("batch " + std::to_string(k) + ": " + what);
        }
    }  // namespace

    void check_index(const data::LoaderCore& ld)
    {
        // 不收 opts：问 loader 要，保证验证的是它【实际在用】的那套。
        // 问的是 core_options()（机器层），不是派生类那份 —— 检查关心的是机器拿到了什么。
        const data::CoreOpts& opt = ld.core_options();
        const int64_t n = ld.size();
        const int64_t bs = opt.batch_size;

        // num_batches 由 drop_last 决定，不由 is_train 推。
        const int64_t expect = opt.drop_last ? n / bs : (n + bs - 1) / bs;

        std::printf("shuffle=%d  drop_last=%d  out=%lldx%lld  workers=%lld\n",
            (int)opt.shuffle, (int)opt.drop_last,
            (long long)opt.out_h, (long long)opt.out_w, (long long)opt.workers);
        std::printf("类数        : %lld\n", (long long)ld.num_classes());
        std::printf("样本数      : %lld\n", (long long)n);
        std::printf("batch_size  : %lld\n", (long long)bs);
        std::printf("num_batches : %lld\n", (long long)ld.num_batches());
        std::printf("期望        : %lld   %s\n", (long long)expect,expect == ld.num_batches() ? "OK" : "✗ 不符");

        if (ld.num_classes() <= 1)
        {
            throw std::runtime_error("索引检查: 类数 " + std::to_string(ld.num_classes()) + " 不合理");
        }
        if (n <= 0)
        {
            throw std::runtime_error("索引检查: 样本数为 0");
        }
        if (expect != ld.num_batches())
        {
            throw std::runtime_error("索引检查: num_batches 与期望不符");
        }
        if (opt.drop_last && n < bs)
        {
            throw std::runtime_error("索引检查: drop_last=true 但样本数 < batch_size，会得到 0 个 batch");
        }
    }

    void check_batch(const data::ImageBatch& b, const data::LoaderCore& ld, int64_t k)
    {
        const int64_t crop = ld.core_options().out_h;
        const int64_t n_classes = ld.num_classes();
        const int64_t n = b.images.size(0);

        // ---- 形状 ----
        if (b.images.dtype() != torch::kFloat32) fail(k, "images 的 dtype 不是 float32");
        if (b.images.dim() != 4) fail(k, "images 不是 4 维");
        if (b.images.size(1) != 3) fail(k, "images 通道数不是 3");
        if (b.images.size(2) != crop || b.images.size(3) != crop)
        {
            fail(k, "images 边长不是 crop=" + std::to_string(crop));
        }
        if (b.labels.dtype() != torch::kInt64) fail(k, "labels 的 dtype 不是 int64");
        if (b.labels.numel() != n) fail(k, "labels 长度与 N 不符");
        if (static_cast<int64_t>(b.indices.size()) != n) fail(k, "indices 长度与 N 不符");

        // ---- 取值范围 ----
        // 越界会让后面的 CrossEntropy 直接报错，或者（更糟）在别处静默算错。
        const int64_t lo = b.labels.min().item<int64_t>();
        const int64_t hi = b.labels.max().item<int64_t>();
        if (lo < 0 || hi >= n_classes)
        {
            fail(k, "label 越界: [" + std::to_string(lo) + ", " +
                    std::to_string(hi) + "] 不在 [0, " +
                    std::to_string(n_classes) + ")");
        }

        // indices 也必须落在样本总数范围内
        for (int64_t i = 0; i < n; ++i)
        {
            const int64_t idx = b.indices[static_cast<size_t>(i)];
            if (idx < 0 || idx >= ld.size()) fail(k, "indices 越界");
        }

        // ---- 有限性（只查前 10 批）----
        // 这一项要把整批 960 万个数扫一遍，intra=1 时单线程约 75 ms/批。
        // 每批都查会把吞吐数字彻底掩盖掉 —— 它不是训练路径的一部分，是测试。
        if (k < 10 && !torch::isfinite(b.images).all().item<bool>())
        {
            fail(k, "images 里有 nan 或 inf");
        }
    }

    namespace
    {
        // 两个实例逐批对拍。只依赖基类 —— ImageNet 与 ALOI 走同一份比较逻辑。
        void compare_batches(data::LoaderCore& a, data::LoaderCore& b, int64_t n_batches)
        {
            a.start_epoch(0);
            b.start_epoch(0);

            for (int64_t k = 0; k < n_batches; ++k)
            {
                const auto x = a.next();
                const auto y = b.next();

                const bool idx_ok = (x.indices == y.indices);
                const bool lab_ok = torch::equal(x.labels, y.labels);
                const bool img_ok = torch::equal(x.images, y.images);

                std::printf("  [对拍] batch %lld  indices %s  labels %s  images %s\n",
                    (long long)k,
                    idx_ok ? "OK" : "✗", lab_ok ? "OK" : "✗", img_ok ? "OK" : "✗");

                if (!idx_ok) throw std::runtime_error("对拍: indices 不一致（第 " + std::to_string(k) + " 批）");
                if (!lab_ok) throw std::runtime_error("对拍: labels 不一致（第 " + std::to_string(k) + " 批）");
                // 精确比较：两边跑的是同一串算子、同样的输入、同样的形状，
                // 每个算子都是确定性的（与被切成几份无关），所以必须逐位相等。
                if (!img_ok) throw std::runtime_error("对拍: images 不一致（第 " + std::to_string(k) + " 批）");
            }

            std::printf("  [对拍] 前 %lld 批逐元素相同\n\n", (long long)n_batches);
        }
    }  // namespace

    void check_workers_equivalence(const std::string& data_dir,
                                   const data::LoaderOpts& opt,
                                   uint64_t seed,
                                   int64_t n_batches)
    {
        data::LoaderOpts o_sync = opt;
        o_sync.workers = 0;

        // 两个独立实例：与调用方的任何 loader 都无关，不会推进别人的 cursor。
        data::ImageNetLoader ld_sync(data_dir, o_sync, seed);
        data::ImageNetLoader ld_thread(data_dir, opt, seed);

        std::printf("  [对拍] workers=0  vs  workers=%lld\n", (long long)opt.workers);
        compare_batches(ld_sync, ld_thread, n_batches);
    }

    void check_workers_equivalence_aloi(const std::string& root,
                                        const data::AloiOpts& opt,
                                        uint64_t seed,
                                        int64_t n_batches)
    {
        data::AloiOpts o_sync = opt;
        o_sync.workers = 0;

        data::AloiLoader ld_sync(root, o_sync, seed);
        data::AloiLoader ld_thread(root, opt, seed);

        std::printf("  [对拍] ALOI  workers=0  vs  workers=%lld  view[%lld,%lld)\n",
            (long long)opt.workers, (long long)opt.view_begin, (long long)opt.view_end);
        compare_batches(ld_sync, ld_thread, n_batches);
    }

    void run_one_epoch(data::LoaderCore& ld)
    {
        using clk = std::chrono::steady_clock;

        ld.start_epoch(0);
        const auto t0 = clk::now();

        int64_t b = 0;
        while (ld.has_next())
        {
            const data::ImageBatch batch = ld.next();

            check_batch(batch, ld, b);              // 每批都查
            if (b < 3) print_batch_head(batch, b);  // 只前 3 批打印
            ++b;

            // 冷读一个 epoch 要十几分钟，没有进度会以为是卡住了
            if (b % 50 == 0) std::printf("  ... 已读 %lld 个 batch\n", (long long)b);
        }

        const double wall = std::chrono::duration<double>(clk::now() - t0).count();

        const auto st = ld.stats();
        std::printf("\n读了 %lld 个 batch / %lld 张\n", (long long)b, (long long)st.images);
        std::printf("墙钟           %.2f 秒   → %.1f img/s\n",
            wall, wall > 0.0 ? (double)st.images / wall : 0.0);
        std::printf("其中 loader 侧  %.2f 秒   → %.1f img/s   （next() 内部累计）\n",
            st.seconds, st.img_per_sec());
        std::printf("（墙钟含每批校验；两者之差 = 校验占的时间）\n");
    }

    void print_batch_head(const data::ImageBatch& b, int64_t k)
    {
        const int64_t n = std::min<int64_t>(5, b.images.size(0));

        std::printf("  batch %lld  N=%lld  min %.3f  max %.3f\n",
            (long long)k, (long long)b.images.size(0),
            b.images.min().item<double>(), b.images.max().item<double>());

        std::printf("    indices[0..4] =");
        for (int64_t i = 0; i < n; ++i)
            std::printf(" %lld", (long long)b.indices[static_cast<size_t>(i)]);

        const int64_t* lab = b.labels.data_ptr<int64_t>();
        std::printf("\n    labels [0..4] =");
        for (int64_t i = 0; i < n; ++i)
            std::printf(" %lld", (long long)lab[i]);
        std::printf("\n");
    }

    void print_decode_health(const data::LoaderStats& st)
    {
        // 两项互斥：不全（更严重）优先于"有警告但读完了"
        const int64_t incomplete = st.incomplete_images;
        const int64_t warned = st.warned_images;
        const int64_t clean = st.images - incomplete - warned;

        std::printf("\n----- 解码体检 -----\n");
        std::printf("干净        : %lld\n", (long long)clean);
        std::printf("有警告      : %lld   （读完了，但 libjpeg 报过警）\n", (long long)warned);
        std::printf("数据不全    : %lld   （提前停住，缺口是 0 黑）\n", (long long)incomplete);
        std::printf("合计        : %lld\n", (long long)st.images);
        std::printf("%s\n", (incomplete + warned) == 0
            ? "→ 这一轮读到的图全部干净"
            : "→ 有坏图（按要求只计数、不跳过）");
    }
    int64_t write_bad_images(const data::LoaderCore& ld, const std::string& path)
    {
        namespace fs = std::filesystem;

        const fs::path p(path);
        if (!p.parent_path().empty())
        {
            std::error_code ec;
            fs::create_directories(p.parent_path(), ec);
        }

        std::ofstream f(p);
        if (!f) throw std::runtime_error("写不了坏图清单: " + path);

        f << "path\twhy\n";
        for (const auto& bad : ld.bad_images())
        {
            f << bad.path << '\t' << bad.why << '\n';
        }

        if (!f) throw std::runtime_error("写坏图清单时出错: " + path);
        return static_cast<int64_t>(ld.bad_images().size());
    }
}  // namespace prkd::test_unit
