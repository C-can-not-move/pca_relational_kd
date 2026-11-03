// src/pca_relational_kd.cpp —— 数据管线冒烟测试
//
// 验的是三件事：
//   ① 索引对不对（类数 / 样本数 / num_batches）
//   ② 每批的形状、dtype、标签范围
//   ③ 数据里有多少张【解码不干净】的图 —— 以及吞吐
//
// ③ 是"只计数、不跳过"：坏图照样进 batch，只是这里会报出来。
// 想知道具体是哪几张，得靠离线体检，不在训练路径里做。
#include "pca_relational_kd.h"

namespace
{
    void print_batch_head(const prkd::data::ImageBatch& b, int64_t k)
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
}

int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    try
    {
        prkd::cfg::RunConfig cfg = prkd::cfg::defaults();

        prkd::data::LoaderOpts opt;
        opt.batch_size = 64;
        opt.workers = 0;              // 同步版
        opt.train = true;

        const std::string data_dir = cfg.data_dir + "/train";
        prkd::data::ImageNetLoader ld(data_dir, opt, /*seed=*/2026);

        std::printf("\n===== %s =====\n", data_dir.c_str());
        std::printf("模式        : %s\n", !opt.train ? "val（不洗牌，末批可短）" : "train（洗牌，drop_last）");
        std::printf("类数        : %lld\n", (long long)ld.num_classes());
        std::printf("样本数      : %lld\n", (long long)ld.size());
        std::printf("batch_size  : %lld\n", (long long)opt.batch_size);
        std::printf("num_batches : %lld\n", (long long)ld.num_batches());

        const int64_t expect_nb = !opt.train
            ? (ld.size() + opt.batch_size - 1) / opt.batch_size
            : ld.size() / opt.batch_size;
        std::printf("期望        : %lld   %s\n", (long long)expect_nb,
            expect_nb == ld.num_batches() ? "OK" : "✗ 不符");
        if (expect_nb != ld.num_batches())
        {
            throw std::runtime_error("num_batches 与期望不符");
        }

        int64_t b = 0;
        while (ld.has_next())
        {
            const prkd::data::ImageBatch batch = ld.next();
            if (b < 3) print_batch_head(batch, b);
            ++b;

            // 冷读一个 epoch 要十几分钟，没有进度会以为是卡住了。
            if (b % 500 == 0) std::printf("  ... 已读 %lld 个 batch\n", (long long)b);
        }

        const auto st = ld.stats();
        std::printf("\n读了 %lld 个 batch / %lld 张\n", (long long)b, (long long)st.images);
        std::printf("吞吐 %.1f img/s   总耗时 %.2f 秒\n",
            st.img_per_sec(), st.seconds);

        // ---- 解码体检 ----
        // 两项互斥：不全（更严重）优先于"有警告但读完了"。
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
            : "→ 有坏图，但按要求没有跳过");

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
