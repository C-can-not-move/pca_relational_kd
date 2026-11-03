// src/pca_relational_kd.cpp —— 数据管线冒烟测试
// 验证的是：索引（类数/样本数）、每批的形状与 dtype、标签范围、
// 有没有 nan/inf、以及吞吐。（顺序可复现要跑两次比 indices，见末尾提示。）
#include "pca_relational_kd.h"

namespace
{
    void print_indices(const prkd::data::ImageBatch& b, int64_t k)
    {
        const int64_t* idx = b.indices.data();
        const int64_t* lab = b.labels.data_ptr<int64_t>();
        std::printf("  batch %lld  N=%lld  min %.3f  max %.3f\n",
            (long long)k, (long long)b.images.size(0),
            b.images.min().item<double>(), b.images.max().item<double>());
        std::printf("    indices[0..4] =");
        for (int64_t i = 0; i < std::min<int64_t>(5, b.images.size(0)); ++i)
            std::printf(" %lld", (long long)idx[i]);
        std::printf("\n    labels [0..4] =");
        for (int64_t i = 0; i < std::min<int64_t>(5, b.images.size(0)); ++i)
            std::printf(" %lld", (long long)lab[i]);
        std::printf("\n");
    }
}

namespace
{
    // 分阶段计时：把每一步单独测一遍，看 7.5 ms 花在哪。
    // 先把图片全部读进内存，这样"读文件"不混进来。
    void bench_stages(const std::string& dir, int n_img)
    {
        namespace fs = std::filesystem;
        using clk = std::chrono::steady_clock;

        std::vector<std::string> paths;
        for (const auto& e : fs::recursive_directory_iterator(dir))
        {
            if (e.is_regular_file() && e.path().extension() == ".JPEG")
            {
                paths.push_back(e.path().string());
                if ((int)paths.size() >= n_img) break;
            }
        }
        if (paths.empty()) { std::printf("[bench] 没有图片\n"); return; }

        // 预读进内存
        std::vector<std::vector<uint8_t>> blobs;
        blobs.reserve(paths.size());
        for (const auto& p : paths)
        {
            std::ifstream in(p, std::ios::binary);
            in.seekg(0, std::ios::end);
            const auto n = (size_t)in.tellg();
            in.seekg(0);
            std::vector<uint8_t> b(n);
            in.read((char*)b.data(), (std::streamsize)n);
            blobs.push_back(std::move(b));
        }

        double t_dec = 0, t_chw = 0, t_rrc = 0, t_flip = 0, t_norm = 0;
        int64_t n_ok = 0;
        const auto norm = prkd::data::imagenet_norm();

        for (size_t i = 0; i < blobs.size(); ++i)
        {
            auto t0 = clk::now();
            auto bmp = prkd::data::decode_rgb8(blobs[i].data(), blobs[i].size());
            auto t1 = clk::now();
            auto x = prkd::data::to_chw01(bmp.data.data(), bmp.h, bmp.w);
            auto t2 = clk::now();
            std::mt19937_64 rng(12345 + i);
            x = prkd::data::random_resized_crop(x, 224, rng);
            auto t3 = clk::now();
            x = prkd::data::random_horizontal_flip(x, rng);
            auto t4 = clk::now();
            x = prkd::data::normalize(x, norm);
            auto t5 = clk::now();

            auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            t_dec += ms(t0, t1); t_chw += ms(t1, t2); t_rrc += ms(t2, t3);
            t_flip += ms(t3, t4); t_norm += ms(t4, t5);
            ++n_ok;
        }

        const double k = 1.0 / (double)n_ok;
        std::printf("\n===== 分阶段（%lld 张，单线程）=====\n", (long long)n_ok);
        std::printf("  解码 decode_rgb8      %7.3f ms/张\n", t_dec * k);
        std::printf("  to_chw01              %7.3f ms/张\n", t_chw * k);
        std::printf("  RRC(含 224 上采样)    %7.3f ms/张\n", t_rrc * k);
        std::printf("  flip                  %7.3f ms/张\n", t_flip * k);
        std::printf("  normalize             %7.3f ms/张\n", t_norm * k);
        std::printf("  --------------------------------\n");
        std::printf("  合计                  %7.3f ms/张\n", (t_dec + t_chw + t_rrc + t_flip + t_norm) * k);
        std::printf("  (不含读文件)\n");
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

		std::string data_dir = cfg.data_dir+"/train";
        prkd::data::ImageNetLoader ld(data_dir, opt, /*seed=*/2026);

        std::printf("\n===== %s =====\n", data_dir.c_str());
        std::printf("模式        : %s\n", !opt.train ? "val（不洗牌，末批可短）" : "train（洗牌，drop_last）");
        std::printf("类数        : %lld\n", (long long)ld.num_classes());
        std::printf("样本数      : %lld\n", (long long)ld.size());
        std::printf("batch_size  : %lld\n", (long long)opt.batch_size);
        std::printf("num_batches : %lld\n", (long long)ld.num_batches());

        // num_batches 与 size/batch_size 的关系（train 整除，val 向上取整）
        const int64_t expect_nb = !opt.train
            ? (ld.size() + opt.batch_size - 1) / opt.batch_size
            : ld.size() / opt.batch_size;
        std::printf("期望        : %lld   %s\n", (long long)expect_nb,
            expect_nb == ld.num_batches() ? "OK" : "✗ 不符");

        ld.start_epoch(0);

        int64_t seen_img = 0;
        int64_t b = 0;
		int64_t limit = -1;  // 只取前几个 batch，-1 = 全部；测吞吐时给大一点
        while (ld.has_next() && (limit <= 0 || b < limit))
        {
            const auto batch = ld.next();
            const int64_t n = batch.images.size(0);

            TORCH_CHECK(batch.images.dim() == 4, "images 应该是 4 维");
            TORCH_CHECK(batch.images.size(1) == 3, "通道数应该是 3，实际 ", batch.images.size(1));
            TORCH_CHECK(batch.images.size(2) == opt.crop && batch.images.size(3) == opt.crop,"crop 尺寸不对: ", batch.images.sizes());
            TORCH_CHECK(batch.images.scalar_type() == torch::kFloat32, "images 应该是 float32");
            TORCH_CHECK(batch.labels.scalar_type() == torch::kInt64, "labels 应该是 int64");
            TORCH_CHECK((int64_t)batch.indices.size() == n, "indices 长度和 N 不一致");

            const auto lmin = batch.labels.min().item<int64_t>();
            const auto lmax = batch.labels.max().item<int64_t>();
            TORCH_CHECK(lmin >= 0 && lmax < ld.num_classes(),"标签越界: [", lmin, ", ", lmax, "] 而类数是 ", ld.num_classes());

            TORCH_CHECK(torch::isfinite(batch.images).all().item<bool>(),"images 里出现 nan / inf");

            if (b < 3) print_indices(batch, b);

            seen_img += n;
            ++b;
        }

        const auto st = ld.stats();
        std::printf("\n读了 %lld 个 batch / %lld 张\n", (long long)b, (long long)seen_img);
        std::printf("吞吐 %.1f img/s   总耗时 %.2f 秒\n", st.img_per_sec(), st.seconds);
        std::printf("\n判据（顺序可复现）：同一条命令跑两次，上面的 indices[0..4] 必须逐位相同。\n");
        bench_stages(cfg.data_dir, 200);

        std::printf("OK\n");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}