#include "train/trainer.h"

#include <chrono>
#include <cstdio>

namespace prkd::train
{
    namespace
    {
        using clk = std::chrono::steady_clock;

        // 一个 epoch 的公共骨架：取批 → 前向 → 记 loss/准确率。
        // 训练与评估只差"要不要梯度"这一段，其余完全相同 —— 所以用 bool 模板参数分开。
        //
        // 每批只前向【一次】：logits 算完既用来算 loss，也用来算准确率。
        // （写成两次 forward 会让训练慢一半，而两次的结果本来就该一样）
        template <bool kTrain>
        EpochStats run_epoch(model::ResNet& model,
                            data::LoaderCore& ld,
                            torch::optim::Optimizer* optimizer,
                            const torch::Device& device,
                            const TrainConfig& cfg)
        {
            TORCH_CHECK(ld.num_classes() > 1, "train: 类别数 ", ld.num_classes(), " 不合理");

            EpochStats st;
            const auto t0 = clk::now();

            // BN 的行为由这里决定：训练用批统计，评估用 running stats。
            if constexpr (kTrain) 
            { 
                model->train(); 
            } 
            else 
            { 
                model->eval(); 
            }

            // 不建图由调用方负责（evaluate() 在自己的作用域里构造 NoGradGuard）。
            // 这里不能用 std::optional<torch::NoGradGuard>：NoGradGuard 不可拷贝也不可移动，
            // optional 的拷贝构造会因此被删除，条件表达式一求值就报"尝试引用已删除的函数"。

            double running = 0.0;

            while (ld.has_next())
            {
                const data::ImageBatch batch = ld.next();
                const int64_t n = batch.n;

                // 数据在 loader 里恒为 CPU float32 —— 只搬这一批
                auto x = batch.images.to(device);
                auto y = batch.labels.to(device);

                torch::Tensor logits;
                if constexpr (kTrain)
                {
                    // ---- 纯净的四步 ----
                    optimizer->zero_grad();
                    logits = model->forward(x);
                    const auto loss = torch::nn::functional::cross_entropy(logits, y);
                    loss.backward();
                    optimizer->step();
                    // ---- 结束 ----
                    running += loss.item<double>();
                }
                else
                {
                    logits = model->forward(x);
                    running += torch::nn::functional::cross_entropy(logits, y).item<double>();
                }

                // 准确率：只读，所以 detach —— 训练时 logits 还挂着图，不 detach 会拖住整条链
                st.correct += (logits.detach().argmax(1) == y).sum().item<int64_t>();
                st.total += n;
                ++st.batches;

                if (cfg.log_every > 0 && (st.batches % cfg.log_every) == 0)
                {
                    std::printf("  [%s] batch %lld  avg loss %.4f  acc %.4f\n",
                        kTrain ? "train" : "eval",
                        (long long)st.batches,
                        running / static_cast<double>(st.batches),
                        static_cast<double>(st.correct) / static_cast<double>(st.total));
                    std::fflush(stdout);
                }
            }

            st.seconds = std::chrono::duration<double>(clk::now() - t0).count();
            st.loss = st.batches > 0 ? running / static_cast<double>(st.batches) : 0.0;
            return st;
        }
    }  // namespace

    EpochStats train_one_epoch(model::ResNet& model,
                              data::LoaderCore& ld,
                              torch::optim::Optimizer& optimizer,
                              const torch::Device& device,
                              const TrainConfig& cfg)
    {
        return run_epoch<true>(model, ld, &optimizer, device, cfg);
    }

    EpochStats evaluate(model::ResNet& model,
                       data::LoaderCore& ld,
                       const torch::Device& device,
                       const TrainConfig& cfg)
    {
        // 守卫放在这里，作用域覆盖整次 run_epoch 调用 —— 函数返回时才解构。
        // 它是"作用域对象"，只能这样用，不能存起来、传参、或放进 optional。
        torch::NoGradGuard no_grad;
        return run_epoch<false>(model, ld, nullptr, device, cfg);
    }

}  // namespace prkd::train
