// src/test_unit/model_check.cpp
#include "test_unit/model_check.h"

#include <torch/torch.h>

#include <cstdio>
#include <stdexcept>
#include <string>

namespace prkd::test_unit
{
    namespace
    {
        // torchvision 发布的参数量 —— 结构写错时这几个数会对不上。
        constexpr int64_t kParamsResNet18  = 11689512;
        constexpr int64_t kParamsResNet34  = 21797672;
        constexpr int64_t kParamsResNet50  = 25557032;
        constexpr int64_t kParamsResNet152 = 60192808;

        // 把 shape 拼成 "(2, 1000)" 这种。报错信息里要能用，
        // 所以单独一个函数，不塞进 throw 的表达式里。
        std::string shape_str(const torch::Tensor& t)
        {
            std::string s = "(";
            for (int64_t i = 0; i < t.dim(); ++i)
            {
                if (i > 0) s += ", ";
                s += std::to_string(t.size(i));
            }
            return s + ")";
        }
    }

    int64_t count_parameters(const model::ResNet& net)
    {
        int64_t n = 0;
        // parameters() 只数 requires_grad 的张量；BN 的 running_mean/var 在 buffers() 里，
        // 所以这个数与 PyTorch 的 sum(p.numel() for p in net.parameters()) 对得上。
        for (const auto& p : net->parameters())
        {
            n += p.numel();
        }
        return n;
    }

    void check_forward(
        model::ResNet net,
        int64_t batch,
        int64_t num_classes,
        int64_t expect_params)
    {
        // ---- ① 参数量 ----
        const int64_t got = count_parameters(net);
        if (got != expect_params)
        {
            throw std::runtime_error(
                "check_forward: 参数量不符，期望 " + std::to_string(expect_params)
                + "，实际 " + std::to_string(got)
                + "（结构写错的最直接信号）");
        }

        // ---- ② 前向 ----
        net->eval();                       // BN 用 running stats，不更新

        auto x = torch::randn({ batch, 3, 224, 224 });

        torch::Tensor y;
        {
            torch::NoGradGuard ng;         // 只验形状，不需要图
            y = net->forward(x);
        }

        if (y.dim() != 2 || y.size(0) != batch || y.size(1) != num_classes)
        {
            throw std::runtime_error(
                "check_forward: 输出形状不符，期望 (" + std::to_string(batch)
                + ", " + std::to_string(num_classes) + ")，实际 " + shape_str(y));
        }

        // ---- ③ 有限性 ----
        // 初始化炸掉（kaiming 参数给错）、BN 除零、fc 权重 NaN —— 都表现在这里。
        if (!torch::isfinite(y).all().item<bool>())
        {
            throw std::runtime_error(
                "check_forward: 输出含 NaN/Inf  —— 检查初始化与 BN（eps、momentum）");
        }

        std::printf("  [forward]  params %-10lld  out (%lld, %lld)  finite  OK\n",
            static_cast<long long>(got),
            static_cast<long long>(y.size(0)),
            static_cast<long long>(y.size(1)));
    }

    void check_backward(
        model::ResNet net,
        int64_t batch,
        int64_t num_classes)
    {
        net->train();                      // BN 用 batch stats

        auto x = torch::randn({ batch, 3, 224, 224 });
        auto y = torch::randint(0, num_classes, { batch }, torch::kLong);

        auto logits = net->forward(x);
        auto loss   = torch::nn::functional::cross_entropy(logits, y);

        if (!torch::isfinite(loss).item<bool>())
        {
            throw std::runtime_error("check_backward: loss 不是有限值");
        }

        loss.backward();

        int64_t n_grad = 0, n_all_zero = 0;
        double  max_abs = 0.0;

        for (const auto& item : net->named_parameters())
        {
            auto p = item.value();
            if (!p.requires_grad()) continue;

            ++n_grad;

            auto g = p.grad();
            if (!g.defined())
            {
                throw std::runtime_error(
                    "check_backward: 参数没有梯度 —— " + item.key()
                    + "（计算图断开，或 loss 不依赖它）");
            }

            if (!torch::isfinite(g).all().item<bool>())
            {
                throw std::runtime_error(
                    "check_backward: 梯度含 NaN/Inf —— " + item.key());
            }

            const double s = g.abs().sum().item<double>();
            if (s == 0.0) ++n_all_zero;
            if (s > max_abs) max_abs = s;
        }

        // 全零梯度意味着「能跑但学不到东西」。允许极少数层为零
        // （某个 stage 可能在这一批次恰好不参与），但不允许大面积为零。
        if (n_all_zero * 2 > n_grad)
        {
            throw std::runtime_error(
                "check_backward: 过半参数梯度为零（" + std::to_string(n_all_zero)
                + "/" + std::to_string(n_grad) + "）—— 计算图可能断在某处");
        }

        std::printf("  [backward] loss %.6f  grad %lld 个  全零 %lld  最大 |dL/dp| %.4g  OK\n",
            loss.item<double>(),
            static_cast<long long>(n_grad),
            static_cast<long long>(n_all_zero),
            max_abs);

        // ★ 按值传的 holder 和调用方共享同一个 Impl —— 梯度已经写进去了。
        //   这里清掉，不把测试的副作用留给调用方。
        net->zero_grad();
    }

    void run_model_checks()
    {
        std::printf("\n===== 模型检查 =====\n");

        const int64_t kBatch = 2;          // 224 输入、152 层 —— batch 先给小，验的是形状不是速度
        const int64_t kClasses = 1000;

        std::printf("--- ResNet-18  (BasicBlock, expansion 1) ---\n");
        {
            auto net = model::h_resnet18(kClasses);
            check_forward(net, kBatch, kClasses, kParamsResNet18);
            check_backward(net, kBatch, kClasses);
        }

        std::printf("--- ResNet-34  (BasicBlock, expansion 1) ---\n");
        {
            auto net = model::h_resnet34(kClasses);
            check_forward(net, kBatch, kClasses, kParamsResNet34);
            check_backward(net, kBatch, kClasses);
        }

        std::printf("--- ResNet-50  (Bottleneck, expansion 4) ---\n");
        {
            auto net = model::h_resnet50(kClasses);
            check_forward(net, kBatch, kClasses, kParamsResNet50);
            check_backward(net, kBatch, kClasses);
        }

        std::printf("--- ResNet-152 (Bottleneck, expansion 4) ---\n");
        {
            auto net = model::h_resnet152(kClasses);
            check_forward(net, kBatch, kClasses, kParamsResNet152);
            check_backward(net, kBatch, kClasses);
        }

        std::printf("模型检查全部通过\n");
    }

}  // namespace prkd::test_unit
