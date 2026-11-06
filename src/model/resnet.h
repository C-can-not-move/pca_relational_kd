// src/model/resnet.h
#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace prkd::model
{
    // ------------------------------------------------------------------------
    // BasicBlock —— torchvision BasicBlock，expansion == 1
    //
    // 两个 3×3 卷积。ResNet-18 / 34 用它。
    //   x → conv1(3×3, s=stride, p=1) → bn1 → relu
    //     → conv2(3×3, s=1,      p=1) → bn2 →(+ identity)→ relu
    //
    // ★ stride 落在 conv1 上 —— BasicBlock 没有 1×1 卷积，不存在 v1/v1.5 的选择。
    // ------------------------------------------------------------------------
    class BasicBlockImpl : public torch::nn::Module
    {
    public:
        static constexpr int64_t kExpansion = 1;

        // out_channel 就是【最终输出通道】（expansion = 1，不乘）
        BasicBlockImpl(
            int64_t in_channel,
            int64_t out_channel,
            int64_t stride = 1,
            torch::nn::Sequential downsample = nullptr);

        torch::Tensor forward(torch::Tensor x);

    private:
        torch::nn::Conv2d      conv1{ nullptr };   // 3x3, stride=stride, pad 1, no bias
        torch::nn::BatchNorm2d bn1{ nullptr };
        torch::nn::Conv2d      conv2{ nullptr };   // 3x3, stride=1,      pad 1, no bias
        torch::nn::BatchNorm2d bn2{ nullptr };
        torch::nn::ReLU        relu{ nullptr };    // inplace = true
        torch::nn::Sequential  downsample{ nullptr };
    };
    TORCH_MODULE(BasicBlock);

    // ------------------------------------------------------------------------
    // Bottleneck —— torchvision Bottleneck，expansion == 4
    //
    // 三个卷积。ResNet-50 / 101 / 152 用它。
    //   x → conv1(1×1) → bn1 → relu
    //     → conv2(3×3, s=stride, p=1) → bn2 → relu
    //     → conv3(1×1) → bn3 →(+ identity)→ relu
    //
    // ★ stride 落在 conv2（中间的 3×3）上 —— 这是 ResNet v1.5，torchvision 用的就是这个。
    //   写成 v1（stride 在 conv1）形状完全一样，load_state_dict 不会报错，但结果会错。
    // ★ 键名已用 resnet152(weights=IMAGENET1K_V1) 的 932 个键逐个验证过。
    // ------------------------------------------------------------------------
    class BottleneckImpl : public torch::nn::Module
    {
    public:
        static constexpr int64_t kExpansion = 4;

        // out_channel 是【不含 expansion 的宽度】，实际输出通道 = out_channel * 4
        BottleneckImpl(
            int64_t in_channel,
            int64_t out_channel,
            int64_t stride = 1,
            torch::nn::Sequential downsample = nullptr);

        torch::Tensor forward(torch::Tensor x);

    private:
        torch::nn::Conv2d      conv1{ nullptr };   // 1x1, stride 1,  no bias
        torch::nn::BatchNorm2d bn1{ nullptr };
        torch::nn::Conv2d      conv2{ nullptr };   // 3x3, stride=stride, pad 1, no bias
        torch::nn::BatchNorm2d bn2{ nullptr };
        torch::nn::Conv2d      conv3{ nullptr };   // 1x1, stride 1,  no bias
        torch::nn::BatchNorm2d bn3{ nullptr };
        torch::nn::ReLU        relu{ nullptr };    // inplace = true
        torch::nn::Sequential  downsample{ nullptr };
    };
    TORCH_MODULE(Bottleneck);

    // ------------------------------------------------------------------------
    // ResNet —— ImageNet 式 stem：7x7/s2 + maxpool/s2 ⇒ 输入必须是 224
    //
    // blocks_num 与 bottleneck 标志共同决定是哪一个：
    //   {2,2,2,2}  + false = ResNet-18
    //   {3,4,6,3}  + false = ResNet-34
    //   {3,4,6,3}  + true  = ResNet-50
    //   {3,4,23,3} + true  = ResNet-101
    //   {3,8,36,3} + true  = ResNet-152
    //
    // ★ bottleneck 是构造参数而不是模板参数：layer1..4 都是 Sequential，
    //   装两种块都行，所以运行时分支就够了，不必生成两个类。
    // ------------------------------------------------------------------------
    class ResNetImpl : public torch::nn::Module
    {
    public:
        explicit ResNetImpl(
            std::vector<int64_t> blocks_num,
            int64_t num_classes = 1000,
            bool    bottleneck  = true);

        torch::Tensor forward(torch::Tensor x);
        torch::Tensor forward_features(torch::Tensor x);   // (N, 2048/512)
        torch::Tensor forward_head(torch::Tensor feat);    // (N, num_classes)

    private:
        bool bottleneck_;

        torch::nn::Conv2d            conv1{ nullptr };   // 7x7, stride 2, pad 3, no bias
        torch::nn::BatchNorm2d       bn1{ nullptr };
        torch::nn::ReLU              relu{ nullptr };    // inplace = true
        torch::nn::MaxPool2d         maxpool{ nullptr }; // 3x3, stride 2, pad 1
        torch::nn::Sequential        layer1{ nullptr };  // 64  ->  256/64,  stride 1
        torch::nn::Sequential        layer2{ nullptr };  // 256/64 ->  512/128, stride 2
        torch::nn::Sequential        layer3{ nullptr };  // 512/128-> 1024/256, stride 2
        torch::nn::Sequential        layer4{ nullptr };  // 1024/256->2048/512, stride 2
        torch::nn::AdaptiveAvgPool2d avgpool{ nullptr }; // (1,1)
        torch::nn::Linear            fc{ nullptr };      // 2048 或 512 -> num_classes
    };
    TORCH_MODULE(ResNet);

    // 工厂 —— 都是 224 输入、ImageNet 口径

	// 18 / 34 是 BasicBlock
    ResNet h_resnet18 (int64_t num_classes = 1000);
    ResNet h_resnet34 (int64_t num_classes = 1000);
	// 50 / 101 / 152 是 Bottleneck
    ResNet h_resnet50 (int64_t num_classes = 1000);
    ResNet h_resnet101(int64_t num_classes = 1000);
    ResNet h_resnet152(int64_t num_classes = 1000);

}  // namespace prkd::model
