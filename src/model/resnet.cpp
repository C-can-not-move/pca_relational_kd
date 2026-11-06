// src/model/resnet.cpp
#include "model/resnet.h"

namespace prkd::model
{

    // ------------------------------------------------------------------------
    // make_layer —— 建一个 stage，两种块通用
    //
    // 模板参数 Block 只要求三件事：
    //   ① 有 kExpansion 静态常量
    //   ② 构造函数签名是 (in, out, stride, downsample)
    //   ③ 有 forward(Tensor) -> Tensor
    // 两个块都满足，所以这一个函数能建全部五种 ResNet。
    //
    // in_channel 用引用传：退出时已更新成这一层的【实际输出通道】，
    // 所以四次调用天然接上。对应 PyTorch 的 self.inplanes。
    // ------------------------------------------------------------------------
    namespace
    {
        template <typename Block>
        torch::nn::Sequential make_layer(
            int64_t& in_channel,
            int64_t  width,          // 设计宽度；Bottleneck 的实际输出是宽度的 4 倍
            int64_t  block_num,
            int64_t  stride)
        {
            const int64_t out_channel = width * Block::kExpansion;

            // ★ 判据是 stride != 1 【或】通道变了 —— 不只是 stride。
            torch::nn::Sequential downsample{ nullptr };
            if (stride != 1 || in_channel != out_channel)
            {
                downsample = torch::nn::Sequential(
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channel, out_channel, /*kernel_size=*/1)
                        .stride(stride).bias(false)),
                    torch::nn::BatchNorm2d(out_channel));
            }

            torch::nn::Sequential layers;
            // push_back 自动命名 "0","1","2" —— 键会是 layer1.0.conv1.weight 这种形式
            layers->push_back(Block(in_channel, width, stride, downsample));

            in_channel = out_channel;   // ★ 副作用：给下一块和下一个 stage 用

            for (int64_t i = 1; i < block_num; ++i)
            {
                // B > 0 时输入是【上一块的输出通道】，不是 width
                layers->push_back(Block(in_channel, width));
            }
            return layers;
        }
    }

    // ------------------------------------------------------------------------
    // BasicBlock
    // ------------------------------------------------------------------------
    BasicBlockImpl::BasicBlockImpl(
        int64_t in_channel,
        int64_t out_channel,
        int64_t stride,
        torch::nn::Sequential downsample)
    {
        this->conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channel, out_channel, /*kernel_size=*/3)
            .stride(stride).padding(1).bias(false)));

        this->bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channel));

        this->conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(out_channel, out_channel, /*kernel_size=*/3)
            .stride(1).padding(1).bias(false)));

        this->bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channel));

        this->relu = register_module("relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

        if (!downsample.is_empty())
        {
            this->downsample = register_module("downsample", downsample);
        }
    }

    torch::Tensor BasicBlockImpl::forward(torch::Tensor x)
    {
        torch::Tensor identity = x;
        if (!downsample.is_empty())
        {
            identity = downsample->forward(x);
        }

        auto out = relu->forward(bn1->forward(conv1->forward(x)));
        out = bn2->forward(conv2->forward(out));

        // 加法用 out + identity 而不是 out += identity：避开原地操作
        out = out + identity;

        // ★ 最后一个 relu 在【加法之后】—— post-ReLU
        return relu->forward(out);
    }

    // ------------------------------------------------------------------------
    // Bottleneck
    // ------------------------------------------------------------------------
    BottleneckImpl::BottleneckImpl(
        int64_t in_channel,
        int64_t out_channel,
        int64_t stride,
        torch::nn::Sequential downsample)
    {
        const int64_t out = out_channel * kExpansion;

        this->conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channel, out_channel, /*kernel_size=*/1)
            .bias(false)));

        this->bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channel));

        this->conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(out_channel, out_channel, /*kernel_size=*/3)
            .stride(stride).padding(1).bias(false)));

        this->bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channel));

        this->conv3 = register_module("conv3", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(out_channel, out, /*kernel_size=*/1)
            .bias(false)));

        this->bn3 = register_module("bn3", torch::nn::BatchNorm2d(out));

        // ReLU 无参数，三次调用共用这一个实例
        this->relu = register_module("relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

        if (!downsample.is_empty())
        {
            this->downsample = register_module("downsample", downsample);
        }
    }

    torch::Tensor BottleneckImpl::forward(torch::Tensor x)
    {
        torch::Tensor identity = x;
        if (!downsample.is_empty())
        {
            identity = downsample->forward(x);
        }

        auto out = relu->forward(bn1->forward(conv1->forward(x)));
        out = relu->forward(bn2->forward(conv2->forward(out)));
        out = bn3->forward(conv3->forward(out));

        out = out + identity;

        return relu->forward(out);
    }

    // ------------------------------------------------------------------------
    // ResNet
    // ------------------------------------------------------------------------
    ResNetImpl::ResNetImpl(
        std::vector<int64_t> blocks_num,
        int64_t num_classes,
        bool    bottleneck)
        : bottleneck_(bottleneck)
    {
        TORCH_CHECK(blocks_num.size() == 4, "blocks_num 必须正好 4 项");

        // stem：224 → conv1/s2 → 112 → maxpool/s2 → 56
        this->conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(/*in=*/3, /*out=*/64, /*kernel_size=*/7)
            .stride(2).padding(3).bias(false)));

        this->bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

        this->relu = register_module("relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

        this->maxpool = register_module("maxpool", torch::nn::MaxPool2d(
            torch::nn::MaxPool2dOptions(/*kernel_size=*/3).stride(2).padding(1)));

        // 通道沿这条链往下传；in_channel 被 make_layer 用引用更新
        int64_t in_channel = 64;

        if (bottleneck_)
        {
            this->layer1 = register_module("layer1", make_layer<BottleneckImpl>(in_channel,  64, blocks_num[0], /*stride=*/1));
            this->layer2 = register_module("layer2", make_layer<BottleneckImpl>(in_channel, 128, blocks_num[1], /*stride=*/2));
            this->layer3 = register_module("layer3", make_layer<BottleneckImpl>(in_channel, 256, blocks_num[2], /*stride=*/2));
            this->layer4 = register_module("layer4", make_layer<BottleneckImpl>(in_channel, 512, blocks_num[3], /*stride=*/2));
        }
        else
        {
            this->layer1 = register_module("layer1", make_layer<BasicBlockImpl>(in_channel,  64, blocks_num[0], /*stride=*/1));
            this->layer2 = register_module("layer2", make_layer<BasicBlockImpl>(in_channel, 128, blocks_num[1], /*stride=*/2));
            this->layer3 = register_module("layer3", make_layer<BasicBlockImpl>(in_channel, 256, blocks_num[2], /*stride=*/2));
            this->layer4 = register_module("layer4", make_layer<BasicBlockImpl>(in_channel, 512, blocks_num[3], /*stride=*/2));
        }

        this->avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(
            torch::nn::AdaptiveAvgPool2dOptions({ 1, 1 })));

        // 最终宽度：Bottleneck 是 512×4 = 2048，BasicBlock 是 512
        const int64_t final_width = 512 * (bottleneck_ ? BottleneckImpl::kExpansion
                                                       : BasicBlockImpl::kExpansion);
        this->fc = register_module("fc", torch::nn::Linear(final_width, num_classes));

        // --------------------------------------------------------------------
        // 初始化：按【模块类型】而不是【参数维度】
        //
        // 旧写法 if (item.value().dim() == 4) 是拿维度当类型用 —— 现在恰好只有
        // conv 是 4 维，但将来加一个非卷积的 4 维参数就会被静默误初始化。
        // 这里直译 torchvision 的 _init_weights：
        //     if isinstance(m, nn.Conv2d):
        //         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        //
        // BN 的 weight=1 / bias=0 是 PyTorch 默认，不重复写。
        // fc 保持默认（kaiming_uniform_(a=sqrt(5))），也不动。
        // --------------------------------------------------------------------
        for (const auto& m : modules(/*include_self=*/false))   // std::vector<std::shared_ptr<Module>>
        {
            if (auto conv = std::dynamic_pointer_cast<torch::nn::Conv2dImpl>(m))
            {
                torch::nn::init::kaiming_normal_(
                    conv->weight, /*a=*/0, torch::kFanOut, torch::kReLU);
            }
        }
    }

    // 特征段：stem + 四个 stage + 全局池化，输出展平后的特征向量
    //   Bottleneck（50/101/152）→ (N, 2048)
    //   BasicBlock（18/34）     → (N,  512)
    torch::Tensor ResNetImpl::forward_features(torch::Tensor x)
    {
        x = maxpool->forward(relu->forward(bn1->forward(conv1->forward(x))));  // (N,   64, 56, 56)
        x = layer1->forward(x);                                                // (N,  W*1, 56, 56)
        x = layer2->forward(x);                                                // (N,  W*2, 28, 28)
        x = layer3->forward(x);                                                // (N,  W*4, 14, 14)
        x = layer4->forward(x);                                                // (N,  W*8,  7,  7)
        x = avgpool->forward(x);                                               // (N,  W*8,  1,  1)
        return torch::flatten(x, 1);                                           // (N, 特征宽度)
    }

    // 分类段：只有一层 fc，把特征映射到 logits
    torch::Tensor ResNetImpl::forward_head(torch::Tensor feat)
    {
        return fc->forward(feat);                                              // (N, num_classes)
    }

    torch::Tensor ResNetImpl::forward(torch::Tensor x)
    {
        // 特征只算一遍。写成 forward_features(x) 和 forward_head(forward_features(x))
        //   分两次调用会让整个卷积网络跑两遍 —— 数值一样，时间是两倍。
        return forward_head(forward_features(x));
    }

    // ------------------------------------------------------------------------
    // 工厂
    // ------------------------------------------------------------------------
    ResNet h_resnet18(int64_t num_classes)
    {
        return ResNet(std::vector<int64_t>{ 2, 2, 2, 2 }, num_classes, /*bottleneck=*/false);
    }

    ResNet h_resnet34(int64_t num_classes)
    {
        return ResNet(std::vector<int64_t>{ 3, 4, 6, 3 }, num_classes, /*bottleneck=*/false);
    }

    ResNet h_resnet50(int64_t num_classes)
    {
        return ResNet(std::vector<int64_t>{ 3, 4, 6, 3 }, num_classes, /*bottleneck=*/true);
    }

    ResNet h_resnet101(int64_t num_classes)
    {
        return ResNet(std::vector<int64_t>{ 3, 4, 23, 3 }, num_classes, /*bottleneck=*/true);
    }

    ResNet h_resnet152(int64_t num_classes)
    {
        return ResNet(std::vector<int64_t>{ 3, 8, 36, 3 }, num_classes, /*bottleneck=*/true);
    }

}  // namespace prkd::model
