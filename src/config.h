#pragma once

#include <cstdint>
#include <string>

namespace mkd::cfg
{
    // 一次运行的设置 —— 在哪训、结果输出到哪、用哪块卡、种子。
    // 刻意不装 epochs / batch_size / out_size 那些。
    //  它们属于 train::TrainConfig（"这一次训练循环用什么参数"）。
    //  分开的理由：一个是"怎么训"，一个是"在哪训"。
    struct RunConfig
    {
        std::string data_dir;              // 由 defaults() 或调用者填；默认是cifar-100-batches-bin
        std::string out_dir = "./run";     // checkpoint 等落在这里
        int         device_index = 0;      // 用哪块卡；-1 = 强制 CPU
        int64_t     seed = 2026;
    };

    RunConfig defaults();

}  // namespace mkd::cfg