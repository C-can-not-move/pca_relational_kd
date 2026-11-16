#pragma once

#include <cstdint>
#include <string>

namespace prkd::cfg
{
    // ------------------------------------------------------------------------
    // 三类设置分开放，各自回答一个问题：
    //   RunConfig      —— 在哪训（数据、输出、卡、种子）
    //   LoaderSettings —— 怎么读（线程、批、池、尺寸）
    //   DebugSettings  —— 诊断（intra-op 线程、是否对拍）
    //
    // 刻意【不】包含 epochs / lr / optimizer —— 那些属于
    // train::TrainConfig（"这一次训练循环用什么参数"），住 train/trainer.h。
    // ------------------------------------------------------------------------

    struct RunConfig
    {
        std::string data_dir;              // 数据集根；ImageNet 会在其下拼 /train 或 /val
        std::string out_dir = "./run";     // checkpoint 等落在这里
        std::string ckpt;                  // 要加载的权重前缀，如 .../run/resnet152_1000；空 = 不加载
        std::string dataset = "imagenet";  // imagenet | aloi（决定用哪个 loader）
        int         device_index = 0;      // 用哪块卡；-1 = 强制 CPU
        int64_t     seed = 2026;
    };

    struct LoaderSettings
    {
        int64_t workers    = 0;      // defaults() 会解析成 hardware_concurrency() / 2
        int64_t batch      = 64;
        int64_t prefetch   = 4;      // 预取池深度：1 = 退化成同步等齐
        int64_t short_side = 256;    // val 的 Resize 短边；V1 权重 256，V2 用 232
        int64_t crop       = 224;

        // 三个开关互相独立，谁也不派生谁。
        //   is_train     只决定变换分支（增强 / 中心裁剪）
        //   shuffle      每 epoch 是否重排
        //   drop_last    是否截尾使批整除
        // 所以评测时必须把三个一起改成 false —— 只改 is_train 会得到
        // "中心裁剪但洗牌且丢尾巴"这种合法又反直觉的组合。启动打印里能看见。
        bool is_train  = true;
        bool shuffle   = true;
        bool drop_last = true;
    };

    // ALOI 没有 /train 与 /val —— 它的 split 就是视角范围（左闭右开）。
    // 只在 run.dataset = aloi 时被读到。
    struct AloiSettings
    {
        int64_t view_begin = 0;      // 含
        int64_t view_end   = 360;    // 不含

        // 半圈训练/测试：{0,180} 与 {180,360}
        // 冒烟测试（只读 r0）：{0,5}
        // 全量看 PCA：{0,360}
    };

    struct DebugSettings
    {
        int  intra    = 1;      // ATen intra-op 线程数；1 = 不让它和 worker 抢核
        bool selftest = true;   // 是否跑模型自检与 workers 对拍
        int64_t log_every = 50; // 长循环的进度间隔（批）；0 = 不打印
    };

    // 训练超参。原先写在 main 的 TrainArgs 里，现在挪进来 ——
    // 理由：换一次配方就改一个 ini，不用重编。
    //
    // 刻意【不】包含 batch：批是 loader 的事（loader.batch），trainer 只收批。
    struct TrainSettings
    {
        int64_t epochs       = 3;
        double  lr           = 0.1;
        double  momentum     = 0.9;
        double  weight_decay = 1e-4;
        int64_t log_every    = 50;   // 每多少批打一行；0 = 不打

        // 续跑时 lr 从哪儿来：
        //   true  = 按本配置的 train.lr（想在续跑时改 lr / 接着做衰减，用这个）
        //   false = 保留 opt.pt 里存的那份（想要"原样接着跑"用这个）
        // 只有 load_optimizer_state 之后才有区别 —— 那个接口会把超参一起恢复。
        bool    resume_lr_from_config = true;

        // best 从训练的哪个位置开始【写盘】（0–1 的比例，默认 0.75 = 最后四分之一）。
        // 前 3/4 的"最好"几乎必然被后面刷掉，写下去只是白占磁盘；
        // 但内存里一直留着最新的 best，收尾时无论如何会落一次盘。
        // 0 = 每次都写；1 = 全程不写（只在收尾写）。
        double  best_write_from = 0.75;
    };

    struct Settings
    {
        RunConfig      run;
        LoaderSettings loader;
        AloiSettings   aloi;
        DebugSettings  debug;
        TrainSettings  train;

        // 读 ini 并覆盖自己的字段。
        //   打不开 → 抛 std::runtime_error（配置是刚需，没有"用默认值跑起来"这种退路）
        //   未知 key / 值非法 → 同样抛，消息带 "路径:行号"
        // 做成成员：读的目标就是自己，传一个 Settings& 进来只是把 this 又写一遍。
        void load_file(const std::string& path);

        // 启动时把配置打一遍 —— 日志里能追溯"这次是用什么参数跑的"。
        // 做成成员而不是自由函数，理由：
        //   ① 打印的内容与字段在同一个类型里，加字段时不容易漏打
        //   ② 调用处是 st.print()，读起来就知道打的是"这一份 st"
        //   ③ 不需要在参数上再写一遍 const Settings&
        void print() const;

        // 只打一行摘要（放日志头部用）：dataset / batch / lr / epochs / seed
        // 与 print() 分开：前者太吵、后者太短，两个用途不该互相迁就。
        // 返回 string 而不是 const char*：调用方可能要拼进别的日志里，
        // 返回裸指针就得管一块谁都不清楚的静态缓冲。
        std::string summary() const;
    };

    // 平台默认值。workers = 0 会在这里解析成实际值 ——
    // 好处是 print() 打出来的永远是"实际会用到的数"，不是 0。
    Settings defaults();

}  // namespace prkd::cfg
