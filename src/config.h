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

    struct Settings
    {
        RunConfig      run;
        LoaderSettings loader;
        AloiSettings   aloi;
        DebugSettings  debug;
    };

    // 平台默认值。workers = 0 会在这里解析成实际值 ——
    // 好处是 print() 打出来的永远是"实际会用到的数"，不是 0。
    Settings defaults();

    // 读 ini 并覆盖 s 里已有的值。
    //   打不开 → 抛 std::runtime_error（配置是刚需，没有"用默认值跑起来"这种退路）
    //   未知 key / 值非法 → 同样抛，消息带 "路径:行号"
    void load_file(const std::string& path, Settings& s);

    // 启动时把配置打一遍 —— 日志里能追溯"这次是用什么参数跑的"
    void print(const Settings& s);

}  // namespace prkd::cfg
