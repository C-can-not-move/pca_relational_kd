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
        std::string data_dir;              // 数据集【某个 split 的父目录】；main 会拼 /train 或 /val
        std::string out_dir = "./run";     // checkpoint 等落在这里
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
        bool    train      = true;
    };

    struct DebugSettings
    {
        int  intra    = 1;      // ATen intra-op 线程数；1 = 不让它和 worker 抢核
        bool selftest = true;   // 是否跑 workers=0 的逐元素对拍
    };

    struct Settings
    {
        RunConfig      run;
        LoaderSettings loader;
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
