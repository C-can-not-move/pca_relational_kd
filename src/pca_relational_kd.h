// src/pca_relational_kd.h —— 伞头：只列已经搬进项目的东西
//
// 各模块的头文件在这里汇总；.cpp 只需要 include 这一份。
// 第三方 C 库（jpeglib）不出现在这里 —— 它只属于 decode 的实现。
//
// 为什么 libtorch 也放进来：
//   项目的头文件（resnet.h / trainer.h / save/*）本身就依赖 torch，
//   所以每个 include 伞头的 .cpp 早就已经编译过一遍 torch 的东西了。
//   在这里集中一次，.cpp 就不会各写一遍 <torch/torch.h>，改依赖时也不会漏。
//   代价：改这个伞头会触发大范围重编 —— 但它本来就很少改。
#pragma once

#include <torch/torch.h>
#include <torch/optim.h>

#include "config.h"
#include "data/decode.h"
#include "data/imagenet_loader.h"
#include "data/transform.h"

#include "model/resnet.h"
#include "model/checkpoint.h"

#include "save/param_index.h"   // dump / restore / names（权重与 buffer 的有序表）
#include "save/slot_dir.h"      // 存档槽：save_slot / load_slot_state / load_model / load_optimizer_state

#include "train/trainer.h"

#include "util/pca.h"

#include "test_unit/loader_check.h"
#include "test_unit/model_check.h"

#include <algorithm>   // std::min
#include <chrono>      // 计时
#include <cmath>       // std::ceil（best 写盘起始轮）
#include <cstdio>      // std::printf / std::fprintf
#include <cstdlib>     // std::strtoll
#include <exception>   // std::exception
#include <filesystem>  // 建目录 / 判断 state.txt 在不在
#include <memory>      // std::make_unique
#include <stdexcept>   // std::runtime_error
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif