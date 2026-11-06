// src/pca_relational_kd.h —— 伞头：只列已经搬进项目的东西
//
// 各模块的头文件在这里汇总；.cpp 只需要 include 这一份。
// 第三方 C 库（jpeglib）不出现在这里 —— 它只属于 decode 的实现。
#pragma once

#include "config.h"
#include "data/decode.h"
#include "data/imagenet_loader.h"
#include "data/transform.h"

#include "model/resnet.h"
#include "model/checkpoint.h"

#include "test_unit/loader_check.h"
#include "test_unit/model_check.h"

#include <cstdio>      // std::fprintf
#include <exception>   // std::exception

#ifdef _WIN32
#include <windows.h>
#endif