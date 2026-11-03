// src/data/decode.h —— 内存字节 → HWC RGB8 像素。一个函数，不管文件从哪来。
//
// 为什么是 HWC：JPEG 解码出来天然就是交错排布（一个像素的 RGB 相邻），
// 而 PIL/numpy 那边也是 (H,W,3)。到 CHW 的转换在 transform.h 的 to_chw01 里做，
// 两件事分开，各自可测。
//
// 平台：用 libjpeg-turbo（CMake 的 find_package(JPEG)）。Windows 和 Linux 同一份代码，
// Python 那边的 Pillow 底下也是它 —— 同一个解码器家族，但不是同一次构建，
// 所以解码结果可能差 1 个灰阶。
#pragma once

#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace prkd::data
{
    struct Bitmap
    {
        int w = 0;
        int h = 0;
        int c = 0;                      // 固定 3
        std::vector<uint8_t> data;      // w*h*3，HWC 交错
    };

    // 任意 JPEG 字节 → RGB8。
    //   · 灰度图会被复制成三通道（等价 PIL 的 convert('RGB')）
    //   · CMYK / YCCK 走 Adobe 反转公式，结果可能与 Pillow 差几个灰阶
    //   · 渐进式 JPEG 原生支持
    // 失败抛 std::runtime_error，不返回空结果。
    Bitmap decode_rgb8(const uint8_t* bytes, size_t n);
}