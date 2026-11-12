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
    //Bitmap 跟 libjpeg / PIL 用(w, h)；张量跟 torch 用(h, w)
    struct Bitmap
    {
        int w = 0;
        int h = 0;
        int c = 0;                      // 固定 3
        std::vector<uint8_t> data;      // w*h*3，HWC 交错

        // 下面两项是解码过程的诊断，不是图像的属性
        int  decode_warnings = 0;       // libjpeg 的 num_warnings
        bool decode_complete = true;    // output_scanline 是否走到 output_height
    };

    // 任意 JPEG / PNG 字节流 → RGB8。按文件头前 8 字节分流：
    //   FF D8 ...        → libjpeg-turbo（灰度复制成三通道，CMYK/YCCK 走 Adobe 反转）
    //   89 50 4E 47 ...  → fpng（只支持 8 位 RGB/RGBA、不隔行）
    //
    // 灰度图会被复制成三通道（等价 PIL 的 convert('RGB'))
    //  CMYK / YCCK 走 Adobe 反转公式，结果可能与 Pillow 差几个灰阶
    //  渐进式 JPEG 原生支持
    //
    // 两级失败：
    //  致命（不是 JPEG、砍在头部、非法 Huffman 表）→ 抛 std::runtime_error，无返回值
    //  有损（数据被截断、有多余字节）→ 正常返回，但 decode_warnings > 0；
    //  若 decode_complete 为 false，说明 libjpeg 提前停住，没被写到的像素仍是最初的 0（黑）；为 true 则它补零比特解完了，缺口是 128 灰。
    //
    // 这两个诊断是 libjpeg 白送的：num_warnings 由 emit_message 自己累加，
    // output_scanline 本来就在读循环的判断里，取它们不增加热路径开销。
    Bitmap decode_rgb8(const uint8_t* bytes, size_t n);
}