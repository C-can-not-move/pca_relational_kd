#include "data/decode.h"

#include <cstring>   // std::memset：MSVC 靠 <stdio.h> 间接带进来，glibc 不会

// PNG 解码：stb_image（通用解码器，单文件、公共领域）。
// 不用 fpng —— 它的解码器只认自己编码的文件，通用 PNG 会返回 NOT_FPNG。
//
// ★ STB_IMAGE_IMPLEMENTATION 必须定义在【唯一】一处 include 之前。
//   stb 的头文件默认只声明接口；定义这个宏的那一份才生成实现（含 stbi_failure_reason 等）。
//   全项目只有这里定义 —— 谁再 include 一次并重复定义，链接期会出现重复符号。
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"


extern "C"
{
#include <jpeglib.h>
}

namespace prkd::data
{
    namespace
    {
        // libjpeg 默认的 error_exit 会直接 exit()。必须换成 longjmp，否则一张坏图就会终止整个进程。
        // 用 C 的方式模拟继承
        // 携带 jmp_buf 的办法是"继承"结构体：JpegErrorMgr 的首成员是 jpeg_error_mgr，
        // 所以 jpeg_std_error() 返回的指针可以安全地 reinterpret_cast 回来。
        struct JpegErrorMgr
        {
            jpeg_error_mgr pub; // 首成员
            std::jmp_buf jb;    // 附加信息
            char msg[JMSG_LENGTH_MAX] = { 0 };
        };

        void on_error(j_common_ptr cinfo)
        {
            auto* e = reinterpret_cast<JpegErrorMgr*>(cinfo->err);
            (*cinfo->err->format_message)(cinfo, e->msg);
            std::longjmp(e->jb, 1);
        }

        // 屏蔽 libjpeg 往 stderr 打印的 warning（ImageNet 里有一部分图会触发）。
        // 注意签名：output_message 只收一个 j_common_ptr；
        // 收两个参数的那个是 emit_message(cinfo, level)。
        void on_output(j_common_ptr) {}

        // a * b / 255，带四舍五入。
        // +127 是舍入补偿：除数 255 的阈值是 127.5，不加会系统性偏小。
        inline uint8_t mul255(uint8_t a, uint8_t b)
        {
            return static_cast<uint8_t>((static_cast<unsigned>(a) * b + 127u) / 255u);
        }

        // ---- PNG ----
        // 为什么不用 fpng：它的解码器只认【自己编码】的文件。
        //   fpng.h 原文："fpng_decode_memory() decompresses 24/32bpp PNG files
        //   ONLY encoded by this module ... otherwise it will return FPNG_DECODE_NOT_FPNG
        //   in which case you should fall back to a general purpose PNG decoder."
        // ALOI 的 PNG 是别的工具产的 ⇒ 每张都会返回 NOT_FPNG。
        // 所以这里用 stb_image：通用 PNG 解码器，单文件、公共领域。
        //
        // 与 libjpeg 那条路的取舍不同：stb 没有"有损继续"，CRC 不对就直接失败，
        // 所以 decode_warnings 恒为 0、decode_complete 恒为 true。
        // 注意：JPEG 仍然走 libjpeg-turbo —— 它更快，而且它的 warning 计数
        // 是 ImageNet 那边做"坏图统计"的依据，不能换掉。
        constexpr uint8_t kPngSig[8] = { 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };

        bool looks_like_png(const uint8_t* p, size_t n)
        {
            return n >= 8 && std::memcmp(p, kPngSig, 8) == 0;
        }

        // 直接要三通道：ALOI 是 color type 2（RGB），本来就没有 alpha。
        // stb 对带 alpha 的图也会按需求降成三通道 —— 省掉一段 RGBA→RGB 的拷贝循环。
        Bitmap decode_png_rgb8(const uint8_t* bytes, size_t n)
        {
            int w = 0, h = 0, ch = 0;
            stbi_uc* px = stbi_load_from_memory(bytes, static_cast<int>(n), &w, &h, &ch, 3);
            if (px == nullptr)
            {
                const char* why = stbi_failure_reason();
                throw std::runtime_error("decode_rgb8: PNG 解码失败 -- "
                                         + std::string(why ? why : "未知原因"));
            }
            if (w <= 0 || h <= 0)
            {
                stbi_image_free(px);
                throw std::runtime_error("decode_rgb8: PNG 尺寸为 0");
            }

            Bitmap bm;
            bm.w = w;
            bm.h = h;
            bm.c = 3;
            bm.decode_warnings = 0;
            bm.decode_complete = true;

            const size_t npix = static_cast<size_t>(w) * h;
            bm.data.resize(npix * 3);
            std::memcpy(bm.data.data(), px, npix * 3);
            stbi_image_free(px);          // 必须还：stb 用它自己的分配器
            return bm;
        }
    }  // namespace

    Bitmap decode_rgb8(const uint8_t* bytes, size_t n)
    {
        if (bytes == nullptr || n == 0)
        {
            throw std::runtime_error("decode_rgb8: 空输入");
        }

        // 按文件头分流：PNG 那条路完全不碰 libjpeg。
        // 放在这里而不是更后面 —— 后面紧跟 cinfo 的 memset 与 setjmp，
        // 插在它们之前才不会留下"初始化完 libjpeg 再返回"的路径。
        if (looks_like_png(bytes, n))
        {
            return decode_png_rgb8(bytes, n);
        }

        jpeg_decompress_struct cinfo;
        std::memset(&cinfo, 0, sizeof(cinfo));      // mem 置空 ⇒ destroy 安全的空操作

        // 这句 memset 合法，前提是 JpegErrorMgr 的成员全部 trivially copyable
        JpegErrorMgr jerr;
        std::memset(&jerr, 0, sizeof(jerr));

        cinfo.err = jpeg_std_error(&jerr.pub);
        jerr.pub.error_exit = on_error;
        jerr.pub.output_message = on_output;

        // longjmp 不跑析构函数。凡是需要析构的局部对象，都必须在这里（setjmp 之前）就构造完毕 —— 这样 longjmp 跳回来时它仍然存在，函数退出时才正常析构。
        // raw 先留空（不分配），后面用 assign 填。
        std::vector<uint8_t> raw;

        if (setjmp(jerr.jb) != 0)
        {
            jpeg_destroy_decompress(&cinfo);
            throw std::runtime_error(std::string("decode_rgb8: ") + jerr.msg);
        }

        jpeg_create_decompress(&cinfo);
        jpeg_mem_src(&cinfo, const_cast<uint8_t*>(bytes), static_cast<unsigned long>(n));

        if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK)
        {
            jpeg_destroy_decompress(&cinfo);
            throw std::runtime_error("decode_rgb8: 不是合法的 JPEG");
        }

        // 四分量（CMYK / YCCK）保持四通道出来，之后再转；其余一律让 libjpeg 转成 RGB
        // （灰度图因此也会变成三通道，等价 PIL 的 convert('RGB')）。
        const bool is_cmyk = (cinfo.jpeg_color_space == JCS_CMYK ||cinfo.jpeg_color_space == JCS_YCCK);
        cinfo.out_color_space = is_cmyk ? JCS_CMYK : JCS_RGB;

        jpeg_start_decompress(&cinfo);

        const int w = static_cast<int>(cinfo.output_width);
        const int h = static_cast<int>(cinfo.output_height);
        const int comps = static_cast<int>(cinfo.output_components);

        if (w <= 0 || h <= 0)
        {
            jpeg_destroy_decompress(&cinfo);
            throw std::runtime_error("decode_rgb8: 图像尺寸为 0");
        }

        // 理论上 out_color_space 已保证是 3 或 4。但少了这句，一旦不是，
        // 下面的 raw[p*4+3] 会越界读 —— 不报错，只是结果乱。
        if (comps != 3 && comps != 4)
        {
            const int got = comps;
            jpeg_destroy_decompress(&cinfo);
            throw std::runtime_error("decode_rgb8: 意外的通道数 " + std::to_string(got));
        }

        raw.assign(static_cast<size_t>(w) * h * comps, 0);

        const size_t row_bytes = static_cast<size_t>(w) * comps;
        while (cinfo.output_scanline < cinfo.output_height)
        {
            JSAMPROW row = raw.data() + static_cast<size_t>(cinfo.output_scanline) * row_bytes;

            // 返回值被丢掉的话，一旦 libjpeg 返回 0 而 output_scanline 不长，
            // 这里就是死循环。jpeg_mem_src 不会悬停（suspend 只发生在非阻塞
            // 数据源），所以理论上跑不到 —— 但接住它只是多一次比较，
            // 换来"hang 变 error"。
            if (jpeg_read_scanlines(&cinfo, &row, 1) == 0)
            {
                break;
            }
        }

        // ★ 这两项必须在 jpeg_finish_decompress 之前取：finish 有可能 longjmp，
        //   一旦跳走就再也拿不到了。
        //   num_warnings 在 jerr.pub 里，destroy 不影响；complete 依赖 cinfo，
        //   所以都趁着 cinfo 还活着读掉。
        const int  warns    = jerr.pub.num_warnings;
        const bool complete = (cinfo.output_scanline >= cinfo.output_height);

        // APP14 的 Adobe 标记。它决定两件事：
        //   ① libjpeg 靠它把四分量识别成 CMYK / YCCK（而不是 RGB）
        //   ② 四分量是【反转】存的 —— 这决定下面走哪条转换公式
        // ★ 必须在 destroy 之前读：destroy 之后 cinfo 全部作废。
        const bool adobe = cinfo.saw_Adobe_marker != 0;
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);

        Bitmap bm;
        bm.w = w;
        bm.h = h;
        bm.c = 3;
        bm.decode_warnings = warns;
        bm.decode_complete = complete;

        if (comps == 3)
        {
            // comps == 3 时 raw 与 bm.data 的尺寸、布局、类型完全相同（都是 w*h*3 的 HWC），
            bm.data = std::move(raw);
            return bm;
        }

        // 只有 comps == 4 才真的需要一块新的三通道缓冲
        bm.data.resize(static_cast<size_t>(w) * h * 3);

        // comps == 4：CMYK → RGB
        //
        // 标准式：R = (255 - C_ink) * (255 - K_ink) / 255
        //
        // Adobe 写的是【反转】值 —— 存的是 255 - 墨量。代入 C_ink = 255 - c_stored，
        // 得到 255 - C_ink = c_stored，于是：
        //     R = c_stored * k_stored / 255
        // 也就是说，带 Adobe 标记的不必再翻转，直接相乘。
        //
        // 不带标记的四分量不是反转的，必须先翻回墨量再套标准式。
        // （实际数据里四分量 JPEG 几乎都带 Adobe 标记，else 那条是防御性的。）
        const size_t npix = static_cast<size_t>(w) * h;
        for (size_t p = 0; p < npix; ++p)
        {
            const uint8_t c0 = raw[p * 4 + 0];
            const uint8_t c1 = raw[p * 4 + 1];
            const uint8_t c2 = raw[p * 4 + 2];
            const uint8_t k = raw[p * 4 + 3];

            uint8_t r, g, b;
            if (adobe)
            {
                r = mul255(c0, k);
                g = mul255(c1, k);
                b = mul255(c2, k);
            }
            else
            {
                r = mul255(static_cast<uint8_t>(255 - c0), static_cast<uint8_t>(255 - k));
                g = mul255(static_cast<uint8_t>(255 - c1), static_cast<uint8_t>(255 - k));
                b = mul255(static_cast<uint8_t>(255 - c2), static_cast<uint8_t>(255 - k));
            }
            bm.data[p * 3 + 0] = r;
            bm.data[p * 3 + 1] = g;
            bm.data[p * 3 + 2] = b;
        }
        return bm;
    }
}  // namespace prkd::data