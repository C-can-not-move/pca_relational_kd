#include "data/decode.h"

#include <csetjmp>
#include <cstring>
#include <stdexcept>
#include <string>

extern "C"
{
#include <jpeglib.h>
}

namespace prkd::data
{
    namespace
    {
        // libjpeg 默认的 error_exit 会直接 exit()。必须换成 longjmp，
        // 否则一张坏图就把整个进程干掉。
        struct JpegErrorMgr
        {
            jpeg_error_mgr pub;
            std::jmp_buf jb;
            char msg[JMSG_LENGTH_MAX] = { 0 };
        };

        void on_error(j_common_ptr cinfo)
        {
            auto* e = reinterpret_cast<JpegErrorMgr*>(cinfo->err);
            (*cinfo->err->format_message)(cinfo, e->msg);
            std::longjmp(e->jb, 1);
        }

        // 屏蔽 libjpeg 往 stderr 打的 warning（ImageNet 里有一部分图会触发）。
        // 注意签名：output_message 只收一个 j_common_ptr；
        // 收两个参数的那个是 emit_message(cinfo, level)。
        void on_output(j_common_ptr) {}

        // Adobe 写的 CMYK 是【反转】的：存的是 255 - 墨量。
        // 标准 CMYK→RGB 是 R = (255-C)*(255-K)/255，
        // 把 C_ink = 255-c_stored 代进去，255-C_ink 就是 c_stored：
        //   R = c_stored * k_stored / 255
        // 非 Adobe 的四分量不是反转的，走另一条。
        inline uint8_t mul255(uint8_t a, uint8_t b)
        {
            return static_cast<uint8_t>((static_cast<unsigned>(a) * b + 127u) / 255u);
        }
    }  // namespace

    Bitmap decode_rgb8(const uint8_t* bytes, size_t n)
    {
        if (bytes == nullptr || n == 0)
        {
            throw std::runtime_error("decode_rgb8: 空输入");
        }

        jpeg_decompress_struct cinfo;
        std::memset(&cinfo, 0, sizeof(cinfo));

        JpegErrorMgr jerr;
        std::memset(&jerr, 0, sizeof(jerr));

        cinfo.err = jpeg_std_error(&jerr.pub);
        jerr.pub.error_exit = on_error;
        jerr.pub.output_message = on_output;

        // ★ setjmp 之后到 longjmp 之间，不能有需要析构的局部对象。
        //   这里只用 POD 和 cinfo；Bitmap 和要抛的异常都在分支里建。
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

        // 四分量（CMYK / YCCK）保持四通道出来，之后再转；其余一律让 libjpeg 转成 RGB。
        const bool is_cmyk = (cinfo.jpeg_color_space == JCS_CMYK ||
                              cinfo.jpeg_color_space == JCS_YCCK);
        cinfo.out_color_space = is_cmyk ? JCS_CMYK : JCS_RGB;

        jpeg_start_decompress(&cinfo);

        const int w = static_cast<int>(cinfo.output_width);
        const int h = static_cast<int>(cinfo.output_height);
        const int comps = static_cast<int>(cinfo.output_components);   // 3 或 4

        if (w <= 0 || h <= 0)
        {
            jpeg_destroy_decompress(&cinfo);
            throw std::runtime_error("decode_rgb8: 图像尺寸为 0");
        }

        std::vector<uint8_t> raw(static_cast<size_t>(w) * h * comps);
        const size_t row_bytes = static_cast<size_t>(w) * comps;

        while (cinfo.output_scanline < cinfo.output_height)
        {
            JSAMPROW row = raw.data() + static_cast<size_t>(cinfo.output_scanline) * row_bytes;
            jpeg_read_scanlines(&cinfo, &row, 1);
        }

        const bool adobe = cinfo.saw_Adobe_marker != 0;
        jpeg_finish_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);

        Bitmap bm;
        bm.w = w;
        bm.h = h;
        bm.c = 3;
        bm.data.resize(static_cast<size_t>(w) * h * 3);

        if (comps == 3)
        {
            std::memcpy(bm.data.data(), raw.data(), raw.size());
            return bm;
        }

        // comps == 4：CMYK → RGB
        for (size_t p = 0; p < static_cast<size_t>(w) * h; ++p)
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