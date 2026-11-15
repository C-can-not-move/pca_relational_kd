// src/save/sha1.cpp
#include "save/sha1.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <sstream>
#include <string>

namespace prkd::save
{
	namespace
	{
		class Sha1
		{
		public:
			Sha1() { reset(); }

			void update(const void* data, size_t len)
			{
				const auto* p = static_cast<const uint8_t*>(data);
				len_ += len;
				while (len > 0)
				{
					const size_t take = std::min(len, size_t(64) - buf_len_);
					std::memcpy(buf_ + buf_len_, p, take);
					buf_len_ += take; p += take; len -= take;
					if (buf_len_ == 64) { process(buf_); buf_len_ = 0; }
				}
			}

			std::string hex()
			{
				const uint64_t bits = len_ * 8;      // 先记下，之后补位会继续累加 len_
				const uint8_t pad = 0x80;
				update(&pad, 1);
				const uint8_t zero = 0x00;
				while (buf_len_ != 56) update(&zero, 1);

				uint8_t tail[8];
				for (int i = 0; i < 8; ++i) tail[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
				std::memcpy(buf_ + 56, tail, 8);
				process(buf_);
				buf_len_ = 0;

				char out[41];
				for (int i = 0; i < 5; ++i) std::snprintf(out + i * 8, 9, "%08x", h_[i]);
				return std::string(out, 40);
			}

		private:
			void reset()
			{
				h_[0] = 0x67452301u; h_[1] = 0xEFCDAB89u; h_[2] = 0x98BADCFEu;
				h_[3] = 0x10325476u; h_[4] = 0xC3D2E1F0u;
				len_ = 0; buf_len_ = 0;
			}

			static uint32_t rol(uint32_t v, int s) { return (v << s) | (v >> (32 - s)); }

			void process(const uint8_t* chunk)
			{
				uint32_t w[80];
				for (int i = 0; i < 16; ++i)
					w[i] = (uint32_t(chunk[i * 4]) << 24) | (uint32_t(chunk[i * 4 + 1]) << 16)
					     | (uint32_t(chunk[i * 4 + 2]) << 8) | uint32_t(chunk[i * 4 + 3]);
				for (int i = 16; i < 80; ++i)
					w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

				uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4];
				for (int i = 0; i < 80; ++i)
				{
					uint32_t f = 0, k = 0;
					if (i < 20)      { f = (b & c) | (~b & d);          k = 0x5A827999u; }
					else if (i < 40) { f = b ^ c ^ d;                   k = 0x6ED9EBA1u; }
					else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCu; }
					else             { f = b ^ c ^ d;                   k = 0xCA62C1D6u; }

					const uint32_t tmp = rol(a, 5) + f + e + k + w[i];
					e = d; d = c; c = rol(b, 30); b = a; a = tmp;
				}
				h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e;
			}

			uint32_t h_[5];
			uint64_t len_;
			uint8_t  buf_[64];
			size_t   buf_len_;
		};
	}  // namespace

	std::string sha1_hex(const void* data, size_t n)
	{
		Sha1 s;
		s.update(data, n);
		return s.hex();
	}

	std::string sha1_hex(const std::string& s)
	{
		return sha1_hex(s.data(), s.size());
	}

	std::string now_iso8601()
	{
		const auto now = std::chrono::system_clock::now();
		const std::time_t t = std::chrono::system_clock::to_time_t(now);

		std::tm local_tm{};
		std::tm utc_tm{};
#if defined(_WIN32)
		localtime_s(&local_tm, &t);
		gmtime_s(&utc_tm, &t);
#else
		localtime_r(&t, &local_tm);
		gmtime_r(&t, &utc_tm);
#endif

		char buf[32];
		std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &local_tm);
		std::ostringstream os;
		os << buf;

		// 时区偏移：不用 tm_gmtoff（非标准，Windows 没有），用本地与 UTC 反算
		const long long local_s = static_cast<long long>(local_tm.tm_hour) * 3600
			+ local_tm.tm_min * 60 + local_tm.tm_sec;
		const long long utc_s = static_cast<long long>(utc_tm.tm_hour) * 3600
			+ utc_tm.tm_min * 60 + utc_tm.tm_sec;

		long long off = local_s - utc_s;
		if (off >  12 * 3600) off -= 24 * 3600;     // 跨日：本地超前（如 +13/+14）
		if (off < -14 * 3600) off += 24 * 3600;     // 跨日：本地落后

		const long long a = off >= 0 ? off : -off;
		os << (off >= 0 ? '+' : '-')
		   << (a / 3600 < 10 ? "0" : "") << a / 3600 << ':'
		   << (a % 3600 / 60 < 10 ? "0" : "") << a % 3600 / 60;
		return os.str();
	}

}  // namespace prkd::save
