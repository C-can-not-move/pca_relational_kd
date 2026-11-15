// src/save/sha1.h
//
// 两个与"留证"有关的小工具，独立无依赖：
//   sha1_hex    比对同一份文本（配置原文、批次指纹）
//   now_iso8601 ISO8601 时间戳，Windows / Linux 都能用
//
// 为什么自己写 sha1 而不引第三方：
//   只为了"两份文本是不是同一份"，不做安全用途；
//   一百行不到的纯函数，比引一个库更容易在两边编过。
#pragma once

#include <cstddef>
#include <string>

namespace prkd::save
{
	std::string sha1_hex(const void* data, size_t n);
	std::string sha1_hex(const std::string& s);

	// 形如 2026-11-15T15:20:03+08:00（本地时区）
	std::string now_iso8601();
}
