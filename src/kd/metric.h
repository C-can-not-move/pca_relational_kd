// src/kd/metric.h —— 关系损失的"矩阵度量"枚举（轻量头，不拉 torch）
//
// 为什么单独一个头：
//   config.h 只想要一个枚举值，不该为了它把 libtorch 拖进来。
//   pcar_loss.h 需要同一份定义来 switch —— 所以放这里，两边都 include。
//
// 两个取值：
//   Huber  逐元素比两张归一化距离矩阵（尺度敏感；RKD-D 那一类）
//   Angle  上三角去心余弦 = Pearson 相关（尺度/平移免疫，只看形状）
//
// 具体的度量函数（要 torch）在 pcar_loss.h 里：
//   metric_huber(Ds, Dt, delta) / metric_angle(Ds, Dt) / matrix_corr(Ds, Dt)

#pragma once

#include <stdexcept>
#include <string>

namespace prkd::kd
{
	enum class Metric
	{
		Huber,
		Angle,
	};

	inline const char* to_string(Metric m)
	{
		switch (m)
		{
		case Metric::Huber: return "huber";
		case Metric::Angle: return "angle";
		}
		return "?";
	}

	// 不认识就抛（调用点是配置解析，抛得越早越好）
	inline Metric from_string(const std::string& s)
	{
		if (s == "huber") return Metric::Huber;
		if (s == "angle") return Metric::Angle;
		throw std::runtime_error("未知的 kd.mode: \"" + s + "\"（只认 huber / angle）");
	}
}
