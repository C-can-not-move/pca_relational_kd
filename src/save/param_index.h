// src/save/param_index.h
//
// 「参数与 buffer 的有序表」——整个项目里只有这一处定义顺序。
//
// 为什么需要它：
//   权重文件按这个顺序写，优化器状态按"第几个参数"编号，批次指纹按 epoch 派生；
//   三处必须用同一份顺序，否则「键名对上了、顺序换了」这种错误极难发现。
//
// 顺序契约（改它等于改文件格式）：
//   named_parameters() 全部，然后 named_buffers() 全部。
//   以 BN 为例 running_mean / running_var / num_batches_tracked 在 buffer 里，
//   只存参数的话，模型在 eval() 下是废的 —— 而且加载时不会报错。
//
// ---------------------------------------------------------------------------
// buffer 到底是什么
// ---------------------------------------------------------------------------
//
// 一个 Module 里其实有三种东西，别混：
//
//   ① parameters      requires_grad=true 的张量
//                     优化器会更新它：conv 权重、bias、BN 的 gamma/beta
//
//   ② buffers         注册过、但不训练的张量
//                     named_buffers() 能列出来；优化器看不见它
//                     典型：BN 的 running_mean / running_var / num_batches_tracked
//                          预训练模型的 mean / std、检测模型的 anchors、attention mask
//
//   ③ 普通属性        纯C++/Python 对象（int、vector、别的模块引用）
//                     不进 state_dict，.to(device) 也不会带走它
//
// 为什么要单独分出 buffer 这一类：它们同时要满足三件事 ——
//
//     要保存        进 state_dict，跟着模型一起落盘
//     要搬设备      model->to(cuda) 时它也会被搬过去
//     不要更新      不求梯度，优化器不碰它
//
// 如果当成普通属性：换设备时会忘了搬（张量留在 CPU，前向直接报错或悄悄变慢）。
// 如果当成参数：优化器会去更新它 —— running_mean 被 SGD 动一下，它就不再是统计量了。
//
// 再细分一层：register_buffer(name, tensor, persistent)
//     persistent=true   进 state_dict（默认）
//     persistent=false  不进 state_dict —— 只当缓存用（临时 mask 之类）
//
// libtorch 里怎么写：
//     register_buffer("running_mean", torch::zeros({channels}));
//     register_buffer("mask", m, /*persistent=*/false);
//     for (auto& b : module->named_buffers()) { b.key(); b.value(); }
//
// ---------------------------------------------------------------------------
// 为什么【必须】包含 buffer
// ---------------------------------------------------------------------------
//
// 一段话：parameters 是"学出来的"，buffers 是"统计出来的"；BN 把这两类分开存，
// 而推理时用的恰恰是统计值。
//
//   参数（named_parameters）
//     conv/bn 的 weight、bias、gamma、beta  —— 反向传播会更新它们
//
//   buffer（named_buffers）
//     running_mean / running_var            —— 前向时按动量滑动更新，不参与梯度
//     num_batches_tracked                   —— 一个 0 维 int64，用来算动量
//
// 前向时 BN 的行为取决于 model.train() / model.eval()：
//
//   train()   用【当前这一批】的均值方差归一化，同时更新 running_mean/var
//   eval()    用 running_mean/var 归一化 —— 也就是"训练期间攒下来的统计量"
//
// 所以只恢复参数、不恢复 buffer 的后果是：
//
//   训练模式    看起来一切正常（根本不用 running_*）
//   评估模式    用一开始的 running_mean=0 / running_var=1 和一个随机的动量计数
//              归一化 —— 输出等于把每层都当成没归一化过，精度会掉一大截
//   而且        加载不报错、形状都对、字节数都对，只是数字不对
//
// 这就是「不报错的错」的典型例子：文件是"完整"的，模型是"坏"的。
// 本项目 h_resnet18 共 122 个张量，其中 40 个是 buffer（20 个 BN 的 running_mean/var）
// —— 漏掉它们，等于每层 BN 都回到出厂默认。
//
// 附带一个说明：num_batches_tracked 到底有没有用
//
//   BN 的 running 统计量是按滑动平均更新的：
//       running_mean ← (1 − momentum) · running_mean + momentum · batch_mean
//
//   momentum = 0.1（默认）      固定值，不读 num_batches_tracked
//   momentum = None（累积平均）  momentum_eff = 1 / num_batches_tracked
//                               第 1 批 = 1.0（直接等于这一批的均值），第 100 批 = 0.01
//
// 本项目用的是默认 momentum = 0.1，所以 num_batches_tracked 不影响精度；
// 存它主要是为了让 state_dict 完整、能和 torchvision 的权重逐键对照。
// 真正必须存的是 running_mean / running_var —— 那两个才是 eval() 用的东西。
//
// ---------------------------------------------------------------------------
// 为什么用【字节数】对账，而不是形状
// ---------------------------------------------------------------------------
//
// num_batches_tracked 是 0 维张量：numel()=1、element_size()=8、字节数 8。
// 别的地方可能把它存成 [1]：numel()=1、字节数同样 8，但 sizes() 不同。
// 比字节数，这种"形状写法不一致但内容等价"的情况自动被接受；
// 比形状，就会在这种无害的差异上报错，而在真正危险的地方（同形状、不同语义）照样看不出来。
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "model/resnet.h"
#include "save/tensor_table.h"

namespace prkd::save
{
	// 按契约顺序列出所有键名（不含张量，开销极小）
	std::vector<std::string> names(const prkd::model::ResNet& net);

	// 按契约顺序导出一份可落盘的表（已 detach、CPU、连续）
	//
	// ⚠ 两个循环缺一不可：named_parameters() 之后必须紧跟 named_buffers()。
	//   只遍历 parameters，BN 的 running_mean / running_var / num_batches_tracked 就丢了；
	//   丢的时候不报错、形状对、字节数对，只是 eval() 下每层 BN 都按"没训练过"处理。
	//   细节见文件头的「buffer 到底是什么」与「为什么必须包含 buffer」两节。
	//
	// .to(kCPU) 可能造出临时张量；emplace_back 按值收下，保证活到写完。
	TensorMap dump(const prkd::model::ResNet& net);

	// 把表搬回模型。每个参数与 buffer 都必须有对应键，且字节数一致。
	//
	// ⚠ 同样两个循环都要走：pull 一次 parameters、一次 buffers。
	//   漏掉 buffers 的后果不是"加载失败"，而是"加载成功但模型坏了" ——
	//   训练模式下看不出问题（BN 用当前批统计），一进评估就掉精度。
	//
	// 用【字节数】对账，不用形状 —— 零维与 [1] 的 numel 都是 1、字节数都是 8，
	// 这样 num_batches_tracked 那个 0 维坑自动被兜住。
	// 返回值 = 实际搬进模型的张量个数（本项目 h_resnet18 = 122）。
	//   ⚠ 这个数由结构决定，不抛就必然对 —— 它不能证明文件里是什么。
	std::size_t restore(prkd::model::ResNet& net, const TensorMap& items);
}
