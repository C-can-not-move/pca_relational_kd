# Windows 本机构建与运行 · 备注

> 这份文件是本机私有的操作笔记，不入库。
> 工具链（14.44 / CUDA 13.3）为什么被固定，写在 `CMakeUserPresets.json` 的 `description` 里 —— 那里是 VS 会显示的位置。

---

## 一、跑长任务：日志一定重定向到文件

**症状**（2026-11-17 夜间，`kd` 第 18 轮）：进程不退出、GPU 掉到 7%、日志停在某一批、看门狗不响。
VS 附加后主线程栈是：

```
printf → _vfprintf_l → WriteFile → NtWaitForSingleObject
```

**根因不是死锁，是 stdout 背压**：

```
printf 写的是 stdout 的缓冲
缓冲满 / 我们每行都 fflush ⇒ 一次真正的 write()
接收方（终端、调试器输出面板、管道）读得慢或暂时不读
⇒ write 阻塞 ⇒ 调 printf 的线程停住 ⇒ 训练循环停住
```

**调试器最容易撞上**：VS 的输出面板要逐行格式化，UI 忙时就不读。

**规矩（本机与服务器都适用）**：

```powershell
.\out\build\x64-release-local\pca_relational_kd.exe > kd.log 2>&1

# 看进度
Get-Content kd.log -Tail 6
```

**三个缓解手段（按代价排）**：

```
① 重定向到文件         零成本，首选
② debug.log_every = 0  只在轮末打一行
③ 去掉每行 fflush，或只在 _isatty(_fileno(stdout)) 时 flush
```

**别做的事**：不要靠"在调试器里看输出"来监控长任务 —— 那是最慢的消费者。

---

## 二、两种"看起来像死锁"的现象，别混

```
① stdout 背压     主线程停在 printf/write；GPU 掉下来；看门狗不响
                   ⇒ 处置：重定向到文件

② 退出期顺序      程序打完 OK 之后崩，报 0xC0000409（堆栈 Cookie）
                  根因：libtorch/CUDA 的静态析构先跑，而 loader 线程与模型还在用
                  ⇒ 处置：return 之前显式 reset 掉 loader 与两个模型
```

**（**判据：**卡在运行中 vs 崩在收尾后**；一个不退出，一个会退出并弹窗。）** **
