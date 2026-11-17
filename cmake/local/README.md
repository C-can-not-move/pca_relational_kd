# Windows 本机构建（MSVC 14.44 + CUDA 13.3）

本目录与 `CMakeUserPresets.json` 都是**本机私有**，不入库（已在 `.gitignore` 里）。
预设里只剩路径，**说明全写在这份文件里** —— 因为它能写注释，而 JSON 不能。

---

## 一、为什么要固定 MSVC 14.44

```
VS 2026 升级后，默认工具集从 MSVC 14.51.36231 变成 14.52.37025
CUDA 13.3 的 cudafe++ 解析 14.52 的头文件时崩溃：
    nvcc error : 'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION)
任何 .cu 都编不了 —— 连空文件 `__global__ void e(){ }` 也一样
```

而 **libtorch 的 `share/cmake/Caffe2/public/cuda.cmake` 是无条件 `enable_language(CUDA)`**
（`Caffe2Config.cmake` 里那句是 `if(1)`，写死的），所以：

```
即使本项目一个 .cu 都没有
CMake 配置阶段也会去调 nvcc
⇒ 直接崩在 "Compiling the CUDA compiler identification source file"
```

**旁证：** 2026-10-27 的 `MultiKD_C++` 能编，看它的 `CMakeCache.txt` 里是
`MSVC 14.51.36231` —— 那正是 CUDA 13.3 还认的版本。

---

## 二、为什么"只把 cl.exe 指到 14.44"没用

试过，还是崩。原因是：

```
cl.exe 换了  →  vcvars 注入的 INCLUDE / LIB 还是 14.52 的
⇒ 编译器与头文件不同版本
⇒ cudafe++ 一样崩
```

所以预设里 `environment` 把 **INCLUDE / LIB / LIBPATH / PATH 四样成套固定**，
`cacheVariables` 再把 C/CXX/CUDA_HOST 三个编译器都指到 14.44。

**判据：** 配置输出里应当是

```
The CXX  compiler identification is MSVC 19.44.35230.0
The CUDA compiler identification is NVIDIA 13.3.73 with host compiler MSVC 19.44.35230.0
```

---

## 三、路径清单（对应预设里的四组）

```
MSVC        D:\Visual Studio\IDE\VC\Tools\MSVC\14.44.35207
Windows SDK C:\Program Files (x86)\Windows Kits\10\          版本 10.0.28000.0
CUDA        D:\CUDA\Toolkit                                  13.3.73
Ninja       D:\Visual Studio\IDE\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
```

**INCLUDE 只要四条就够**（ATLMFC / winrt / cppwinrt / NETFX 都去掉了，本项目不用）：

```
…\14.44.35207\include
…\Windows Kits\10\include\10.0.28000.0\ucrt
…\Windows Kits\10\include\10.0.28000.0\um
…\Windows Kits\10\include\10.0.28000.0\shared
```

**LIB 三条、LIBPATH 一条**同理。

---

## 四、验证方法

```powershell
# ① 配置
cmake --preset x64-release-local

# ② 编译（注意：--build 不会重放预设的 environment，
#    要在 VS 里生成，或者在设好 14.44 环境的 shell 里跑）
cmake --build out/build/x64-release-local --parallel 8
```

**手工验证 nvcc 是否可用：**

```cmd
call "D:\Visual Studio\IDE\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.44.35207
"D:\CUDA\Toolkit\bin\nvcc.exe" -c 任意.cu -o out.obj -arch=sm_75
```

能出 obj 就说明整条链通了（退出码 0）。

---

## 五、遗留风险

```
libtorch 是 cu126，而本机 toolkit 是 13.3
只链接、不自编 .cu          → 能过（已实测 43 个目标，编译 + 链接均通过）
真要写 .cu 或直接调 CUDA API → 必须换成 CUDA 12.6，与 libtorch 对齐
```

**换 12.6 时只改一处：** 预设里 `CMAKE_CUDA_COMPILER` 指到 `…\v12.6\bin\nvcc.exe`，
`environment` 与其它 cacheVariables 都不用动。
