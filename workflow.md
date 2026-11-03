# 工作日记与记录

## 11月3日更新
C++上面读取JPEG图片，为了确保win和linux都能读取，使用了libjpeg-turbo库。这个会在CMakeLists.txt中添加libjpeg-turbo的依赖。

Python 读 JPEG 的底层是 libjpeg-turbo。和 C++ 要用的 libjpeg-turbo 属于同一个解码器家族，但不是同一次构建会有影响。

