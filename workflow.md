# 工作日记与记录

## 11月3日更新
***
C++上面读取JPEG图片，为了确保win和linux都能读取，使用了libjpeg-turbo库。这个会在CMakeLists.txt中添加libjpeg-turbo的依赖。

Python 读 JPEG 的底层是 libjpeg-turbo。和 C++ 要用的 libjpeg-turbo 属于同一个解码器家族，但不是同一次构建会有影响。
***
transform类应该有的功能，为了让增强方法更加明显，采用的python类似的一个操作一个函数的形式
```
def get_imagenet_train_transform(mean, std):
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

def get_imagenet_test_transform(mean, std):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
```

数据流程应该是imagenet_loader调用decode来读取jpeg，然后交给transform类的函数进行数据增强。

transform类把各种增强方式封装成一个个函数，调用时传入参数即可。transform类的函数应该是可以直接调用的。
***
**待验证的假设，非结论**Python 的 transform 跑在 CPU 的 worker 进程里。这不是遗漏：ImageNet 的图解码后尺寸不一，必须先逐张变换到同一尺寸才能拼 batch，所以变换天然落在 CPU。

真正的优化空间只在批量形状固定之后：

1. 验证路径（Resize + CenterCrop，整批同形状）→ 可以整批上 GPU，是净收益
2. 训练路径（RandomResizedCrop，每张的框不同）→ 逐张上 GPU 反而更慢，kernel launch 开销占主导
3. 若将来引入固定尺寸的 uint8 缓存（当前方案已排除），那时整批上传再变换才是净收益。

**结论**：现在不改。 等实测出「数据管线成为瓶颈，且瓶颈在 CPU 变换」时再动。

*** 
imageNet 1K 里面存在坏图，存在读取到脏数据的情况。需要在 decode 里面加上异常处理，至少需要个计数器。

imageNet 1K 上面使用普通的蒸馏手段效果不好，难道是因为脏数据的原因？
```
----- tinyimagenet 解码体检 -----
干净        : 99968
有警告      : 0   （读完了，但 libjpeg 报过警）
数据不全    : 0   （提前停住，缺口是 0 黑）
合计        : 99968
→ 这一轮读到的图全部干净
```

## 11月4日更新
***
### 数据读取多线程设计
一般来说，torch库会有多线程，但这会和自己起多线程冲突→ 所以这个时候必须设置库的intra-op为1