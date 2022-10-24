# 目录

<!-- TOC -->

- [目录](#目录)
- [Bi-Real-Net 描述](#Bi-Real-Net描述)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
- [训练和测试](#训练和测试)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [ImageNet-1k 上的 Bi-Real-Net-34layers](#imagenet-1k上的Bi-Real-Net-34layers)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# [Bi-Real-Net 描述](#目录)

Bi-Real-Net 的目的是提升二值化卷积神经网络（1-bit CNN）的精度。虽然 1-bit CNN 压缩程度高，但是其当前在大数据集上的分类精度与对应的实值 CNN 相比有较大的精度下降。作者提出的 Bi-Real net 用 shortcut 传递网络中已有的实数值，从而提高二值化网络的表达能力。并且改进了现有的 1-bit CNN 训练方法，原有训练方法在对于激活值的求导和对于参数的更新存在导数不匹配的问题，作者采用二阶拟合 sign 的 ApproxSign 的导数来作为 sign 的导数，从而缩小导数值的不匹配问题，在实现网络权重与输出二值化的同时，确保了较高的推理精度，尤其是在大型数据集（ILSVRC ImageNet）上的表现。

本文复现的是 Bi-Real-Net-34layers.

# [数据集](#目录)

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像
- 数据格式：JPEG
    - 注：数据在dataset.py中处理。
- 下载数据集，目录结构如下：

 ```text
└─dataset
    ├─train                 # 训练数据集
    └─val                   # 评估数据集
```

# [特性](#目录)

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html)
的训练方法，使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。

# [环境要求](#目录)

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r1.3/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/r1.3/index.html)

# [脚本说明](#目录)

## 脚本及样例代码

```bash
├── Bi-Real-Net
  ├── README_CN.md                        // Bi-Real-Net相关说明
  ├── src
      ├──configs                          // Bi-Real-Net的配置文件
      ├──data                             // 数据集配置文件
          ├──imagenet.py                  // imagenet配置文件
          ├──augment                      // 数据增强函数文件
          ┕──data_utils                   // modelarts运行时数据集复制函数文件
  │   ├──models                           // 模型定义文件夹
          ┕──birealnet                    // Bi-Real-Net模型定义文件
  │   ├──trainers                         // 自定义TrainOneStep文件
  │   ├──tools                            // 工具文件夹
          ├──callback.py                  // 自定义回调函数，训练结束测试
          ├──cell.py                      // 一些关于cell的通用工具函数
          ├──criterion.py                 // 关于损失函数的工具函数
          ├──get_misc.py                  // 一些其他的工具函数
          ├──optimizer.py                 // 关于优化器和参数的函数
          ┕──schedulers.py                // 学习率衰减的工具函数
  ├── train.py                            // 训练文件
  ├── eval.py                             // 评估文件
  ├── export.py                           // 导出模型文件
  ├── postprocess.py                      // 推理计算精度文件
  ├── preprocess.py                       // 推理预处理图片文件

```

## 脚本参数

- 配置 Bi-Real-Net 和 ImageNet-1k 数据集。

  ```python
    # Architecture
    arch: birealnet34                   # Bi-Real-Net结构选择
    # ===== Dataset ===== #
    data_url: ./data/imagenet           # 数据集地址
    set: ImageNet                       # 数据集名字
    num_classes: 1000                   # 数据集分类数目
    interpolation: bicubic              # 图像缩放插值方法
    # ===== Learning Rate Policy ======== #
    optimizer: adam                     # 优化器类别
    base_lr: 0.001                      # 基础学习率
    min_lr: 0.000001                    # 最小学习率
    lr_scheduler: lambda_lr             # 学习率衰减策略
    # ===== Network training config ===== #
    amp_level: O1                       # 混合精度策略(创建黑白名单)
    clip_global_norm_value: 5.          # 全局梯度范数裁剪阈值
    is_dynamic_loss_scale: True         # 是否使用动态缩放
    epochs: 256                         # 训练轮数
    label_smoothing: 0.1                # 标签平滑参数
    weight_decay: 0.                    # 权重衰减参数
    momentum: 0.9                       # 优化器动量
    batch_size: 128                     # 批次大小
    # ===== Hardware setup ===== #
    num_parallel_workers: 32            # 数据预处理线程数
    device_target: Ascend               # Ascend npu
  ```

# [训练和测试](#目录)

- Ascend处理器环境运行

  ```bash
  # 在openi平台使用多卡训练
  python train.py --device_num 8 --dataset_sink_mode True --run_openi True --device_target Ascend

  # 在openi平台使用单卡评估
  python eval.py --device_num 1 --device_target Ascend --run_openi True --pretrained True
  ```

# [模型描述](#目录)

## 性能

### 评估性能

#### ImageNet-1k 上的 Bi-Real-Net-layers34

| 参数 | Ascend |
| ---- | ------ |
| 模型 | Bi-Real-Net |
| 模型版本 | Bi-Real-Net-layers34 |
| 资源 | Ascend 910 |
| 上传日期 | 2022-8-18 |
| MindSpore版本 | 1.5.1 |
| 数据集 | ImageNet-1k Train，共1,281,167张图像 |
| 训练参数| epoch=256, batch_size=128 |
| 优化器 | Adam |
| 损失函数 | CrossEntropySmooth |
| 损失|  2.8036785 |
| 输出 | 概率 |
| 分类准确率 | 八卡：top1: 62.68229166666667% , top5: 83.90625% |
| 速度 | 八卡：196毫秒/步 |
| 训练耗时 | 26h19min20s（run on OpenI）|

# ModelZoo主页

请浏览官网[主页](https://gitee.com/mindspore/models)