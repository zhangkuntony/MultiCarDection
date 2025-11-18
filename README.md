# MultiCarDection
多车辆检测与跟踪系统

## 项目简介

MultiCarDection是一个基于YOLOv3目标检测和卡尔曼滤波的多目标跟踪系统，主要用于视频中的车辆检测、跟踪和计数。该项目实现了SORT(Simple Online and Realtime Tracking)算法，结合YOLOv3目标检测器，实现了对视频中车辆的实时检测与跟踪。

## 功能特点

- 🚗 基于YOLOv3的车辆检测
- 📊 使用卡尔曼滤波进行运动预测
- 🔄 实现SORT算法的多目标跟踪
- 📈 虚拟线圈车辆计数
- 🎯 支持双向车道车辆计数
- 🎬 视频输入与输出处理

## 算法原理

### 目标检测
项目使用YOLOv3(You Only Look Once)目标检测算法来检测视频中的车辆。YOLOv3是一种单阶段目标检测算法，具有高速度和高精度的特点。

### 目标跟踪
跟踪部分基于卡尔曼滤波器实现SORT算法，主要包含以下步骤：

1. **卡尔曼滤波预测**：使用等速模型预测目标在下一帧的位置
2. **数据关联**：通过IOU(Intersection Over Union)匹配检测结果与跟踪目标
3. **状态更新**：根据匹配结果更新卡尔曼滤波器的状态
4. **生命周期管理**：管理跟踪器的创建、更新和删除

### 车辆计数
使用虚拟线圈技术，通过检测车辆中心点与虚拟检测线的交叉来计数车辆。支持双向车道分别计数。

## 文件结构

```
MultiCarDection/
├── kalman.py          # 卡尔曼滤波器和SORT算法实现
├── yolo.py            # 主程序，YOLO检测与跟踪流程
├── input/             # 输入视频文件夹
│   └── test_1.mp4     # 测试视频
├── output/            # 输出视频文件夹
│   └── output.mp4     # 处理后的视频
├── yolo-coco/         # YOLO模型文件
│   ├── coco.names     # COCO数据集类别名称
│   ├── yolov3.cfg     # YOLOv3配置文件
│   └── yolov3.weights # YOLOv3预训练权重
├── requirements.txt   # 项目依赖
└── README.md          # 项目说明
```

## 核心组件

### KalmanBoxTracker类
单个目标的卡尔曼滤波跟踪器，维护一个目标的状态估计。使用7个状态变量和4个观测输入：
- 状态变量：[x, y, s, r, vx, vy, vs] (位置、尺寸、速度)
- 观测变量：[x, y, s, r] (位置、尺寸)

### Sort类
多目标跟踪器，管理多个KalmanBoxTracker对象，实现SORT算法。主要参数：
- max_age: 目标未被检测到的最大帧数，超过后会被删除
- min_hits: 目标被连续检测到多少次后才开始输出跟踪结果

### IOU函数
计算两个边界框的交并比，用于数据关联。使用Numba JIT编译器加速计算。

## 安装与使用

### 环境要求
- Python 3.7+
- OpenCV 4.x
- NumPy
- SciPy
- filterpy
- numba

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行项目
1. 准备输入视频文件，放入`input/`文件夹
2. 确保YOLOv3模型文件在`yolo-coco/`文件夹中
3. 运行主程序：
```bash
python yolo.py
```

### 输出结果
处理后的视频将保存在`output/output.mp4`中，视频中包含：
- 检测框和跟踪ID
- 虚拟检测线
- 车辆计数显示

## 参数调整

### YOLO检测参数
- 置信度阈值：0.3 (代码中可调整)
- NMS阈值：0.5 (代码中可调整)
- IOU阈值：0.3 (代码中可调整)

### 跟踪参数
- max_age: 默认1帧
- min_hits: 默认3次检测
- 虚拟检测线位置：可通过修改`line`变量调整

## 性能优化

- 使用Numba JIT编译器加速IOU计算
- 使用OpenCV的DNN模块进行YOLO推理
- 优化数据关联过程，减少计算复杂度

## 扩展应用

该项目可以扩展应用于：
- 交通流量监测
- 智能停车场管理
- 车辆行为分析
- 道路安全监测

## 参考与致谢

- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
- [Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)
- [filterpy - Kalman filters and other optimal Bayesian filters](https://github.com/rlabbe/filterpy)

## 许可证

本项目仅供学习和研究使用。
