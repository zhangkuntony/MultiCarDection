from __future__ import print_function
from numba import jit
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

@jit
def iou(bb_test, bb_gt):
    """
    在两个box间计算IOU
    :param bb_test: box1 = [x1, y1, x2, y2]
    :param bb_gt: box2 = [x1, y1, x2, y2]
    :return: 交并比IOU
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) + (bb_gt[2] - bb_gt[0]) * (
            bb_gt[3] - bb_gt[1]) - wh)
    return o

def convert_bbox_to_z(bbox):
    """
    将[x1,y1,x2,y2]形式的检测框转为滤波器的状态标识形式[x,y,s,r],其中x, y是框的中心，s是w*h, r是宽高比
    :param bbox: [x1,y1,x2,y2] 分别是左上角坐标和右下角坐标
    :return: [x,y,s,r] 4行1列，其中x,y是box中心位置的坐标，s是面积，r是宽高w/h
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)
    return np.array([x, y, s, r]).reshape(4, 1)

def convert_x_to_bbox(x, score=None):
    """
    将[cx, cy, s, r]的目标框表示转为[x_min, y_min, x_max, y_max]的形式
    :param x: [x,y,s,r], 其中x,y是box中心位置的坐标，s是面积, r是宽高比
    :param score: 置信度
    :return: [x1,y1,x2,y2],左上角坐标和右下角坐标
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape(1, 4)
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape(1, 5)

"""
# 表示观测目标框bbox所对应的单个跟踪对象的内部状态 
"""
class KalmanBoxTracker(object):
    count = 0

    def __init__(self, bbox):
        """
        初始化边界框和跟踪器
        :param bbox:
        """
        # 定义等速模型
        # 内部使用KalmanFilter, 7个状态变量和4个观测输入
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # F是状态变换矩阵
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]]
        )
        # H是观测函数
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
        )
        # R是观测噪声协方差矩阵
        self.kf.R[2:, 2:] *= 10.
        # P是协方差矩阵
        self.kf.P[4:, 4:] *= 1000.      # 对无法观测的初始速度赋予高度不确定性
        self.kf.P *= 10.
        # Q是过程噪声矩阵
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        # 内部状态估计
        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        使用观察到的目标框更新状态向量。filterpy.kalman.KalmanFilter.update 会根据观测修改内部状态估计self.kf.x.
        重置self.time_since_update, 清空self.history
        :param bbox: 目标框
        :return:
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        推进状态向量并返回预测的边界框估计。
        将预测结果追加到self.history。由于get_state直接访问self.kf.x, 所以self.history没有用到
        :return:
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        返回当前边界框估计值
        :return:
        """
        return convert_x_to_bbox(self.kf.x)