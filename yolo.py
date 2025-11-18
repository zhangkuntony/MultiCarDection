from kalman import *
import time
import cv2
import numpy as np

line = [(0, 150), (2560, 150)]
# 车辆总数
counter = 0
# 正向车道的车辆数据
counter_up = 0
# 逆向车道的车辆数据
counter_down = 0

# 创建跟踪器对象
tracker = Sort()
memory = {}

# 线与线的碰撞检测：叉乘的方法判断两条线是否相交
# 计算叉乘符号
def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

# 检测AB和CD两条直线是否相交
def intersect(a, b, c, d):
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


# 利用yolov3模型进行目标检测
# 加载模型相关信息
# 加载可以检测的目标的类型
label_path = "./yolo-coco/coco.names"
labels = open(label_path).read().strip().split("\n")
# 生成多种不同的颜色
np.random.seed(42)
colors = np.random.randint(0, 255, size=(200, 3), dtype='uint8')
# 加载预训练的模型：权重 配置信息，进行恢复
weights_path = "./yolo-coco/yolov3.weights"
config_path = "./yolo-coco/yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
# 获取yolo中每一层的名称
ln = net.getLayerNames()
# 获取输出层的名称: [yolo-82, yolo-94, yolo-106]
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# 读取图像
# frame = cv2.imread('./images/car2.jpg')
# (W,H) = (None,None)
# (H,W) = frame.shape[:2]
# 视频
vs = cv2.VideoCapture('./input/test_1.mp4')
(W, H) = (None, None)
writer = None
try:
    prop = cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("INFO:{} total Frame in video".format(total))
except Exception as e:
    print("ERROR: could not determine in video:", e)

# 遍历每一帧图像
while True:
    (grabed, frame) = vs.read()
    if not grabed:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    # 将图像转换为blob，进行前向传播
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    # 将blob送入网络
    net.setInput(blob)
    start = time.time()
    # 前向传播，进行预测，返回目标框边界和相应的概率
    layerOutputs = net.forward(ln)
    end = time.time()

    # 存放目标的检测框
    boxes = []
    # 置信度
    confidences = []
    # 目标类别
    class_ids = []

    # 遍历每个输出
    for output in layerOutputs:
        # 遍历检测结果
        for detection in output:
            # detection: 1*85 [5:]表示类别，[0:4]bbox的位置信息 [5]置信度
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                # 将检测结果与原图片进行适配
                box = detection[0:4] * np.array([W, H, W, H])
                (center_x, center_y, width, height) = box.astype('int')
                # 左上角坐标
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                # 更新目标框，置信度，类别
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # 非极大值抑制
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    # 检测框：左上角和右下角
    dets = []
    if len(idxs) > 0:
        idxs = idxs.flatten() if hasattr(idxs, 'flatten') else np.array(idxs).flatten()
        for i in idxs:
            if labels[class_ids[i]] == "car":
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                dets.append([x, y, x + w, y + h, confidences[i]])

    dets = np.asarray(dets)

    # # 显示
    # plt.imshow(frame[:,:,::-1])
    # plt.show()

    # SORT 目标跟踪
    if np.size(dets) == 0:
        continue
    else:
        tracks = tracker.update(dets)

    # 跟踪框
    boxes = []
    # 置信度
    index_ids = []
    # 前一帧跟踪结果
    previous = memory.copy()
    memory = {}
    for track in tracks:
        boxes.append([track[0], track[1], track[2], track[3]])
        index_ids.append(int(track[4]))
        memory[index_ids[-1]] = boxes[-1]

    # 碰撞检测
    if len(boxes) > 0:
        i = int(0)
        # 遍历跟踪框
        for box in boxes:
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))
            color = [int(c) for c in colors[index_ids[i] % len(colors)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            # 根据在上一帧和当前帧的检测结果，利用虚拟线圈完成车辆计数
            if index_ids[i] in previous:
                previous_box = previous[index_ids[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                p1 = (int(x2 + (w2 - x2) / 2), int(y2 + (h2 - y2) / 2))
                p0 = (int(x + (w - x) / 2), int(y + (h - y) / 2))

                # 利用p0, p1与line进行碰撞检测
                if intersect(p0, p1, line[0], line[1]):
                    counter += 1
                    # 判断方向
                    if y2 > y:
                        counter_down += 1
                    else:
                        counter_up += 1
            i += 1

    # 将车辆计数的相关结果放在视频上
    cv2.line(frame, line[0], line[1], (0, 255, 0), 3)
    cv2.putText(frame, str(counter), (30, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 0, 0), 3)
    cv2.putText(frame, str(counter_up), (130, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 255, 0), 3)
    cv2.putText(frame, str(counter_down), (230, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 3)

    # 将检测结果保存在视频
    if writer is None:
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter("./output/output.mp4", fourcc, 30, (frame.shape[1], frame.shape[0]), True)
    writer.write(frame)
    cv2.imshow("", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
writer.release()
vs.release()
cv2.destroyAllWindows()
