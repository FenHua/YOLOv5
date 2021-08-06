# YOLOv5 image augmentation functions
import cv2
import math
import random
import logging
import numpy as np
from utils.metrics import bbox_ioa
from utils.general import colorstr, segment2box, resample_segments, check_version


# Albu数据增强（必须要装albu的程序包）
class Albumentations:
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A
            check_version(A.__version__, '1.0.0')                                        # 版本检查
            self.transform = A.Compose([A.Blur(p=0.1), A.MedianBlur(p=0.1),A.ToGray(p=0.01)],
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))  # 模糊处理
            logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms))
        except ImportError:
            # 没有安装程序包
            pass
        except Exception as e:
            logging.info(colorstr('albumentations: ') + f'{e}')

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # 执行数据增广
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        # 返回增广后的照片和对应的label
        return im, labels


# HSV空间的数据增广
def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1   # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))  # 融合操作
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=im)             # 返回需要的数据类型


# 均衡化直方图处理图片
def hist_equalize(im, clahe=True, bgr=False):
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    # 将YUV格式转为RGB格式
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


# 复制粘贴目标
def replicate(im, labels):
    h, w = im.shape[:2]                 # 图片的长宽
    boxes = labels[:, 1:].astype(int)   # bbox信息
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2     # box的预估的边长(pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:   # 针对小目标采用复制粘贴增广
        x1b, y1b, x2b, y2b = boxes[i]   # 原始的bbox信息
        bh, bw = y2b - y1b, x2b - x1b   # 原始的bbox长宽信息
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # 随机选一个左上角点
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]   # 在左上角点确定的情况下生成一个新的box
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]       # 粘贴新的目标到图片中
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)   # 新增复制产生的label
    return im, labels


# 图片缩放填充操作（居中padding）
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]                      # 图片的长宽
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)    # 正方形
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])    # 计算缩放率
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)                       # 只缩小不放大
    ratio = r, r  # 长宽缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))     # 新的图片大小
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 宽长需要填充的像素值
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)                # 宽长需要填充的最少像素数
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]       # 宽和长的缩放率
    dw /= 2  # padding左右边界尺度
    dh /= 2  # padding尺度上下界
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)   # 添加padding
    # 返回图片，宽长缩放率，padding大小（居中padding）
    return im, ratio, (dw, dh)


# 随机透视和仿射变换
def random_perspective(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # 中心点
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # 透视变换
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # 旋转和缩放操作
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # 平移操作
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))    # 透视变换
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))     # 仿射变换

    # 对标注信息的增广变换
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)    # 分割数据标注
        new = np.zeros((n, 4))                           # 新生成的标注
        if use_segments:
            segments = resample_segments(segments)       # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T                            # 坐标变换
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine
                new[i] = segment2box(xy, width, height)  # 裁剪操作
        else:
            # 检测boxes变换
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]
    return im, targets


# 复制粘贴（segments应该是目标图片）
def copy_paste(im, labels, segments, probability=0.5):
    n = len(segments)        # 需要复制粘贴目标的个数
    if probability and n:
        h, w, c = im.shape   # 长, 宽, 通道数
        im_new = np.zeros(im.shape, np.uint8)  # 构建新的图片
        for j in random.sample(range(n), k=round(probability * n)):
            l, s = labels[j], segments[j]          # label 和 segments
            box = w - l[3], l[2], w - l[1], l[4]   # 随机一个位置
            ioa = bbox_ioa(box, labels[:, 1:5])    # IoU
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)             # 添加新的标注信息
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)
        result = cv2.bitwise_and(src1=im, src2=im_new)   # 将目标添加到原始图片中
        result = cv2.flip(result, 1)                     # 从左往右翻转增广
        i = result > 0  # pixels to replace
        im[i] = result[i]  # cv2.imwrite('debug.jpg', img)  # debug
    # 返回增广后的图片，label，以及对应的拟粘贴图片
    return im, labels, segments


# cut out
def cutout(im, labels):
    h, w = im.shape[:2]   # 图片长宽
    # 创建随机掩膜
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))   # 掩膜的长
        mask_w = random.randint(1, int(w * s))   # 掩膜的宽
        # 该掩膜生成的位置
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)
        # 掩膜内随机生成数值
        im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]
        # 保留清晰可见的标注信息
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)    # 掩膜位置
            ioa = bbox_ioa(box, labels[:, 1:5])                           # IoU
            labels = labels[ioa < 0.60]                                   # 遮挡小于0.6保留
    # 返回更新后的标注信息以及cutout的图片
    return labels


# 混合数据增强
def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)                  # 混合稀疏，beta分布
    im = (im * r + im2 * (1 - r)).astype(np.uint8)  # 混合
    labels = np.concatenate((labels, labels2), 0)   # 将两者的标注信息进行合并
    return im, labels


# 根据增广与原始bbox的一系列变化
def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]      # 原始bbox 宽 长
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]      # 增广bbox 宽 长
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates