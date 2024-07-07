import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
import random
from math import cos, sin
from py_path_signature.data_models.stroke import Stroke
from py_path_signature.path_signature_extractor import PathSignatureExtractor
from rdp import rdp


def removePoint(strokes):
    # draw_text_line(strokes, 'origin points')
    new_strokes = []
    for stroke in strokes:
        new_stroke = rdp(stroke, epsilon=2)
        new_strokes.append(new_stroke)
    # draw_text_line(new_strokes, 'remove points')
    return new_strokes

def draw_xys_seq_sample(seq):
    seq = np.array(seq)
    seq[:, 0:2] = np.cumsum(seq[:, 0:2], axis=0)
    new_seq = np.split(seq, np.where(seq[:, 2] > 0)[0] + 1)
    plt.figure(figsize=(15, 2))
    for s in new_seq:
        plt.plot(s[:, 0], -s[:, 1], color='k', linewidth=1, marker='o', markersize='3', markerfacecolor='cornflowerblue',
                 markeredgecolor='cornflowerblue')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def draw_path_signature(path_signature):
    """path_signature：(C, H, W)"""
    # plt.figure(figsize=(15, 8))
    fig, axs = plt.subplots(7, 1)
    for index, channel in enumerate(path_signature):
        img = Image.fromarray(channel)
        plt_img = np.array(img)
        axs[index].axis('off')
        axs[index].imshow(plt_img)
    plt.show()
    pass

def draw_text_line(Strokes):
    plt.figure(figsize=(20, 3))
    for stroke in Strokes:
        x = [point[0] for point in stroke]
        y = [-point[1] for point in stroke]
        plt.plot(x, y, '-bx')
    plt.show()

def calu_path_signature(dict_strokes, size):
    strokes = [Stroke(**stroke) for stroke in dict_strokes]
    path_signature_extractor = PathSignatureExtractor(
        order=2, rendering_size=size, min_rendering_dimension=5, max_aspect_ratio=30, delta=5
    )
    path_signature = path_signature_extractor.extract_signature(strokes=strokes)
    return path_signature




def slantStroke(strokes):
    factor = random.uniform(3, 8)
    if random.random() < 0.3:
        factor = - factor
    angle = factor * (np.pi / 180)
    new_strokes = []
    for stroke in strokes:
        new_stroke = []
        for point in stroke:
            new_stroke.append(
                [cos(angle) * point[0] - sin(angle) * point[1], point[1]])
        new_strokes.append(new_stroke)
    return new_strokes

# 轨迹轻微扰动
def disturbanceStroke(strokes):
    mean = 0
    std = 0.1
    new_strokes = []
    for stroke in strokes:
        size = np.array(stroke).shape
        g_noise = np.random.normal(loc=mean, scale=std, size=size)
        p_noise = np.random.poisson(lam=0.1, size=size)
        new_strokes.append(stroke + p_noise + g_noise)
    return new_strokes

def strokes_to_stroke_4(strokes):
    """所有序列点按照[x1, y1, p1, p2]的方式存于一个列表"""
    sample = []
    for stroke in strokes:
        for point_index in range(0, len(stroke)):
            if point_index == len(stroke) - 1:
                sample.append([stroke[point_index][0], stroke[point_index][1], 1, 0])  # 最后一个表示笔划结束
            else:
                sample.append([stroke[point_index][0], stroke[point_index][1], 0, 1])  # 笔划继续

    target_sample = []
    target_sample.append([sample[0][0], sample[0][1], 0, 1])  # 不能误失第一个点
    for point_index in range(0, len(sample)):
        if point_index == len(sample) - 1:
            break
        else:
            delta_x = sample[point_index + 1][0] - sample[point_index][0]
            delta_y = sample[point_index + 1][1] - sample[point_index][1]
            p1 = sample[point_index + 1][2]
            p2 = sample[point_index + 1][3]
            target_sample.append([delta_x, delta_y, p1, p2])
    return target_sample

def padding_sequence(sequence: list, padding_len=512):
    if len(sequence) <= padding_len:
        padding_strokes = sequence
        for i in range(len(sequence), padding_len):
            padding_strokes.append([0, 0, 1, 1])
    else:
        padding_strokes = sequence[0:padding_len]
    assert len(padding_strokes) == padding_len
    return np.array(padding_strokes)



def corrds2dict(strokes):
    new_strokes = []
    for stroke in strokes:
        new_strokes.append({'x': [point[0] for point in stroke], 'y': [point[1] for point in stroke]})
    return new_strokes

def dict2corrds(strokes):
    new_strokes = []
    for stroke in strokes:
        new_stroke = []
        y = stroke['y']
        for index, x in enumerate(stroke['x']):
            new_stroke.append([x, y[index]])
        new_strokes.append(new_stroke)
    return new_strokes

def normalize_lines(line_strokes):
    new_strokes = corrds2xys(line_strokes)
    normalized_strokes = normalize_xys(new_strokes)
    new_strokes = xys2corrds(normalized_strokes)
    return new_strokes

'''
description: 
    [x, y] --> [x, y, p1, p2, p3]
    see 'A NEURAL REPRESENTATION OF SKETCH DRAWINGS' for more details
'''
def corrds2xys(coordinates):
    new_strokes = []
    for stroke in coordinates:
        for (x, y) in np.array(stroke).reshape((-1, 2)):
            p = np.array([x, y, 1, 0], np.float32)
            new_strokes.append(p.tolist())
        try:
            new_strokes[-1][2:] = [0, 1]  # set the end of a stroke
        except IndexError:
            print(stroke)
            return None
    new_strokes = np.stack(new_strokes, axis=0)
    return new_strokes


'''
description: Normalize the xy-coordinates into a standard interval.
Refer to "Drawing and Recognizing Chinese Characters with Recurrent Neural Network".
'''


def normalize_xys(xys):
    stroken_state = np.cumsum(np.concatenate((np.array([0]), xys[:, -2]))[:-1])
    px_sum = py_sum = len_sum = 0
    for ptr_idx in range(0, xys.shape[0] - 2):
        if stroken_state[ptr_idx] == stroken_state[ptr_idx + 1]:
            xy_1, xy = xys[ptr_idx][:2], xys[ptr_idx + 1][:2]
            temp_len = np.sqrt(np.sum(np.power(xy - xy_1, 2)))
            temp_px, temp_py = temp_len * (xy_1 + xy) / 2
            px_sum += temp_px
            py_sum += temp_py
            len_sum += temp_len
    if len_sum == 0:
        print(xys)
        return None
        # raise Exception("Broken online characters")
    else:
        pass

    mux, muy = px_sum / len_sum, py_sum / len_sum
    dx_sum, dy_sum = 0, 0
    for ptr_idx in range(0, xys.shape[0] - 2):
        if stroken_state[ptr_idx] == stroken_state[ptr_idx + 1]:
            xy_1, xy = xys[ptr_idx][:2], xys[ptr_idx + 1][:2]
            temp_len = np.sqrt(np.sum(np.power(xy - xy_1, 2)))
            temp_dx = temp_len * (
                    np.power(xy_1[0] - mux, 2) + np.power(xy[0] - mux, 2) + (xy_1[0] - mux) * (xy[0] - mux)) / 3
            temp_dy = temp_len * (
                    np.power(xy_1[1] - muy, 2) + np.power(xy[1] - muy, 2) + (xy_1[1] - muy) * (xy[1] - muy)) / 3
            dx_sum += temp_dx
            dy_sum += temp_dy
    sigma = np.sqrt(dx_sum / len_sum)
    if sigma == 0:
        sigma = np.sqrt(dy_sum / len_sum)
    xys[:, 0], xys[:, 1] = (xys[:, 0] - mux) / sigma, (xys[:, 1] - muy) / sigma
    return xys

def xys2corrds(xys):
    strokes = np.split(xys, np.where(xys[:, 3] > 0)[0] + 1)
    new_strokes = []
    for stroke in strokes:
        new_stroke = []
        if len(stroke) == 0:
            continue
        else:
            for point in stroke:
                new_stroke.append([point[0], point[1]])
        new_strokes.append(new_stroke)
    return new_strokes

def equidistance_sample(strokes, interval):
    new_strokes = []
    for stroke in strokes:
        stroke = np.array(stroke)
        # 计算轨迹点之间的距离
        distances = np.sqrt(np.sum(np.diff(stroke, axis=0) ** 2, axis=1))
        total_distances = np.sum(distances)

        # 计算等距采样点数
        num_samples = int(total_distances / interval)

        sample_points = [stroke[0]]  # 加入起始点

        # 从起始点开始，按照等距采样间隔选取新的样本点
        current_distance = 0
        for i in range(1, len(stroke)):
            distance = np.linalg.norm(stroke[i] - stroke[i - 1])
            current_distance += distance
            while current_distance >= interval:
                overshoot = current_distance - interval
                ratio = overshoot / distance
                new_point = stroke[i - 1] + ratio * (stroke[i] - stroke[i - 1])
                sample_points.append(new_point)
                current_distance -= interval
        sample_points.append(stroke[-1])
        new_strokes.append(sample_points)
    return new_strokes


# 计算两点间距离的辅助函数
def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# 插值函数，用于在两点之间产生一个等速点
def interpolate(p1, p2, t):
    return [p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t]


# 笔速归一化的函数
def normalize_pen_speed(strokes, target_speed):
    normalized_trajectory = []
    for stroke in strokes:
        normalized_stroke = [stroke[0]]  # 从每个笔画的第一个点开始
        accumulated_distance = 0  # 累计距离
        for i in range(1, len(stroke)):
            p1 = stroke[i - 1]
            p2 = stroke[i]
            segment_distance = distance(p1, p2)
            accumulated_distance += segment_distance

            # 当累计距离超过目标速度时，插入新的点并重置累计距离
            while accumulated_distance >= target_speed:
                accumulated_distance -= target_speed
                t = (segment_distance - accumulated_distance) / segment_distance
                new_point = interpolate(p1, p2, t)
                normalized_stroke.append(new_point)
                p1 = new_point  # 更新下个插值的起始点

        normalized_stroke.append(stroke[-1])  # 添加笔画的最后一个点
        normalized_trajectory.append(normalized_stroke)
    return normalized_trajectory

def coordToOriginPoint(strokes):
    min_x = min_y = 1000000
    max_x = max_y = 0
    for stroke in strokes:
        x = [point[0] for point in stroke]
        y = [point[1] for point in stroke]
        x1 = min(x)
        y1 = min(y)
        x2 = max(x)
        y2 = max(y)
        if x1 < min_x:
            min_x = x1
        if y1 < min_y:
            min_y = y1
        if x2 > max_x:
            max_x = x2
        if y2 > max_y:
            max_y = y2
    new_strokes = []
    for stroke in strokes:
        new_stroke = []
        for point in stroke:
            new_stroke.append([point[0] - min_x, point[1] - min_y])
        new_strokes.append(new_stroke)
    return new_strokes, max_x, max_y, min_x, min_y


def scale_points(strokes, radio):
    new_strokes = []
    for stroke in strokes:
        new_stroke = []
        for point in stroke:
            new_stroke.append([point[0] / radio, point[1] / radio])
        new_strokes.append(new_stroke)
    return new_strokes





