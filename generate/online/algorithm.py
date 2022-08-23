import pickle
import time
import numpy as np

from generate.utils import getPED4GPS, getMaxError
from preprocess.SpatialRegionTools import cell2gps
import os
from preprocess.SpatialRegionTools import SpacialRegion
from sys_config import DATA_DIR
from generate.online.squish_e import squish_e


# squish压缩算法
def squish(points, max_buffer_size):
    buffer = []
    # 坐标、err、下标
    buffer.append([points[0], 0, 0])
    max_err = 0
    if max_buffer_size > 2:
        for i in range(1, len(points)):
            buffer.append([points[i], 0, i])
            if len(buffer) <= 2:
                continue
            segment_start = buffer[-3][0]
            segment_end = buffer[-1][0]
            buffer[-2][1] += getPED4GPS(buffer[-2][0], segment_start, segment_end)
            if len(buffer) > max_buffer_size:
                to_remove = len(buffer) - 2
                for j in range(1, len(buffer) - 1):
                    if buffer[j][1] < buffer[to_remove][1]:
                        to_remove = j
                buffer[to_remove - 1][1] += buffer[to_remove][1]
                buffer[to_remove + 1][1] += buffer[to_remove][1]
                err = getPED4GPS(buffer[to_remove][0], buffer[to_remove - 1][0], buffer[to_remove + 1][0])
                max_err = max(max_err, err)
                buffer.pop(to_remove)
    else:
        buffer.append([points[-1], 0, len(points) - 1])
        segment_start = buffer[0][0]
        segment_end = buffer[-1][0]
        for i in range(1, len(points) - 1):
            max_err = max(max_err, getPED4GPS(points[i], segment_start, segment_end))
    idx = [p[2] for p in buffer]
    pp = [p[2] for p in buffer]
    return pp, idx, max_err


def STTrace(points, max_buffer_size, mode):
    buffer = []
    buffer.append([points[0], 0])
    max_err = 0
    if max_buffer_size > 2:
        buffer.append([points[1], 0])
        for i in range(2, len(points)):
            buffer.append([points[i], 0])
            st = buffer[len(buffer) - 3]
            en = buffer[len(buffer) - 1]
            buffer[len(buffer) - 2][1] = getPED4GPS(buffer[len([buffer])][0], st[0], en[0])
            if len(buffer) > max_buffer_size:
                start = buffer[0]
                to_remove = buffer[len(buffer) - 1]
                min_idx = 1
                k = 1
                for i in range(1, len(buffer) - 1):
                    p = buffer[i]
                    if to_remove == buffer[len(buffer) - 1] or p[1] < to_remove[1]:
                        to_remove = p
                        k += 1
                if min_idx - 1 > 0:
                    buffer[min_idx - 1][1] = getPED4GPS(buffer[min_idx - 1][0], buffer[min_idx - 2][0],
                                                        buffer[min_idx + 1][0])
                if min_idx + 1 < len(buffer) - 1:
                    buffer[min_idx + 1][1] = getPED4GPS(buffer[min_idx + 1][0], buffer[min_idx - 1][0],
                                                        buffer[min_idx + 2][0])
                buffer.remove(to_remove)

    else:
        buffer.append([points[len(points) - 1], 0])
    idx = []
    for i in range(len(buffer)):
        idx.append(points.index(buffer[i][0]))
    idx.sort()

    for i in range(len(idx) - 1):
        st = idx[i]
        en = idx[i + 1]
        _, err = getMaxError(st, en, points, mode)
        max_err = max(max_err, err)
    return None, idx, max_err


def readData(src_file, region):
    with open(src_file, "r") as f:
        ss = f.readlines()
    points = []
    src = []
    for s in ss:
        s = s.strip("\n")
        s = s.split(" ")
        point = []
        src_ = []
        for p in s:
            if p == " " or p == "UNK":
                continue
            point.append(cell2gps(region, int(p)))
            src_.append(p)
        points.append(point)
        src.append(src_)
    return points, src


def compress_squish(src, points, max_ratio, metric):
    # 计时开始

    err = []
    compRes = []
    timelist = []
    timelist2 = []
    for idseq, seq in zip(src, points):
        tic1 = time.time()
        _, idx, maxErr = squish(seq, int(max_ratio * len(seq)))
        tic2 = time.time()
        timelist.append((tic2 - tic1) / len(seq))
        maxErr_ = 0
        for i in range(len(idx) - 1):
            _, e = getMaxError(idx[i], idx[i + 1], seq, metric)
            maxErr_ = max(maxErr_, e)
        err.append(maxErr_)

    print(f"squish压缩率 {max_ratio},耗时 {np.min(timelist)}, error {np.mean(err)}")
    # print(f"squish-e压缩率 {max_ratio},耗时 {np.min(timelist2)}")
    return compRes


def compress_squish_e(src, points, max_ratio, metric):
    # 计时开始

    err = []
    compRes = []
    timelist = []
    for idseq, seq in zip(src, points):
        tic1 = time.time()
        try:
            _, idx, maxErr = squish_e(seq, int(max_ratio * len(seq)), mode='ped')
        except Exception as e:
            continue
        tic2 = time.time()
        timelist.append((tic2 - tic1) / len(seq))
        maxErr_ = 0
        for i in range(len(idx) - 1):
            _, e = getMaxError(idx[i], idx[i + 1], seq, metric)
            maxErr_ = max(maxErr_, e)
        err.append(maxErr_)

    print(f"squish_e压缩率 {max_ratio},耗时 {0}, error {np.mean(err)}")
    return compRes


def compress_sttrace(src, points, max_ratio, metric):
    # 计时开始

    err = []
    compRes = []
    timelist = []
    for idseq, seq in zip(src, points):
        tic1 = time.time()
        _, idx, maxErr = STTrace(seq, int(max_ratio * len(seq)), mode='ped')
        tic2 = time.time()
        timelist.append((tic2 - tic1) / len(seq))
        maxErr_ = 0
        for i in range(len(idx) - 1):
            _, e = getMaxError(idx[i], idx[i + 1], seq, metric)
            maxErr_ = max(maxErr_, e)
        err.append(maxErr_)

    print(f"sttrace压缩率 {max_ratio},耗时 {np.min(timelist)}, error {np.mean(err)}")
    return compRes


if __name__ == '__main__':
    src_file = os.path.join(DATA_DIR, "infer.src")
    with open('../../preprocess/pickle.txt', 'rb') as f:
        var_a = pickle.load(f)
    region = pickle.loads(var_a)
    points, src = readData(src_file, region)
    metric = 'dad'
    for i in range(0, 5):
        ratio = 0.1 * (i + 1)
        compress_sttrace(src, points, ratio, metric)
        compress_squish(src, points, ratio, metric)
        compress_squish_e(src, points, ratio, metric)

