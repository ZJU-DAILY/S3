import pickle
import time
import numpy as np

from generate.utils import getPED4GPS, getMaxError
from preprocess.SpatialRegionTools import cell2gps
import os
from preprocess.SpatialRegionTools import SpacialRegion
from sys_config import DATA_DIR

# 0:priority, 1:pi, 2:point



def adjust_priority(Q, pre_index, Q_index, succ_index):
    if Q_index == len(Q) - 1 or Q_index == 0:
        return
    p = Q[Q_index][1] + getPED4GPS(Q[Q_index][2], Q[pre_index][2], Q[succ_index][2])
    Q[Q_index][0] = p


def reduce(Q, min_index, min_p):
    Q[min_index - 1][1] = max(min_p, Q[min_index - 1][1])
    Q[min_index + 1][1] = max(min_p, Q[min_index + 1][1])
    adjust_priority(Q,min_index - 2, min_index - 1, min_index + 1)
    adjust_priority(Q,min_index - 1, min_index + 1, min_index + 2)
    q = Q[min_index]
    Q.remove(q)


def find_mini_priority(Q):
    st = Q[0]
    to_remove = Q[len(Q) - 1]
    min_idx = 1
    k = 1
    for i in range(1, len(Q) - 1):
        p = Q[i]
        if to_remove == Q[len(Q) - 1] or p[0] < to_remove[0]:
            to_remove = p
            min_idx = k
        k += 1
    return min_idx


def squish_e(points, buffer_size, mode, cmp_ratio=1, capacity=4):
    i = 0
    Q = []
    while i < len(points):
        if (i // cmp_ratio) >= capacity:
            capacity += 1
        Q.append([float('inf'), 0, points[i]])
        if i > 0:
            adjust_priority(Q, len(Q) - 3, len(Q) - 2, len(Q) - 1)
        if len(Q) == capacity:
            min_index = find_mini_priority(Q)
            min_p = Q[min_index][0]
            reduce(Q, min_index, min_p)
        i += 1
    min_index = find_mini_priority(Q)
    min_p = Q[min_index][0]
    while len(Q) > buffer_size:
        reduce(Q,min_index, min_p)
        min_index = find_mini_priority(Q)
        min_p = Q[min_index][0]
    idx = []
    for i in range(len(Q)):
        idx.append(points.index(Q[i][2]))
    idx.sort()
    max_err = 0
    for i in range(len(idx) - 1):
        st = idx[i]
        en = idx[i + 1]
        _, err = getMaxError(st, en, points, mode)
        max_err = max(max_err, err)
    return None, idx, max_err
