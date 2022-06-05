# -*- coding: utf-8 -*-
# @Time    : 2022/5/15 9:49
# @Author  : HeAlec
# @FileName: exp.py
# @Desc: description
# @blogs ：https://segmentfault.com/u/alec_5e8962e4635ca
import pickle

from modules.helpers import getSED4GPS
from matplotlib import pyplot as plt
import numpy as np
from preprocess.SpatialRegionTools import cell2gps


def energy_(trj, size_1, window_size):
    ener = [0] * size_1
    for i in range(1, len(trj) - 1):
        # window取到1
        p = trj[i]
        priority = []
        lambda_ = 1
        for window in range(1, 4):
            # window = window_size
            # st = trj[i - window if i - window > 0 else 0]
            st = trj[i - 1 if i - 1 > 0 else 0]
            en = trj[i + window if i + window < len(trj) - 1 else len(trj) - 1]
            pri = getSED4GPS(p, st, en)
            priority.append(pri * lambda_)
            lambda_ *= 0.1
        # window = 1
        # st = trj[i - window if i - window > 0 else 0]
        # en = trj[i + window if i + window < len(trj) - 1 else len(trj) - 1]
        # pri = getSED4GPS(p, st, en)
        # priority.append(pri*2)
        #
        # window = 2
        # st = trj[i - window if i - window > 0 else 0]
        # en = trj[i + window if i + window < len(trj) - 1 else len(trj) - 1]
        # pri = getSED4GPS(p, st, en)
        # priority.append(pri)
        #
        # # window取到 len(trj) / 2
        # window = window_size
        # st = trj[i - window if i - window > 0 else 0]
        # en = trj[i + window if i + window < len(trj) - 1 else len(trj) - 1]
        # pri = getSED4GPS(p, st, en)
        # priority.append(pri)

        ener[i] = np.mean(priority)
    sum = 0
    for r in ener:
        sum += r
    res = []
    for r in ener:
        res.append(round(r / sum, 2))
    return res


def compression_hch(order, points, maxlen):
    pp = list(range(0, len(points)))
    pp.sort(key=lambda j: -order[j])
    res = pp[0:maxlen - 2]
    res.append(0)
    res.append(len(points) - 1)
    res.sort()
    result = [points[p] for p in res]
    return result


def SEDsimilarity(src, trg):
    # p为慢指针（指向trg），f为快指针（指向src）。src的长度应该大于等于trg
    p = 0
    f = 0
    idx = -1
    maxSED = -1
    while p < len(trg) and f < len(src):
        if trg[p][0] == src[f][0] and trg[p][1] == src[f][1]:
            p += 1
            f += 1
        else:
            st = trg[p - 1]
            en = trg[p]
            while p < len(trg) and f < len(src) and (trg[p][0] != src[f][0] or trg[p][1] != src[f][1]):
                in_ = src[f]
                dis = getSED4GPS(in_, st, en)
                if dis > maxSED:
                    maxSED = dis
                    idx = f
                f += 1
    return maxSED, idx


with open('../preprocess/pickle.txt', 'rb') as f:
    var_a = pickle.load(f)
region = pickle.loads(var_a)

with open("../datasets/infer.src", "r") as f:
    ss = f.readlines()
seqs = []
for s in ss:
    s = s.strip("\n")
    s = s.split(" ")
    seq = []
    for p in s:
        x, y = cell2gps(region, int(p))
        seq.append([x, y])
    seqs.append(seq)

for i in range(1, 2):
    window_size = i
    errs = []
    for seq in seqs:
        res_ = energy_(seq, len(seq), window_size)
        res = compression_hch(res_, seq, int(len(seq) * 0.5))

        # print(len(res))
        maxErr, idx = SEDsimilarity(seq, res)
        errs.append(maxErr)
    print(f"window_size:{window_size} maErr:{np.mean(errs)}")
