# -*- coding: utf-8 -*-
# @Time    : 2022/7/10 11:21
# @Author  : HeAlec
# @FileName: POI.py
# @Desc: description
# @blogs ：https://segmentfault.com/u/alec_5e8962e4635ca
import csv
import pickle
from sklearn.neighbors import KDTree
import numpy as np
from preprocess.SpatialRegionTools import cell2gps

# 可以在这里预处理每一个GPS点对应的poi点，那么就可以获取一条原始轨迹和一条poi轨迹
# 具体规则如下：设定一个e代表阈值，若GPS点最近的poi点的距离小于e，那么可用当前poi点进行表示。否则，则用unk（id为0）表示。

data = []
# 构建poi KD树
with open("../datasets/node.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)
    for s in reader:
        data.append([s[1], s[2]])
data = np.array(data)

tree = KDTree(data)
# 打开GPS所需的id -> gps映射
with open('pickle.txt', 'rb') as f:
    var_a = pickle.load(f)
region = pickle.loads(var_a)

epi = 150
gid2poi = dict()
dist_all = []

file = ['train.src', 'val.src', 'infer.src']
for _f in file:
    print(_f)
    with open("../datasets/" + _f, 'r') as f:
        for trj in f:
            trj = trj.strip("\n")
            trj = trj.split(" ")
            for p in trj:
                p = int(p)
                x, y = cell2gps(region, p)
                dists, idxs = tree.query(np.array([[x, y]]), 1)
                dist_all.append(dists[0])
                if dists[0] > epi:
                    id = -1
                else:
                    id = idxs[0].tolist()[0]
                gid2poi[str(p)] = id

var_b = pickle.dumps(gid2poi)
with open('../datasets/gid2poi.txt', 'wb') as f:
    pickle.dump(var_b, f)


with open('../datasets/gid2poi.txt', 'rb') as f:
    var_a = pickle.load(f)
gid2poi = pickle.loads(var_a)
print(gid2poi['178794'])

print(np.mean(dist_all))