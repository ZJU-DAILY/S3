from SpatialRegionTools import cell2gps
from SpatialRegionTools import str2trip
# from SpatialRegionTools import getRegionInstance
import numpy as np
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt
import math
import pickle


def plotGPS(region, trj):
    trg = trj.split(" ")
    x = []
    y = []
    for p in trg:
        x_, y_ = cell2gps(region, int(p))
        x.append(x_)
        y.append(y_)
    plt.plot(x, y)
    plt.scatter(x, y)
    plt.show()


def getSED(region, p, start, end):
    x, y = cell2gps(region, p)
    st_x, st_y = cell2gps(region, start)
    en_x, en_y = cell2gps(region, end)
    #     Ax + By + C = 0
    A = (en_y - st_y) / (en_x - st_x)
    B = -1
    C = st_y - st_x * A
    return abs(A * x + B * y + C) / math.sqrt(A * A + B * B)


def SEDsimilarity(region, src, trg):
    # p为慢指针（指向trg），f为快指针（指向src）。src的长度应该大于等于trg
    p = 0
    f = 0
    maxSED = -1
    while p < len(trg) and f < len(src):
        if trg[p] == src[f]:
            p += 1
            f += 1
        else:
            st = trg[p - 1]
            en = trg[p]
            while trg[p] != src[f]:
                in_ = src[f]
                dis = getSED(region, in_, st, en)
                maxSED = max(maxSED, dis)
                f += 1


def getCompress(region, filepath):
    with open(filepath, "r") as f:
        ss = f.readlines()

    src = ss[0].split(" ")
    trg = ss[1].split(" ")
    data = np.zeros([len(src), 2])
    # 给id一个时间顺序
    mp = {}
    i = 0
    for p in src:
        x, y = cell2gps(region, int(p))
        data[i, :] = [x, y]
        mp[int(p)] = i
        i += 1
    tree = KDTree(data)
    # 存放压缩后的轨迹的id
    resTrj = []
    for p in trg:
        x, y = cell2gps(region, int(p))
        dists, idxs = tree.query(np.array([[x, y]]), 1)
        id = idxs[0].tolist()[0]
        resTrj.append(id)
    resTrj.sort(key=lambda j: mp[j])
    return resTrj


def plotCompress(region, filepath):
    # 为画图直观，将原始轨迹x轴向下平移offset
    offset = 0.001
    with open(filepath, "r") as f:
        ss = f.readlines()

    src = ss[0].split(" ")
    trg = ss[1].split(" ")
    data = np.zeros([len(src), 2])
    data_offset = np.zeros([len(src), 2])
    print("原始轨迹的长度： ", len(src))
    i = 0
    for p in src:
        x, y = cell2gps(region, int(p))
        data[i, :] = [x, y]
        data_offset[i, :] = [x + offset, y]
        i += 1
    # plt.plot(data, color='r')
    tree = KDTree(data)
    trg_x = []
    trg_x_ori = []
    trg_y = []
    trg_y_ori = []
    i = 0
    st = set()
    for p in trg:
        x, y = cell2gps(region, int(p))
        trg_x_ori.append(x), trg_y_ori.append(y)
        dists, idxs = tree.query(np.array([[x, y]]), 1)
        id = idxs[0].tolist()[0]
        if id not in st:
            st.add(id)
            trg_x.append(data[id, 0]), trg_y.append(data[id, 1])
            plt.text(data[id, 0], data[id, 1], str(i + 1), fontdict={'size': '8', 'color': 'b'})
            i += 1
    print("压缩后轨迹的长度： ", i)
    # data_offset = data
    # data_offset[:, 0] += offset

    # plt.scatter(data_offset[:, 0].tolist(), data_offset[:, 1].tolist(), color='r')
    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(), color='r')
    plt.plot(data[:, 0].tolist(), data[:, 1].tolist(), color='r')
    plt.scatter(trg_x, trg_y, color='b')

    # plt.scatter(trg_x_ori, trg_y_ori, color='g')
    plt.show()


def plotHotcellTrj(region, filepath):
    with open(filepath, "r") as f:
        ss = f.readlines()

    trg = ss[0].split(" ")
    x = []
    y = []
    for p in trg:
        x_, y_ = cell2gps(region, int(p))
        x.append(x_)
        y.append(y_)
    xx = []
    yy = []

    src = str2trip(ss[1])
    for p in src:
        xx.append(p[0])
        yy.append(p[1])
    plt.plot(x, y, color='r')
    plt.plot(xx, yy)
    plt.show()


# with open('pickle.txt', 'rb') as f:
#     reg = pickle.load(f)
# region = getRegionInstance()
#
# plotCompress(region, "../datasets/compress")

getCompress("../datasets/compress")
# plotHotcellTrj(region,"dataset/test")
# plotGPS(region,
#         "182028 183318 183965 185256 186547 187838 189774 190420 191065 193000 194935 196870 197516")
