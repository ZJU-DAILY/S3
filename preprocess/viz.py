from SpatialRegionTools import cell2gps
from SpatialRegionTools import str2trip
# from SpatialRegionTools import getRegionInstance
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib as mpl

mpl.use('Agg')
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
    x, y = cell2gps(region, int(p))
    st_x, st_y = cell2gps(region, start)
    en_x, en_y = cell2gps(region, end)
    #     Ax + By + C = 0
    if en_x == st_x:
        return abs(x - en_x)
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
                dis = getSED(region, int(in_), int(st), int(en))
                maxSED = max(maxSED, dis)
                f += 1
    return maxSED


def getCompress(region, src, trg):
    epi = 0.005
    # epi = 0
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
    trg_x_ori = []
    trg_y_ori = []
    for p in trg:
        x, y = cell2gps(region, int(p))
        trg_x_ori.append(x)
        trg_y_ori.append(y)
        dists, idxs = tree.query(np.array([[x, y]]), 1)
        if dists[0] > epi:
            continue
        id = idxs[0].tolist()[0]
        if src[id] not in resTrj:
            resTrj.append(src[id])
    resTrj.sort(key=lambda j: mp[int(j)])
    return resTrj, trg_x_ori, trg_y_ori


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
    trg_x = []
    trg_y = []

    i = 0

    trg, trg_x_ori, trg_y_ori = getCompress(region, src, trg)
    for p in trg:
        x, y = cell2gps(region, int(p))
        trg_x.append(x), trg_y.append(y)
        plt.text(x, y, str(i + 1), fontdict={'size': '8', 'color': 'b'})
        i += 1
    print("压缩后轨迹的长度： ", i)
    loss = SEDsimilarity(region, src, trg)
    print("轨迹SED相似度误差", loss)
    # data_offset = data
    # data_offset[:, 0] += offset

    # plt.scatter(data_offset[:, 0].tolist(), data_offset[:, 1].tolist(), color='r')
    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(), color='r')
    plt.plot(data[:, 0].tolist(), data[:, 1].tolist(), color='r')
    plt.plot(trg_x, trg_y, color='y', ls='dotted')
    plt.scatter(trg_x, trg_y, color='b')

    # plt.scatter(trg_x_ori, trg_y_ori, color='g')
    # plt.show()
    plt.savefig("image.png")
    plt.close()


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


with open('pickle.txt', 'rb') as f:
    var_a = pickle.load(f)
region = pickle.loads(var_a)

plotCompress(region, "../datasets/compress")

# getCompress("../datasets/compress")
# plotHotcellTrj(region,"dataset/test")
# plotGPS(region,
#         "182028 183318 183965 185256 186547 187838 189774 190420 191065 193000 194935 196870 197516")
