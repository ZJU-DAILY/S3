import datetime

from models.seq3_losses import sed_loss
from preprocess.SpatialRegionTools import cell2gps, cell2coord, coord2cell
from preprocess.SpatialRegionTools import str2trip
# from SpatialRegionTools import getRegionInstance
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib as mpl
from modules.helpers import getSED, getCompress

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


def SEDsimilarity(region, src, trg):
    # p为慢指针（指向trg），f为快指针（指向src）。src的长度应该大于等于trg
    p = 0
    f = 0
    idx = -1
    maxSED = -1
    while p < len(trg) and f < len(src):
        if src[f] == '' or src[f] == 'UNK':
            f += 1
            continue
        if trg[p] == src[f]:
            p += 1
            f += 1
        else:
            st = trg[p - 1]
            en = trg[p]
            while trg[p] != src[f]:
                if src[f] == '' or src[f] == 'UNK':
                    f += 1
                    continue
                in_ = src[f]
                dis = getSED(region, int(in_), int(st), int(en))
                if dis > maxSED:
                    maxSED = dis
                    idx = f
                f += 1
    return round(maxSED * 1000, 5), idx


# def getCompress(region, src, trg):
#     # epi = 0.1
#     epi = 0
#     data = np.zeros([len(src), 2])
#     # 给id一个时间顺序
#     mp = {}
#     i = 0
#     for p in src:
#         x, y = cell2gps(region, int(p))
#         data[i, :] = [x, y]
#         mp[int(p)] = i
#         i += 1
#     tree = KDTree(data)
#     # 存放压缩后的轨迹的id
#     resTrj = []
#     trg_x_ori = []
#     trg_y_ori = []
#     for p in trg:
#         x, y = cell2gps(region, int(p))
#         trg_x_ori.append(x)
#         trg_y_ori.append(y)
#         dists, idxs = tree.query(np.array([[x, y]]), 1)
#         if dists[0] > epi:
#             continue
#         id = idxs[0].tolist()[0]
#         if src[id] not in resTrj:
#             resTrj.append(src[id])
#     if src[-1] not in resTrj:
#         resTrj.append(src[-1])
#     resTrj.sort(key=lambda j: mp[int(j)])
#     return resTrj, trg_x_ori, trg_y_ori


def plotCompress(region, filepath):
    # 为画图直观，将原始轨迹x轴向下平移offset
    offset = 0.001
    with open(filepath, "r") as f:
        ss = f.readlines()

    src = ss[0].strip("\n").split(" ")
    trg = ss[1].strip("\n").split(" ")
    ratio = ss[3].strip("\n").split(" ")
    data = np.zeros([len(src), 2])
    data_offset = np.zeros([len(src), 2])
    print("原始轨迹的长度： ", len(src))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 2)
    i = 0
    for p in src:
        x, y = cell2gps(region, int(p))
        plt.text(x, y, str(ratio[i]), fontdict={'size': '15', 'color': 'b'})
        data[i, :] = [x, y]
        data_offset[i, :] = [x + offset, y]
        i += 1
    # plt.plot(data, color='r')
    trg_x = []
    trg_y = []

    i = 0

    plt.subplot(1, 3, 3)
    trg, trg_x_ori, trg_y_ori = getCompress(region, src, trg)
    for p in trg:
        x, y = cell2gps(region, int(p))
        trg_x.append(x), trg_y.append(y)
        plt.text(x, y, str(i + 1), fontdict={'size': '15', 'color': 'b'})
        i += 1
    print("压缩后轨迹的长度： ", i)
    loss, max_loss_point = SEDsimilarity(region, src, trg)
    print("轨迹SED相似度误差", loss)
    # data_offset = data
    # data_offset[:, 0] += offset

    # plt.scatter(data_offset[:, 0].tolist(), data_offset[:, 1].tolist(), color='r')

    # 原始轨迹 + 压缩轨迹 + 噪声点
    plt.subplot(1, 3, 1)

    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(), color='r')
    plt.plot(data[:, 0].tolist(), data[:, 1].tolist(), color='r')
    plt.plot(trg_x, trg_y, color='k', ls='dotted')
    plt.scatter(trg_x, trg_y, color='b')

    plt.scatter(trg_x_ori, trg_y_ori, color='g')
    # 原轨迹 + 压缩轨迹
    plt.subplot(1, 3, 2)
    plt.scatter(data[:, 0].tolist(), data[:, 1].tolist(), color='r')
    plt.plot(data[:, 0].tolist(), data[:, 1].tolist(), color='r')
    plt.plot(trg_x, trg_y, color='k', ls='dotted')
    plt.scatter(trg_x, trg_y, color='b')
    plt.scatter(data[max_loss_point, 0], data[max_loss_point, 1], color='y')
    # 压缩轨迹，最终效果图
    plt.subplot(1, 3, 3)
    plt.plot(trg_x, trg_y, color='k', ls='dotted')
    plt.scatter(trg_x, trg_y, color='b')
    # plt.show()
    date = datetime.datetime.now()
    plt.suptitle(f"Distance Error: {loss},  Point Number: {len(trg)}", fontsize=25)
    plt.savefig(f"{date.month}_{date.day}_{date.hour}_{date.minute}_{date.second}_image.png")

    plt.tight_layout()
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


def plotTest(region):
    with open("../datasets/train.src", "r") as f:
        ss = f.readlines()
    x = []
    y = []
    for s in ss:
        trj = s.split(" ")
        for p in trj:
            if p != '':
                x_, y_ = cell2gps(region, int(p))
                x.append(x_), y.append(y_)
    plt.scatter(x, y, color='r')

    # with open("../datasets/val.src", "r") as f:
    #     ss = f.readlines()
    # x = []
    # y = []
    # for s in ss:
    #     trj = s.split(" ")
    #     for p in trj:
    #         x_, y_ = cell2gps(region, int(p))
    #         x.append(x_), y.append(y_)
    # plt.scatter(x, y, color='b')

    plt.savefig("image_test.png")
    plt.close()


def plotArea(region):
    with open("../datasets/compress", "r") as f:
        ss = f.readlines()
    ss[0].strip('\n')
    src = ss[0].split(" ")

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    xx = []
    yy = []
    for p in src:
        if p == " ":
            continue
        cell_x, cell_y = cell2coord(region, int(p))
        xx.append(cell_x)
        yy.append(cell_y)
        min_x = min(min_x, int(cell_x))
        max_x = max(max_x, int(cell_x))
        min_y = min(min_y, int(cell_y))
        max_y = max(max_y, int(cell_y))
    res = []
    plt.plot([min_x, max_x, max_x, min_x, min_x], [min_y, min_y, max_y, max_y, min_y])
    plt.plot(xx, yy)

    # 噪声点观测站
    trg = ss[1].split(" ")
    trg_x_ori = []
    trg_y_ori = []

    for p in trg:
        x, y = cell2coord(region, int(p))
        trg_x_ori.append(x)
        trg_y_ori.append(y)
    plt.scatter(trg_x_ori, trg_y_ori, color='g')

    min_x = int(min_x)
    min_y = int(min_y)
    offset = 3
    for y in range(min_y - offset * int(region.ystep), max_y + (offset + 1) * int(region.ystep), int(region.ystep)):
        for x in range(min_x - offset * int(region.ystep), max_x + (offset + 1) * int(region.ystep), int(region.xstep)):
            cell = coord2cell(region, x, y)
            # hotcell = cell2vocab(region, cell)
            # x_,y_ = cell2coord(region,hotcell)
            plt.scatter(x, y, color='k')
            res.append(cell)
    plt.savefig("image_grid.png")
    return res


def genGPS(region, file, outfile):
    with open(file, "r") as f:
        ss = f.readlines()
    src = ss[0].split(" ")
    with open(outfile, "w") as f:
        for p in src:
            x, y = cell2gps(region, int(p))
            strr = str(x) + " " + str(y) + "\n"
            f.write(strr)


def plotOri_seq3_adp(region, no, src, seq3, adp, squish):
    plt.figure(figsize=(17, 5))
    data_src = np.zeros([len(src), 2])
    data_seq3 = np.zeros([len(seq3), 2])
    data_adp = np.zeros([len(adp), 2])
    data_squish = np.zeros([len(squish), 2])

    for i, p in enumerate(src):
        x, y = cell2gps(region, int(p))
        data_src[i, :] = [x, y]

    for i, p in enumerate(seq3):
        x, y = cell2gps(region, int(p))
        data_seq3[i, :] = [x, y]

    for i, p in enumerate(adp):
        x, y = cell2gps(region, int(p))
        data_adp[i, :] = [x, y]

    for i, p in enumerate(squish):
        x, y = cell2gps(region, int(p))
        data_squish[i, :] = [x, y]


    # 原始轨迹 + seq3压缩后的轨迹
    plt.subplot(1, 3, 1)
    loss1, max_loss_point = SEDsimilarity(region, src, seq3)
    plt.scatter(data_src[:, 0].tolist(), data_src[:, 1].tolist(), color='r')
    plt.plot(data_src[:, 0].tolist(), data_src[:, 1].tolist(), color='r')
    plt.plot(data_seq3[:, 0].tolist(), data_seq3[:, 1].tolist(), color='k', ls='dotted')
    plt.scatter(data_seq3[:, 0].tolist(), data_seq3[:, 1].tolist(), color='b')
    plt.scatter(data_src[max_loss_point, 0], data_src[max_loss_point, 1], color='y')
    plt.xlabel(f"Ours (err={loss1})")

    # 原始轨迹 + adp压缩后的轨迹
    plt.subplot(1, 3, 2)
    loss2, max_loss_point = SEDsimilarity(region, src, adp)
    plt.scatter(data_src[:, 0].tolist(), data_src[:, 1].tolist(), color='r')
    plt.plot(data_src[:, 0].tolist(), data_src[:, 1].tolist(), color='r')
    plt.plot(data_adp[:, 0].tolist(), data_adp[:, 1].tolist(), color='k', ls='dotted')
    plt.scatter(data_adp[:, 0].tolist(), data_adp[:, 1].tolist(), color='b')
    plt.scatter(data_src[max_loss_point, 0], data_src[max_loss_point, 1], color='y')
    plt.xlabel(f"TDTR (err={loss2})")

    # 原始轨迹 + squish压缩后的轨迹
    plt.subplot(1, 3, 3)
    loss3, max_loss_point = SEDsimilarity(region, src, squish)
    plt.scatter(data_src[:, 0].tolist(), data_src[:, 1].tolist(), color='r')
    plt.plot(data_src[:, 0].tolist(), data_src[:, 1].tolist(), color='r')
    plt.plot(data_squish[:, 0].tolist(), data_squish[:, 1].tolist(), color='k', ls='dotted')
    plt.scatter(data_squish[:, 0].tolist(), data_squish[:, 1].tolist(), color='b')
    plt.scatter(data_src[max_loss_point, 0], data_src[max_loss_point, 1], color='y')
    plt.xlabel(f"SQUISH (err={loss3})")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.suptitle(f"预期轨迹长度{round(len(data_seq3) / len(data_src),1)}(%|T|)", fontsize=15)
    path = "../evaluation/exp_image/" + str(no) + "_image.png"
    plt.savefig(path)

    plt.tight_layout()
    plt.close()


def plotCompare(region):
    file = "../datasets/exp"
    with open(file, "r") as f:
        ss = f.readlines()
    N = len(ss) // 4
    for i in range(N):
        print(i)
        src = ss[i * 4 + 0].strip("\n")
        src = src.split(" ")
        seq = ss[i * 4 + 1].strip("\n")
        seq = seq.split(" ")
        adp = ss[i * 4 + 2].strip("\n")
        adp = adp.split(" ")
        squish = ss[i * 4 + 3].strip("\n")
        squish = squish.split(" ")
        plotOri_seq3_adp(region, i, src, seq, adp, squish)


with open('pickle.txt', 'rb') as f:
    var_a = pickle.load(f)
region = pickle.loads(var_a)

plotCompare(region)
# genGPS(region, "../datasets/compress", "./gps.txt")
# plotCompress(region, "../datasets/compress")
# plotArea(region)
# plotTest(region)
# getCompress("../datasets/compress")
# plotHotcellTrj(region,"dataset/test")
# plotGPS(region,
#         "182028 183318 183965 185256 186547 187838 189774 190420 191065 193000 194935 196870 197516")
