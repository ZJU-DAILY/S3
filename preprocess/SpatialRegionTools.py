# div 整除 //
# y =sin(cos(abs(-1.0))) 可以写成 y =-1.0 |>abs |>cos |>sin
# 用 collect() 将 tuple 转换为 array
import math
import numpy as np
from sklearn.neighbors import KDTree
import pickle
import sys


sys.setrecursionlimit(10000000)


def lonlat2meters(lon, lat):
    semimajoraxis = 6378137.0
    east = lon * 0.017453292519943295
    north = lat * 0.017453292519943295
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))


def coord2cell(region, x, y):
    xoffset = round(x - region.minx, 6) / region.xstep
    yoffset = round(y - region.miny, 6) / region.ystep
    xoffset = int(math.floor(xoffset))
    yoffset = int(math.floor(yoffset))
    return yoffset * region.numx + xoffset


def cell2coord(region, cell):
    yoffset = cell // region.numx
    xoffset = cell % region.numx
    y = region.miny + (yoffset + 0.5) * region.ystep
    x = region.minx + (xoffset + 0.5) * region.xstep
    return x, y


def gps2cell(region, lon, lat):
    # 经纬度坐标和网格内的坐标之间还有一层对应关系
    x, y = lonlat2meters(lon, lat)
    return coord2cell(region, x, y)


def meters2lonlat(x, y):
    semimajoraxis = 6378137.0
    lon = x / semimajoraxis / 0.017453292519943295
    t = math.exp(y / 3189068.5)
    lat = math.asin((t - 1) / (t + 1)) / 0.017453292519943295
    return lon, lat


def cell2gps(region, cell):
    x, y = cell2coord(region, cell)
    return meters2lonlat(x, y)


def gps2offset(region, lon, lat):
    x, y = lonlat2meters(lon, lat)
    xoffset = round(x - region.minx, 6) / region.xstep
    yoffset = round(y - region.miny, 6) / region.ystep
    return xoffset, yoffset


def inregion(region, lon, lat):
    return region.minlon <= lon < region.maxlon \
           and region.minlat <= lat < region.maxlat


# def inregion(region, trip):
#     pass
#     # for i in range(trip):
#     #     inregion(region, trip[:, i]...) || return false
#     #
#     # true


"""
Build the vocabulary from the raw trajectories stored in the hdf5 file.
For a trip (trajectory), each point lies in a column when reading with Julia
while it lies in a row if reading with Python.
"""


def makeVocab(region, trjfile):
    region.cellcount = []
    region.hotcell = []
    ## useful for evaluating the size of region bounding box
    num_out_region = 0
    ## scan all trips (trajectories)
    with open(trjfile, "r") as f:
        ss = f.readlines()
    num = len(ss)
    print("共有", num, "条轨迹")
    countDict = dict()
    i = 0
    for trip in ss:
        i += 1
        if i % 1000 == 0:
            print("no %i", i)
        s1 = trip[1: len(trip) - 2]
        s2 = s1.split("]")
        for j in s2:
            j = j.strip(',')
            j = j.strip('[')
            if j == '':
                continue
            x, y = j.split(',')
            x = float(x)
            y = float(y)
            if not inregion(region, x, y):
                num_out_region += 1
            else:
                cell = gps2cell(region, x, y)
                region.cellcount.append(cell)
                if countDict.get(cell) is None:
                    countDict[cell] = 1
                else:
                    countDict[cell] += 1

    ## filter out all hot cells,热点cell的原因是轨迹中存在一些噪声数据，若将每一个点都作为轨迹中的点的话，那么生成的轨迹序列存在很多噪声。所以我们只保留出现点比较多（ex. e > p）的cell
    max_num_hotcells = min(region.maxvocab_size, len(region.cellcount))

    # for i in set(region.cellcount):
    #     countDict[i] = region.cellcount.count(i)
    sorted(countDict.items(), key=lambda d: d[1])
    cnt = 0
    for key in countDict:
        if cnt > max_num_hotcells:
            break
        elif countDict[key] > region.minfreq:
            cnt += 1
            region.hotcell.append(key)

    ## build the map between cell and vocab id
    region.hotcell2vocab = dict([(cell, i - 1 + region.vocab_start)
                                 for (i, cell) in enumerate(region.hotcell)])
    # region.vocab2hotcell = map(reverse, region.hotcell2vocab)
    region.vocab2hotcell = dict([(region.hotcell2vocab[cell], cell)
                                 for cell in region.hotcell2vocab])
    ## vocabulary size
    region.vocab_size = region.vocab_start + len(region.hotcell)
    ## build the hot cell kdtree to facilitate search

    data = np.zeros([len(region.hotcell), 2])
    i = 0
    for cell_ in region.hotcell:
        x, y = cell2coord(region, cell_)
        data[i, :] = [x, y]
        i += 1
    region.hotcell_kdtree = KDTree(data)
    # length = len(xs)
    # xs = np.array([xs])
    # ys = np.array(ys)
    # x_grid = np.tile(xs.transpose(), (1, length))
    # y_grid = np.tile(ys, (length, 1))
    # A = list(zip(x_grid.ravel(), y_grid.ravel()))
    # region.hotcell_kdtree = spatial.KDTree(A)
    region.built = True

    return num_out_region


def knearestHotcells(region, cell, k):
    # @assert region.built == true "Build index for region first"
    coord = cell2coord(region, cell)
    #     idx搜索到点的索引，dists返回的是coord到这些近邻点的欧氏距离
    dists, idxs = region.hotcell_kdtree.query(np.array([[coord[0], coord[1]]]), k)
    # 取对应下标表示的cell id,因为从上层可以得出此处的k一直为1，所以偷懒写法
    res = region.hotcell[idxs[0].tolist()[0]]
    return res, dists


def nearestHotcell(region, cell):
    # @assert region.built == true "Build index for region first"
    hotcell, _ = knearestHotcells(region, cell, 1)
    return hotcell


"""
Return the vocab id for a cell in the region.
If the cell is not hot cell, the function will first search its nearest
hotcell and return the corresponding vocab id.
"""


def cell2vocab(region, cell):
    # @assert region.built == true "Build index for region first"
    #     若落入了hot cell，则直接返回hot cell对应的vocab id
    if region.hotcell2vocab.get(cell):
        # return region.hotcell2vocab[cell]
        return cell
    else:
        # 否则找一个最近的hot cell，返回其对应的vocab id
        hotcell = nearestHotcell(region, cell)
        return hotcell
        # return region.hotcell2vocab[hotcell]


"""
Mapping a gps point to the vocab id in the vocabulary consists of hot cells,
each hot cell has an unique vocab id (hotcell2vocab).

If the point falls out of the region, `UNK` will be returned.
If the point falls into the region, but out of the hot cells its nearest hot cell
will be used.
"""


def gps2vocab(region, lon, lat):
    if not inregion(region, lon, lat):
        return "UNK"
    return cell2vocab(region, gps2cell(region, lon, lat))


# 将raw trip（trajectory）中的点进行重新映射，将点映射到vocab id。由vocab id构成的序列就是seq.
def trip2seq(region, trip):
    seq = []
    for i in range(len(trip)):
        lon, lat = trip[i]
        x = gps2vocab(region, lon, lat)
        if x == "UNK":
            continue
        seq.append(x)
    return seq


def seq2trip(region, seq):
    trip = np.zeros(2, len(seq))
    for i in range(len(seq)):
        cell = region.vocab2hotcell.get(seq[i], -1)
        if cell == -1:
            print("i=%i is out of vocabulary", i)
            continue
        lon, lat = cell2gps(region, cell)
        trip[:, i] = [lon, lat]
    return trip


class SpacialRegion:
    # xstep、ystep默认就是cellsize
    def __init__(self, minlon, minlat, maxlon, maxlat, xstep, ystep,
                 minfreq, maxvocab_size, k, vocab_start):
        self.minlon = minlon
        self.minlat = minlat
        self.maxlon = maxlon
        self.maxlat = maxlat
        self.xstep = xstep
        self.ystep = ystep
        self.minfreq = minfreq
        self.maxvocab_size = maxvocab_size
        self.k = k
        self.vocab_start = vocab_start

        self.minx, self.miny = lonlat2meters(minlon, minlat)
        self.maxx, self.maxy = lonlat2meters(maxlon, maxlat)
        numx = round(self.maxx - self.minx, 6) / xstep
        self.numx = int(math.ceil(numx))
        numy = round(self.maxy - self.miny, 6) / ystep
        self.numy = int(math.ceil(numy))

def createVocab_save():
    # 构建词表以及kdtree
    region = SpacialRegion(minlon=-8.735152, minlat=40.953673, maxlon=-8.156309,
                           maxlat=41.307945, xstep=100.0, ystep=100.0, minfreq=100,
                           maxvocab_size=40000, k=10, vocab_start=4)
    makeVocab(region, "../datasets/porto.pos")

    var_b = pickle.dumps(region)
    with open('pickle.txt', 'wb') as f:
        pickle.dump(var_b, f)
# with open('pickle.txt', 'rb') as f:
#     region = pickle.load(f)
# region = pickle.loads(region)

# def getRegionInstance():
#     with open('pickle.txt', 'rb') as f:
#         region = pickle.load(f)
#     region = pickle.loads(region)
#     return region

# 将轨迹变成对应的字符串输出
def seq2str(seq):
    res = ""
    last = ""
    leng = 0
    for i in range(len(seq)):
        if last == "" or last != str(seq[i]):
            last = str(seq[i])
            leng += 1
            if i == len(seq) - 1:
                res += str(seq[i]) + '\n'
            else:
                res += str(seq[i]) + ' '
    return res,leng


def str2trip(ss):
    trip = []
    s1 = ss[1: len(trip) - 2]
    s2 = s1.split("]")
    for j in s2:
        j = j.strip(',')
        j = j.strip('[')
        if j == '':
            continue
        x, y = j.split(',')
        x = float(x)
        y = float(y)
        trip.append([x, y])
    return trip





