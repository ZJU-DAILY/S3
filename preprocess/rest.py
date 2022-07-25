import datetime

from models.seq3_losses import sed_loss
from preprocess.SpatialRegionTools import cell2gps, cell2coord, coord2cell, lonlat2meters
from preprocess.SpatialRegionTools import str2trip
# from SpatialRegionTools import getRegionInstance
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib as mpl
from modules.helpers import getDistance, getCompress

mpl.use('Agg')
from matplotlib import pyplot as plt
import math
import pickle


def plotTest(region):
    with open("../datasets/eval.src", "r") as f:
        ss = f.readlines()
    x = []
    y = []

    minlon, minlat = lonlat2meters(115.7001, 39.4)
    maxlon, maxlat = lonlat2meters(117.39994, 41.59471)
    x.append(minlon)
    x.append(maxlon)
    y.append(minlat)
    y.append(maxlat)
    for s in ss:
        trj = s.split(" ")
        for p in trj:
            if p != '':
                lon, lat = cell2gps(region, int(p))
                x_, y_ = lonlat2meters(lon, lat)
                x.append(x_), y.append(y_)
    print(len(ss))
    plt.scatter(x, y, color='r')
    plt.grid(True)
    plt.savefig('test_eval.png')
    plt.close()


def case(region, gl_gid2poi, src_file, out_file):
    with open(src_file, "r") as f:
        ss = f.readlines()
    res = ""

    for s in ss:
        trj = s.strip('\n').split(" ")
        for p in trj:
            if p != '':
                lon, lat = cell2gps(region, int(p))
                poi = gl_gid2poi.get(p, -1)
                res += str(lon) + " " + str(lat) + " " + str(poi) + '\n'
        res += '\n'
    with open(f"./{out_file}", "w") as f:
        f.write(res)


with open('pickle.txt', 'rb') as f:
    var_a = pickle.load(f)
region = pickle.loads(var_a)

# plotTest(region)
with open('../datasets/gid2poi.txt', 'rb') as f:
    var_a = pickle.load(f)
gl_gid2poi = pickle.loads(var_a)

# case(region, gl_gid2poi, '../datasets/eval.src', 'gps.txt')
case(region, gl_gid2poi, '../evaluation/seq3.full_-valid_preds.txt', 'gps_comp.txt')