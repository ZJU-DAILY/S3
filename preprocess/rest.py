import datetime

from models.seq3_losses import sed_loss
from preprocess.SpatialRegionTools import cell2gps, cell2coord, coord2cell, lonlat2meters
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

def plotTest(region):
    with open("../datasets/eval.src", "r") as f:
        ss = f.readlines()
    x = []
    y = []

    minlon,minlat = lonlat2meters(115.7001, 39.4)
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

with open('pickle.txt', 'rb') as f:
    var_a = pickle.load(f)
region = pickle.loads(var_a)

plotTest(region)