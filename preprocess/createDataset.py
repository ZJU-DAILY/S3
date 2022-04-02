#
# with open('porto.pos', 'r') as f:
#     ss = f.readlines()
#     for s in ss:
#         s1 = s[1:len(s) - 2]
#         s2 = s1.split("]")
#         for j in s2:
#             j = j.strip(',')
#             j = j.strip('[')
#             if j is '':
#                 continue
#             x, y = j.split(',')
#             # print(float(x), float(y))
import pickle

from SpatialRegionTools import trip2seq, str2trip, seq2str
import numpy as np
from SpatialRegionTools import SpacialRegion




def createTrainVal(region, trjfile,
                   ntrain=60000, nval=20000, neval=20000,
                   min_length=15, max_length=50):
    # seq2str(seq) = join(map(string, seq), " ") * "\n"

    with open(trjfile, "r") as f:
        ss = f.readlines()
        trainsrc = open("dataset/train.src", "w")
        validsrc = open("dataset/val.src", "w")
        evalsrc = open("dataset/eval.src", "w")

        for i in range(ntrain + nval + neval):
            trip = str2trip(ss[i])
            if not (min_length <= len(trip) <= max_length):
                continue

            trg = trip2seq(region, trip)
            trg_str, leng = seq2str(trg)
            if not (min_length <= leng <= max_length):
                continue
            if i <= ntrain:
                trainsrc.write(trg_str)
            elif i <= ntrain + nval:
                validsrc.write(trg_str)
            else:
                evalsrc.write(trg_str)

            if i % 10000 == 0:
                print("Scaned %i trips...", i)
        trainsrc.close(), validsrc.close(), evalsrc.close()


with open('pickle.txt', 'rb') as f:
    var_a = pickle.load(f)
region = pickle.loads(var_a)
createTrainVal(region, "porto.pos")