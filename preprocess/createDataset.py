import pickle

from SpatialRegionTools import trip2seq, str2trip, seq2str, createVocab_save
import numpy as np
from SpatialRegionTools import SpacialRegion




def createTrainVal(region, trjfile,
                   ntrain=480000, nval=160000, neval=160000,
                   min_length=15, max_length=30):
    # seq2str(seq) = join(map(string, seq), " ") * "\n"

    with open(trjfile, "r") as f:
        ss = f.readlines()
        trainsrc = open("../datasets/train.src", "w")
        validsrc = open("../datasets/val.src", "w")
        evalsrc = open("../datasets/eval.src", "w")

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
                print("Scaned ",i," trips...")
        trainsrc.close(), validsrc.close(), evalsrc.close()

createVocab_save()
with open('pickle.txt', 'rb') as f:
    var_a = pickle.load(f)
region = pickle.loads(var_a)
createTrainVal(region, "../datasets/porto.pos")