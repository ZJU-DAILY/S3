import os
import sys


sys.path.append('/home/hch/Desktop/trjcompress/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle

from preprocess.SpatialRegionTools import trip2seq, str2trip, seq2str, createVocab_save


# todo 完成数据集的重新生成
def createTrainVal(region, trjfile,
                   # 60%,20%,20%
                   ntrain=0, nval=0, neval=1000,
                   min_length=400, max_length=512):
    # seq2str(seq) = join(map(string, seq), " ") * "\n"

    with open(trjfile, "r") as f:
        ss = f.readlines()
        trainsrc = open("../datasets/train.src", "w")
        validsrc = open("../datasets/val.src", "w")
        evalsrc = open("../datasets/eval.src", "w")
        cnt = 1
        sum_ = 0

        for i in range(len(ss)):
            trip_ = str2trip(ss[i])
            trips = []
            if len(trip_) > max_length:
                st = 0
                while st + max_length < len(trip_):
                    trips.append(trip_[st:st + max_length])
                    st += max_length
            else:
                trips.append(trip_)

            if len(trips) == 0:
                continue
            for trip in trips:
                trg = trip2seq(region, trip)
                trg_str, leng = seq2str(trg)
                # print(leng)
                # if leng == 0:
                #     print()
                if not (min_length <= leng <= max_length):
                    continue
                # trg_str = [str(k) for k in trg]
                # trg_str = " ".join(trg_str)
                sum_ += 1
                trg_str = trg_str.strip(" ")
                trg_str = trg_str.strip("\n")
                trg_str = trg_str + "\n"
                if cnt <= ntrain:
                    trainsrc.write(trg_str)
                    cnt += 1
                elif cnt <= ntrain + nval:
                    validsrc.write(trg_str)
                    cnt += 1
                elif cnt <= ntrain + nval + neval:
                    evalsrc.write(trg_str)
                    cnt += 1
                else:
                    break
            if cnt > ntrain + nval + neval:
                break
            if i % 1000 == 0:
                print("Scaned ", i, " trips...")
        trainsrc.close(), validsrc.close(), evalsrc.close()
    print(sum_)
path = "../datasets/beijing.pos"
# path = "../datasets/porto.pos"
# createVocab_save(path)
with open('pickle.txt', 'rb') as f:
    var_a = pickle.load(f)
region = pickle.loads(var_a)
createTrainVal(region, path)