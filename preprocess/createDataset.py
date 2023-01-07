import csv
import os
import sys

from sys_config import DATA_DIR

sys.path.append('/home/hch/Desktop/trjcompress/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle

from preprocess.SpatialRegionTools import trip2seq, str2trip, seq2str, createVocab_save

# you can change the datasets size and trajectory length from here.
def createTrainVal(region, trjfile,
                   # 60%,20%,20%
                   ntrain=3000, nval=1000, neval=1000,
                   min_length=30, max_length=400):

    with open(trjfile, "r") as f:
        ss = f.readlines()
        trainsrc = open(os.path.join(DATA_DIR, "train.src"), "w")
        validsrc = open(os.path.join(DATA_DIR, "val.src"), "w")
        evalsrc = open(os.path.join(DATA_DIR, "eval.src"), "w")
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
                if not (min_length <= leng <= max_length):
                    continue
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


def parseCSV():
    res = []
    i = 0
    with open('../datasets/porto.csv', 'r') as f:

        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv:
            pos = row[8]
            res.append(pos)

    with open('../datasets/porto.pos', 'w') as f:
        for it in res:
            i += 1
            if i > 100000:
                break
            f.write(it + "\n")


if __name__ == '__main__':
    path = "your_pos_file_dir"
    createVocab_save(path)
    with open(os.path.join(DATA_DIR, "pickle.txt"), 'rb') as f:
        var_a = pickle.load(f)
    region = pickle.loads(var_a)
    createTrainVal(region, path)
