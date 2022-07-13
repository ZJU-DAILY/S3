import pickle

from preprocess.SpatialRegionTools import trip2seq, str2trip, seq2str, createVocab_save

# todo 完成数据集的重新生成
def createTrainVal(region, trjfile,
                   ntrain=3600, nval=1200, neval=1200,
                   min_length=30, max_length=50):
    # seq2str(seq) = join(map(string, seq), " ") * "\n"

    with open(trjfile, "r") as f:
        ss = f.readlines()
        trainsrc = open("../datasets/train.src", "w")
        validsrc = open("../datasets/val.src", "w")
        evalsrc = open("../datasets/eval.src", "w")
        cnt = 1
        sum_ = 0
        for i in range(len(ss)):
            trip = str2trip(ss[i])
            if len(trip) > max_length:
                trip = trip[:max_length]
            if not (min_length <= len(trip) <= max_length):
                continue

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

            if i % 1000 == 0:
                print("Scaned ",i," trips...")
        trainsrc.close(), validsrc.close(), evalsrc.close()
    print(sum_)
path = "../datasets/beijing.pos"
createVocab_save(path)
with open('pickle.txt', 'rb') as f:
    var_a = pickle.load(f)
region = pickle.loads(var_a)
createTrainVal(region, path)