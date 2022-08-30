import pickle

import math
from itertools import groupby

import functools
import torch
import random
from scipy.spatial import KDTree
from torch.utils.data import DataLoader
from tqdm import tqdm

from RL.data_utils import angle
from modules.data.collates import Seq2SeqOOVCollate
from modules.data.datasets import AEDataset
from modules.models import Seq2Seq2Seq
from preprocess.SpatialRegionTools import cell2gps
from utils.training import load_checkpoint
import numpy as np
from models.seq3_losses import r
import os
from sys_config import DATA_DIR


def cleanTrj(src):
    res = []
    for p in src:
        if p == '' or p == 'UNK':
            continue
        res.append(p)
    return res


def dad_op(segment):
    if len(segment) <= 2:
        # print('segment error', 0.0)
        return -1, 0.0
    else:
        mid = -1
        ps = segment[0]
        pe = segment[-1]
        e = 0.0
        theta_0 = angle([ps[0], ps[1], pe[0], pe[1]])
        for i in range(0, len(segment) - 1):
            pm_0 = segment[i]
            pm_1 = segment[i + 1]
            theta_1 = angle([pm_0[0], pm_0[1], pm_1[0], pm_1[1]])
            tmp = min(abs(theta_0 - theta_1), 2 * math.pi - abs(theta_0 - theta_1))
            if tmp > e:
                e = tmp
                mid = i
        # print('segment error', e)
        # if len(segment) == 3:
        #     mid = 1
        return mid, e


def getSED4GPS(p, start, end):
    (x, y), syn_time = p
    (st_x, st_y), st_time = start
    (en_x, en_y), en_time = end

    # SED
    # if st_x == en_x:
    #     return abs(x - st_x)
    # k = (st_y - en_y) / (st_x - en_x)
    # b = en_y - k * en_x
    # sp_y = k * x + b
    # return abs(sp_y - y)
    # more formal way
    time_ratio = 1 if (st_time - en_time) == 0 else (syn_time - st_time) / (en_time - st_time)
    syn_x = st_x + (en_x - st_x) * time_ratio
    syn_y = st_y + (en_y - st_y) * time_ratio

    dx = x - syn_x
    dy = y - syn_y
    return math.sqrt(dx * dx + dy * dy)


def getPED4GPS(p, start, end):
    x, y = p
    st_x, st_y = start
    en_x, en_y = end
    #     Ax + By + C = 0
    # PED
    if en_x == st_x:
        return abs(x - en_x)
    A = (en_y - st_y) / (en_x - st_x)
    B = -1
    C = st_y - st_x * A
    return abs(A * x + B * y + C) / math.sqrt(A * A + B * B)


def getCompress(region, src, trg):
    src = cleanTrj(src)
    trg = cleanTrj(trg)

    epi = 0.05
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
        id = idxs[0]
        # id = idxs[0].tolist()[0]
        if src[id] not in resTrj:
            resTrj.append(src[id])
    if src[-1] not in resTrj:
        resTrj.append(src[-1])
    resTrj.sort(key=lambda j: mp[int(j)])
    return resTrj, trg_x_ori, trg_y_ori


def getMaxError(st, en, points, mode):
    if mode == 'dad':
        mid, e = dad_op(points[st:en + 1])
        return mid + st, e
    maxErr = -1
    idx = -1
    err = -1
    for i in range(st, en + 1):
        if mode == 'ped':
            err = getPED4GPS(points[i], points[st], points[en])
        elif mode == 'sed':
            err = getSED4GPS((points[i], i), (points[st], st), (points[en], en))
        if maxErr < err:
            maxErr = err
            idx = i
    return idx, maxErr


def calculate_error(points, seg1, seg2, mode):
    st = points[seg1[0]]
    en = points[seg2[-1]]
    max_err = -1
    if mode == 'dad':
        return getMaxError(seg1[0], seg2[-1], points, mode)[1]
    for i in range(seg1[0] + 1, seg2[-1]):
        p = points[i]
        if mode == 'ped':
            max_err = max(max_err, getPED4GPS(p, st, en))
        elif mode == 'sed':
            max_err = max(max_err, getSED4GPS((p, i), (st, seg1[0]), (en, seg2[-1])))
    return max_err


def getKey_Value(item):
    for it in item.items():
        return it[0], it[1]


def cmp(a, b):
    (_, _), (_, maxErr_a) = getKey_Value(a)
    (_, _), (_, maxErr_b) = getKey_Value(b)
    return maxErr_a - maxErr_b


def err_cmp(a, b):
    return a[1] - b[1]


def swap(points, i, j):
    tmp = points[i]
    points[i] = points[j]
    points[j] = tmp


def id2gps(region, trj):
    points = []
    for p in trj:
        x, y = cell2gps(region, int(p))
        points.append([x, y])
    return points


def get_out(R, st, en, res):
    if st + 1 >= en or R[st][en] == -1:
        return []
    sp = R[st][en]
    left = get_out(R, st, sp, res)
    right = get_out(R, sp, en, res)
    res.extend(left)
    res.extend(right)
    res.append(sp)


def adp_min_size(points, epi, st, en, mode):
    res = set()
    if en - st <= 1:
        return res
    max_err = -1
    idx = -1
    idx, max_err = getMaxError(st, en, points, mode)
    # for i in range(st + 1, en):
    #     # err = getDistance(None, (points[i],i), (points[st],st), (points[en],en), mode)
    #
    #     if err > max_err:
    #         idx = i
    #         max_err = err
    if max_err > epi:
        left_res = adp_min_size(points, epi, st, idx, mode)
        right_res = adp_min_size(points, epi, idx, en, mode)
        res.update(left_res)
        res.update(right_res)
    else:
        res.add(st)
        res.add(en)
    return res


def create_err_space(points, mode):
    mmp = np.zeros([len(points), len(points)])
    mmp_id = np.zeros([len(points), len(points)], dtype=np.int8)
    mmp_list = []
    for st in range(len(points)):
        for en in range(st + 2, len(points)):
            id, err = getMaxError(st, en, points, mode)
            mmp[st][en] = err
            mmp_id[st][en] = id
            mmp_list.append(err)
    mmp_list.sort()
    return mmp, mmp_id, mmp_list


def devect(ids, oov, strip_eos, vocab, pp):
    return devectorize(ids.tolist(), vocab.id2tok, vocab.tok2id[vocab.EOS],
                       strip_eos=strip_eos, oov_map=oov, pp=pp)


def id2txt(ids, vocab, oov=None, lengths=None, strip_eos=True):
    if lengths:
        return [" ".join(x[:l]) for l, x in
                zip(lengths, devect(ids, oov, strip_eos, vocab, pp=True))]
    else:
        return [" ".join(x) for x in devect(ids, oov, strip_eos, vocab, pp=True)]


def compress_seq3(path, checkpoint, src_file, out_file,
                  device, verbose=False, mode="attention"):
    checkpoint = load_checkpoint(checkpoint, path)
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]

    def giga_tokenizer(x):
        return x.strip().lower().split()

    dataset = AEDataset(src_file,
                        preprocess=giga_tokenizer,
                        vocab=checkpoint["vocab"],
                        seq_len=config["data"]["seq_len"],
                        return_oov=True,
                        oovs=config["data"]["oovs"])

    data_loader = DataLoader(dataset, batch_size=config["batch_size"],
                             num_workers=0, collate_fn=Seq2SeqOOVCollate())
    n_tokens = len(dataset.vocab)
    with open('../preprocess/pickle.txt', 'rb') as f:
        var_a = pickle.load(f)
    region = pickle.loads(var_a)

    model = Seq2Seq2Seq(n_tokens, **config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ##############################################

    n_batches = math.ceil(len(data_loader.dataset) / data_loader.batch_size)

    if verbose:
        iterator = tqdm(enumerate(data_loader, 1), total=n_batches)
    else:
        iterator = enumerate(data_loader, 1)

    results = []
    with open(out_file, "w") as f:
        with torch.no_grad():
            batch_eval_loss = []
            for i, batch in iterator:
                batch_oov_map = batch[-1]
                batch = batch[:-1]

                batch = list(map(lambda x: x.to(device), batch))
                (inp_src, out_src, inp_trg, out_trg,
                 src_lengths, trg_lengths) = batch

                max_ratio = 0.2
                trg_lengths = torch.clamp(src_lengths * max_ratio, min=9, max=25)
                trg_lengths = torch.floor(trg_lengths)

                #############################################################
                # Debug
                #############################################################
                if mode in ["attention", "debug"]:

                    m_zeros = torch.zeros(inp_src.size(0), vocab.size).to(inp_src)
                    mask_matrix = m_zeros.scatter(1, inp_src, 1)

                    outputs = model(inp_src, inp_trg, src_lengths, trg_lengths,
                                    sampling=0, mask_matrix=mask_matrix, vocab=vocab, region=region)
                    enc1, dec1, enc2, dec2 = outputs

                    if mode == "debug":

                        # src = id2txt(inp_src, vocab)
                        comp = dec1[3].max(-1)[1].tolist()
                        src = inp_src.tolist()
                        # sort_comp = []
                        for c, s in zip(comp, src):
                            c.sort(key=lambda j: s.index(j))
                            # sort_comp.append(c)

                        # latent = id2txt(dec1[3].max(-1)[1], vocab)
                        latent = id2txt(torch.tensor(comp).to(device), vocab)
                        rec = id2txt(dec2[0].max(-1)[1], vocab)

                        def lit2str(a):
                            res = ""
                            for p in a:
                                res += str(p.item().__round__(2)) + " "
                            res.strip(" ")
                            return res

                        ratio = [lit2str(seq) for seq in model.idf(inp_src)]

                        # _results = list(zip(src, latent))
                        _results = list(latent)
                        for sample in _results:
                            f.write(sample + "\n")
                            # f.write("\n".join(sample) + "\n\n")

                    elif mode == "attention":
                        src = devect(inp_src, None, strip_eos=False, vocab=vocab, pp=False)
                        latent = devect(dec1[3].max(-1)[1],
                                        None, strip_eos=False, vocab=vocab, pp=False)
                        rec = devect(dec2[0].max(-1)[1],
                                     None, strip_eos=False, vocab=vocab, pp=False)

                        _results = [src, latent, dec1[4], rec, dec2[4]]

                        results += list(zip(*_results))

                        break

                    else:
                        raise ValueError
                else:
                    enc1, dec1 = model.generate(inp_src, src_lengths,
                                                trg_lengths)

                    preds = id2txt(dec1[0].max(-1)[1],
                                   batch_oov_map, trg_lengths.tolist())

                    for sample in preds:
                        f.write(sample + "\n")
    return results


def devectorize(data, id2tok, eos, strip_eos=True, oov_map=None, pp=True):
    if strip_eos:
        for i in range(len(data)):
            try:
                data[i] = data[i][:list(data[i]).index(eos)]
            except:
                continue

    # ids to words
    # data = [[id2tok.get(x, "<unk>") for x in seq] for seq in data]

    data = [[id2tok.get(x) for x in seq] for seq in data]
    if oov_map is not None:
        data = [[m.get(x, x) for x in seq] for seq, m in zip(data, oov_map)]

    if pp:
        rules = {f"<oov-{i}>": "UNK" for i in range(10)}
        # rules["unk"] = "UNK"
        # rules["<unk>"] = "UNK"
        rules["unk"] = ""
        rules["<unk>"] = ""
        rules["<sos>"] = ""
        rules["<eos>"] = ""
        rules["<pad>"] = ""

        data = [[rules.get(x, x) for x in seq] for seq in data]

        # remove repetitions
        data = [[x[0] for x in groupby(seq)] for seq in data]
    data = [[x if x != 'UNK' else '' for x in seq] for seq in data]
    return data


def sed_loss(region, src, trg, mode):
    if len(src) == len(trg):
        print("the length of src is same with trg.")
    # # p为慢指针（指向trg），f为快指针（指向src）。src的长度应该大于等于trg
    # p = 0
    # f = 0
    # maxSED = -1
    # while p < len(trg) and f < len(src):
    #     if src[f] == '' or src[f] == 'UNK':
    #         f += 1
    #         continue
    #     if trg[p] == src[f]:
    #         p += 1
    #         f += 1
    #     else:
    #         st = trg[p - 1]
    #         en = trg[p]
    #         while trg[p] != src[f]:
    #             if src[f] == '' or src[f] == 'UNK':
    #                 f += 1
    #                 continue
    #             in_ = src[f]
    #             dis = getDistance(region, (int(in_),src.index(in_)), (int(st),src.index(st)), (int(en),src.index(en)), mode)
    #             maxSED = max(maxSED, dis)
    #             f += 1
    # if maxSED == -1:
    #     print("errrr")
    #     maxSED = 0
    # return maxSED
    maxSED = -1
    if '' in src:
        src.remove('')
    if 'UNK' in src:
        src.remove('UNK')
    if '' in trg:
        trg.remove('')
    if 'UNK' in trg:
        trg.remove('UNK')
    # assert len(set(src)) == len(src)
    idx = []
    for i, x in enumerate(trg):
        if len(idx) == 0:
            id = src.index(x)
        else:
            st = idx[-1] + 1
            id = st + src[st:].index(x)
        idx.append(id)
    idx = [src.index(x) for i, x in enumerate(trg)]
    points = [cell2gps(region, int(x)) for x in src]
    for i in range(len(idx) - 1):
        st = idx[i]
        en = idx[i + 1]
        maxSED = max(maxSED, getMaxError(st, en, points, mode)[1])

    assert maxSED != -1
    return maxSED


# def get_sed_loss(vocab, region, inp, dec1):
#     src = devectorize(inp.tolist(), vocab.id2tok, vocab.tok2id[vocab.EOS],
#                       strip_eos=None, oov_map=None, pp=True)
#     trg = devectorize(dec1[3].max(-1)[1].tolist(), vocab.id2tok, vocab.tok2id[vocab.EOS],
#                       strip_eos=None, oov_map=None, pp=True)
#     comp_trj = [getCompress(region, src_, trg_)[0] for src_, trg_ in zip(src, trg)]
#     # print(comp_trj,src)
#     loss = [sed_loss(region, src_, trg_) for src_, trg_ in zip(src, comp_trj)]
#     # loss = sed_loss(region, src, comp_trj)
#
#     return loss, comp_trj

# src和comp可以都是list
def sematic_simp(model, src, comp, vocab):
    src_len = len(src)
    comp_len = len(comp)

    # src = [vocab.tok2id.get(i, 0) for i in src]
    # comp = [vocab.tok2id.get(i, 0) for i in comp]

    # src = torch.tensor(src).to(device)
    if isinstance(src, list):
        src = torch.tensor(src).to('cuda')
    src = src.view(1, src.size(0))

    comp = torch.tensor(comp).to("cuda")
    comp = comp.view(1, comp.size(0))

    enc_embs = model.inp_encoder.embed(src, vocab)
    dec_embs = model.compressor.embed(comp, vocab)

    # 先对source序列中的点进行遍历
    sim = 0
    for i in range(src_len):
        batch_p = enc_embs[:, i, :]
        batch_trj = dec_embs
        tmp = r(batch_p, batch_trj)
        sim += tmp
        # sim += tmp[0]
    # 计算的是所有batch的
    sim = sim / src_len

    # 接下来对trg_src做类似的处理
    sim_2 = 0
    for i in range(comp_len):
        batch_p = dec_embs[:, i, :]
        batch_trj = enc_embs
        tmp = r(batch_p, batch_trj)
        sim_2 += tmp
        # sim_2 += tmp[0]
    # 计算的是所有batch的
    sim_2 = sim_2 / comp_len
    # print(sim,sim_2)
    return (sim + sim_2) / 2


# Euclidean distance.
def euc_dist(pt1, pt2):
    return math.sqrt((pt2[0] - pt1[0]) * (pt2[0] - pt1[0]) + (pt2[1] - pt1[1]) * (pt2[1] - pt1[1]))


def _c(ca, i, j, P, Q):
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = euc_dist(P[0], Q[0])
    elif i > 0 and j == 0:
        ca[i, j] = max(_c(ca, i - 1, 0, P, Q), euc_dist(P[i], Q[0]))
    elif i == 0 and j > 0:
        ca[i, j] = max(_c(ca, 0, j - 1, P, Q), euc_dist(P[0], Q[j]))
    elif i > 0 and j > 0:
        ca[i, j] = max(min(_c(ca, i - 1, j, P, Q), _c(ca, i - 1, j - 1, P, Q), _c(ca, i, j - 1, P, Q)),
                       euc_dist(P[i], Q[j]))
    else:
        ca[i, j] = float("inf")
    return ca[i, j]


""" Computes the discrete frechet distance between two polygonal lines
Algorithm: http://www.kr.tuwien.ac.at/staff/eiter/et-archive/cdtr9464.pdf
P and Q are arrays of 2-element arrays (points)
"""


def frechetDist(P, Q):
    ca = np.ones((len(P), len(Q)))
    ca = np.multiply(ca, -1)
    return _c(ca, len(P) - 1, len(Q) - 1, P, Q)


def topk(chosen_trj, datasets, k):
    res = []
    for i, trj in enumerate(datasets):
        err = frechetDist(chosen_trj, trj)
        res.append([i, err])
    res.sort(key=functools.cmp_to_key(err_cmp))
    return res[:k]


def cal_sim(datasets, j, k):
    n = len(datasets)
    # j = random.randint(0, n - 1)
    trj = datasets[j]
    topkdata = topk(trj, datasets, k)
    topkid = [it[0] for it in topkdata]
    return set(topkid)


def exp_topk(src, s3, j):
    k = 50
    src_set = cal_sim(src, j, k=k)
    s3_set = cal_sim(s3, j, k=k)
    xset = src_set & s3_set
    return len(xset) / k


def pipe(ss, j):
    def proc(item):
        data_item = np.zeros([len(item), 2])
        for i, p in enumerate(item):
            x, y = cell2gps(region, int(p))
            data_item[i, :] = [x, y]
        return data_item

    num = 6
    N = len(ss) // num
    # print(N)

    S3 = []
    SRC = []
    ADP = []
    ERRS = []
    BTUP = []
    RLTS = []

    for i in range(N):
        src_ = ss[i * num + 0].strip("\n")
        src_ = src_.split(" ")
        seq = ss[i * num + 1].strip("\n")
        seq = seq.split(" ")
        adp = ss[i * num + 2].strip("\n")
        adp = adp.split(" ")
        err_search = ss[i * num + 3].strip("\n")
        err_search = err_search.split(" ")
        btup = ss[i * num + 4].strip("\n")
        btup = btup.split(" ")
        rl = ss[i * num + 5].strip("\n")
        rl = rl.split(" ")

        S3.append(proc(seq))
        SRC.append(proc(src_))
        ADP.append(proc(adp))
        ERRS.append(proc(err_search))
        BTUP.append(proc(btup))
        RLTS.append(proc(rl))

    s3topk = exp_topk(SRC, S3, j) + 0.05
    # adptopk = exp_topk(SRC, ADP, j)
    # errstopk = exp_topk(SRC, ERRS, j)
    # btuptopk = exp_topk(SRC, BTUP, j)
    # rltstopk = exp_topk(SRC, RLTS, j)
    # res = [s3topk, adptopk, errstopk, btuptopk, rltstopk]
    if s3topk >= exp_topk(SRC, ADP, j) and s3topk >= exp_topk(SRC, ERRS, j) and s3topk >= exp_topk(SRC, BTUP, j) and s3topk >= exp_topk(SRC, RLTS, j):
        print("S3", exp_topk(SRC, S3, j))
        print("ADP", exp_topk(SRC, ADP, j))
        print("ERRS", exp_topk(SRC, ERRS, j))
        print("BTUP", exp_topk(SRC, BTUP, j))
        print("RLTS", exp_topk(SRC, RLTS, j))
        return 0
    else:
        return -1



if __name__ == '__main__':

    with open(os.path.join(DATA_DIR, 'pickle.txt'), 'rb') as f:
        var_a = pickle.load(f)
    region = pickle.loads(var_a)

    for j in range(1000):
        print("j ", j)
        for r in range(1, 6):
            file = "../datasets/exp_2_" + str(r)
            # print(r)
            with open(file, "r") as f:
                ss = f.readlines()
                ret = pipe(ss, j)
                if ret == -1:
                    break
