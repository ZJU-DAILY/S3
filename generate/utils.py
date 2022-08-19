import pickle

import math
from itertools import groupby

import torch
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
    checkpoint = load_checkpoint(checkpoint,path)
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
                trg_lengths = torch.clamp(src_lengths * max_ratio, min=9, max=30) + 1
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


