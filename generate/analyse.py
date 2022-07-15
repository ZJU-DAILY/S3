# -*- coding: utf-8 -*-
# @Time    : 2022/5/13 20:18
# @Author  : HeAlec
# @FileName: analyse.py
# @Desc: Analyse the exp result
# @blogs ：https://segmentfault.com/u/alec_5e8962e4635ca

import functools
import os
import pickle
import time

from modules.data.samplers import BucketBatchSampler
from modules.helpers import getSED4GPS, squish, sequence_mask
from preprocess.SpatialRegionTools import cell2gps
from preprocess.SpatialRegionTools import SpacialRegion
from sys_config import DATA_DIR

import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.data.collates import Seq2SeqOOVCollate
from modules.data.datasets import AEDataset
from modules.models import Seq2Seq2Seq
from utils.training import load_checkpoint
import numpy as np
from generate.utils import get_sed_loss
from modules.helpers import adp
from models.seq3_losses import r


def readData(src_file, region):
    with open(src_file, "r") as f:
        ss = f.readlines()
    points = []
    src = []
    for s in ss:
        s = s.strip("\n")
        s = s.split(" ")
        point = []
        src_ = []
        for p in s:
            if p == " " or p == "UNK":
                continue
            point.append(cell2gps(region, int(p)))
            src_.append(p)
        points.append(point)
        src.append(src_)
    return points, src

def sematic_cal(model, inp, dec1, src_lengths, trg_lengths,vocab):
    enc_mask = sequence_mask(src_lengths).unsqueeze(-1).float()
    dec_mask = sequence_mask(trg_lengths - 1).unsqueeze(-1).float()

    enc_embs = model.inp_encoder.embed(inp, vocab) * enc_mask
    if dec1[3] is None:
        print("dec1[3] is none")
    dec_embs = model.compressor.embed.expectation(dec1[3], vocab) * dec_mask

    # 先对source序列中的点进行遍历
    max_len = enc_embs.size(1)
    sim = np.zeros(enc_embs.size(0))
    for i in range(max_len):
        batch_p = enc_embs[:, i, :]
        batch_trj = dec_embs
        tmp = r(batch_p, batch_trj)
        sim += tmp
    # 计算的是所有batch的
    sim = [sim[i] / length.item() for i, length in enumerate(src_lengths)]
    # 将所有batch的结果求平均
    sim = np.mean(sim)

    # 接下来对trg_src做类似的处理
    max_len = dec_embs.size(1)
    sim_2 = np.zeros(dec_embs.size(0))
    for i in range(max_len):
        batch_p = dec_embs[:, i, :]
        batch_trj = enc_embs
        tmp = r(batch_p, batch_trj)
        sim_2 += tmp
    # 计算的是所有batch的
    sim_2 = [sim_2[i] / length.item() for i, length in enumerate(trg_lengths)]
    # 将所有batch的结果求平均
    sim_2 = np.mean(sim_2)
    # print(sim,sim_2)
    return (sim + sim_2) / 2

# src和comp可以都是list
def sematic_simp(model, src, comp,vocab):
    src_len = len(src)
    comp_len = len(comp)

    src = [vocab.tok2id.get(i,0) for i in src]
    comp = [vocab.tok2id.get(i,0) for i in comp]

    src = torch.tensor(src).to(device)
    src = src.view(1,src.size(0))

    comp = torch.tensor(comp).to(device)
    comp = comp.view(1, comp.size(0))

    enc_embs = model.inp_encoder.embed(src, vocab)
    dec_embs = model.compressor.embed(comp, vocab)

    # 先对source序列中的点进行遍历
    sim = 0
    for i in range(src_len):
        batch_p = enc_embs[:, i, :]
        batch_trj = dec_embs
        tmp = r(batch_p, batch_trj)
        sim += tmp[0]
    # 计算的是所有batch的
    sim = sim / src_len

    # 接下来对trg_src做类似的处理
    sim_2 = 0
    for i in range(comp_len):
        batch_p = dec_embs[:, i, :]
        batch_trj = enc_embs
        tmp = r(batch_p, batch_trj)
        sim_2 += tmp[0]
    # 计算的是所有batch的
    sim_2 = sim_2 / comp_len
    # print(sim,sim_2)
    return (sim + sim_2) / 2


def seq2str(seq):
    res = ""
    for i, s in enumerate(seq):
        if i != len(seq) - 1:
            res = res + s + " "
        else:
            res = res + s + "\n"
    return res


def compress_adp(src, points, max_ratio, key_dict, tlen=None, AError=None, score_adp=None, score_squish=None,model=None, vocab=None):
    # 计时开始
    tic1 = time.perf_counter()
    err = []
    key = []
    compRes = []
    sematical_loss = []
    if tlen is None:
        for idseq, seq in zip(src, points):
            _, idx, maxErr = adp(seq, int(max_ratio * len(seq)))
            r_ = 0
            for id in idx:
                r_ += key_dict.get(idseq[id], 0)
            key.append(r_ / len(idx))
            err.append(maxErr)
    else:
        for idseq, seq, L, err_trj in zip(src, points, tlen, AError):
            # pp为压缩后的GPS点
            _, idx, maxErr = adp(seq, L)
            comp = [idseq[i] for i in idx]
            s_loss = sematic_simp(model,idseq,comp,vocab)
            sematical_loss.append(s_loss)
            r_ = 0
            for id in idx:
                r_ += key_dict.get(idseq[id], 0)
            key.append(r_ / len(idx))
            err.append(maxErr)

            _, idx_squish, maxErr_squish = squish(seq, L)
            r_squish = 0
            for id in idx_squish:
                r_squish += key_dict.get(idseq[id], 0)

            # 计算真实的压缩结果
            ratio = round(L / len(idseq), 1)
            # adp部分
            if ratio not in score_adp:
                # 当前压缩率下对应的：loss、关键点程度、个数
                score_adp[ratio] = [maxErr, r_ / len(idx), 1]
            else:
                score_adp[ratio][0] += maxErr
                score_adp[ratio][1] += r_ / len(idx)
                score_adp[ratio][2] += 1
            # squish部分
            if ratio not in score_squish:
                # 当前压缩率下对应的：loss、关键点程度、个数
                score_squish[ratio] = [maxErr_squish, r_squish / len(idx), 1]
            else:
                score_squish[ratio][0] += maxErr_squish
                score_squish[ratio][1] += r_squish / len(idx)
                score_squish[ratio][2] += 1

            # 写压缩轨迹部分
            a_err = err_trj[0]
            seq3_trj = err_trj[1]
            adp_trj = [idseq[id] for id in idx]
            squish_trj = [idseq[id] for id in idx_squish]
            if maxErr >= a_err and maxErr_squish >= a_err:
                seq3_trj = seq2str(seq3_trj)
                adp_trj = seq2str(adp_trj)
                squish_trj = seq2str(squish_trj)
                compRes.append(seq2str(idseq) + seq3_trj + adp_trj + squish_trj)

    # 计时结束
    tic2 = time.perf_counter()
    print(f"压缩率 {max_ratio},耗时 {tic2 - tic1},误差 {np.mean(err)},关键程度 {np.mean(key)},语义相似度 {np.mean(sematical_loss)}")
    return compRes, score_adp, score_squish


def compress_squish(src, points, max_ratio, key_dict, tlen=None, AError=None):
    # 计时开始
    tic1 = time.perf_counter()
    err = []
    key = []
    compRes = []
    if tlen is None:
        for idseq, seq in zip(src, points):
            _, idx, maxErr = squish(seq, int(max_ratio * len(seq)))
            r_ = 0
            for id in idx:
                r_ += key_dict.get(idseq[id], 0)
            key.append(r_ / len(idx))
            err.append(maxErr)
    else:
        for idseq, seq, L, err_trj in zip(src, points, tlen, AError):
            # pp为压缩后的GPS点
            pp, idx, maxErr = squish(seq, L)
            r_ = 0
            for id in idx:
                r_ += key_dict.get(idseq[id], 0)
            key.append(r_ / len(idx))
            err.append(maxErr)
            a_err = err_trj[0]
            seq3_trj = err_trj[1]
            adp_trj = [idseq[id] for id in idx]
            if maxErr > a_err:
                seq3_trj = seq2str(seq3_trj)
                adp_trj = seq2str(adp_trj)
                compRes.append(seq2str(idseq) + seq3_trj + adp_trj)

    # 计时结束
    tic2 = time.perf_counter()
    print(f"压缩率 {max_ratio},耗时 {tic2 - tic1},误差 {np.mean(err)},关键程度 {np.mean(key)}")
    return compRes


def load_model(path, checkpoint, src_file, device):
    checkpoint = load_checkpoint(checkpoint, path=path)
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
    lengths = [len(x) for x in dataset.data]
    sampler = BucketBatchSampler(lengths, config["batch_size"])
    # data_loader = DataLoader(dataset, batch_sampler=sampler,
    #                          num_workers=0, collate_fn=Seq2SeqOOVCollate())
    data_loader = DataLoader(dataset, batch_size=config["batch_size"],
                             num_workers=0, collate_fn=Seq2SeqOOVCollate())
    n_tokens = len(dataset.vocab)

    model = Seq2Seq2Seq(n_tokens, **config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ##############################################

    return data_loader, model, vocab


def load_dict_key(data_loader, model, vocab):
    diction = {}
    iterator = enumerate(data_loader, 1)
    with torch.no_grad():
        for i, batch in iterator:
            batch = batch[:-1]
            batch = list(map(lambda x: x.to(device), batch))
            (inp_src, out_src, inp_trg, out_trg,
             src_lengths, trg_lengths) = batch
            for seq, r_list in zip(inp_src, model.idf(inp_src)):
                for p, r in zip(seq, r_list):
                    diction[vocab.id2tok[p.item()]] = r.item()
    return diction


def compress_seq3(data_loader, max_ratio, model, vocab, region, key_dict, score):
    results = []
    batch_eval_loss = []
    batch_eval_semantic_loss = []
    # 平均每个点的重要程度
    key_info = []
    time_sum = 0
    delta = []
    acc_tlen = []
    iterator = enumerate(data_loader, 1)
    with torch.no_grad():
        for i, batch in iterator:
            batch_oov_map = batch[-1]
            batch = batch[:-1]

            batch = list(map(lambda x: x.to(device), batch))
            (inp_src, out_src, inp_trg, out_trg,
             src_lengths, trg_lengths) = batch

            trg_lengths = torch.clamp(src_lengths * max_ratio, min=9, max=30)
            trg_lengths = torch.floor(trg_lengths).int()

            m_zeros = torch.zeros(inp_src.size(0), vocab.size).to(inp_src)
            mask_matrix = m_zeros.scatter(1, inp_src, 1)
            # mask_matrix = None

            tic1 = time.perf_counter()
            outputs = model(inp_src, inp_trg, src_lengths, trg_lengths,
                            sampling=0, mask_matrix=mask_matrix, vocab=vocab, region=region)
            tic2 = time.perf_counter()
            time_sum += tic2 - tic1
            enc1, dec1, enc2, dec2 = outputs
            loss, compTrj = get_sed_loss(vocab, region, inp_src, dec1)
            semantic_loss = sematic_cal(model, inp_src, dec1, src_lengths, trg_lengths,vocab)
            # print(semantic_loss)
            for trj, tlen, srcL, ll in zip(compTrj, trg_lengths, src_lengths, loss):
                n = len(trj)
                acc_tlen.append(n)
                results.append([ll, trj])
                key = 0
                for p in trj:
                    key += key_dict[p]
                # if key < 0:
                #     print()
                key_info.append(key / n)
                # print(tlen)
                delta.append(tlen.item() / n)

                ratio = round(n / srcL.item(), 1)
                if ratio not in score:
                    # 当前压缩率下对应的：loss、关键点程度、个数
                    score[ratio] = [ll, key / n, 1]
                else:
                    score[ratio][0] += ll
                    score[ratio][1] += key / n
                    score[ratio][2] += 1

            batch_eval_loss.append(np.mean(loss))
            batch_eval_semantic_loss.append(semantic_loss)

    print(f"压缩率 {max_ratio},耗时 {time_sum},误差 {np.mean(batch_eval_loss)},关键程度 {np.mean(key_info)},语义相似度 {np.mean(batch_eval_semantic_loss)},失真 {np.mean(delta)}")
    return acc_tlen, results, score


# main
# path = os.path.join(BASE_DIR, "checkpoints/best_model")
path = None
checkpoint = "seq3.full_-prior"
seed = 1
device = "cuda"
verbose = True
out_file = ""
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

src_file = os.path.join(DATA_DIR, "eval.src")
with open('../preprocess/pickle.txt', 'rb') as f:
    var_a = pickle.load(f)
region = pickle.loads(var_a)

data_loader, model, vocab = load_model(path, checkpoint, src_file, device)
d = load_dict_key(data_loader, model, vocab)

score = {}
score_adp = {}
score_squish = {}
src_tlen = []
all_error = []
range_ = range(3,4)
for ratio in range_:
    tlen, results, score = compress_seq3(data_loader, ratio / 10, model, vocab, region, d, score)
    src_tlen.append(tlen)
    all_error.append(results)
for x in score.items():
    rt = x[0]
    err_sum = x[1][0]
    key = x[1][1]
    num = x[1][2]
    print(rt, err_sum / num, key / num)

print()
points, src = readData(src_file, region)

exp = []
for ratio, tlen, AError in zip(range_, src_tlen, all_error):
    res, score_adp, score_squish = compress_adp(src, points, ratio / 10, d, tlen, AError, score_adp, score_squish,model, vocab)
    exp.append(res)

# for x in score_adp.items():
#     rt = x[0]
#     err_sum = x[1][0]
#     key = x[1][1]
#     num = x[1][2]
#     print(rt, err_sum / num, key / num)
# print()
# for x in score_squish.items():
#     rt = x[0]
#     err_sum = x[1][0]
#     key = x[1][1]
#     num = x[1][2]
#     print(rt, err_sum / num, key / num)
# print()
# exp_squish = []
# for ratio, tlen, AError in zip(range_, src_tlen, all_error):
#     res = compress_squish(src, points, ratio / 10, d, tlen, AError)
#     exp_squish.append(res)

# save_path = "../datasets/exp"
# with open(save_path, 'w') as f:
#     for item in exp:
#         for i in item:
#             f.write(i)
