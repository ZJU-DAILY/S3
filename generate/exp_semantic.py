# -*- coding: utf-8 -*-
# @Time    : 2022/5/13 20:18
# @Author  : HeAlec
# @FileName: analyse.py
# @Desc: Analyse the exp result
# @blogs ：https://segmentfault.com/u/alec_5e8962e4635ca

import os
import sys

sys.path.append('/home/hch/Desktop/trjcompress/modules/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import time

from modules.helpers import sequence_mask, getCompress, dp, btup
from preprocess.SpatialRegionTools import cell2gps, cell2meters
from sys_config import DATA_DIR

import torch
from torch.utils.data import DataLoader

from modules.data.collates import Seq2SeqOOVCollate
from modules.data.datasets import AEDataset
from modules.models import Seq2Seq2Seq
from utils.training import load_checkpoint
import numpy as np
from modules.helpers import adp
from models.seq3_losses import r, sed_loss
from models.constants import minerr

def sematic_cal(model, inp, dec1, src_lengths, trg_lengths, vocab):
    enc_mask = sequence_mask(src_lengths).unsqueeze(-1).float()
    dec_mask = sequence_mask(trg_lengths - 1).unsqueeze(-1).float()

    enc_embs = model.inp_encoder.embed(inp, vocab) * enc_mask
    if dec1[3] is None:
        print("dec1[3] is none")
    # dec_embs = model.compressor.embed.expectation(dec1[3], vocab) * dec_mask

    dec_embs = model.compressor.embed(dec1[3].max(-1)[1], vocab)
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
    # sim = np.mean(sim)

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
    # sim_2 = np.mean(sim_2)
    # print(sim,sim_2)
    return (np.array(sim) + np.array(sim_2)) / 2


# src和comp可以都是list
def sematic_simp(model, src, comp, vocab):
    src_len = len(src)
    comp_len = len(comp)

    # src = [vocab.tok2id.get(i, 0) for i in src]
    # comp = [vocab.tok2id.get(i, 0) for i in comp]

    # src = torch.tensor(src).to(device)
    src = src.view(1, src.size(0))

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
    # sampler = BucketBatchSampler(lengths, config["batch_size"])
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


# def cal_metric(metric):
#     if meritc == 'ped':
#         pass
#     elif meritc == 'sed':
#         pass
#     elif meritc == 'ss':
#         pass

def compress_seq3(data_loader, max_ratio, model, vocab, region, metric):
    batch_eval_loss = []
    batch_eval_metric_loss_seq3 = []
    batch_eval_metric_loss_adp = []
    batch_eval_metric_loss_btup = []
    batch_eval_metric_loss_dp = []
    # 平均每个点的重要程度

    time_adp = 0
    time_btup = 0
    time_dp = 0

    iterator = enumerate(data_loader, 1)
    time_list = [0, 0]
    rollback_time_list = [0, 0]

    with torch.no_grad():
        for i, batch in iterator:
            print(f"batch {i}")
            batch = batch[:-1]

            batch = list(map(lambda x: x.to(device), batch))
            (inp_src, out_src, inp_trg, out_trg,
             src_lengths, trg_lengths) = batch

            trg_lengths = torch.clamp(src_lengths * max_ratio, min=9, max=30)
            trg_lengths = torch.floor(trg_lengths).int()

            m_zeros = torch.zeros(inp_src.size(0), vocab.size).to(inp_src)
            mask_matrix = m_zeros.scatter(1, inp_src, 1)

            rollback_time_list = time_list
            outputs = model(inp_src, inp_trg, src_lengths, trg_lengths,
                            sampling=0, mask_matrix=mask_matrix, vocab=vocab, region=region,
                            decoder_time_list=time_list)

            enc1, dec1, enc2, dec2 = outputs

            ik = 0
            for src_vid, _src_len, comp_vid, _trg_len in zip(inp_src, src_lengths, dec1[3].max(-1)[1], trg_lengths):
                comp_vid = comp_vid[:_trg_len].tolist()
                if 0 in comp_vid:
                    comp_vid.remove(0)
                src_vid = src_vid[:_src_len]
                complen = len(comp_vid)

                comp_gid = [vocab.id2tok[p] for p in comp_vid]
                src_gid = [vocab.id2tok[p.item()] for p in src_vid]
                try:
                    # points = [cell2meters(region, int(p)) for p in src_vid]
                    points = [cell2meters(region, int(p)) for p in src_gid]
                    # Tea
                    if metric == "ss":
                        s_loss_seq3 = sematic_simp(model, src_vid, comp_vid, vocab)
                    else:
                        comp_sort_gid = getCompress(region, src_gid, comp_gid)[0]
                        s_loss_seq3 = sed_loss(region, src_gid, comp_sort_gid, metric)
                except Exception as e:
                    print("exception occured")
                    time_list = rollback_time_list
                    continue
                batch_eval_metric_loss_seq3.append(s_loss_seq3)

                # tdtr，一个完整的算法运算和获取

                if metric == "ss":
                    tic1 = time.perf_counter()
                    _, idx, maxErr = adp(points, complen, 'sed')
                    tic2 = time.perf_counter()
                    time_adp += tic2 - tic1
                    comp_vid_adp = [src_vid[i].item() for i in idx]
                    s_loss_adp = sematic_simp(model, src_vid, comp_vid_adp, vocab)
                else:
                    tic1 = time.perf_counter()
                    _, idx, maxErr = adp(points, complen, metric)
                    tic2 = time.perf_counter()
                    time_adp += tic2 - tic1
                    s_loss_adp = maxErr
                batch_eval_metric_loss_adp.append(s_loss_adp)

                # # dp
                # mp_src = {num: i for i, num in enumerate(src_gid)}
                # src_copy = src_gid.copy()
                # tic1 = time.perf_counter()
                # minerr = float('inf')
                # _, idx = dp(src_copy, src_gid, mp_src, complen, metric, region)
                # tic2 = time.perf_counter()
                # time_dp += tic2 - tic1
                # comp_vid_adp = [src_vid[i].item() for i in idx]
                # s_loss_adp = sematic_simp(model, src_vid, comp_vid_adp, vocab)
                #
                # bottom-up
                if metric == "ss":
                    tic1 = time.perf_counter()
                    _, idx, maxErr = btup(points, complen,'sed')
                    tic2 = time.perf_counter()
                    time_btup += tic2 - tic1
                    comp_vid_btup = [src_vid[i].item() for i in idx]
                    s_loss_btup = sematic_simp(model, src_vid, comp_vid_btup, vocab)
                else:
                    tic1 = time.perf_counter()
                    _, idx, maxErr = btup(points, complen, metric)
                    tic2 = time.perf_counter()
                    time_btup += tic2 - tic1
                    s_loss_btup = maxErr
                batch_eval_metric_loss_btup.append(s_loss_btup)


            # batch_eval_loss.append(np.mean(loss))

    # print(f"压缩率 {max_ratio},耗时 {time_sum},误差 {np.mean(batch_eval_loss)},关键程度 {np.mean(key_info)},语义相似度 {np.mean(batch_eval_semantic_loss_seq3)},失真 {np.mean(delta)}")
    print(f"Tea\t|\t推理用时:\t{time_list[0]}\t|\t{metric}:\t{np.mean(batch_eval_metric_loss_seq3)}")
    print(f"TDTR\t|\t推理用时:\t{time_adp}\t|\t{metric}:\t{np.mean(batch_eval_metric_loss_adp)}")
    print(f"dp\t|\t推理用时:\t{time_dp}\t|\t{metric}:\t{np.mean(batch_eval_metric_loss_dp)}")
    print(f"bottom up\t|\t推理用时:\t{time_btup} |\t{metric}:\t{np.mean(batch_eval_metric_loss_btup)}")

    res = f"Tea\t|\t推理用时:\t{time_list[0]}\t|\t{metric}:\t{np.mean(batch_eval_metric_loss_seq3)}\n" \
          f"TDTR\t|\t推理用时:\t{time_adp}\t|\t{metric}:\t{np.mean(batch_eval_metric_loss_adp)}\n" \
          f"dp\t|\t推理用时:\t{time_dp}\t|\t{metric}:\t{np.mean(batch_eval_metric_loss_dp)}\n" \
          f"bottom up\t|\t推理用时:\t{time_btup} |\t{metric}:\t{np.mean(batch_eval_metric_loss_btup)}\n"
    return res


path = None
seed = 1
device = "cuda"
verbose = True
out_file = ""
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# metrics = ['ped','sed','ss']
# datasets = ['geolife','tdrive']
#
# for meritc in metrics:
#     for dataset in datasets:
#         checkpoint = "seq3.full_" + dataset + "-" + meritc
metric = sys.argv[1]
checkpoint = "seq3.full_-ped"

src_file = os.path.join(DATA_DIR, "infer.src")
with open('../preprocess/pickle.txt', 'rb') as f:
    var_a = pickle.load(f)
region = pickle.loads(var_a)

data_loader, model, vocab = load_model(path, checkpoint, src_file, device)
with open(f"../experiments/result_{checkpoint}_{metric}", "w") as f:
    # 1-5对应90%-50%的压缩率
    range_ = range(1, 6)
    for ratio in range_:
        print(f"压缩率: {ratio / 10} \n------------------------")
        head = f"压缩率: {ratio / 10} \n------------------------\n"
        res = compress_seq3(data_loader, ratio / 10, model, vocab, region, metric)
        f.write(head + res + "\n")
