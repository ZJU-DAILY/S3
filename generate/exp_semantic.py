# -*- coding: utf-8 -*-
# @Time    : 2022/5/13 20:18
# @Author  : HeAlec
# @FileName: analyse.py
# @Desc: Analyse the exp result
# @blogs ：https://segmentfault.com/u/alec_5e8962e4635ca

import os
import sys
import csv

sys.path.append('/home/hch/Desktop/trjcompress/modules/')
sys.path.append('/home/hch/Desktop/trjcompress/RL/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import time

from modules.helpers import sequence_mask
from generate.utils import getCompress, sed_loss, sematic_simp

from preprocess.SpatialRegionTools import cell2gps, cell2meters
from sys_config import DATA_DIR

import torch
from torch.utils.data import DataLoader

from modules.data.collates import Seq2SeqOOVCollate
from modules.data.datasets import AEDataset
from modules.models import Seq2Seq2Seq
from utils.training import load_checkpoint
import numpy as np
from generate.batch.algorithm import adp, error_search_algorithm, bellman, btup

from models.constants import minerr

from RL.rl_env_inc import TrajComp
from RL.rl_brain import PolicyGradient
from RL.data_utils import dad_op, dad_error
from generate.batch.spanSearch import span_search, error_search
import wandb


# import torch_tensorrt

def RL_algorithm(buffer_size, episode):
    steps, observation = env.reset(episode, buffer_size)
    for index in range(buffer_size, steps):
        if index == steps - 1:
            done = True
        else:
            done = False
        action = RL.quick_time_action(observation)  # matrix implementation for fast efficiency when the model is ready
        observation_, _ = env.step(episode, action, index, done, 'V')  # 'T' means Training, and 'V' means Validation
        observation = observation_
    idx, max_err = env.output(episode, 'V')
    if idx[-1] == steps:
        idx, max_err = env.output(episode, 'V')
    return idx, max_err


# def sematic_cal(model, inp, dec1, src_lengths, trg_lengths, vocab):
#     enc_mask = sequence_mask(src_lengths).unsqueeze(-1).float()
#     dec_mask = sequence_mask(trg_lengths - 1).unsqueeze(-1).float()
#
#     enc_embs = model.inp_encoder.embed(inp, vocab) * enc_mask
#     if dec1[3] is None:
#         print("dec1[3] is none")
#     # dec_embs = model.compressor.embed.expectation(dec1[3], vocab) * dec_mask
#
#     dec_embs = model.compressor.embed(dec1[3].max(-1)[1], vocab)
#     # 先对source序列中的点进行遍历
#     max_len = enc_embs.size(1)
#     sim = np.zeros(enc_embs.size(0))
#     for i in range(max_len):
#         batch_p = enc_embs[:, i, :100]
#         batch_trj = dec_embs
#         tmp = r(batch_p, batch_trj)
#         sim += tmp
#     # 计算的是所有batch的
#     sim = [sim[i] / length.item() for i, length in enumerate(src_lengths)]
#     # 将所有batch的结果求平均
#     # sim = np.mean(sim)
#
#     # 接下来对trg_src做类似的处理
#     max_len = dec_embs.size(1)
#     sim_2 = np.zeros(dec_embs.size(0))
#     for i in range(max_len):
#         batch_p = dec_embs[:, i, :100]
#         batch_trj = enc_embs
#         tmp = r(batch_p, batch_trj)
#         sim_2 += tmp
#     # 计算的是所有batch的
#     sim_2 = [sim_2[i] / length.item() for i, length in enumerate(trg_lengths)]
#     # 将所有batch的结果求平均
#     # sim_2 = np.mean(sim_2)
#     # print(sim,sim_2)
#     return (np.array(sim) + np.array(sim_2)) / 2


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
    # checkpoint['config']['data']['seq_len'] = 1024
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
    # trt_model = torch_tensorrt.compile(model,
    #                                    inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
    #                                    enabled_precisions={torch_tensorrt.dtype.half}  # Run with FP16
    #                                    )
    model.load_state_dict(checkpoint["model"])

    # model.compressor.attention_used = False
    # model.compressor.coverage = False
    # model.decompressor.attention_used = False
    # model.decompressor.coverage = False

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
    vis_res = ""

    batch_eval_loss = []
    batch_eval_metric_loss_seq3 = []
    batch_eval_metric_loss_adp = []
    batch_eval_metric_loss_btup = []
    batch_eval_metric_loss_error_search = []
    batch_eval_metric_loss_RL = []
    batch_eval_metric_loss_bellman = []
    batch_eval_metric_loss_span_search = []
    # 平均每个点的重要程度

    time_seq3 = 0
    time_adp = 0
    time_btup = 0
    time_error_search = 0
    time_RL = 0
    time_bellman = 0
    time_res = ""

    iterator = enumerate(data_loader, 1)
    time_list = []
    rollback_time_list = []
    ik = 0
    with torch.no_grad():
        for i, batch in iterator:
            # if i == 20:
            #     break
            print(f"batch {i}")
            batch = batch[:-1]

            batch = list(map(lambda x: x.to(device), batch))
            (inp_src, out_src, inp_trg, out_trg,
             src_lengths, trg_lengths) = batch

            trg_lengths = torch.clamp(src_lengths * max_ratio, min=5, max=25)
            # trg_lengths = torch.clamp(src_lengths * max_ratio, min=min(5, int((src_lengths.max().item() - 50) * 0.5)),
            #                           max=int((src_lengths.max().item() + 50) * 0.5))
            trg_lengths = torch.floor(trg_lengths).int()

            m_zeros = torch.zeros(inp_src.size(0), vocab.size).to(inp_src)
            mask_matrix = m_zeros.scatter(1, inp_src, 1)
            mask_matrix[:, 0] = 0

            rollback_time_list = time_list.copy()
            outputs = model(inp_src, inp_trg, src_lengths, trg_lengths,
                            sampling=0, mask_matrix=mask_matrix, vocab=vocab, region=region,
                            decoder_time_list=time_list)

            enc1, dec1, enc2, dec2 = outputs

            for src_vid, _src_len, comp_vid, _trg_len in zip(inp_src, src_lengths, dec1[3].max(-1)[1], trg_lengths):

                comp_vid = comp_vid[:_trg_len].tolist()
                if 0 in comp_vid:
                    print("comp_vid has zero,so we rollback...")
                    time_list = rollback_time_list.copy()
                    ik += 1
                    continue
                src_vid = src_vid[:_src_len]
                complen = len(comp_vid)

                comp_gid = [vocab.id2tok[p] for p in comp_vid]
                src_gid = [vocab.id2tok[p.item()] for p in src_vid]
                try:
                    # points = [cell2meters(region, int(p)) for p in src_vid]
                    points = [cell2gps(region, int(p)) for p in src_gid]
                    # print("Our: \n",points)
                    mp_src = {num: i for i, num in enumerate(src_gid)}
                    comp_sort_gid = comp_gid.copy()
                    # comp_sort_gid = getCompress(region, src_gid, comp_gid)[0]
                    comp_sort_gid.sort(key=lambda j: mp_src[j])
                    if src_gid[-1] not in comp_sort_gid:
                        comp_sort_gid.append(src_gid[-1])
                        complen = len(comp_sort_gid)
                    if src_gid[0] not in comp_sort_gid:
                        comp_sort_gid.insert(0, src_gid[0])
                        complen = len(comp_sort_gid)
                    # Tea
                    if metric == "ss":
                        s_loss_seq3 = sematic_simp(model, src_vid, comp_vid, vocab)
                    else:
                        s_loss_seq3 = sed_loss(region, src_gid, comp_sort_gid, metric)
                    src_gid_str = [str(i) for i in src_gid]
                    comp_sort_gid_str = [str(i) for i in comp_sort_gid]
                    vis_res += " ".join(src_gid_str) + "\n"
                    vis_res += " ".join(comp_sort_gid_str) + "\n"
                except Exception as e:
                    print("exception occured")
                    # time_list = rollback_time_list
                    ik += 1
                    continue
                # time_seq3 += time_list[ik]
                batch_eval_metric_loss_seq3.append(s_loss_seq3)

                # tdtr，一个完整的算法运算和获取

                if metric == "ss":
                    tic1 = time.perf_counter()
                    _, idx, maxErr = adp(points, complen, 'sed')
                    tic2 = time.perf_counter()
                    time_adp += tic2 - tic1
                    idx.sort()
                    comp_vid_adp = [src_vid[i].item() for i in idx]
                    # s_loss_adp = sematic_simp(model, src_vid, comp_vid_adp, vocab)
                else:
                    tic1 = time.perf_counter()
                    _, idx, maxErr = adp(points, complen, metric)
                    tic2 = time.perf_counter()
                    time_adp += tic2 - tic1
                    s_loss_adp = maxErr
                comp_sort_gid_str = [str(src_gid[i]) for i in idx]
                vis_res += " ".join(comp_sort_gid_str) + "\n"
                batch_eval_metric_loss_adp.append(s_loss_adp)

                # # error-search
                if metric == "ss":
                    tic1 = time.perf_counter()
                    idx, maxErr = error_search(points, complen)
                    tic2 = time.perf_counter()
                    time_error_search += tic2 - tic1
                    comp_vid_ersh = [src_vid[i].item() for i in idx]
                    # s_loss_ersh = sematic_simp(model, src_vid, comp_vid_ersh, vocab)
                else:
                    tic1 = time.perf_counter()
                    idx, maxErr = error_search(points, complen)
                    tic2 = time.perf_counter()
                    time_error_search += tic2 - tic1
                    s_loss_ersh = maxErr
                o = 0
                while len(set(idx)) < complen:
                    idx.append((idx[o] + idx[o + 2]) // 2)
                    o += 1
                idx = list(set(idx))
                idx.sort()
                # print(idx, len(src_gid))
                comp_sort_gid_str = [str(src_gid[min(i, len(src_gid) - 1)]) for i in idx]
                vis_res += " ".join(comp_sort_gid_str) + "\n"
                batch_eval_metric_loss_error_search.append(s_loss_ersh)

                # # bottom-up
                if metric == "ss":
                    tic1 = time.perf_counter()
                    _, idx, maxErr = btup(points, complen, 'sed')
                    tic2 = time.perf_counter()
                    time_btup += tic2 - tic1
                    comp_vid_btup = [src_vid[i].item() for i in idx]

                    # s_loss_btup = sematic_simp(model, src_vid, comp_vid_btup, vocab)
                else:
                    tic1 = time.perf_counter()
                    _, idx, maxErr = btup(points, complen, metric)
                    tic2 = time.perf_counter()
                    time_btup += tic2 - tic1
                    s_loss_btup = maxErr
                idx.sort()
                comp_sort_gid_str = [str(src_gid[i]) for i in idx]
                vis_res += " ".join(comp_sort_gid_str) + "\n"
                batch_eval_metric_loss_btup.append(s_loss_btup)

                # RLOnline
                if metric == "ss":
                    tic1 = time.perf_counter()
                    idx, maxErr = RL_algorithm(complen, ik)
                    tic2 = time.perf_counter()
                    time_RL += tic2 - tic1
                    idx.sort()
                    comp_vid_rl = [src_vid[i].item() for i in idx]
                    # s_loss_rl = sematic_simp(model, src_vid, comp_vid_rl, vocab)
                else:
                    tic1 = time.perf_counter()
                    idx, maxErr = RL_algorithm(complen, ik)
                    tic2 = time.perf_counter()
                    time_RL += tic2 - tic1
                    s_loss_rl = maxErr
                comp_sort_gid_str = [str(src_gid[i]) for i in idx]
                vis_res += " ".join(comp_sort_gid_str) + "\n"
                batch_eval_metric_loss_RL.append(s_loss_rl)

                # # bellman
                # if metric == "ss":
                #     tic1 = time.perf_counter()
                #     # idx, maxErr = bellman(points, complen, 'sed')
                #     tic2 = time.perf_counter()
                #     time_bellman += tic2 - tic1
                #     # comp_vid_bell = [src_vid[i].item() for i in idx]
                #     # s_loss_bell = sematic_simp(model, src_vid, comp_vid_bell, vocab)
                # else:
                #     tic1 = time.perf_counter()
                #     # idx, maxErr = bellman(points, complen, 'sed')
                #     tic2 = time.perf_counter()
                #     time_bellman += tic2 - tic1
                #     s_loss_bell = maxErr
                # batch_eval_metric_loss_bellman.append(s_loss_bell)

                # # span-search
                # if metric == "ss":
                #     tic1 = time.perf_counter()
                #     # idx, maxErr = span_search(points, complen)
                #     tic2 = time.perf_counter()
                #     time_bellman += tic2 - tic1
                #     # comp_vid_bell = [src_vid[i].item() for i in idx]
                #     # s_loss_span_search = sematic_simp(model, src_vid, comp_vid_bell, vocab)
                # else:
                #     tic1 = time.perf_counter()
                #     idx, maxErr = span_search(points, complen)
                #     tic2 = time.perf_counter()
                #     time_bellman += tic2 - tic1
                #     s_loss_span_search = maxErr
                # batch_eval_metric_loss_span_search.append(s_loss_span_search)

                ik += 1
            if (i + 1) % 10 == 0:
                time_res += f"Tea {np.sum(time_list)} tdtr {time_adp} errSea {time_error_search} btup {time_btup} rl {time_RL} bell {time_bellman}\n"

            # batch_eval_loss.append(np.mean(loss))

    # print(f"压缩率 {max_ratio},耗时 {time_sum},误差 {np.mean(batch_eval_loss)},关键程度 {np.mean(key_info)},语义相似度 {np.mean(batch_eval_semantic_loss_seq3)},失真 {np.mean(delta)}")
    # print(f"Tea\t|\t推理用时:\t{np.sum(time_list)}\t|\t{metric}:\t{np.mean(batch_eval_metric_loss_seq3)}")
    # print(f"TDTR\t|\t推理用时:\t{time_adp}\t|\t{metric}:\t{np.mean(batch_eval_metric_loss_adp)}")
    # print(f"Error Search\t|\t推理用时:\t{time_error_search}\t|\t{metric}:\t{np.mean(batch_eval_metric_loss_error_search)}")
    # print(f"Bottom up\t|\t推理用时:\t{time_btup} |\t{metric}:\t{np.mean(batch_eval_metric_loss_btup)}")
    # print(f"RLOnline\t|\t推理用时:\t{time_RL} |\t{metric}:\t{np.mean(batch_eval_metric_loss_RL)}")
    # print(f"bellman\t|\t推理用时:\t{time_bellman} |\t{metric}:\t{np.mean(batch_eval_metric_loss_bellman)}")
    #

    res = f"Tea\t|\t推理用时:\t{np.sum(time_list)}\t|\t{metric}:\t{np.mean(batch_eval_metric_loss_seq3)}\n" \
          f"TDTR\t|\t推理用时:\t{time_adp}\t|\t{metric}:\t{np.mean(batch_eval_metric_loss_adp)}\n" \
          f"Error Search\t|\t推理用时:\t{time_error_search}\t|\t{metric}:\t{np.mean(batch_eval_metric_loss_error_search)}\n" \
          f"Bottom up\t|\t推理用时:\t{time_btup} |\t{metric}:\t{np.mean(batch_eval_metric_loss_btup)}\n" \
          f"RLOnline\t|\t推理用时:\t{time_RL} |\t{metric}:\t{np.mean(batch_eval_metric_loss_RL)}\n" \
          f"bellman\t|\t推理用时:\t{time_bellman} |\t{metric}:\t{np.mean(batch_eval_metric_loss_bellman)}\n" \
          f"span_search\t|\t推理用时:\t{time_bellman} |\t{metric}:\t{np.mean(batch_eval_metric_loss_span_search)}\n"
    print(res)
    print(f"{time_btup}\n{time_adp}\n{time_RL}\n{np.sum(time_list)}")
    with open(f"../datasets/{metric}/exp_2_{ratio}", "w") as f:
        f.write(vis_res)

    csv_writer.writerow([time_btup, time_adp, time_RL, np.sum(time_list)])
    return time_res + res


if __name__ == '__main__':
    path = None
    seed = 1
    device = "cpu"
    verbose = True
    out_file = ""
    torch.manual_seed(seed)
    # wandb.login(key='071c6a599f6f404d5fc01850bf0e47f6fd660bf4')
    # wandb.init(project="S3", entity="healec")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # metrics = ['ped','sed','ss']
    # datasets = ['geolife','tdrive']
    #
    # for meritc in metrics:
    #     for dataset in datasets:
    #         checkpoint = "seq3.full_" + dataset + "-" + meritc
    metric = sys.argv[1]
    datasets = sys.argv[2]

    if metric == 'ped' or metric == 'dad':
        # checkpoint = "seq3.full_-ped-tdrive"
        # checkpoint = "seq3.full_-sed-tdrive"
        checkpoint = "seq3.full_-ped"
    elif metric == 'sed':
        # checkpoint = "seq3.full_-ped-tdrive"
        # checkpoint = "seq3.full_-sed-tdrive"
        # checkpoint = "seq3.full_-sed-tdrive-train"
        checkpoint = "seq3.full_-sed"
        # checkpoint = "seq3.full_-noAttn"
    elif metric == 'ss':
        # checkpoint = "seq3.full_-noAttn"
        checkpoint = "seq3.full_-valid"
        # checkpoint = "seq3.full_-ped-tdrive"
        # checkpoint = "seq3.full_-noGraph"

    src_file = os.path.join(DATA_DIR, datasets + ".src")
    with open(os.path.join(DATA_DIR, 'pickle.txt'), 'rb') as f:
        var_a = pickle.load(f)
    region = pickle.loads(var_a)

    data_loader, model, vocab = load_model(path, checkpoint, src_file, device)

    # ----------------------------------------------------------------
    # RLOnline model init
    traj_path = src_file
    test_amount = 1000
    elist = [i for i in range(test_amount)]
    a_size = 3  # RLTS 3, RLTS-Skip 5
    s_size = 3  # RLTS 3, RLTS-Skip 5
    ratio = 0.1
    if metric == 'ss':
        env = TrajComp(traj_path, 1000, region, a_size, s_size, 'sed')
    else:
        env = TrajComp(traj_path, 1000, region, a_size, s_size, metric)
    RL = PolicyGradient(env.n_features, env.n_actions)
    RL.load('../RL/save/0.00039190653824900003_ratio_0.1/')  # your_trained_model your_trained_model_skip

    # --------------------------------------------------------------------
    file = open(f"../experiments/result_{checkpoint}_{metric}_{datasets}_x_1218_time.csv", 'a+', encoding='utf-8',
                newline='')
    range_ = range(1, 6)
    csv_writer = csv.writer(file)
    csv_writer.writerow([f'v1', 'v2', 'v3', 'v4', 'v5'])
    for ratio in range_:
        # csv_writer.writerow(['','','',''])
        print(f"压缩率: {ratio / 10} \n------------------------")
        head = f"压缩率: {ratio / 10} \n------------------------\n"
        res = compress_seq3(data_loader, ratio / 10, model, vocab, region, metric)
    file.close()
    # with open(f"../experiments/result_{checkpoint}_{metric}_{datasets}_x_0818_time", "a") as f:
    #     # 1-5对应90%-50%的压缩率
    #     range_ = range(1, 6)
    #     for ratio in range_:
    #         print(f"压缩率: {ratio / 10} \n----------x--------------")
    #         head = f"压缩率: {ratio / 10} \n------------------------\n"
    #         res = compress_seq3(data_loader, ratio / 10, model, vocab, region, metric)
    #         f.write(head + res + "\n")
