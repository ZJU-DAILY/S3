import os
from random import sample
import sys

sys.path.append('/home/hch/Desktop/trjcompress/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracemalloc import start
import numpy as np

from generate.online.OnlineCED import CEDer
from generate.utils import sed_loss
from models.constants import BOS

import pickle
import time

from modules.helpers import sequence_mask
from preprocess.SpatialRegionTools import cell2gps, cell2meters
from sys_config import DATA_DIR

import torch
from torch.utils.data import DataLoader

from modules.data.collates import Seq2SeqOOVCollate
from modules.data.datasets import AEDataset
from modules.models import Seq2Seq2Seq
from utils.training import load_checkpoint


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
    checkpoint['config']['data']['seq_len'] = 1024
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
    data_loader = DataLoader(dataset, batch_size=1,
                             num_workers=0, collate_fn=Seq2SeqOOVCollate())
    n_tokens = len(dataset.vocab)

    model = Seq2Seq2Seq(n_tokens, **config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    model.compressor.layer_norm = False

    ##############################################

    return data_loader, model, vocab


def compress_seq3_online(data_loader, max_ratio, model, vocab, region, metric):
    timelist = []
    batch_eval_metric_loss_seq3 = []
    iterator = enumerate(data_loader, 1)
    with torch.no_grad():
        for i, batch in iterator:

            print(f"batch {i}")
            # if i == 10:
            #     break
            batch = batch[:-1]

            batch = list(map(lambda x: x.to(device), batch))
            (inp_src, out_src, inp_trg, out_trg,  # 1,L
             src_lengths, trg_lengths) = batch

            trg_lengths = torch.clamp(src_lengths * max_ratio, min=2, max=25)
            trg_lengths = torch.floor(trg_lengths).int()

            m_zeros = torch.zeros(inp_src.size(0), vocab.size).to(inp_src)
            mask_matrix = m_zeros.scatter(1, inp_src, 1)
            mask_matrix[:, 0] = 0
            batch = (inp_src, out_src, inp_trg, out_trg,  # 1,L
                     src_lengths, trg_lengths)

            _, dec1, _, _ = stream4batch(batch, model, mask_matrix, vocab, region, timelist)

            for src_vid, _src_len, comp_vid, _trg_len in zip(inp_src, src_lengths, dec1[3].max(-1)[1], trg_lengths):

                comp_vid = comp_vid[:_trg_len].tolist()
                if 0 in comp_vid:
                    print("comp_vid has zero,so we rollback...")
                    continue
                src_vid = src_vid[:_src_len]
                complen = len(comp_vid)

                comp_gid = [vocab.id2tok[p] for p in comp_vid]
                src_gid = [vocab.id2tok[p.item()] for p in src_vid]
                try:
                    # points = [cell2meters(region, int(p)) for p in src_vid]
                    points = [cell2gps(region, int(p)) for p in src_gid]
                    # print("Our: \n",points)
                    # Tea
                    mp_src = {num: i for i, num in enumerate(src_gid)}
                    comp_sort_gid = comp_gid.copy()
                    # comp_sort_gid = getCompress(region, src_gid, comp_gid)[0]
                    comp_sort_gid.sort(key=lambda j: mp_src[j])
                    if src_gid[-1] not in comp_sort_gid:
                        comp_sort_gid.append(src_gid[-1])
                        comp_vid.append(src_vid[-1])
                        complen = len(comp_sort_gid)
                    if src_gid[0] not in comp_sort_gid:
                        comp_sort_gid.insert(0, src_gid[0])
                        comp_vid.insert(0, src_vid[0])
                        complen = len(comp_sort_gid)

                    if metric == "ss":
                        # s_loss_seq3 = sematic_simp(model, src_vid, comp_vid, vocab)
                        idx = [src_gid.index(i) for i in comp_sort_gid]
                        s_loss_seq3 = ceder.CED_op(idx, src_gid)
                    else:
                        s_loss_seq3 = sed_loss(region, src_gid, comp_sort_gid, metric)

                except Exception as e:
                    print("exception occured")
                    continue

                batch_eval_metric_loss_seq3.append(s_loss_seq3)

    print(np.mean(batch_eval_metric_loss_seq3))
    return np.mean(timelist)


def stream4batch(batch, model, mask_matrix, vocab, region, timelist):
    inp_src, out_src, inp_trg, out_trg, src_lengths, trg_lengths = batch
    threshold = trg_lengths
    cache_h = None
    cache_out = None
    cache_id = None
    sos_id = torch.tensor(BOS).view(1, 1).to(inp_src)  # id of <SOS>
    is_stream = False

    batch, max_length = inp_src.size()
    start_time = time.time()
    for i in range(src_lengths):
        if i + 1 < threshold:
            cache_id = inp_src[:, :i + 1]

        else:
            input_id = inp_src[:, i].unsqueeze(0)
            cache_id = torch.cat([cache_id, input_id], dim=-1)
            cache_id_trg = torch.cat([sos_id, cache_id], dim=-1)
            _src_length = torch.tensor([cache_id.shape[1]]).to(src_lengths)
            _trg_length = torch.tensor([cache_id_trg.shape[1]]).to(trg_lengths)

            outputs = model.generate_online(cache_id, cache_id_trg,
                                            _src_length,
                                            _trg_length,
                                            vocab=vocab,
                                            region=region,
                                            stream=is_stream,
                                            Cache_h=cache_h,
                                            Cache_out=cache_out,
                                            mask_matrix=mask_matrix)

            excute_time = time.time() - start_time
            # print(excute_time)
            timelist.append(excute_time / (i + 1))
            _, _, cache_h, cache_out = outputs
            is_stream = True
            break
    return outputs


path = None
seed = 1
device = "cuda"
verbose = True
out_file = ""
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
metric = "dad"
datasets = "eval"

checkpoint = "seq3.full_-ped-tdrive"
# checkpoint = "seq3.full_-ped"

src_file = os.path.join(DATA_DIR, datasets + ".src")
with open(os.path.join(DATA_DIR, 'pickle.txt'), 'rb') as f:
    var_a = pickle.load(f)
region = pickle.loads(var_a)

data_loader, model, vocab = load_model(path, checkpoint, src_file, device)
# ceder = CEDer(datasets="eval", checkpoint="seq3.full_-ped-tdrive")
ceder = None
# --------------------------------------------------------------------
# 1-5对应90%-50%的压缩率
range_ = range(1, 6)
for ratio in range_:
    print(f"压缩率: {ratio / 10} \n------------------------")
    head = f"压缩率: {ratio / 10} \n------------------------\n"
    res = compress_seq3_online(data_loader, ratio / 10, model, vocab, region, metric)
    print(res)
