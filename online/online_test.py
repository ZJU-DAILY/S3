import os
from random import sample
import sys
from tracemalloc import start
import numpy as np
from models.constants import BOS

sys.path.append('/home/hch/Desktop/trjcompress/modules/')
sys.path.append('/home/hch/Desktop/trjcompress/RL/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    ##############################################

    return data_loader, model, vocab


def compress_seq3_online(data_loader, max_ratio, model, vocab, region, metric):
    timelist = []

    iterator = enumerate(data_loader, 1)
    with torch.no_grad():
        for i, batch in iterator:

            # print(f"batch {i}")
            if i == 10:
                break
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
            start_time = time.time()

            stream4batch(batch, model, mask_matrix, vocab, region, timelist)
            excute_time = time.time() - start_time
            timelist.append(excute_time / trg_lengths[0].item())

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
    for i in range(src_lengths):
        if i + 1 < threshold:
            cache_id = inp_src[:, :i + 1]

        else:
            input_id = inp_src[:, i].unsqueeze(0)
            cache_id = torch.cat([cache_id, input_id], dim=-1)
            cache_id_trg = torch.cat([sos_id, cache_id], dim=-1)
            _src_length = torch.tensor([cache_id.shape[1]]).to(src_lengths)
            _trg_length = torch.tensor([cache_id_trg.shape[1]]).to(trg_lengths)
            # start_time = time.time()
            outputs = model.generate_online(cache_id, cache_id_trg,
                                            _src_length,
                                            _trg_length,
                                            vocab=vocab,
                                            region=region,
                                            stream=is_stream,
                                            Cache_h=cache_h,
                                            Cache_out=cache_out,
                                            mask_matrix=mask_matrix)

            # excute_time = time.time() - start_time
            # print(excute_time)
            # timelist.append(excute_time)
            _, _, cache_h, cache_out = outputs
            is_stream = True
            break


path = None
seed = 1
device = "cuda"
verbose = True
out_file = ""
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
metric = "ss"
datasets = "infer_small"

checkpoint = "seq3.full_-ped"

src_file = os.path.join(DATA_DIR, datasets + ".src")
with open('../preprocess/pickle.txt', 'rb') as f:
    var_a = pickle.load(f)
region = pickle.loads(var_a)

data_loader, model, vocab = load_model(path, checkpoint, src_file, device)
# --------------------------------------------------------------------
# 1-5对应90%-50%的压缩率
range_ = range(1, 6)
for ratio in range_:
    print(f"压缩率: {ratio / 10} \n------------------------")
    head = f"压缩率: {ratio / 10} \n------------------------\n"
    res = compress_seq3_online(data_loader, ratio / 10, model, vocab, region, metric)
    print(res)