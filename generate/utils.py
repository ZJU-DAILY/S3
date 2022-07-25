import pickle

import math
from itertools import groupby

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.data.collates import Seq2SeqOOVCollate
from modules.data.datasets import AEDataset
from modules.models import Seq2Seq2Seq
from utils.training import load_checkpoint
import numpy as np
from models.seq3_losses import sed_loss
from modules.helpers import getCompress


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


def get_sed_loss(vocab, region, inp, dec1):
    src = devectorize(inp.tolist(), vocab.id2tok, vocab.tok2id[vocab.EOS],
                      strip_eos=None, oov_map=None, pp=True)
    trg = devectorize(dec1[3].max(-1)[1].tolist(), vocab.id2tok, vocab.tok2id[vocab.EOS],
                      strip_eos=None, oov_map=None, pp=True)
    comp_trj = [getCompress(region, src_, trg_)[0] for src_, trg_ in zip(src, trg)]
    # print(comp_trj,src)
    loss = [sed_loss(region, src_, trg_) for src_, trg_ in zip(src, comp_trj)]
    # loss = sed_loss(region, src, comp_trj)

    return loss, comp_trj
