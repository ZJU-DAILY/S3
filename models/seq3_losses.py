import torch
from torch.nn import functional as F
from modules.helpers import sequence_mask, getSED, adp, id2gps

import numpy as np


def _kl_div(inp_logits, trg_logits, lengths, tau=1):
    """
    Compute the prior loss using a pretrained "oracle" LM.
    The loss is computed using the produced posteriors over the vocabulary
    produced by a generator and the posteriors of the "oracle" LM.

    Args:
        logits: the logits of the generator
        words: the argmax of the logits
        oracle: the oracle LM
        tau: the temperature of the softmax
        lengths: the lengths of the target sequence. Used for masking the loss.


    Debug = -F.softmax(_logits, -1) * torch.log(F.softmax(logits, -1) /
                                                F.softmax(_logits, -1))

    Returns:
        the average KL Divergence per timestep (word)

    """
    mask = sequence_mask(lengths).unsqueeze(-1).float()

    input_logp = F.log_softmax(inp_logits * mask / tau, -1)
    target_p = F.softmax(trg_logits * mask / tau, -1)

    # shape: batch x seq_length x tokens
    loss = F.kl_div(input_logp, target_p, reduction='none')

    # sum over words/vocab (KL per word/timestep !)
    # shape: batch x length
    loss = loss.sum(-1)

    # zero losses for padded timesteps
    loss = loss * mask.squeeze()

    total_loss = loss.sum() / mask.sum()

    return total_loss, loss


def _global_prior(logits, word_idx, lengths):
    """
    Evaluate the probability of a sequence, under a language model

    """

    mask = sequence_mask(lengths)
    labels = (word_idx * mask.long()).contiguous().view(-1)
    _logits = logits.contiguous().view(-1, logits.size(-1))
    loss = F.cross_entropy(_logits, labels, ignore_index=0, reduction='none')

    # normalize by length to avoid mode collapse
    total = loss.sum() / mask.float().sum()

    return total, loss.view(mask.size())


def kl_length(logits, lengths, eos):
    """
    Length control loss, using a sequence of length labels (with eos token).

    Args:
        logits:
        lengths:
        eos:

    Returns:

    """
    mask = sequence_mask(lengths - 1, lengths.max())
    eos_labels = ((~mask) * eos).long().contiguous().view(-1)

    _logits = logits.contiguous().view(-1, logits.size(-1))
    loss = F.cross_entropy(_logits, eos_labels, ignore_index=0)
    anti_loss = F.cross_entropy(_logits, eos_labels, ignore_index=2)

    return loss / (anti_loss + 1.0)


def pairwise_loss(a, b, dist="cosine"):
    if dist == "euclidean":
        return F.pairwise_distance(a, b).mean()
    elif dist == "cosine":
        return 1 - F.cosine_similarity(a, b).mean()
    elif dist == "dot":
        dot = torch.bmm(a.unsqueeze(1), b.unsqueeze(-1)).squeeze()
        scaled_dot = dot.mean() / a.size(1)
        return - scaled_dot
    else:
        raise ValueError


def sed_loss(region, src, trg):
    if len(src) == len(trg):
        print("the length of src is same with trg.")
    # p为慢指针（指向trg），f为快指针（指向src）。src的长度应该大于等于trg
    p = 0
    f = 0
    maxSED = -1
    while p < len(trg) and f < len(src):
        if src[f] == '' or src[f] == 'UNK':
            f += 1
            continue
        if trg[p] == src[f]:
            p += 1
            f += 1
        else:
            st = trg[p - 1]
            en = trg[p]
            while trg[p] != src[f]:
                if src[f] == '' or src[f] == 'UNK':
                    f += 1
                    continue
                in_ = src[f]
                dis = getSED(region, int(in_), int(st), int(en))
                maxSED = max(maxSED, dis)
                f += 1
    if maxSED == -1:
        print("errrr")
        maxSED = 0
    return maxSED


def energy_(region, inp, size_1):
    enc1_energies = []
    for trj in inp:
        ener = [0] * size_1
        for i in range(1, len(trj) - 1):
            # window取到1
            p = int(trj[i])
            st = int(trj[i - 1])
            if trj[i + 1] == '':
                break
            en = int(trj[i + 1])
            pri = getSED(region, p, st, en)
            # window取到2
            st = int(trj[i - 2 if i - 2 > 0 else 0])
            if trj[i + 1 if i + 1 < len(trj) - 1 else len(trj) - 1] == '':
                break
            en = int(trj[i + 1 if i + 1 < len(trj) - 1 else len(trj) - 1])
            pri2 = getSED(region, p, st, en)

            ener[i] = (pri + pri2) / 2
        enc1_energies.append(ener)
    return enc1_energies

def energy_2(region, inp, size_1,src_lengths):
    enc1_energies = []
    for trj,_len in zip(inp, src_lengths):
        ener = [0] * size_1
        trj = trj[0:_len]
        try:
            _, idx, _ = adp(id2gps(region, trj), _len)
        except Exception as e:
            print(trj)
            idx = []

        for id in idx:
            ener[id] = 1
        enc1_energies.append(ener)
    return enc1_energies