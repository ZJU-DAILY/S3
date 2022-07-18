import functools

import torch
from numpy import mean
from torch.nn import functional as F
from torch.nn.functional import gumbel_softmax
from preprocess.SpatialRegionTools import cell2gps
from sklearn.neighbors import KDTree
import numpy as np
import math


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .unsqueeze(0).expand(batch_size, max_len)
            .lt(lengths.unsqueeze(1)))


def masked_normalization(logits, mask):
    scores = F.softmax(logits, dim=-1)

    # apply the mask - zero out masked timesteps
    masked_scores = scores * mask.float()

    # re-normalize the masked scores
    normed_scores = masked_scores.div(masked_scores.sum(-1, keepdim=True))

    return normed_scores


def masked_mean(vecs, mask):
    masked_vecs = vecs * mask.float()

    mean = masked_vecs.sum(1) / mask.sum(1)

    return mean


def masked_normalization_inf(logits, mask):
    # ~表示取反操作，mask是一个逻辑值
    logits.masked_fill_(~mask, float('-inf'))
    # energies.masked_fill_(1 - mask, -1e18)

    scores = F.softmax(logits, dim=-1)

    return scores


def expected_vecs(dists, vecs):
    flat_probs = dists.contiguous().view(dists.size(0) * dists.size(1),
                                         dists.size(2))
    flat_embs = flat_probs.mm(vecs)
    embs = flat_embs.view(dists.size(0), dists.size(1), flat_embs.size(1))
    return embs


def straight_softmax(logits, tau=1, hard=False, target_mask=None):
    y_soft = F.softmax(logits.squeeze() / tau, dim=1)

    if target_mask is not None:
        y_soft = y_soft * target_mask.float()
        y_soft.div(y_soft.sum(-1, keepdim=True))

    if hard:
        shape = logits.size()
        _, k = y_soft.max(-1)
        y_hard = logits.new_zeros(*shape).scatter_(-1, k.view(-1, 1), 1.0)
        y = y_hard - y_soft.detach() + y_soft
        return y
    else:
        return y_soft


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, target_mask=None):
    r"""
    Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
      logits: `[batch_size, num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd

    Returns:
      Sampled tensor of shape ``batch_size x num_features`` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across features

    Constraints:

    - Currently only work on 2D input :attr:`logits` tensor of shape ``batch_size x num_features``

    Based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    # return None

    shape = logits.size()
    assert len(shape) == 2
    y_soft = torch.nn.functional.gumbel_softmax(logits, tau=tau, eps=eps)

    if target_mask is not None:
        y_soft = y_soft * target_mask.float()
        y_soft.div(y_soft.sum(-1, keepdim=True))

    if hard:
        _, k = y_soft.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = logits.new_zeros(*shape).scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft
    return y


def avg_vectors(vectors, mask, energies=None):
    if energies is None:
        centroid = masked_mean(vectors, mask)
        return centroid, None

    else:
        masked_scores = energies * mask.float()
        normed_scores = masked_scores.div(masked_scores.sum(1, keepdim=True))
        centroid = (vectors * normed_scores).sum(1)
    return centroid, normed_scores


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def module_grad_wrt_loss(optimizers, module, loss, prefix=None):
    loss.backward(retain_graph=True)

    grad_norms = [(n, p.grad.norm().item())
                  for n, p in module.named_parameters() if p.requires_grad]

    if prefix is not None:
        grad_norms = [g for g in grad_norms if g[0].startswith(prefix)]

    mean_norm = mean([gn for n, gn in grad_norms])

    for optimizer in optimizers:
        optimizer.zero_grad()

    return mean_norm


def getSED(region, p, start, end):
    x, y = cell2gps(region, int(p))
    st_x, st_y = cell2gps(region, int(start))
    en_x, en_y = cell2gps(region, int(end))
    #     Ax + By + C = 0
    return getSED4GPS((x, y), (st_x, st_y), (en_x, en_y))


def getSED4GPS(p, start, end):
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
    # SED
    # if st_x == en_x:
    #     return abs(x - st_x)
    # k = (st_y - en_y) / (st_x - en_x)
    # b = en_y - k * en_x
    # sp_y = k * x + b
    # return abs(sp_y - y)


def SEDsimilarity(region, src, trg):
    # p为慢指针（指向trg），f为快指针（指向src）。src的长度应该大于等于trg
    p = 0
    f = 0
    idx = -1
    maxSED = -1
    while p < len(trg) and f < len(src):
        if trg[p] == src[f]:
            p += 1
            f += 1
        else:
            st = trg[p - 1]
            en = trg[p]
            while trg[p] != src[f]:
                in_ = src[f]
                dis = getSED(region, int(in_), int(st), int(en))
                if dis > maxSED:
                    maxSED = dis
                    idx = f
                f += 1
    return maxSED, idx


def cleanTrj(src):
    res = []
    for p in src:
        if p == '' or p == 'UNK':
            continue
        res.append(p)
    return res


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
        id = idxs[0].tolist()[0]
        if src[id] not in resTrj:
            resTrj.append(src[id])
    if src[-1] not in resTrj:
        resTrj.append(src[-1])
    resTrj.sort(key=lambda j: mp[int(j)])
    return resTrj, trg_x_ori, trg_y_ori


def getMaxError(st, en, points):
    maxErr = -1
    idx = -1
    for i in range(st, en + 1):
        if maxErr < getSED4GPS(points[i], points[st], points[en]):
            maxErr = getSED4GPS(points[i], points[st], points[en])
            idx = i
    return idx, maxErr


def getKey_Value(item):
    for it in item.items():
        return it[0], it[1]


def cmp(a, b):
    (_, _), (_, maxErr_a) = getKey_Value(a)
    (_, _), (_, maxErr_b) = getKey_Value(b)
    return maxErr_a - maxErr_b


# adaptive douglas-peucker
def adp(points, max_len):
    if max_len > len(points):
        return None
    q = []
    st = 0
    en = len(points) - 1
    q.append({(st, en): getMaxError(st, en, points)})
    cnt = 2
    res = [st, en]
    maxErr = -1
    while cnt < max_len:
        q.sort(key=functools.cmp_to_key(cmp))
        solu = q.pop()
        cnt += 1
        (st, en), (split, maxErr) = getKey_Value(solu)
        q.append({(st, split): getMaxError(st, split, points)})
        q.append({(split, en): getMaxError(split, en, points)})
        res.append(split)
    q.sort(key=functools.cmp_to_key(cmp))
    solu = q.pop()
    (_, _), (_, maxErr) = getKey_Value(solu)
    pp = []
    res.sort()
    for idx in res:
        pp.append(points[idx])
    # 坐标点，下标，最大误差
    return pp, res, maxErr


# squish压缩算法
def squish(points, max_buffer_size):
    buffer = []
    # 坐标、err、下标
    buffer.append([points[0], 0, 0])
    max_err = 0
    if max_buffer_size > 2:
        for i in range(1, len(points)):
            buffer.append([points[i], 0, i])
            if len(buffer) <= 2:
                continue
            segment_start = buffer[-3][0]
            segment_end = buffer[-1][0]
            buffer[-2][1] += getSED4GPS(buffer[-2][0], segment_start, segment_end)
            if len(buffer) > max_buffer_size:
                to_remove = len(buffer) - 2
                for j in range(1, len(buffer) - 1):
                    if buffer[j][1] < buffer[to_remove][1]:
                        to_remove = j
                buffer[to_remove - 1][1] += buffer[to_remove][1]
                buffer[to_remove + 1][1] += buffer[to_remove][1]
                err = getSED4GPS(buffer[to_remove][0], buffer[to_remove - 1][0], buffer[to_remove + 1][0])
                max_err = max(max_err, err)
                buffer.pop(to_remove)
    else:
        buffer.append([points[-1], 0, len(points) - 1])
        segment_start = buffer[0][0]
        segment_end = buffer[-1][0]
        for i in range(1, len(points) - 1):
            max_err = max(max_err, getSED4GPS(points[i], segment_start, segment_end))
    idx = [p[2] for p in buffer]
    pp = [p[2] for p in buffer]
    return pp, idx, max_err


def id2gps(region, trj):
    points = []
    for p in trj:
        x, y = cell2gps(region, int(p))
        points.append([x, y])
    return points


def getErr(region, vocab, inp_src, seqs, cur):
    res = []

    def deal(p):
        return int(vocab.id2tok[src[p].item()])

    for seq, src in zip(seqs, inp_src):
        p = seq[cur - 1]
        # 1.选到了轨迹外的点，也无共享，直接为-1 2.选到了之前选过的节点，那么这个节点对于整个轨迹的误差减少没有任何贡献，直接为-1
        if p == -1 or src[p] == 0:
            res.append(-1)
            continue
        # 最小的
        st_value = 100
        st = -1
        # 第二小的
        en_value = 100
        en = -1
        for i in range(0, cur - 1):
            if seq[i] == -1 or seq[i] == p:
                continue
            if abs(p - seq[i]) < st_value:
                if st_value < en_value:
                    en_value = st_value
                    en = st
                st_value = abs(p - seq[i])
                st = i
            elif seq[i] == seq[st]:
                continue
            elif abs(p - seq[i]) < en_value:
                en_value = abs(p - seq[i])
                en = i
        try:
            st = seq[st]
            en = seq[en]
            res.append(
                getSED(region, vocab.id2tok[src[p].item()], vocab.id2tok[src[st].item()], vocab.id2tok[src[en].item()]))
        except Exception as e:
            print(seq)
            print(src)
            res.append(-1)
    return res
