from pprint import pprint
import numpy as np
import torch
import constants
from funcy import merge
from collections import namedtuple
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from generate.utils import devectorize


def compute_dataset_idf(dataset, vocab):
    def identity_func(doc):
        return doc

    my_data = [dataset.read_sample(i) for i in range(len(dataset))]

    tf = TfidfVectorizer(lowercase=False,
                         tokenizer=identity_func,
                         preprocessor=identity_func,
                         use_idf=True,
                         vocabulary=vocab)
    tf.fit(my_data)
    return tf.idf_


def sample2text(word_ids, vocab):
    tokens = devectorize(word_ids,
                         vocab.id2tok,
                         vocab.tok2id[vocab.EOS],
                         strip_eos=False,
                         pp=False)
    text = [" ".join(out) for out in tokens]
    lengths = [len(t) for t in tokens]
    return text, lengths


def str2tree(ls):
    tree = {}
    for item in ls:
        t = tree
        for part in item.split('.'):
            t = t.setdefault(part, {})
    pprint(tree)


def sample_lengths(src_lengths,
                   min_ratio, max_ratio,
                   min_length, max_length):
    """
    Sample summary lengths from a list of source lengths.

    """
    t = torch.empty(len(src_lengths), device=src_lengths.device)
    samples = t.uniform_(min_ratio, max_ratio)
    lengths = (src_lengths.float() * samples).long()
    lengths = lengths.clamp(min=min_length, max=max_length)
    return lengths


def argsort(seq):
    """
    sort by length in reverse order
    ---
    seq (list[array[int32]])
    """
    return [x for x, y in sorted(enumerate(seq),
                                 key=lambda x: len(x[1]),
                                 reverse=True)]


def pad_array(a, max_length, PAD=constants.PAD):
    """
    a (array[int32])
    """
    return np.concatenate((a, [PAD] * (max_length - len(a))))


def pad_arrays(a):
    max_length = max(map(len, a))
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a).astype(np.int)
    return torch.LongTensor(a)


def pad_arrays_pair(src, trg, keep_invp=False):
    """
    Input:
    src (list[array[int32]])
    trg (list[array[int32]])
    ---
    Output:
    src (seq_len1, batch)
    trg (seq_len2, batch)
    lengths (1, batch)
    invp (batch,): inverse permutation, src.t()[invp] gets original order
    """
    TD = namedtuple('TD', ['src', 'lengths', 'trg', 'invp'])

    assert len(src) == len(trg), "source and target should have the same length"
    idx = argsort(src)
    src = list(np.array(src)[idx])
    trg = list(np.array(trg)[idx])

    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    src = pad_arrays(src)
    trg = pad_arrays(trg)
    if keep_invp == True:
        invp = torch.LongTensor(invpermute(idx))
        # (batch, seq_len) => (seq_len, batch)
        return TD(src=src.t().contiguous(), lengths=lengths.view(1, -1), trg=trg.t().contiguous(), invp=invp)
    else:
        # (batch, seq_len) => (seq_len, batch)
        return TD(src=src.t().contiguous(), lengths=lengths.view(1, -1), trg=trg.t().contiguous(), invp=[])


def invpermute(p):
    """
    inverse permutation
    """
    p = np.asarray(p)
    invp = np.empty_like(p)
    for i in range(p.size):
        invp[p[i]] = i
    return invp


def pad_arrays_keep_invp(src):
    """
    Pad arrays and return inverse permutation

    Input:
    src (list[array[int32]])
    ---
    Output:
    src (seq_len, batch)
    lengths (1, batch)
    invp (batch,): inverse permutation, src.t()[invp] gets original order
    """
    idx = argsort(src)
    src = list(np.array(src)[idx])
    lengths = list(map(len, src))
    lengths = torch.LongTensor(lengths)
    src = pad_arrays(src)
    invp = torch.LongTensor(invpermute(idx))
    return src.t().contiguous(), lengths.view(1, -1), invp


def random_subseq(a, rate):
    """
    Dropping some points between a[3:-2] randomly according to rate.

    Input:
    a (array[int])
    rate (float)
    """
    idx = np.random.rand(len(a)) < rate
    idx[0], idx[-1] = True, True
    return a[idx]


class DataLoader():
    """
    srcfile: source file name
    trgfile: target file name
    batch: batch size
    validate: if validate = True return batch orderly otherwise return
        batch randomly
    """

    def __init__(self, srcfile, batch, bucketsize, validate=False):
        self.srcfile = srcfile

        self.batch = batch
        self.validate = validate
        # self.bucketsize = [(30, 30), (30, 50), (50, 50), (50, 70), (70, 70)]
        self.bucketsize = bucketsize

    def insert(self, s):
        for i in range(len(self.bucketsize)):
            if len(s) <= self.bucketsize[i][0]:
                self.srcdata[i].append(np.array(s, dtype=np.int32))
                return 1
        return 0

    def load(self, max_num_line=0):
        self.srcdata = [[] for _ in range(len(self.bucketsize))]

        srcstream = open(self.srcfile, 'r')
        num_line = 0
        for s in srcstream:
            s = [int(x) for x in s.split()]

            num_line += self.insert(s)
            if num_line >= max_num_line and max_num_line > 0: break
            if num_line % 500000 == 0:
                print("Read line {}".format(num_line))
        ## if vliadate is True we merge all buckets into one
        if self.validate == True:
            self.srcdata = np.array(merge(*self.srcdata))

            self.start = 0
            self.size = len(self.srcdata)
        else:
            self.srcdata = list(map(np.array, self.srcdata))

            self.allocation = list(map(len, self.srcdata))
            self.p = np.array(self.allocation) / sum(self.allocation)
        srcstream.close()

    def getbatch_one(self):
        if self.validate == True:
            src = self.srcdata[self.start:self.start + self.batch]

            # update `start` for next batch
            self.start += self.batch
            if self.start >= self.size:
                self.start = 0
            return list(src)
        else:
            # select bucket
            sample = np.random.multinomial(1, self.p)
            bucket = np.nonzero(sample)[0][0]
            # select data from the bucket
            idx = np.random.choice(len(self.srcdata[bucket]), self.batch)
            src = self.srcdata[bucket][idx]
            return list(src)

    def getbatch_generative(self):
        src, trg, _ = self.getbatch_one()
        # src (seq_len1, batch), lengths (1, batch), trg (seq_len2, batch)
        return pad_arrays_pair(src, trg, keep_invp=False)

    def getbatch_discriminative_cross(self):
        a = self.getbatch_one()
        p = self.getbatch_one()
        n = self.getbatch_one()

        return a, p, n

    def getbatch_discriminative_inner(self):
        """
        Test Case:
        a, p, n = dataloader.getbatch_discriminative_inner()
        i = 2
        idx_a = torch.nonzero(a[2].t()[a[3]][i])
        idx_p = torch.nonzero(p[2].t()[p[3]][i])
        idx_n = torch.nonzero(n[2].t()[n[3]][i])
        a_t = a[2].t()[a[3]][i][idx_a].view(-1).numpy()
        p_t = p[2].t()[p[3]][i][idx_p].view(-1).numpy()
        n_t = n[2].t()[n[3]][i][idx_n].view(-1).numpy()
        print(len(np.intersect1d(a_t, p_t)))
        print(len(np.intersect1d(a_t, n_t)))
        """
        a_src, a_trg = [], []
        p_src, p_trg = [], []
        n_src, n_trg = [], []

        _, trgs, _ = self.getbatch_one()
        for i in range(len(trgs)):
            trg = trgs[i][1:-1]
            if len(trg) < 10: continue
            a1, a3, a5 = 0, len(trg) // 2, len(trg)
            a2, a4 = (a1 + a3) // 2, (a3 + a5) // 2
            rate = np.random.choice([0.5, 0.6, 0.8])
            if np.random.rand() > 0.5:
                a_src.append(random_subseq(trg[a1:a4], rate))
                a_trg.append(np.r_[constants.BOS, trg[a1:a4], constants.EOS])
                p_src.append(random_subseq(trg[a2:a5], rate))
                p_trg.append(np.r_[constants.BOS, trg[a2:a5], constants.EOS])
                n_src.append(random_subseq(trg[a3:a5], rate))
                n_trg.append(np.r_[constants.BOS, trg[a3:a5], constants.EOS])
            else:
                a_src.append(random_subseq(trg[a2:a5], rate))
                a_trg.append(np.r_[constants.BOS, trg[a2:a5], constants.EOS])
                p_src.append(random_subseq(trg[a1:a4], rate))
                p_trg.append(np.r_[constants.BOS, trg[a1:a4], constants.EOS])
                n_src.append(random_subseq(trg[a1:a3], rate))
                n_trg.append(np.r_[constants.BOS, trg[a1:a3], constants.EOS])

        a = pad_arrays_pair(a_src, a_trg, keep_invp=True)
        p = pad_arrays_pair(p_src, p_trg, keep_invp=True)
        n = pad_arrays_pair(n_src, n_trg, keep_invp=True)
        return a, p, n


class DataOrderScaner():
    def __init__(self, srcfile, batch):
        self.srcfile = srcfile
        self.batch = batch
        self.srcdata = []
        self.start = 0

    def load(self, max_num_line=0):
        num_line = 0
        with open(self.srcfile, 'r') as srcstream:
            for s in srcstream:
                s = [int(x) for x in s.split()]
                self.srcdata.append(np.array(s, dtype=np.int32))
                num_line += 1
                if max_num_line > 0 and num_line >= max_num_line:
                    break
        self.size = len(self.srcdata)
        self.start = 0

    def getbatch(self):
        """
        Output:
        src (seq_len, batch)
        lengths (1, batch)
        invp (batch,): inverse permutation, src.t()[invp] gets original order
        """
        if self.start >= self.size:
            return None, None, None
        src = self.srcdata[self.start:self.start + self.batch]
        ## update `start` for next batch
        self.start += self.batch
        return pad_arrays_keep_invp(src)
