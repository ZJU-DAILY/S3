import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from modules.helpers import straight_softmax, gumbel_softmax, getErr, getDistance
from modules.layers import Embed, Attention
from utils.gcn_emb import gcn_emb


class RecurrentHelper:
    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()

    def last_timestep(self, outputs, lengths, bi=False):
        if bi:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    def pad_outputs(self, out_packed, max_length):

        out_unpacked, _lengths = pad_packed_sequence(out_packed,
                                                     batch_first=True)

        # pad to initial max length
        pad_length = max_length - out_unpacked.size(1)
        out_unpacked = F.pad(out_unpacked, (0, 0, 0, pad_length))
        return out_unpacked

    @staticmethod
    def project2vocab(output, projection):
        # output_unpacked.size() = batch_size, max_length, hidden_units
        # flat_outputs = (batch_size*max_length, hidden_units),
        # which means that it is a sequence of *all* the outputs (flattened)
        flat_output = output.contiguous().view(output.size(0) * output.size(1),
                                               output.size(2))

        # the sequence of all the output projections
        decoded_flat = projection(flat_output)

        # reshaped the flat sequence of decoded words,
        # in the original (reshaped) form (3D tensor)
        decoded = decoded_flat.view(output.size(0), output.size(1),
                                    decoded_flat.size(1))
        return decoded

    @staticmethod
    def sort_by(lengths):
        """
        Sort batch data and labels by length.
        Useful for variable length inputs, for utilizing PackedSequences
        Args:
            lengths (nn.Tensor): tensor containing the lengths for the data

        Returns:
            - sorted lengths Tensor
            - sort (callable) which will sort a given iterable
                according to lengths
            - unsort (callable) which will revert a given iterable to its
                original order

        """
        batch_size = lengths.size(0)

        sorted_lengths, sorted_idx = lengths.sort()
        _, original_idx = sorted_idx.sort(0, descending=True)
        reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()

        if lengths.data.is_cuda:
            reverse_idx = reverse_idx.cuda()

        sorted_lengths = sorted_lengths[reverse_idx]

        def sort(iterable):

            if iterable is None:
                return None

            if len(iterable.shape) > 1:
                return iterable[sorted_idx][reverse_idx]
            else:
                return iterable

        def unsort(iterable):

            if iterable is None:
                return None

            if len(iterable.shape) > 1:
                return iterable[reverse_idx][original_idx][reverse_idx]
            else:
                return iterable

        return sorted_lengths, sort, unsort


def transfer_weigths(target, source):
    target_params = target.named_parameters()
    source_params = source.named_parameters()

    dict_target_params = dict(target_params)

    for name, param in source_params:
        if name in dict_target_params:
            dict_target_params[name].data.copy_(param.data)


def tie_weigths(target, source):
    target_params = target.named_parameters()
    source_params = source.named_parameters()

    dict_target_params = dict(target_params)

    for name, param in source_params:
        if name in dict_target_params:
            setattr(target, name, getattr(source, name))


def length_countdown(lengths):
    batch_size = lengths.size(0)
    max_length = max(lengths)
    desired_lengths = lengths - 1

    _range = torch.arange(0, -max_length, -1, device=lengths.device)
    _range = _range.repeat(batch_size, 1)
    _countdown = _range + desired_lengths.unsqueeze(-1)

    return _countdown


def drop_tokens(embeddings, word_dropout):
    batch, length, size = embeddings.size()
    mask = embeddings.new_empty(batch, length)
    mask = mask.bernoulli_(1 - word_dropout)
    embeddings = embeddings * mask.unsqueeze(-1).expand_as(embeddings).float()
    return embeddings, mask


#         self.encoder = RNNModule(input_size=self.emb_size,
#                                  rnn_size=self.rnn_size,
#                                  num_layers=self.rnn_layers,
#                                  bidirectional=self.rnn_bidirectional,
#                                  dropout=self.rnn_dropout,
#                                  pack=self.pack,
#                                  countdown=self.countdown)

class RNNModule(nn.Module, RecurrentHelper):
    def __init__(self, input_size,
                 rnn_size,
                 rnn_type,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.,
                 pack=True, last=False, countdown=False):
        """
        A simple RNN Encoder, which produces a fixed vector representation
        for a variable length sequence of feature vectors, using the output
        at the last timestep of the RNN.
        Args:
            input_size (int): the size of the input features
            rnn_size (int): default 300
            num_layers (int):
            bidirectional (bool):
            dropout (float):
        """
        super(RNNModule, self).__init__()

        self.pack = pack
        self.last = last
        self.countdown = countdown

        if self.countdown:
            self.Wt = nn.Parameter(torch.rand(1))
            input_size += 1
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=input_size,
                               hidden_size=rnn_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               batch_first=True)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(input_size=input_size,
                              hidden_size=rnn_size,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first=True)

        # the dropout "layer" for the output of the RNN
        self.dropout = nn.Dropout(dropout)

        # define output feature size
        self.feature_size = rnn_size

        # double if bidirectional
        if bidirectional:
            self.feature_size *= 2

    @staticmethod
    def reorder_hidden(hidden, order):
        if isinstance(hidden, tuple):
            hidden = hidden[0][:, order, :], hidden[1][:, order, :]
        else:
            hidden = hidden[:, order, :]

        return hidden

    def forward(self, x, hidden=None, lengths=None):

        batch, max_length, feat_size = x.size()

        if lengths is not None and self.pack:

            ###############################################
            # sorting
            ###############################################
            lenghts_sorted, sorted_i = lengths.sort(descending=True)
            _, reverse_i = sorted_i.sort()

            x = x[sorted_i]

            if hidden is not None:
                hidden = self.reorder_hidden(hidden, sorted_i)

            ###############################################
            # forward
            ###############################################

            if self.countdown:
                ticks = length_countdown(lenghts_sorted).float() * self.Wt
                x = torch.cat([x, ticks.unsqueeze(-1)], -1)
            lenghts_sorted = lenghts_sorted.to('cpu')
            packed = pack_padded_sequence(x, lenghts_sorted, batch_first=True)

            self.rnn.flatten_parameters()
            out_packed, hidden = self.rnn(packed, hidden)

            out_unpacked, _lengths = pad_packed_sequence(out_packed,
                                                         batch_first=True,
                                                         total_length=max_length)

            out_unpacked = self.dropout(out_unpacked)

            ###############################################
            # un-sorting
            ###############################################
            outputs = out_unpacked[reverse_i]
            hidden = self.reorder_hidden(hidden, reverse_i)

        else:
            # todo: make hidden return the true last states
            self.rnn.flatten_parameters()
            outputs, hidden = self.rnn(x, hidden)
            outputs = self.dropout(outputs)

        if self.last:
            return outputs, hidden, self.last_timestep(outputs, lengths,
                                                       self.rnn.bidirectional)

        return outputs, hidden


# 模型1/2，即模型的encoder部分
class SeqReader(nn.Module, RecurrentHelper):
    def __init__(self, ntokens, _gcn_emb_weights, **kwargs):
        super(SeqReader, self).__init__()

        ############################################
        # Attributes
        ############################################
        self.ntokens = ntokens
        self.emb_size = kwargs.get("emb_size", 100)
        self.embed_noise = kwargs.get("embed_noise", .0)
        self.embed_dropout = kwargs.get("embed_dropout", .0)
        self.rnn_size = kwargs.get("rnn_size", 100)
        # 配置参数为两层
        self.rnn_layers = kwargs.get("rnn_layers", 1)
        self.rnn_dropout = kwargs.get("rnn_dropout", .0)
        self.rnn_type = kwargs.get("rnn_type", "LSTM")
        self.rnn_bidirectional = kwargs.get("rnn_bidirectional", False)
        self.decode = kwargs.get("decode", False)
        self.tie_weights = kwargs.get("tie_weights", False)
        self.pack = kwargs.get("pack", True)
        self.countdown = kwargs.get("countdown", False)
        self.gnn_latent_dim = kwargs.get("gnn_latent_dim", 128)
        self.gnn_used = kwargs.get("gnn_used", True)

        ############################################
        # Layers
        ############################################

        # 此处自己设计的词嵌入
        self.embed = Embed(ntokens, self.emb_size, _gcn_emb_weights, self.gnn_latent_dim,
                           noise=self.embed_noise, dropout=self.embed_dropout, gnn_used=self.gnn_used)
        if self.gnn_used:
            self.emb_size += self.gnn_latent_dim
        self.encoder = RNNModule(input_size=self.emb_size,
                                 rnn_size=self.rnn_size,
                                 rnn_type=self.rnn_type,
                                 num_layers=self.rnn_layers,
                                 bidirectional=self.rnn_bidirectional,
                                 dropout=self.rnn_dropout,
                                 pack=self.pack,
                                 countdown=self.countdown)

        if self.rnn_bidirectional:
            self.rnn_size *= 2

        if self.decode:
            if self.rnn_bidirectional:
                raise ValueError("Can't decode with bidirectional RNNs!")

            if self.tie_weights and self.rnn_size != self.emb_size:
                rnn_out = self.emb_size
                self.down = nn.Linear(self.rnn_size, rnn_out)
            else:
                rnn_out = self.rnn_size

            self.out = nn.Linear(rnn_out, ntokens)

            if self.tie_weights:
                # if self.rnn_size != self.emb_size:
                #     raise ValueError("if `tie_weights` is True,"
                #                      "emb_size has to be equal to rnn_size")
                self.out.weight = self.embed.embedding.weight

    def init_hidden(self, bsz):
        weight = next(self.parameters())

        if self.encoder.rnn.mode == 'LSTM':
            return (weight.new_zeros(self.rnn_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.rnn_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(self.rnn_layers, bsz, self.rnn_size)

    def encode(self, embeds, hidden=None, lengths=None):
        outputs, hidden = self.encoder(embeds, hidden, lengths)

        return outputs, hidden

    def read_embs(self, embeds, hidden=None, lengths=None):

        outputs, hidden = self.encode(embeds, hidden, lengths)

        if self.decode:

            if self.tie_weights and self.rnn_size != self.emb_size:
                outputs = self.down(outputs)

            logits = self.project2vocab(outputs, self.out)
            return logits, outputs, hidden
        else:
            return outputs, hidden

    def forward(self, src, hidden=None, lengths=None, word_dropout=0.0, vocab=None):
        embeds = self.embed(src, vocab)

        if word_dropout > 0:
            embeds, mask = drop_tokens(embeds, word_dropout)

        outputs, hidden = self.encode(embeds, hidden, lengths)

        if self.decode:

            if self.tie_weights and self.rnn_size != self.emb_size:
                outputs = self.down(outputs)

            logits = self.project2vocab(outputs, self.out)
            return logits, outputs, hidden
        else:
            return outputs, hidden


# 模型2/2，即模型的decoder部分
class AttSeqDecoder(nn.Module):
    def __init__(self, trg_ntokens, enc_size, err_control_, _gcn_emb_weights, **kwargs):
        super(AttSeqDecoder, self).__init__()

        ############################################
        # Attributes
        ############################################
        self.trg_ntokens = trg_ntokens
        # 输入特征维度，即词嵌入的维度
        emb_size = kwargs.get("emb_size", 100)
        embed_noise = kwargs.get("embed_noise", .0)
        embed_dropout = kwargs.get("embed_dropout", .0)
        # 输出的特征维度，即h的维度
        rnn_size = kwargs.get("rnn_size", 100)
        # rnn的层数
        rnn_layers = kwargs.get("rnn_layers", 1)
        rnn_dropout = kwargs.get("rnn_dropout", .0)
        tie_weights = kwargs.get("tie_weights", False)
        attention_fn = kwargs.get("attention_fn", "general")
        self.input_feeding = kwargs.get("input_feeding", False)
        self.learn_tau = kwargs.get("learn_tau", False)
        self.length_control = kwargs.get("length_control", False)
        self.err_control = err_control_
        self.gumbel = kwargs.get("gumbel", False)
        self.out_non_linearity = kwargs.get("out_non_linearity", None)
        self.layer_norm = kwargs.get("layer_norm", None)
        self.input_feeding_learnt = kwargs.get("input_feeding_learnt", False)
        self.coverage = kwargs.get("attention_coverage", False)
        self.rnn_type = kwargs.get("rnn_type", "LSTM")
        self.gnn_latent_dim = kwargs.get("gnn_latent_dim", 128)
        self.gnn_used = kwargs.get("gnn_used", True)

        ############################################
        # Layers
        ############################################
        # _gcn_emb_weights = gcn_emb()
        # 轨迹点嵌入
        self.embed = Embed(trg_ntokens, emb_size, _gcn_emb_weights, self.gnn_latent_dim,
                           noise=embed_noise,
                           dropout=embed_dropout)
        if self.gnn_used:
            emb_size += self.gnn_latent_dim
        # the output size of the ho token: ho = [ h || c]
        if tie_weights:
            self.ho_size = emb_size
        else:
            self.ho_size = rnn_size

        dec_input_size = emb_size
        if self.input_feeding:
            dec_input_size += self.ho_size
        if self.length_control:
            dec_input_size += 2
            # length scaling parameter
            self.W_tick = nn.Parameter(torch.rand(1))
        if self.err_control:
            dec_input_size += 1
            self.W_err = nn.Parameter(torch.rand(1))

        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=dec_input_size,
                               hidden_size=rnn_size,
                               num_layers=rnn_layers,
                               batch_first=True)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(input_size=dec_input_size,
                              hidden_size=rnn_size,
                              num_layers=rnn_layers,
                              batch_first=True)

        self.rnn_dropout = nn.Dropout(rnn_dropout)

        self.attention = Attention(enc_size, rnn_size, method=attention_fn, coverage=self.coverage)

        # learnt temperature parameter
        if self.learn_tau:
            self.softplus = nn.Sequential(nn.Linear(self.ho_size, 1,
                                                    bias=False),
                                          nn.Softplus())
            self.tau_0 = kwargs.get("tau_0", 1)

        # 此处我还是想将隐形量最终映射为tmp_size大小的，也就是word2vec。目的也仅仅是能够兼容tie_weights
        tmp_size = kwargs.get("emb_size", 100)

        # initial input feeding
        if self.input_feeding_learnt:
            self.Wi = nn.Linear(enc_size, self.ho_size)

        # source context-aware output projection
        self.Wc = nn.Linear(rnn_size + enc_size, self.ho_size)

        # self.Wc = nn.Linear(rnn_size + enc_size, tmp_size)

        # projection layer to the vocabulary
        # self.Wo = nn.Linear(self.ho_size, trg_ntokens)
        self.Wo = nn.Linear(tmp_size, trg_ntokens)

        if self.layer_norm:
            # self.norm_ctx = nn.LayerNorm(tmp_size)
            self.norm_ctx = nn.LayerNorm(self.ho_size)

            if self.input_feeding_learnt:
                # self.norm_input_feed = nn.LayerNorm(tmp_size)
                self.norm_input_feed = nn.LayerNorm(self.ho_size)

        if tie_weights:
            # if rnn_size != emb_size:
            #     raise ValueError("if `tie_weights` is True,"
            #                      "emb_size has to be equal to rnn_size")
            self.Wo.weight = self.embed.embedding.weight

    @staticmethod
    def _top_hidden(hidden):
        """
        Get the hidden state from the top RNN layer.
        Used as a query for attention mechanisms.
        Args:
            hidden:

        Returns:

        """
        if isinstance(hidden, tuple):
            return hidden[0][-1]
        else:
            return hidden[-1]

    @staticmethod
    def _coin_flip(prob):
        """
        Return the outcome of a biased coin flip.
        Args:
            prob: the probability of True.

        Returns: bool

        """
        return prob > 0 and torch.rand(1).item() < prob

    def get_embedding(self, step, trg, logits, dists, sampling_prob, argmax, hard,
                      tau, mask_matrix, vocab):
        """
        Get the token embedding for the current timestep. Possible options:
        - select the embedding by a given index
        - sample a token from a probability distribution and embed
        - construct a "fuzzy" embedding, by taking a convex combination of all
        the token embeddings, parameterized by a probability distribution

        Note: In the first step (step==0) select the embedding
        of the actual target word (usually the <sos> token).

        Args:
            step: the i-th timestep
            trg: the true token at the given step
            logits: the unormalized probability distribution over the tokens
                from the previous timestep.
            sampling_prob: how often to sample a word instead of using
                the gold one. (free-run vs. teacher-forcing)
            argmax: take the argmax of the distribution
            hard: (Straight-Trough Estimator) discretize the probability
                distribution and compute a convex combination
            tau:

        Returns: the word embedding and its index.

        """
        # in sample is `True`, then feed the prediction back to the model,
        # instead of the true target word
        use_init_hidden = True
        sample = sampling_prob == 1 or self._coin_flip(sampling_prob)

        if step > 0 and sample:

            if argmax:  # get the argmax
                maxv, maxi = logits[-1].max(dim=2)
                e_i = self.embed(maxi, vocab)
                return e_i, None

            else:  # get the expected embedding, parameterized by the posterior
                if not use_init_hidden and (step == 1 or step == 2):
                    batch_size = logits[-1].size(0)
                    class_num = logits[-1].size(2)
                    label = trg[:, step].unsqueeze(1)
                    m_zeros = torch.zeros(batch_size, class_num, device=logits[-1].device)
                    dist = m_zeros.scatter(1, label, 1)
                elif self.gumbel and self.training:
                    if mask_matrix is not None and self._coin_flip(sampling_prob):
                        dist = gumbel_softmax(logits[-1].squeeze(), tau, hard, target_mask=mask_matrix)
                    else:
                        dist = gumbel_softmax(logits[-1].squeeze(), tau, hard)
                else:
                    if mask_matrix is not None:
                    # if mask_matrix is not None and self._coin_flip(sampling_prob):
                        # mask_vector = mask_matrix
                        # logit = logits[-1].squeeze()
                        # mask_logit = mask_vector * logit
                        # if mask_vector[0][0] == 1:
                        #     print()
                        dist = straight_softmax(logits[-1].squeeze(), tau, hard, mask_matrix)
                    else:
                        dist = straight_softmax(logits[-1].squeeze(), tau, hard)

                # A = logits[-1].squeeze()
                # print(A.max(-1)[1])
                # print(dist.max(-1)[1])

                e_i = self.embed.expectation(dist.unsqueeze(1), vocab)
                return e_i, dist
        else:

            w_i = trg[:, step].unsqueeze(1)
            e_i = self.embed(w_i, vocab)
            return e_i, None

    # 初始化context向量
    def _init_input_feed(self, enc_states, lengths):

        batch = enc_states.size(0)
        if self.input_feeding_learnt:

            mean = enc_states.sum(1) / lengths.unsqueeze(1).float()
            ho = self.Wi(mean).squeeze().unsqueeze(1)

            if self.layer_norm:
                ho = self.norm_input_feed(ho)

            if self.out_non_linearity == "relu":
                ho = torch.relu(ho)
            elif self.out_non_linearity == "tanh":
                ho = torch.tanh(ho)
        else:
            ho = torch.zeros((batch, 1, self.ho_size),
                             device=enc_states.device,
                             dtype=enc_states.dtype)

        return ho

    def step(self, embs, enc_outputs, state, enc_lengths, ho=None, tick=None, ec=None, attn_coverage=None):
        """
        Perform one decoding step.
        1. Construct the input. If input-feeding is used, then the input is the
            concatenation of the current embedding and previous context vector.
        2. Feed the input to the decoder and obtain the contextualized
            token representations.
        3. generate a context vector. It is a convex combination of the
            states of the encoder, the weights of which are a function of each
            state of the encoder and the current state of the decoder.
        4. Re-weight the decoder's state with the context vector.
        5. Project the context-aware vector to the vocabulary.

        Args:
            embs:
            enc_outputs:
            state:
            ho:
            enc_lengths:
            tick:
      Returns:

        """

        # 1. Construct the input
        decoder_input = embs
        # 默认是true
        if self.input_feeding:
            if ho is None:
                ho = self._init_input_feed(enc_outputs, enc_lengths)
            decoder_input = torch.cat([embs, ho], -1)
        if self.length_control:
            decoder_input = torch.cat([decoder_input, tick], -1)
        if self.err_control and ec is not None:
            decoder_input = torch.cat([decoder_input, ec], -1)

        # 2. Feed the input to the decoder
        self.rnn.flatten_parameters()
        outputs, state = self.rnn(decoder_input, state)
        outputs = self.rnn_dropout(outputs)

        # 3. generate the context vector
        query = outputs.squeeze(1)
        contexts, att_scores = self.attention(enc_outputs, query, enc_lengths, coverage=attn_coverage)
        contexts = contexts.unsqueeze(1)

        # 4. Re-weight the decoder's state with the context vector.
        ho = self.Wc(torch.cat([outputs, contexts], -1))

        if self.layer_norm:
            ho = self.norm_ctx(ho)

        if self.out_non_linearity == "relu":
            ho = torch.relu(ho)
        elif self.out_non_linearity == "tanh":
            ho = torch.tanh(ho)
        # 5. cut the ho vector for top tmp_size if gnn_used

        if self.gnn_used:
            input_size = self.Wo.in_features
            dec_logits = self.Wo(ho[:,:,:input_size])
        # 6. Project the context-aware vector to the vocabulary.gumbel_softmax
        # print(ho.shape,self.Wo)
        else:
            dec_logits = self.Wo(ho)

        return dec_logits, outputs, state, ho, att_scores

    def forward(self, gold_tokens, enc_outputs, init_hidden, enc_lengths,
                sampling_prob=0.0, argmax=False, hard=False, tau=1.0,
                desired_lengths=None, word_dropout=0, mask_matrix=None, inp_src=None,
                region=None, vocab=None,time_list=None):
        """

        Args:
            gold_tokens:应该是decoder的初始输入
            enc_outputs:指的是encoder的所有输出，应该是用于attention
            init_hidden:encoder的最后一个隐状态
            enc_lengths:
            sampling_prob:
            argmax:
            hard:
            tau:
            desired_lengths:
            word_dropout:

        Returns:
            Note: dists contain one less element than logits, because
            we do not care about sampling from the last timestep as it will not
            be used for sampling another token. The last timestep should
            correspond to the EOS token, and the corresponding logit will be
            used only for computing the NLL loss of the EOS token.

        """

        batch, max_length = gold_tokens.size()

        logits = []
        outputs = []
        attentions = []
        dists = []
        taus = []

        # initial hidden state of the decoder, and initial context
        state = init_hidden
        ho = None
        tick = None
        ec = None
        attn_covergae = None

        # length_control在配置文件中为true
        if self.length_control:
            countdown = length_countdown(desired_lengths).float() * self.W_tick
            ratio = desired_lengths.float() / enc_lengths.float()

        if self.err_control:
            idx = [[] for i in range(batch)]

        for i in range(max_length):
            # obtain the input word embedding，返回的词向量和对应词id（即经过softmax之后转化为one-hot的词）
            e_i, d_i = self.get_embedding(i, gold_tokens, logits,dists,
                                          sampling_prob, argmax, hard, tau, mask_matrix, vocab)
            if d_i is not None:
                tmp = d_i.max(-1)[1].unsqueeze(1)
                mask_matrix = mask_matrix.scatter(1, tmp, 0)

            if word_dropout > 0 and i > 0:
                e_i, mask = drop_tokens(e_i, word_dropout)

            # the number of remaining tokens
            if self.length_control:
                tick = torch.stack([countdown[:, i], ratio], -1).unsqueeze(1)

            # if self.err_control:
            #     if d_i is not None:
            #         curP = d_i.max(-1)[-1]
            #         id = [(seq == p).nonzero().flatten().tolist() if p in seq and p != 0 else [-1] for p, seq in
            #               zip(curP, inp_src)]
            #
            #         for x, ii in enumerate(id):
            #             idx[x].append(ii[0])
            #
            #     if i <= 2:
            #         ec = torch.ones([batch, 1, 1]).to(e_i)
            #     else:
            #         err = getErr(region, vocab, inp_src, idx, i)
            #         ec = torch.tensor(err).to(e_i)
            #         ec = ec.view(batch, 1, 1)
            #         ec = ec * self.W_err

            if self.coverage:
                if len(attentions) == 0:
                    attn_coverage = torch.zeros([enc_outputs.size(0), enc_outputs.size(1)]).to(e_i)
                else:
                    attn_coverage = torch.stack(attentions).sum(0)
            # if i == 2:
            #     state = init_hidden

            # perform one decoding step
            # logits就是输出对应词表中的词，output就是当前输出的隐向量，state就是最后一个状态的隐向量。
            # 因为此处的rnn层数只有一层，所以output和state理论上来说应该是一样的。ho和att是和attention相关的一些参数，可暂时不考虑。
            tic1 = time.perf_counter()
            _logits, outs, state, ho, att = self.step(e_i, enc_outputs, state,
                                                      enc_lengths, ho, tick, ec, attn_coverage)
            tic2 = time.perf_counter()

            if time_list is not None:
                time_list[0] += tic2 - tic1

            if self.learn_tau and self.training:
                tau = 1 / (self.softplus(ho.squeeze()) + self.tau_0)
                taus.append(tau)

            logits.append(_logits)
            outputs.append(outs)
            attentions.append(att)

            if i > 0 and sampling_prob == 1 and not argmax:
                dists.append(d_i)

        outputs = torch.cat(outputs, dim=1).contiguous()
        logits = torch.cat(logits, dim=1).contiguous()
        attentions = torch.stack(attentions, dim=1).contiguous()

        if len(dists) > 0:
            dists = torch.stack(dists, dim=1).contiguous()
        else:
            dists = None

        if len(taus) > 0:
            taus = torch.stack(taus, dim=1).squeeze()

        return logits, outputs, state, dists, attentions, taus
