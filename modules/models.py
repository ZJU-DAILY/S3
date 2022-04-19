import torch
from torch import nn
from torch.nn import functional as F

from modules.modules import RecurrentHelper, AttSeqDecoder, SeqReader


class Seq2Seq2Seq(nn.Module, RecurrentHelper):

    def __init__(self, n_tokens, **kwargs):
        super(Seq2Seq2Seq, self).__init__()

        ############################################
        # Attributes
        ############################################
        # 词表的大小
        self.n_tokens = n_tokens
        self.bridge_hidden = kwargs.get("bridge_hidden", False)
        self.bridge_non_linearity = kwargs.get("bridge_non_linearity", None)
        self.detach_hidden = kwargs.get("detach_hidden", False)
        self.input_feeding = kwargs.get("input_feeding", False)
        self.length_control = kwargs.get("length_control", False)
        self.bi_encoder = kwargs.get("rnn_bidirectional", False)
        self.bi_encoder = kwargs.get("rnn_bidirectional", False)
        self.rnn_type = kwargs.get("rnn_type", "LSTM")
        self.layer_norm = kwargs.get("layer_norm", False)
        self.sos = kwargs.get("sos", 1)
        self.sample_embed_noise = kwargs.get("sample_embed_noise", 0)
        self.topic_idf = kwargs.get("topic_idf", False)
        self.dec_token_dropout = kwargs.get("dec_token_dropout", .0)
        self.enc_token_dropout = kwargs.get("enc_token_dropout", .0)
        self.embedding_size = kwargs.get("embedding_size", 100)

        # tie embedding layers to output layers (vocabulary projections)
        kwargs["tie_weights"] = kwargs.get("tie_embedding_outputs", False)

        ############################################
        # Layers
        ############################################
        self.embedding = nn.Embedding(self.n_tokens, self.embedding_size,
                                      padding_idx=0)

        # backward-compatibility for older version of the project
        kwargs["rnn_size"] = kwargs.get("enc_rnn_size", kwargs.get("rnn_size"))
        self.inp_encoder = SeqReader(self.n_tokens, **kwargs)
        self.cmp_encoder = SeqReader(self.n_tokens, **kwargs)

        # backward-compatibility for older version of the project
        kwargs["rnn_size"] = kwargs.get("dec_rnn_size", kwargs.get("rnn_size"))
        enc_size = self.inp_encoder.rnn_size
        # self.n_tokens实际上存的是词向量，用于初始化嵌入层
        self.compressor = AttSeqDecoder(self.n_tokens, enc_size, **kwargs)
        self.decompressor = AttSeqDecoder(self.n_tokens, enc_size, **kwargs)

        # create a dummy embedding layer, which will retrieve the idf values
        # of each word, given the word ids
        if self.topic_idf:
            self.idf = nn.Embedding(num_embeddings=n_tokens, embedding_dim=1)
            self.idf.weight.requires_grad = False

        if self.bridge_hidden:
            self._initialize_bridge(enc_size,
                                    kwargs["dec_rnn_size"],
                                    kwargs["rnn_layers"])

    def _initialize_bridge(self, enc_hidden_size, dec_hidden_size, num_layers):
        """
        adapted from
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/encoders/rnn_encoder.py#L85

        """

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if self.rnn_type == "LSTM" else 1

        if self.length_control:
            # add a parameter, for scaling the absolute target length
            self.Wl = nn.Parameter(torch.rand(1))
            # the length information will contain 2 additional dimensions,
            # - the target length
            # - the expansion / compression ratio given the source length
            enc_hidden_size += 2

        # Build a linear layer for each
        self.src_bridge = nn.ModuleList([nn.Linear(enc_hidden_size,
                                                   dec_hidden_size)
                                         for _ in range(number_of_states)])
        self.trg_bridge = nn.ModuleList([nn.Linear(enc_hidden_size,
                                                   dec_hidden_size)
                                         for _ in range(number_of_states)])

    def _bridge(self, bridge, hidden, src_lengths=None, trg_lengths=None):
        """Forward hidden state through bridge."""

        def _fix_hidden(_hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            fwd_final = _hidden[0:_hidden.size(0):2]
            bwd_final = _hidden[1:_hidden.size(0):2]
            final = torch.cat([fwd_final, bwd_final], dim=2)
            return final

        def bottle_hidden(linear, states, length_feats=None):
            if length_feats is not None:
                lf = length_feats.unsqueeze(0).repeat(states.size(0), 1, 1)
                _states = torch.cat([states, lf], -1)
                result = linear(_states)
            else:
                result = linear(states)

            if self.bridge_non_linearity == "tanh":
                result = torch.tanh(result)
            elif self.bridge_non_linearity == "relu":
                result = F.relu(result)

            return result

        if self.length_control:
            ratio = trg_lengths.float() / src_lengths.float()
            lengths = trg_lengths.float() * self.Wl
            L = torch.stack([ratio, lengths], -1)
        else:
            L = None

        if isinstance(hidden, tuple):  # LSTM
            # concat directions
            hidden = tuple(_fix_hidden(h) for h in hidden)
            outs = tuple([bottle_hidden(state, hidden[ix], L)
                          for ix, state in enumerate(bridge)])
        else:
            outs = bottle_hidden(bridge[0], hidden)

        return outs

    def get_embeddings(self):
        return self.embedding

    def initialize_embeddings_idf(self, idf):
        idf_embs = torch.from_numpy(idf).float().unsqueeze(-1)
        self.idf = nn.Embedding.from_pretrained(idf_embs, freeze=True)

    def set_embedding_gradient_mask(self, mask):
        self.inp_encoder.embed.set_grad_mask(mask)
        self.cmp_encoder.embed.set_grad_mask(mask)
        self.compressor.embed.set_grad_mask(mask)
        self.decompressor.embed.set_grad_mask(mask)

    def _fake_inputs(self, inputs, latent_lengths, pad=1):
        batch_size, seq_len = inputs.size()

        if latent_lengths is not None:
            max_length = int(max(latent_lengths).item())
        else:
            max_length = seq_len + pad

        fakes = torch.zeros(batch_size, max_length, device=inputs.device)
        fakes = fakes.type_as(inputs)
        fakes[:, 0] = self.sos
        return fakes

    def generate(self, inputs, src_lengths, trg_seq_len):
        # ENCODER
        enc1_results = self.inp_encoder(inputs, None, src_lengths)
        outs_enc1, hn_enc1 = enc1_results[-2:]

        # DECODER
        dec_init = self._bridge(self.src_bridge, hn_enc1, src_lengths,
                                trg_seq_len)
        inp_fake = self._fake_inputs(inputs, trg_seq_len)
        # dec1_results = self.compressor(inp_fake, outs_enc1, dec_init,
        #                                argmax=True,
        #                                enc_lengths=src_lengths,
        #                                sampling_prob=1.,
        #                                desired_lengths=trg_seq_len)
        dec1_results = self.compressor(inp_fake, outs_enc1, dec_init,
                                       enc_lengths=src_lengths,
                                       sampling_prob=1.,
                                       desired_lengths=trg_seq_len)
        return enc1_results, dec1_results
    # latent_length应该就是论文中的M
    def forward(self, inp_src, inp_trg,
                src_lengths, latent_lengths,
                sampling, tau=1, hard=True):

        """
        This approach utilizes 4 RNNs. The latent representation is obtained
        using an L2 decoder, which decodes the L1 in its own language
        and feeds the

                      L2-encoder -> L2-decoder
                           ^
                        e1,  e2,  ...  decoder-embs   (sampled embeddings)
                        p1,  p2,  ...  decoder-dists  (distribution)
                        c1,  c2,  ...  context-states (re-weighted)
                        h1,  h2,  ...  decoder-states
                           ^
        L1-encoder -> L2-decoder
        Args:
            inp_src:输入的轨迹，其中是vocab中的id
            inp_trg:在inp_src的前面加上了一个<sos>起始符
        """
        # --------------------------------------------
        # ENCODER-1 (Compression)
        # --------------------------------------------
        enc1_results = self.inp_encoder(inp_src, None, src_lengths,
                                        word_dropout=self.enc_token_dropout)
        # encoder的输出以及最后一个的隐向量.注意此处是双向的lstm。num_layer = 2
        # outs_enc1(batch,seq_len,hid_size * bidirection), hn_enc1 (num_layer,batch,hid_size) * bidirection
        outs_enc1, hn_enc1 = enc1_results[-2:]

        # --------------------------------------------
        # DECODER-1 (Compression)
        # --------------------------------------------
        # 大概是将hn_enc1进行加工之后的隐状态 _dec1_init (num_layer/2,batch,hid_size) * bidirection
        _dec1_init = self._bridge(self.src_bridge, hn_enc1, src_lengths,
                                  latent_lengths)
        # inp_fake (batch,seq_len + 1),因为在decoder中，输入是未知的，即需要上一个时刻的输出来作为下一时刻的输入，所以此处相当于先声明一个向量，先给他先送进去。
        inp_fake = self._fake_inputs(inp_src, latent_lengths)
        dec1_results = self.compressor(inp_fake, outs_enc1, _dec1_init,
                                       enc_lengths=src_lengths,
                                       sampling_prob=1., hard=hard, tau=tau,
                                       desired_lengths=latent_lengths)
        # logits_dec1(batch,latent_size,vocab_size -> 此处还未映射) outs_dec1 (batch,seq_len+1,hid_size), dists_dec1 (batch,seq_len,vocab_size):已经one-hot的结果
        logits_dec1, outs_dec1, _, dists_dec1, _, _ = dec1_results
        # --------------------------------------------
        # ENCODER-2 (Reconstruction)
        # --------------------------------------------
        # cmp_embeddings (batch,seq_len,emb_size)
        cmp_embeddings = self.compressor.embed.expectation(dists_dec1)
        cmp_lengths = latent_lengths - 1 # 删除之前的<sos>起始符，因为作为下一个encoder的输入是不需要起始符的

        # !!! Limit the communication only through the embs
        # The compression encoder reads only the sampled embeddings
        # so it is initialized with a zero state
        enc2_init = None
        enc2_results = self.cmp_encoder.encode(cmp_embeddings, enc2_init,
                                               cmp_lengths)
        outs_enc2, hn_enc2 = enc2_results[-2:]

        # --------------------------------------------
        # DECODER-2 (Reconstruction)
        # --------------------------------------------
        dec2_lengths = src_lengths + 1  # <sos> + src
        _dec2_init = self._bridge(self.trg_bridge, hn_enc2, cmp_lengths,
                                  dec2_lengths)
        # 此处的inp_trg和上面的inp_fake其实是一样的，里面的值是啥都无所谓，主要是维度。那么此处需要恢复成最开始的长度，为了方便就直接将inp_trg给入了。
        dec2_results = self.decompressor(inp_trg, outs_enc2, _dec2_init,
                                         enc_lengths=cmp_lengths,
                                         sampling_prob=sampling,
                                         tau=tau,
                                         desired_lengths=dec2_lengths,
                                         word_dropout=self.dec_token_dropout)

        return enc1_results, dec1_results, enc2_results, dec2_results
