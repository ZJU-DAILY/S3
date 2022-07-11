import torch
from torch.nn import functional as F

from models.seq3_losses import _kl_div, kl_length, pairwise_loss, sed_loss, energy_, energy_2
from models.seq3_utils import sample_lengths
from modules.helpers import sequence_mask, avg_vectors, module_grad_wrt_loss
from modules.training.trainer import Trainer
from generate.utils import devectorize
from numpy import mean
from utils.data_parsing import getArea
import numpy as np
from generate.utils import getCompress


class Seq3Trainer(Trainer):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.oracle = kwargs.get("oracle", None)
        self.top = self.config["model"]["top"]
        self.hard = self.config["model"]["hard"]
        self.sampling = self.anneal_init(self.config["model"]["sampling"])
        self.tau = self.anneal_init(self.config["model"]["tau"])
        self.len_min_rt = self.anneal_init(self.config["model"]["min_ratio"])
        self.len_max_rt = self.anneal_init(self.config["model"]["max_ratio"])
        self.len_min = self.anneal_init(self.config["model"]["min_length"])
        self.len_max = self.anneal_init(self.config["model"]["max_length"])
        self.mask = self.config["model"]["mask"]
        self.mask_fn = self.config["model"]["mask_fn"]

    def _debug_grads(self):
        return list(sorted([(n, p.grad) for n, p in
                            self.model.named_parameters() if p.requires_grad]))

    def _debug_grad_norms(self, reconstruct_loss, local_topic_loss, topic_loss):
        c_grad_norm = []
        c_grad_norm.append(
            module_grad_wrt_loss(self.optimizers, self.model.compressor,
                                 reconstruct_loss,
                                 "rnn"))

        if self.config["model"]["topic_loss"]:
            c_grad_norm.append(
                module_grad_wrt_loss(self.optimizers, self.model.compressor,
                                     topic_loss,
                                     "rnn"))

        if self.config["model"]["local_topic_loss"]:
            c_grad_norm.append(
                module_grad_wrt_loss(self.optimizers, self.model.compressor,
                                     local_topic_loss,
                                     "rnn"))
        return c_grad_norm

    def _topic_loss(self, inp, dec1, src_lengths, trg_lengths):
        """
        Compute the pairwise distance of various outputs of the seq^3 architecture.
        Args:
            enc1: the outputs of the first encoder (input sequence)
            dec1: the outputs of the first decoder (latent sequence)
            src_lengths: the lengths of the input sequence
            trg_lengths: the lengths of the targer sequence (summary)

        """

        enc_mask = sequence_mask(src_lengths).unsqueeze(-1).float()
        dec_mask = sequence_mask(trg_lengths - 1).unsqueeze(-1).float()

        enc_embs = self.model.inp_encoder.embed(inp)
        if dec1[3] is None:
            print("dec1[3] is none")
        dec_embs = self.model.compressor.embed.expectation(dec1[3])

        if self.config["model"]["topic_idf"]:
            enc1_energies = self.model.idf(inp)
            # dec1_energies = expected_vecs(dec1[3], self.model.idf.weight)

            x_emb, att_x = avg_vectors(enc_embs, enc_mask, enc1_energies)
            # y_emb, att_y = avg_vectors(dec_reps, dec_mask, dec1_energies)
            y_emb, att_y = avg_vectors(dec_embs, dec_mask)

        else:
            x_emb, att_x = avg_vectors(enc_embs, enc_mask)
            y_emb, att_y = avg_vectors(dec_embs, dec_mask)

        distance = self.config["model"]["topic_distance"]
        loss = pairwise_loss(x_emb, y_emb, distance)

        return loss, (att_x, att_y)

    def _local_topic_loss(self, inp, dec1, src_lengths, trg_lengths):
        """
        Compute the pairwise distance of various outputs of the seq^3 architecture.
        Args:
            enc1: the outputs of the first encoder (input sequence)
            dec1: the outputs of the first decoder (latent sequence)
            src_lengths: the lengths of the input sequence
            trg_lengths: the lengths of the targer sequence (summary)

        """

        enc_mask = sequence_mask(src_lengths).unsqueeze(-1).float()
        dec_mask = sequence_mask(trg_lengths - 1).unsqueeze(-1).float()
        # 头尾向量不参与计算
        enc_mask[:, 0, :] = 0
        enc_mask[:, -1, :] = 0
        dec_mask[:, 0, :] = 0
        dec_mask[:, -1, :] = 0

        enc_embs = self.model.inp_encoder.embed(inp)
        if dec1[3] is None:
            print("dec1[3] is none")
        dec_embs = self.model.compressor.embed.expectation(dec1[3])

        vocab = self._get_vocab()
        region = self._get_region()
        # src = [[vocab.id2tok.get(p.item()) for p in seq] for seq in inp]
        src = devectorize(inp.tolist(), vocab.id2tok, vocab.tok2id[vocab.EOS],
                          strip_eos=None, oov_map=None, pp=True)
        # energies = energy_2(region, src, enc_mask.size(1),src_lengths)
        energies = energy_(region, src, enc_mask.size(1))
        enc1_energies = torch.tensor(energies, dtype=float).view(enc_mask.size()).to(enc_mask.device)
        enc1_energies += 0.1

        x_emb, att_x = avg_vectors(enc_embs, enc_mask, enc1_energies)
        y_emb, att_y = avg_vectors(dec_embs, dec_mask)

        distance = self.config["model"]["local_topic_distance"]
        loss = pairwise_loss(x_emb, y_emb, distance)

        return loss, (att_x, att_y)

    def _prior_loss(self, outputs, latent_lengths):
        """
        Prior Loss
        Args:
            outputs:
            latent_lengths:

        Returns:

        """
        enc1, dec1, enc2, dec2 = outputs
        _vocab = self._get_vocab()

        logits_dec1, outs_dec1, hn_dec1, dists_dec1, _, _ = dec1

        # dists_dec1 contain the distributions from which
        # the samples were taken. It contains one less element than the logits
        # because the last logit is only used for computing the NLL of EOS.
        words_dec1 = dists_dec1.max(-1)[1]

        # sos + the sampled sentence
        sos_id = _vocab.tok2id[_vocab.SOS]
        sos = torch.zeros_like(words_dec1[:, :1]).fill_(sos_id)
        oracle_inp = torch.cat([sos, words_dec1], -1)

        logits_oracle, _, _ = self.oracle(oracle_inp, None,
                                          latent_lengths)

        prior_loss, prior_loss_time = _kl_div(logits_dec1,
                                              logits_oracle,
                                              latent_lengths)

        return prior_loss, prior_loss_time, logits_oracle

    def _sed_loss(self, inp, dec1):
        vocab = self._get_vocab()
        region = self._get_region()
        src = devectorize(inp.tolist(), vocab.id2tok, vocab.tok2id[vocab.EOS],
                          strip_eos=None, oov_map=None, pp=True)
        trg = devectorize(dec1[3].max(-1)[1].tolist(), vocab.id2tok, vocab.tok2id[vocab.EOS],
                          strip_eos=None, oov_map=None, pp=True)
        comp_trj = [getCompress(region, src_, trg_)[0] for src_, trg_ in zip(src, trg)]
        # print(comp_trj,src)
        loss = [sed_loss(region, src_, trg_) for src_, trg_ in zip(src, comp_trj)]
        # loss = sed_loss(region, src, comp_trj)

        return mean(loss)


    def r(self,p,trj):
        batch = p.size(0)
        ll = torch.zeros(batch).to(p)
        max_len = trj.size(1)
        for i in range(max_len):
            p2 = trj[:,i,:]
            l = pairwise_loss(p, p2, dist="cosine")
            ll.add(l)

        # for i in range(batch):
        #     p_vector = p[i]
        #     trj_vector = trj[i]
        #     tmp = []
        #     for p2_vector in trj_vector:
        #         l = pairwise_loss(p_vector, p2_vector, dist="cosine")
        #         tmp.append(l)
        #     ll.append(np.max(tmp))
        # return np.array(ll)
        return ll.tolist()

    def _sematic_loss(self, inp, dec1, src_lengths, trg_lengths):

        enc_mask = sequence_mask(src_lengths).unsqueeze(-1).float()
        dec_mask = sequence_mask(trg_lengths - 1).unsqueeze(-1).float()

        enc_embs = self.model.inp_encoder.embed(inp) * enc_mask
        if dec1[3] is None:
            print("dec1[3] is none")
        dec_embs = self.model.compressor.embed.expectation(dec1[3]) * dec_mask

        # 先对source序列中的点进行遍历
        max_len = enc_embs.size(1)
        sim = np.zeros(enc_embs.size(0))
        for i in range(max_len):
            batch_p = enc_embs[:,i,:]
            batch_trj = dec_embs
            tmp = self.r(batch_p,batch_trj)
            sim += tmp
        # 计算的是所有batch的
        sim = [sim[i] / length.item() for i, length in enumerate(src_lengths)]
        # 将所有batch的结果求平均
        sim = np.mean(sim)

        # 接下来对trg_src做类似的处理
        max_len = dec_embs.size(1)
        sim_2 = np.zeros(dec_embs.size(0))
        for i in range(max_len):
            batch_p = dec_embs[:, i, :]
            batch_trj = enc_embs
            tmp = self.r(batch_p, batch_trj)
            sim_2 += tmp
        # 计算的是所有batch的
        sim_2 = [sim_2[i] / length.item() for i, length in enumerate(trg_lengths)]
        # 将所有batch的结果求平均
        sim_2 = np.mean(sim_2)
        print()
        return (sim + sim_2) / 2

    def _process_batch(self, inp_x, out_x, inp_xhat, out_xhat,
                       x_lengths, xhat_lengths):

        self.model.train()

        tau = self.anneal_step(self.tau)
        sampling = self.anneal_step(self.sampling)
        len_min_rt = self.anneal_step(self.len_min_rt)
        len_max_rt = self.anneal_step(self.len_max_rt)
        len_min = self.anneal_step(self.len_min)
        len_max = self.anneal_step(self.len_max)
        vocab = self._get_vocab()
        region = self._get_region()

        latent_lengths = sample_lengths(x_lengths,
                                        len_min_rt, len_max_rt,
                                        len_min, len_max)

        if self.mask:
            if self.mask_fn == "area":
                # Schema 1 for mask: A area
                m_zeros = torch.zeros(inp_x.size(0), vocab.size).to(inp_x)
                idxs = np.zeros([inp_x.size(0), vocab.size], dtype='int64')
                i = 0
                for seq in inp_x:
                    idx = getArea(region, vocab, seq.tolist())
                    idxs[i, :] = idx
                    i += 1
                mask = torch.tensor(idxs).to(inp_x)
                mask_matrix = m_zeros.scatter(1, mask, 1)
            elif self.mask_fn == "trj":
                # Schema 2 for mask: A trj
                m_zeros = torch.zeros(inp_x.size(0), vocab.size).to(inp_x)
                mask_matrix = m_zeros.scatter(1, inp_x, 1)
                mask_matrix[:, 0] = 0
        else:
            mask_matrix = None

        outputs = self.model(inp_x, inp_xhat,
                             x_lengths, latent_lengths, sampling, tau, mask_matrix, region=region, vocab=vocab)

        enc1, dec1, enc2, dec2 = outputs

        batch_outputs = {"model_outputs": outputs}

        # --------------------------------------------------------------
        # 1 - RECONSTRUCTION
        # --------------------------------------------------------------
        # reconstruct_loss = self._seq_loss(dec2[0], out_xhat)
        # 维度变化，将向量拉直（batch * max_length）
        _dec2_logits = dec2[0].contiguous().view(-1, dec2[0].size(-1))
        _x_labels = out_xhat.contiguous().view(-1)
        # 将_dec2_logits中的概率转化为对应的one-hot（即对应的词）的过程也包含在交叉熵计算中了
        reconstruct_loss = F.cross_entropy(_dec2_logits, _x_labels,
                                           ignore_index=0,
                                           reduction='none')
        # F.mse_loss()

        reconstruct_loss_token = reconstruct_loss.view(out_xhat.size())
        batch_outputs["reconstruction"] = reconstruct_loss_token
        mean_rec_loss = reconstruct_loss.sum() / xhat_lengths.float().sum()
        losses = [mean_rec_loss]

        # --------------------------------------------------------------
        # 2 - LOCAL_TOPIC
        # --------------------------------------------------------------
        if self.config["model"]["local_topic_loss"]:
            local_topic_loss, weight = self._local_topic_loss(inp_x, dec1,
                                                              x_lengths,
                                                              latent_lengths)
            batch_outputs["local_weight"] = weight
            losses.append(local_topic_loss)
        else:
            local_topic_loss = None

        # --------------------------------------------------------------
        # 3 - TOPIC
        # --------------------------------------------------------------
        if self.config["model"]["topic_loss"]:
            topic_loss, attentions = self._topic_loss(inp_x, dec1,
                                                      x_lengths,
                                                      latent_lengths)
            batch_outputs["attention"] = attentions
            losses.append(topic_loss)
        else:
            topic_loss = None

        # --------------------------------------------------------------
        # 4 - LENGTH
        # --------------------------------------------------------------
        if self.config["model"]["length_loss"]:
            _vocab = self._get_vocab()
            eos_id = _vocab.tok2id[_vocab.EOS]
            length_loss = kl_length(dec1[0], latent_lengths, eos_id)
            losses.append(length_loss)

        # --------------------------------------------------------------
        # Plot Norms of loss gradient wrt to the compressor
        # --------------------------------------------------------------
        if self.config["plot_norms"] and self.step % self.config[
            "log_interval"] == 0:
            batch_outputs["grad_norm"] = self._debug_grad_norms(
                mean_rec_loss,
                local_topic_loss,
                topic_loss)

        return losses, batch_outputs

    def eval_epoch(self):
        """
        Evaluate the network for one epoch and return the average loss.

        Returns:
            loss (float, list(float)): list of mean losses

        """
        self.model.eval()

        results = []
        oov_maps = []
        loss_sum = 0

        self.len_min_rt = self.anneal_init(
            self.config["model"]["test_min_ratio"])
        self.len_max_rt = self.anneal_init(
            self.config["model"]["test_max_ratio"])
        self.len_min = self.anneal_init(
            self.config["model"]["test_min_length"])
        self.len_max = self.anneal_init(
            self.config["model"]["test_max_length"])
        vocab = self._get_vocab()
        region = self._get_region()

        iterator = self.valid_loader
        with torch.no_grad():
            for i_batch, batch in enumerate(iterator, 1):
                batch_oov_map = batch[-1]
                batch = batch[:-1]

                batch = list(map(lambda x: x.to(self.device), batch))
                (inp_src, out_src, inp_trg, out_trg,
                 src_lengths, trg_lengths) = batch

                latent_lengths = sample_lengths(src_lengths,
                                                self.len_min_rt,
                                                self.len_max_rt, self.len_min,
                                                self.len_max)
                # if self.mask:
                #     # Schema 2 for mask: A trj
                #     m_zeros = torch.zeros(inp_src.size(0), vocab.size).to(inp_src)
                #     mask_matrix = m_zeros.scatter(1, inp_src, 1)
                # else:
                #     mask_matrix = None

                # 在推理的时候默认使用mask
                m_zeros = torch.zeros(inp_src.size(0), vocab.size).to(inp_src)
                mask_matrix = m_zeros.scatter(1, inp_src, 1)

                # 模型推理
                enc, dec = self.model.generate(inp_src, src_lengths,
                                               latent_lengths, mask_matrix=mask_matrix, inp_src=inp_src, vocab=vocab,
                                               region=region)
                losses = []
                # --------------------------------------------------------------
                # 1 - SED loss
                # --------------------------------------------------------------
                loss_sed = self._sed_loss(inp_src, dec)

                self._sematic_loss(inp_src, dec, src_lengths, latent_lengths)
                # --------------------------------------------------------------
                # 2 - LENGTH Penalty
                # --------------------------------------------------------------
                # if self.config["model"]["length_loss"]:
                #     _vocab = self._get_vocab()
                #     eos_id = _vocab.tok2id[_vocab.EOS]
                #     length_loss = kl_length(dec[0], latent_lengths, eos_id)
                #     losses.append(length_loss)
                # --------------------------------------------------------------
                # 3 - TOPIC
                # --------------------------------------------------------------
                # if self.config["model"]["topic_loss"]:
                #     topic_loss, _ = self._topic_loss(inp_src, dec,
                #                                      src_lengths,
                #                                      latent_lengths)
                #     losses.append(topic_loss)
                loss_sum = loss_sed  # + loss_sum + sum(losses).item()
        return loss_sum

    def _get_vocab(self):
        if isinstance(self.train_loader, (list, tuple)):
            dataset = self.train_loader[0].dataset
        else:
            dataset = self.train_loader.dataset

        if dataset.subword:
            _vocab = dataset.subword_path
        else:
            _vocab = dataset.vocab

        return _vocab

    def _get_region(self):
        if isinstance(self.train_loader, (list, tuple)):
            dataset = self.train_loader[0].dataset
        else:
            dataset = self.train_loader.dataset

        _region = dataset.region

        return _region

    def get_state(self):

        state = {
            "config": self.config,
            "epoch": self.epoch,
            "step": self.step,
            "model": self.model.state_dict(),
            "model_class": self.model.__class__.__name__,
            "optimizers": [x.state_dict() for x in self.optimizers],
            "vocab": self._get_vocab(),
        }

        return state
