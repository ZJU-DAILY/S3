import itertools
import math
import os
import warnings

import numpy
import torch
from tabulate import tabulate
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from generate.utils import devectorize
from models.seq3_trainer import Seq3Trainer
from models.seq3_utils import compute_dataset_idf
from modules.data.collates import Seq2SeqCollate, Seq2SeqOOVCollate
from modules.data.datasets import AEDataset
from modules.data.samplers import BucketBatchSampler
from modules.models import Seq2Seq2Seq
from modules.module import SeqReader
# from mylogger.attention import samples2html
# from mylogger.experiment import Experiment
from sys_config import EXP_DIR, EMBS_PATH, MODEL_CNF_DIR
from utils.eval import rouge_file_list, pprint_rouge_scores
from utils.generic import number_h
from utils.opts import seq2seq2seq_options
from utils.training import load_checkpoint
from utils.transfer import freeze_module
from seq3_utils import DataLoader

####################################################################
# Settings
####################################################################
opts, config = seq2seq2seq_options()

####################################################################
#
# Weight Transfer
#
####################################################################
vocab = None

####################################################################
#
# Data Loading and Preprocessing
#
####################################################################



print("Building training dataset...")
trainsrc = os.path.join(config["data"]["base_dir"], "train.src")
trainData = DataLoader(trainsrc, 128,
                       [(20, 30), (30, 30), (30, 50), (50, 50), (50, 70), (70, 70), (70, 100), (100, 100)])
# 加载的最大轨迹条数
trainData.load(5000)

print("Building validation dataset...")
valsrc = os.path.join(config["data"]["base_dir"], "val.src")
if os.path.isfile(valsrc):
    valData = DataLoader(valsrc, 128,
                         [(20, 30), (30, 30), (30, 50), (50, 50), (50, 70), (70, 70), (70, 100), (100, 100)], True)
    print("Reading validation data...")
    valData.load()
    assert valData.size > 0, "Validation data size must be greater than 0"
    print("Loaded validation data size {}".format(valData.size))
else:
    print("No validation data found, training without validating...")

# train_lengths = len(trainData)


####################################################################
#
# Model Definition
# - additional layer initializations
# - weight / layer tying
#
####################################################################

# Define the model
n_tokens = 10
model = Seq2Seq2Seq(n_tokens, **config["model"])
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Load Pretrained Word Embeddings
embs = model.get_embeddings()
try:
    model.compressor.Wo.weight.data.copy_(torch.from_numpy(embs))
    model.decompressor.Wo.weight.data.copy_(torch.from_numpy(embs))
except:
    print("Can't init outputs from embeddings. Dim mismatch!")

####################################################################
#
# Tie Models
#
####################################################################

# tie the output layers of the decoders
if config["model"]["tie_decoder_outputs"]:
    model.compressor.Wo = model.decompressor.Wo

# tie the embedding to the output layers
if config["model"]["tie_embedding_outputs"]:
    emb_size = model.compressor.embed.weight.size(1)
    rnn_size = model.compressor.Wo.weight.size(1)

    if emb_size != rnn_size:
        warnings.warn("Can't tie outputs, since emb_size != rnn_size.")
    else:
        model.compressor.Wo.weight = model.inp_encoder.embed.weight
        model.decompressor.Wo.weight = model.inp_encoder.embed.weight

if config["model"]["tie_decoders"]:
    model.compressor = model.decompressor

if config["model"]["tie_encoders"]:
    model.cmp_encoder = model.inp_encoder

# then we need only one bridge
if config["model"]["tie_encoders"] and config["model"]["tie_decoders"]:
    model.src_bridge = model.trg_bridge

####################################################################
#
# Experiment Logging and Visualization
#
####################################################################

parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters,
                             lr=config["lr"],
                             weight_decay=config["weight_decay"])

model.to(opts.device)
print(model)

total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters()
                             if p.requires_grad)

print("Total Params:", number_h(total_params))
print("Total Trainable Params:", number_h(total_trainable_params))
trainable_params = sorted([[n] for n, p in model.named_parameters()
                           if p.requires_grad])
