import os

import torch

from sys_config import DATA_DIR, BASE_DIR
# from generate.utils import compress_seq3
from generate.utils import compress_seq3
from utils.viz import seq3_attentions

# path = os.path.join(BASE_DIR, "checkpoints/best_model")
path = None
checkpoint = "seq3.full_-valid"
# checkpoint = "seq3.full"
seed = 1
device = "cpu"
verbose = True
out_file = ""
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

src_file = os.path.join("../datasets/geolife/infer.src")

out_file = os.path.join(BASE_DIR, f"evaluation/{checkpoint}_preds.txt")

results = compress_seq3(path, checkpoint, src_file, out_file, device, True,
                        mode="debug")

