from generate.utils import sematic_simp
from generate.exp_semantic import load_model
import torch
import os
from sys_config import DATA_DIR

class CEDer:
    def __init__(self, path=None, datasets="infer", checkpoint="seq3.full_-valid", device="cuda"):
        src_file = os.path.join(DATA_DIR, datasets + ".src")
        _, model, vocab = load_model(path, checkpoint, src_file, device)
        self.model = model
        self.vocab = vocab

    def CED_op(self, idx, src_vid):
        src_vid = [self.vocab.tok2id[i] for i in src_vid]
        comp_vid_adp = [src_vid[i] for i in idx]
        loss = sematic_simp(self.model, src_vid, comp_vid_adp, self.vocab)
        return loss


if __name__ == '__main__':
    seed = 1
    device = "cuda"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    ceder = CEDer()
    # _, idx, _ = method(points, buffer_size, 'ped')