import os
import pickle

from sys_config import DATA_DIR

global gl_gid2poi
gl_gid2poi = None

global gl_vocab
gl_vocab = None

def get_gid2poi():
    global gl_gid2poi
    if gl_gid2poi is None:
        with open(os.path.join(DATA_DIR, 'gid2poi.txt'), 'rb') as f:
            print("loading get_gid2poi")
            var_a = pickle.load(f)
        gl_gid2poi = pickle.loads(var_a)
    return gl_gid2poi

def get_vocab():
    return gl_vocab