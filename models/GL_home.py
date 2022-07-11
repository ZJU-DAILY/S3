# todo 完成全局变量的配置
import pickle

global gl_gid2poi
gl_gid2poi = None

def get_gid2poi():
    global gl_gid2poi
    if gl_gid2poi is None:
        with open('../datasets/gid2poi.txt', 'rb') as f:
            print("loading get_gid2poi")
            var_a = pickle.load(f)
        gl_gid2poi = pickle.loads(var_a)
    return gl_gid2poi
