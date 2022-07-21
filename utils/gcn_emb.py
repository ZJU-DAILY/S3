import numpy as np
# from pars_args import args
from scipy import sparse
from scipy.sparse import lil_matrix

from models.constants import zero_poi


def load_weight(emb_file):
    # 简单处理一下，开一个74671 + 1(oov向量)大小的emb_w
    weights_l = np.loadtxt(emb_file,skiprows=1)
    emb_w = np.zeros((zero_poi + 1,weights_l.shape[1]-1))
    # emb_w = np.zeros((weights_l.shape[0]+1,weights_l.shape[1]-1))
    for weight in weights_l:
        idx = int(weight[0])
        emb_w[idx,:] = weight[1:]
    return emb_w


def normalize_adj(adj, alpha=1.):
    adj = sparse.eye(adj.shape[0]) + alpha * adj
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sparse.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocsr()

def generate_graph(alpha,graph_file,len_n):

    graph = lil_matrix((len_n,len_n))
    edge_list = np.loadtxt(graph_file,dtype=int)
    for edge in edge_list:
        graph[edge[0],edge[1]] = 1
        graph[edge[1],edge[0]] = 1


    return normalize_adj(graph.tocsr(),alpha)

# alpha默认为0.5，相当于是论文中的拉普拉斯变化
def gcn_emb(emb_file="../datasets/beijing.emd",alpha=0.5,graph_file="../datasets/edge.edgelist"):


    emb = load_weight(emb_file)
    len_n = emb.shape[0] - 1
    A = generate_graph(alpha,graph_file,len_n)
    emb[:len_n,:] = A.dot(emb[:len_n,:])

    return emb


# emb = gcn_emb("../datasets/beijing.emd",0.5,"../datasets/edge.edgelist")
# # emb = gcn_emb("../datasets/karate.emd",0.5,"../datasets/edge.edgelist")
# gcn_emb()
# print()