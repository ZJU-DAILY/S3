from sys_config import DATA_DIR
# adaptive douglas-peucker
import functools
import os
from generate.utils import *


def readData(src_file, region):
    with open(src_file, "r") as f:
        ss = f.readlines()
    points = []
    src = []
    for s in ss:
        s = s.strip("\n")
        s = s.split(" ")
        point = []
        src_ = []
        for p in s:
            if p == " " or p == "UNK":
                continue
            point.append(cell2gps(region, int(p)))
            src_.append(p)
        points.append(point)
        src.append(src_)
    return points, src


def adp(points, max_len, mode):
    if max_len > len(points):
        return None
    q = []
    st = 0
    en = len(points) - 1
    q.append({(st, en): getMaxError(st, en, points, mode)})
    cnt = 2
    res = [st, en]
    maxErr = -1
    while cnt < max_len:
        q.sort(key=functools.cmp_to_key(cmp))
        solu = q.pop()
        cnt += 1
        (st, en), (split, maxErr) = getKey_Value(solu)
        q.append({(st, split): getMaxError(st, split, points, mode)})
        q.append({(split, en): getMaxError(split, en, points, mode)})
        res.append(split)
    q.sort(key=functools.cmp_to_key(cmp))
    solu = q.pop()
    (_, _), (_, maxErr) = getKey_Value(solu)
    pp = []
    res.sort()
    for idx in res:
        pp.append(points[idx])
    # 坐标点，下标，最大误差
    return pp, res, maxErr


# todo 查找相关bug，err递增很诡异
def btup(points, max_len, mode):
    # def cal_len(segs):
    #     res = set()
    #     for i in segs:

    segs = []
    for i in range(0, len(points) - 1, 2):
        st = i
        en = min(len(points) - 1, i + 1)
        segs.append([st, en])

    # merge_cost = []
    # for i in range(len(segs) - 1):
    #     merge_cost.append(calculate_error(points,segs[i],segs[i+1],mode))
    #
    while len(segs) + 1 > (max_len // 2):
        merge_cost = []
        min_cost = float('inf')
        min_idx = -1
        for i in range(len(segs) - 1):
            err = calculate_error(points, segs[i], segs[i + 1], mode)
            merge_cost.append(err)
            if err < min_cost:
                min_cost = err
                min_idx = i
        head = segs[min_idx]
        tail = segs[min_idx + 1]
        merge = [head[0], tail[-1]]
        segs.insert(min_idx, merge)
        segs.remove(head)
        segs.remove(tail)
        merge_cost.pop(min_idx)

    max_err = -1
    pp = []
    idx = []
    for i in range(len(segs)):
        max_err = max(max_err, calculate_error(points, [segs[i][0]], [segs[i][-1]], mode))
        # if min_cost == float('inf'):
        idx.append(segs[i][0])
        pp.append(points[idx[-1]])
        if i == len(segs) - 1:
            idx.append(segs[i][1])
            pp.append(points[idx[-1]])
    if len(points) - 1 not in idx:
        if len(idx) == max_len:
            idx[-1] = len(points) - 1
        else:
            idx.append(len(points) - 1)
    min_cost = max_err
    return pp, idx, min_cost


def bellman(points, max_len, mode):
    err_map, err_id, err_list = create_err_space(points, mode)
    n = len(points)
    E = np.zeros([n + 1, n + 1])
    R = np.zeros([n + 1, n + 1])
    for i in range(1, n):
        E[i][2] = err_map[i - 1][n - 1]
    for k in range(3, max_len + 1):
        for i in range(1, n - k + 1):
            min_err = float('inf')
            id = -1
            for h in range(i + 1, n):
                error_v = max(err_map[i - 1][h - 1], E[h][k - 1])
                if error_v < min_err:
                    min_err = error_v
                    id = h
            R[i][k] = id
            E[i][k] = min_err
    max_err = E[1][max_len]
    res = []
    # get_out(R, 1, max_len, res)
    res.sort()
    return res, max_err


def error_search_algorithm(points, max_len, mode):
    err_map, err_id, err_list = create_err_space(points, mode)
    st = 0
    en = len(err_list) - 1
    optim_comp = None
    cur_err = -1
    gap = len(points)
    while st < en:
        mid = (st + en) // 2
        err = err_list[mid]
        comp = adp_min_size(points, err, st, len(points) - 1, mode)

        if len(comp) <= max_len:
            en = mid
            if abs(len(comp) - max_len) < gap:
                optim_comp = comp
                cur_err = err
                gap = abs(len(comp) - max_len)
        else:
            st = mid + 1
    optim_comp = list(optim_comp)
    # print(gap)
    while gap > 0:
        optim_comp.sort()
        new = -1
        max_err = -1
        for i in range(len(optim_comp) - 1):
            err = err_map[optim_comp[i]][optim_comp[i + 1]]
            id = err_id[optim_comp[i]][optim_comp[i + 1]]
            if err > max_err:
                new = id
                max_err = err
        optim_comp.append(new)
        cur_err = max_err
        gap -= 1
    return optim_comp, cur_err


if __name__ == '__main__':
    src_file = "../../datasets/geolife/infer.src"
    with open(os.path.join(DATA_DIR, 'pickle.txt'), 'rb') as f:
        var_a = pickle.load(f)
    region = pickle.loads(var_a)
    points, src = readData(src_file, region)
    ss = ""
    mer = "ped"
    for k in range(len(points)):
        _, idx, _ = adp(points[k], 9, mer)
        idx.sort()
        res = [str(src[k][i]) for i in idx]
        ss += " ".join(res) + "\n"
    with open(f"../../datasets/casestudy/{mer}.txt", "w") as f:
        f.write(ss)
