# -*- coding: utf-8 -*-
# @Time    : 2022/7/23 13:27
# @Author  : HeAlec
# @FileName: error_search.py
# @Desc: description
# @blogs ：https://segmentfault.com/u/alec_5e8962e4635ca
from modules.helpers import getDistance, getMaxError
import numpy as np


def adp_min_size(points, epi, st, en, mode):
    res = set()
    if en - st <= 1:
        return res
    max_err = -1
    idx = -1
    idx, max_err = getMaxError(st, en, points, mode)
    # for i in range(st + 1, en):
    #     # err = getDistance(None, (points[i],i), (points[st],st), (points[en],en), mode)
    #
    #     if err > max_err:
    #         idx = i
    #         max_err = err
    if max_err > epi:
        left_res = adp_min_size(points, epi, st, idx, mode)
        right_res = adp_min_size(points, epi, idx, en, mode)
        res.update(left_res)
        res.update(right_res)
    else:
        res.add(st)
        res.add(en)
    return res


def create_err_space(points, mode):
    mmp = np.zeros([len(points), len(points)])
    mmp_id = np.zeros([len(points), len(points)], dtype=np.int8)
    mmp_list = []
    for st in range(len(points)):
        for en in range(st + 2, len(points)):
            id, err = getMaxError(st, en, points, mode)
            mmp[st][en] = err
            mmp_id[st][en] = id
            mmp_list.append(err)
    mmp_list.sort()
    return mmp, mmp_id, mmp_list


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


def get_out(R, st, en, res):
    if st + 1 >= en or R[st][en] == -1:
        return []
    sp = R[st][en]
    left = get_out(R, st, sp, res)
    right = get_out(R, sp, en, res)
    res.extend(left)
    res.extend(right)
    res.append(sp)


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


# if __name__ == '__main__':
#     points = [(1, 2), (3, 7), (5, 4), (8, 9), (5, 1), (3, 6)]
#     max_len = 3
#     bellman(points, max_len, 'sed')
