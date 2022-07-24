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
    for i in range(st + 1, en):
        err = getDistance(None, points[i], points[st], points[en], mode)
        if err > max_err:
            idx = i
            max_err = err
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
    err_map,err_id, err_list = create_err_space(points, mode)
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
