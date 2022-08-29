import os
import sys

sys.path.append('/home/hch/Desktop/trjcompress/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import time
import numpy as np

from generate.online.RLOnline.rl_brain import PolicyGradient
from generate.online.RLOnline.rl_env_inc import TrajComp
from generate.utils import getPED4GPS, getMaxError
from preprocess.SpatialRegionTools import cell2gps
import os
from preprocess.SpatialRegionTools import SpacialRegion
from sys_config import DATA_DIR
from generate.online.squish_e import squish_e
from generate.online.OnlineCED import CEDer


# squish压缩算法
def squish(points, max_buffer_size):
    buffer = []
    # 坐标、err、下标
    buffer.append([points[0], 0, 0])
    max_err = 0
    if max_buffer_size > 2:
        for i in range(1, len(points)):
            buffer.append([points[i], 0, i])
            if len(buffer) <= 2:
                continue
            segment_start = buffer[-3][0]
            segment_end = buffer[-1][0]
            buffer[-2][1] += getPED4GPS(buffer[-2][0], segment_start, segment_end)
            if len(buffer) > max_buffer_size:
                to_remove = len(buffer) - 2
                for j in range(1, len(buffer) - 1):
                    if buffer[j][1] < buffer[to_remove][1]:
                        to_remove = j
                buffer[to_remove - 1][1] += buffer[to_remove][1]
                buffer[to_remove + 1][1] += buffer[to_remove][1]
                err = getPED4GPS(buffer[to_remove][0], buffer[to_remove - 1][0], buffer[to_remove + 1][0])
                max_err = max(max_err, err)
                buffer.pop(to_remove)
    else:
        buffer.append([points[-1], 0, len(points) - 1])
        segment_start = buffer[0][0]
        segment_end = buffer[-1][0]
        for i in range(1, len(points) - 1):
            max_err = max(max_err, getPED4GPS(points[i], segment_start, segment_end))
    idx = [p[2] for p in buffer]
    pp = [p[2] for p in buffer]
    return pp, idx, max_err


def STTrace(points, max_buffer_size, mode):
    buffer = []
    buffer.append([points[0], 0])
    max_err = 0
    if max_buffer_size > 2:
        buffer.append([points[1], 0])
        for i in range(2, len(points)):
            buffer.append([points[i], 0])
            st = buffer[len(buffer) - 3]
            en = buffer[len(buffer) - 1]
            buffer[len(buffer) - 2][1] = getPED4GPS(buffer[len([buffer])][0], st[0], en[0])
            if len(buffer) > max_buffer_size:
                start = buffer[0]
                to_remove = buffer[len(buffer) - 1]
                min_idx = 1
                k = 1
                for i in range(1, len(buffer) - 1):
                    p = buffer[i]
                    if to_remove == buffer[len(buffer) - 1] or p[1] < to_remove[1]:
                        to_remove = p
                        k += 1
                if min_idx - 1 > 0:
                    buffer[min_idx - 1][1] = getPED4GPS(buffer[min_idx - 1][0], buffer[min_idx - 2][0],
                                                        buffer[min_idx + 1][0])
                if min_idx + 1 < len(buffer) - 1:
                    buffer[min_idx + 1][1] = getPED4GPS(buffer[min_idx + 1][0], buffer[min_idx - 1][0],
                                                        buffer[min_idx + 2][0])
                buffer.remove(to_remove)

    else:
        buffer.append([points[len(points) - 1], 0])
    idx = []
    for i in range(len(buffer)):
        idx.append(points.index(buffer[i][0]))
    idx.sort()

    for i in range(len(idx) - 1):
        st = idx[i]
        en = idx[i + 1]
        _, err = getMaxError(st, en, points, mode)
        max_err = max(max_err, err)
    return None, idx, max_err


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


def compress_squish(src, points, max_ratio, metric):
    # 计时开始

    err = []
    compRes = []
    timelist = []
    timelist2 = []
    for idseq, seq in zip(src, points):
        tic1 = time.time()
        _, idx, maxErr = squish(seq, int(max_ratio * len(seq)))
        tic2 = time.time()
        timelist.append((tic2 - tic1) / len(seq))
        maxErr_ = 0
        if metric != 'ss':
            for i in range(len(idx) - 1):
                _, e = getMaxError(idx[i], idx[i + 1], seq, metric)
                maxErr_ = max(maxErr_, e)
        else:
            try:
                maxErr_ = ceder.CED_op(idx, idseq)
            except Exception as e:
                print("exception")
                continue
        err.append(maxErr_)

    print(f"squish压缩率 {max_ratio},耗时 {np.min(timelist)}, error {np.mean(err)}")
    # print(f"squish-e压缩率 {max_ratio},耗时 {np.min(timelist2)}")
    return compRes


def compress_squish_e(src, points, max_ratio, metric):
    # 计时开始

    err = []
    compRes = []
    timelist = []
    for idseq, seq in zip(src, points):
        tic1 = time.time()
        try:
            _, idx, maxErr = squish_e(seq, int(max_ratio * len(seq)), mode=metric)
        except Exception as e:
            continue
        tic2 = time.time()
        timelist.append((tic2 - tic1) / len(seq))
        maxErr_ = 0
        if metric != 'ss':
            for i in range(len(idx) - 1):
                _, e = getMaxError(idx[i], idx[i + 1], seq, metric)
                maxErr_ = max(maxErr_, e)
        else:
            try:
                maxErr_ = ceder.CED_op(idx, idseq)
            except Exception as e:
                print("exception")
                continue
        err.append(maxErr_)

    print(f"squish_e压缩率 {max_ratio},耗时 {np.mean(timelist)}, error {np.mean(err)}")
    return compRes


def compress_sttrace(src, points, max_ratio, metric):
    # 计时开始

    err = []
    compRes = []
    timelist = []
    for idseq, seq in zip(src, points):
        tic1 = time.time()
        _, idx, maxErr = STTrace(seq, int(max_ratio * len(seq)), mode='ped')
        tic2 = time.time()
        timelist.append((tic2 - tic1) / len(seq))
        maxErr_ = 0
        if metric != 'ss':
            for i in range(len(idx) - 1):
                _, e = getMaxError(idx[i], idx[i + 1], seq, metric)
                maxErr_ = max(maxErr_, e)
        else:
            try:
                maxErr_ = ceder.CED_op(idx, idseq)
            except Exception as e:
                print("exception")
                continue
        err.append(maxErr_)

    print(f"sttrace压缩率 {max_ratio},耗时 {np.min(timelist)}, error {np.mean(err)}")
    return compRes


class RLAgent:
    def __init__(self, datasets, region, dataSize, metric):
        self.traj_path = os.path.join(DATA_DIR, datasets + ".src")
        if metric == "ss":
            metric = "ped"

        a_size = 3  # RLTS 3, RLTS-Skip 5
        s_size = 3  # RLTS and RLTS-Skip are both 3 online
        self.env = TrajComp(self.traj_path, dataSize, region, a_size, s_size, metric)
        self.RL = PolicyGradient(self.env.n_features, self.env.n_actions)
        self.RL.load(
            '/home/hch/Desktop/trjcompress/generate/online/online-rlts/save/0.00051169259094067_ratio_0.1/')  # your_trained_model your_trained_model_skip

    def RL_online(self, buffer_size, episode):
        if buffer_size < 3:
            return
        steps, observation = self.env.reset(episode, buffer_size)
        tic1 = time.perf_counter()
        for index in range(buffer_size, steps):
            if index == steps - 1:
                done = True
            else:
                done = False
            action = self.RL.quick_time_action(observation)  # matrix implementation for fast efficiency when the model is ready
            observation_, _ = self.env.step(episode, action, index, done,
                                            'V')  # 'T' means Training, and 'V' means Validation
            observation = observation_
        tic2 = time.perf_counter()
        idx, max_err = self.env.output(episode, 'V')
        if idx[-1] == steps:
            idx, max_err = self.env.output(episode, 'V')
        tm = (tic2 - tic1) / len(self.env.ori_traj_set[episode])
        return None, idx, max_err, tm

    def run_all(self, size, ratio, metric):
        err = []
        timelist = []
        for i in range(size):
            buffer_size = int(ratio * len(self.env.ori_traj_set[i]))
            _, idx, max_err, tm = self.RL_online(buffer_size, i)
            timelist.append(tm)

            if metric == 'ss':
                try:
                    max_err = ceder.CED_op(idx, self.env.ori_trajID_set[i])
                except Exception as e:
                    print("exception")
                    continue
            err.append(max_err)
        print(f"rlts压缩率 {ratio}, 耗时 {np.mean(timelist)}, error {np.mean(err)}")


if __name__ == '__main__':
    datasets = "len250.src"
    src_file = os.path.join(DATA_DIR, datasets + ".src")
    with open(os.path.join(DATA_DIR, 'pickle.txt'), 'rb') as f:
        var_a = pickle.load(f)
    region = pickle.loads(var_a)
    points, src = readData(src_file, region)
    metric = 'dad'
    agent = RLAgent(datasets, region, len(points), metric)
    ceder = CEDer()

    for i in range(4, 5):
        ratio = 0.1 * (i + 1)
        compress_sttrace(src, points, ratio, metric)
        compress_squish(src, points, ratio, metric)
        compress_squish_e(src, points, ratio, metric)
        agent.run_all(len(points), ratio, metric)
