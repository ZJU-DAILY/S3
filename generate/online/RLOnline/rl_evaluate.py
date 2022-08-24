import pickle

from generate.online.RLOnline.rl_env_inc import TrajComp
# from rl_env_inc_skip import TrajComp
from generate.online.RLOnline.rl_brain import PolicyGradient
import time
import numpy as np
import os
from sys_config import DATA_DIR


def evaluate(elist):  # Evaluation
    effectiveness = []
    timeList = []
    st = time.time()
    for episode in elist:
        # print('online episode', episode)
        buffer_size = int(ratio * len(env.ori_traj_set[episode]))
        if buffer_size < 3:
            continue
        steps, observation = env.reset(episode, buffer_size)
        for index in range(buffer_size, steps):
            if index == steps - 1:
                done = True
            else:
                done = False
            action = RL.quick_time_action(
                observation)  # matrix implementation for fast efficiency when the model is ready
            observation_, _ = env.step(episode, action, index, done,
                                       'V')  # 'T' means Training, and 'V' means Validation
            observation = observation_
        effectiveness.append(env.output(episode,
                                        'V'))  # 'T' means Training, 'V' means Validation, and 'V-VIS' for visualization on Validation
        timeList.append((time.time() - st) / steps)
    return sum(effectiveness) / len(effectiveness), np.mean(timeList)


def RL_online(buffer_size, episode, ratio=0.1):
    buffer_size = int(ratio * len(env.ori_traj_set[episode]))
    if buffer_size < 3:
        return
    steps, observation = env.reset(episode, buffer_size)
    tic1 = time.perf_counter()
    for index in range(buffer_size, steps):
        if index == steps - 1:
            done = True
        else:
            done = False
        action = RL.quick_time_action(observation)  # matrix implementation for fast efficiency when the model is ready
        observation_, _ = env.step(episode, action, index, done, 'V')  # 'T' means Training, and 'V' means Validation
        observation = observation_
    tic2 = time.perf_counter()
    idx, max_err = env.output(episode, 'V')
    if idx[-1] == steps:
        idx, max_err = env.output(episode, 'V')
    tm = (tic2 - tic1) / len(env.ori_traj_set[episode])
    return idx, max_err, tm


def evaluate_skip(elist):
    effectiveness = []
    total_len = []
    skip_pts = 0
    step_pts = 0
    for episode in elist:
        # print('online episode', episode)
        total_len.append(len(env.ori_traj_set[episode]))
        buffer_size = int(ratio * len(env.ori_traj_set[episode]))
        if buffer_size < 3:
            continue
        steps, observation = env.reset(episode, buffer_size)
        step_pts = step_pts + steps
        for index in range(buffer_size, steps):
            if index == steps - 1:
                done = True
            else:
                done = False
            if index < env.INX:
                # print('test skip')
                skip_pts = skip_pts + 1
                continue
            action = RL.quick_time_action(
                observation)  # matrix implementation for fast efficiency when the model is ready
            observation_, _ = env.step(episode, action, index, done,
                                       'V')  # 'T' means Training, and 'V' means Validation
            observation = observation_
        effectiveness.append(env.output(episode,
                                        'V'))  # 'T' means Training, 'V' means Validation, and 'V-VIS' for visualization on Validation
    return sum(effectiveness) / len(effectiveness)


if __name__ == "__main__":
    datasets = 'infer'
    with open(os.path.join(DATA_DIR, 'pickle.txt'), 'rb') as f:
        var_a = pickle.load(f)
    region = pickle.loads(var_a)

    # building subtrajectory env 
    traj_path = os.path.join(DATA_DIR, datasets + ".src")

    a_size = 3  # RLTS 3, RLTS-Skip 5
    s_size = 3  # RLTS and RLTS-Skip are both 3 online
    ratio = 0.1
    env = TrajComp(traj_path, 1000, region, a_size, s_size, 'dad')
    RL = PolicyGradient(env.n_features, env.n_actions)
    RL.load(
        '/home/hch/Desktop/trjcompress/generate/online/online-rlts/save/0.00051169259094067_ratio_0.1/')  # your_trained_model your_trained_model_skip
    for ratio in range(5):
        r = (ratio + 1) / 10
        err = 0
        for i in range(1000):
            _, max_err, tm = RL_online(-1, i, r)  # evaluate evaluate_skip
            err += max_err
        print(f"r: {r} err {err / 1000}")
