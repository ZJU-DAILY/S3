from rl_env_inc import TrajComp
#from rl_env_inc_skip import TrajComp
from rl_brain import PolicyGradient
import matplotlib.pyplot as plt
import time
import numpy as np


def evaluate(elist): # Evaluation
    effectiveness = []
    timeList = []
    st = time.time()
    for episode in elist:
        #print('online episode', episode)
        buffer_size = int(ratio*len(env.ori_traj_set[episode]))
        if buffer_size < 3:
            continue
        steps, observation = env.reset(episode, buffer_size)
        for index in range(buffer_size, steps):
            if index == steps - 1:
                done = True
            else:
                done = False
            action = RL.quick_time_action(observation) #matrix implementation for fast efficiency when the model is ready
            observation_, _ = env.step(episode, action, index, done, 'V') #'T' means Training, and 'V' means Validation
            observation = observation_
        effectiveness.append(env.output(episode, 'V')) #'T' means Training, 'V' means Validation, and 'V-VIS' for visualization on Validation
        timeList.append((time.time() - st) / steps)
    return sum(effectiveness)/len(effectiveness), np.mean(timeList)

def evaluate_skip(elist):
    effectiveness = []
    total_len = []
    skip_pts = 0
    step_pts = 0
    for episode in elist:
        #print('online episode', episode)
        total_len.append(len(env.ori_traj_set[episode]))
        buffer_size = int(ratio*len(env.ori_traj_set[episode]))
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
                #print('test skip')
                skip_pts = skip_pts + 1
                continue
            action = RL.quick_time_action(observation) #matrix implementation for fast efficiency when the model is ready
            observation_, _ = env.step(episode, action, index, done, 'V') #'T' means Training, and 'V' means Validation
            observation = observation_
        effectiveness.append(env.output(episode, 'V')) #'T' means Training, 'V' means Validation, and 'V-VIS' for visualization on Validation
    return sum(effectiveness)/len(effectiveness)

if __name__ == "__main__":
    # building subtrajectory env 
    traj_path = '../TrajData/Geolife_out/'
    test_amount = 100
    elist = [i for i in range(1100, 1100 + test_amount - 1)]
    a_size = 3 #RLTS 3, RLTS-Skip 5
    s_size = 3 #RLTS and RLTS-Skip are both 3 online
    ratio = 0.1
    env = TrajComp(traj_path, 3000, a_size, s_size)
    RL = PolicyGradient(env.n_features, env.n_actions)
    RL.load('./save/your_trained_model/') #your_trained_model your_trained_model_skip
    effectiveness, avg_time= evaluate(elist) #evaluate evaluate_skip
    print("%e" %effectiveness)
    print(avg_time)
