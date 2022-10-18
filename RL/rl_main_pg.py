import os
import pickle
import sys

from sys_config import DATA_DIR

sys.path.append('/home/hch/RLTS/')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate.online.RLOnline.rl_env_inc import TrajComp
from generate.online.RLOnline.rl_brain import PolicyGradient
import time

def run_online(elist): # Validation
    eva = []
    total_len = []
    for episode in elist:
        #print('online episode', episode)
        total_len.append(len(env.ori_traj_set[episode]))
        buffer_size = int(ratio*len(env.ori_traj_set[episode]))
        if buffer_size < 3:
            continue
        steps, observation = env.reset(episode, buffer_size)
        for index in range(buffer_size, steps):
            if index == steps - 1:
                done = True
            else:
                done = False
            action = RL.fix_choose_action(observation)
            #action = RLOnline.quick_time_action(observation) #use it when your model is ready for efficiency
            observation_, _ = env.step(episode, action, index, done, 'V') #'T' means Training, and 'V' means Validation
            observation = observation_
        eva.append(env.output(episode, 'V')) #'T' means Training, 'V' means Validation, and 'V-VIS' for visualization on Validation
    return eva
        
def run_comp(): #Training
    check = 999999
    training = []
    validation = []
    Round = 5
    while Round!=0:
        Round = Round - 1
        for episode in range(0, traj_amount):
            #print('training', episode)
            buffer_size = int(ratio*len(env.ori_traj_set[episode]))
            # extreme cases
            if buffer_size < 3:
                continue
            steps, observation = env.reset(episode, buffer_size)
            for index in range(buffer_size, steps):
                #print('index', index)
                if index == steps - 1:
                    done = True
                else:
                    done = False
                
                # RLOnline choose action based on observation
                action = RL.pro_choose_action(observation)
                #print('action', action)
                # RLOnline take action and get next observation and reward
                observation_, reward = env.step(episode, action, index, done, 'T') #'T' means Training, and 'V' means Validation
                
                RL.store_transition(observation, action, reward)
                
                if done:
                    vt = RL.learn()
                    break
                # swap observation
                observation = observation_
            train_e = env.output(episode, 'T') #'T' means Training, 'V' means Validation, and 'V-VIS' for visualization on Validation
            show = 50
            if episode % show == 0:
                 eva_ = run_online([i for i in range(2500, 2500 + valid_amount - 1)])
                 #print('eva', eva)
                 eva = [i[1] for i in eva_]
                 res = sum(eva)/len(eva)
                 training.append(train_e)
                 validation.append(res)
                 print('Training error: {}, Validation error: {}'.format(sum(training[-show:])/len(training[-show:]), res))
                 RL.save('./save/'+ str(res) + '_ratio_' + str(ratio) + '/trained_model.ckpt')
                 print('Save model at round {} episode {} with error {}'.format(10 - Round, episode, res))
                 if res < check:
                     check = res
                 print('==>current best model is {} with ratio {}'.format(check, ratio))
    return training, validation

if __name__ == "__main__":
    # building subtrajectory env
    traj_path = '../datasets/RLtrj.src'
    with open(os.path.join(DATA_DIR, 'pickle.txt'), 'rb') as f:
        var_a = pickle.load(f)
    region = pickle.loads(var_a)

    traj_amount = 2500
    valid_amount = 1000
    a_size = 3
    s_size = 3
    ratio = 0.3
    env = TrajComp(traj_path, traj_amount + valid_amount, region, a_size, s_size)
    RL = PolicyGradient(env.n_features, env.n_actions)
    #RLOnline.load('./save/your_model/')
    start = time.time()
    training, validation = run_comp()
    print("Training elapsed time = %s", float(time.time() - start))
