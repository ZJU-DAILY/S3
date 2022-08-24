from generate.online.RLOnline.rl_env_inc import TrajComp
# from rl_env_inc_skip import TrajComp
from generate.online.RLOnline.rl_brain import PolicyGradient


def evaluate(elist):  # Evaluation
    effectiveness = []
    for episode in elist:
        print('online episode', episode)
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
    return sum(effectiveness) / len(effectiveness)


def RL(points, buffer_size, metric, episode):
    steps, observation = env.reset(episode, buffer_size)
    for index in range(buffer_size, steps):
        if index == steps - 1:
            done = True
        else:
            done = False
        action = RL.quick_time_action(observation)  # matrix implementation for fast efficiency when the model is ready
        observation_, _ = env.step(episode, action, index, done, 'V')  # 'T' means Training, and 'V' means Validation
        observation = observation_
    idx, max_err = env.output(episode, metric, 'V')
    return idx, max_err


if __name__ == "__main__":
    # building subtrajectory env
    traj_path = '../datasets/'
    test_amount = 1000
    elist = [i for i in range(1100, 1100 + test_amount - 1)]
    a_size = 3  # RLTS 3, RLTS-Skip 5
    s_size = 3  # RLTS 3, RLTS-Skip 5
    ratio = 0.1
    env = TrajComp(traj_path, 3000, a_size, s_size)
    RL = PolicyGradient(env.n_features, env.n_actions)
    RL.load('./save/0.00039190653824900003_ratio_0.1/')  # your_trained_model your_trained_model_skip
    effectiveness = evaluate(elist)  # evaluate evaluate_skip
    print("%e" % effectiveness)
