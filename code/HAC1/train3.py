import argparse
import copy
import csv
import os
import time

import torch
import gym
import numpy as np
from HAC import HAC
from LowerEnv1 import LowerEnv
from UpperEnv1 import UpperEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(env_name, instance, dfa_text, accepting_state, dfa_index, training_time, log):
    #################### Hyperparameters ####################
    save_episode = 5  # keep saving every n episodes
    max_episodes = 10000000000  # max num of training episodes

    """
     Actions (both primitive and subgoal) are implemented as follows:
       action = ( network output (Tanh) * bounds ) + offset
       clip_high and clip_low bound the exploration noise
    """
    # HAC parameters:
    k_level = 2  # num of levels in hierarchy
    H = 100  # time horizon to achieve subgoal
    ########################################################

    lower_env = LowerEnv(env_name, instance, dfa_text, accepting_state)
    upper_env = UpperEnv(env_name, instance, dfa_text, accepting_state)
    # creating HAC agent and setting parameters
    agent = HAC(k_level, upper_env, lower_env, 500)

    # logging file:
    log_path = 'log/' + env_name + "/HAC/" + lower_env.problem_fname + "/dfa" + str(dfa_index) + "/" + str(log) + '.csv'
    data = [
        ['n_steps', 'time', 'mean_reward', 'mean_length'],
    ]
    directory = os.path.dirname(log_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(log_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)
    n_steps = 0
    mean_reward = []
    mean_length = []
    time_start = time.time()

    state, _ = lower_env.reset()
    state['goal'] = lower_env.final_goal.reshape(1, -1)
    # training procedure
    for i_episode in range(1, max_episodes + 1):
        agent.reward = 0
        agent.timestep = 0
        agent.length = 0

        # collecting experience in environment
        last_state, done, info, step = agent.run_HAC(lower_env, k_level - 1, copy.deepcopy(state), lower_env.final_goal,
                                                     False)
        goal_obs = info[0]['goal_obs'][0]
        achieved_goal_num = 0
        for i in goal_obs[: lower_env.non_dfa_goal_num]:
            if i == 1:
                achieved_goal_num += 1
        if goal_obs[-1] == 1:
            achieved_goal_num += 1
        if len(mean_reward) > 50:
            mean_reward.pop(0)
            mean_length.pop(0)
        mean_reward.append(achieved_goal_num / (lower_env.non_dfa_goal_num + 1))
        mean_length.append(agent.length)

        n_steps += agent.length
        if agent.check_goal1(info[0]['goal_obs'], lower_env.final_goal):
            print("################ Solved! ################")
        # logging updates:
        if i_episode % 1 == 0:
            '''last_state, done, info = agent.run_HAC(lower_env, k_level-1, copy.deepcopy(state), lower_env.final_goal, True)
            goal_obs = info[0]['goal_obs'][0]
            achieved_goal_num = 0
            for i in goal_obs[: lower_env.non_dfa_goal_num]:
                if i == 1:
                    achieved_goal_num += 1
            if goal_obs[-1] == 1:
                achieved_goal_num += 1
            if len(mean_reward) > 50:
                mean_reward.pop(0)
                mean_length.pop(0)
            mean_reward.append(achieved_goal_num / (lower_env.non_dfa_goal_num + 1))
            mean_length.append(agent.length)'''

            time_now = time.time()
            seconds = int(time_now - time_start)
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            second = int(seconds % 60)
            hour = str(hours) + "h" + str(minutes) + "min" + str(second) + 's'
            data = [
                [n_steps, hour, round(sum(mean_reward) / len(mean_reward), 2),
                 round(sum(mean_length) / len(mean_length), 2)],
            ]
            with open(log_path, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(data)
            print('time:', hours, "h", minutes, "min", second, 's')
            print('n_step:', n_steps, 'mean_reward:', round(sum(mean_reward) / len(mean_reward), 2), 'mean_length:',
                  round(sum(mean_length) / len(mean_length), 2))
            if seconds > training_time:
                break

        # print("Episode: {}\t Reward: {}\t Length: {}".format(i_episode, agent.reward, agent.length))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, default=1, help='log path')
    parser.add_argument('-d', type=int, default=1, help='log path')
    parser.add_argument('-t', type=int, default=3600, help='log path')
    parser.add_argument('-l', type=int, default=1, help='log path')
    args = parser.parse_args()
    from dfa.dfa4 import dfa_text, accepting_state

    train("PDDLEnvCitycar", args.i, dfa_text, accepting_state, args.d, args.t, args.l)

