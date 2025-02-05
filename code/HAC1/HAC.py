import copy
import random

import torch
import numpy as np
from UpperModel import UpperModel
from UpperEnv import UpperEnv
from LowerModel import LowerModel
from LowerEnv import LowerEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HAC:
    def __init__(self, k_level, upper_env, lower_env, H):
        # adding lowest level
        self.HAC = [LowerModel("MultiInputPolicy", lower_env, verbose=1, learning_starts=1000,
              learning_rate=0.001, batch_size=256, train_freq=1, device='cpu',)]
        # adding remaining levels
        for _ in range(k_level-1):
            self.HAC.append(UpperModel("MultiInputPolicy", upper_env, verbose=1, learning_starts=20,
              learning_rate=0.001, batch_size=256, train_freq=1, device='cpu',))
        for model in self.HAC:
            model._setup_learn(
                total_timesteps=10000000,
                callback=None,
                reset_num_timesteps=True,
                tb_log_name="run",
                progress_bar=False,
            )
        # set some parameters
        self.k_level = k_level
        self.H_high = 30
        self.H_low = H
        
        # logging parameters
        self.reward = 0
        self.timestep = 0
        self.length = 0
    
    def check_goal1(self, goal_obs, goal):
        return not np.any(goal_obs != goal)

    def check_goal2(self, goal_obs, goal):
        return goal_obs[0][goal] == 1

    def select_goal(self, state, action, dfa_index, goal_index, dfa):
        state = torch.FloatTensor(state).reshape((1, -1)).to('cpu')
        # 合并状态和目标进行actor计算
        index = torch.FloatTensor(action).reshape((1, -1)).to('cpu')
        batch_size = state.size(0)

        # 获取 dfa 的观察值
        dfa_obs = state[:, dfa_index]  # [batch_size, dfa.node_num]

        # 计算每个样本的 dfa_state 索引，假设每个样本在 dfa_obs 中只有一个 1
        # 使用 torch.argmax 选择第一个为 1 的位置，如果可能有多个为 1，需要调整逻辑
        dfa_state_indices = torch.argmax(dfa_obs, dim=1)  # [batch_size]

        # 将 dfa_state_indices 转换为字符串列表
        dfa_states = [str(i.item()) for i in dfa_state_indices]

        # 获取每个 dfa_state 对应的 out_edges
        out_states_batch = []
        for state_str in dfa_states:
            out_edges = dfa.dfa.out_edges(state_str, data=True)

            out_states = [int(edge[1]) + len(goal_index) - dfa.node_num
                          for edge in out_edges if edge[1] != state_str and edge[1] not in dfa.error_states]
            out_states_batch.append(out_states)

        # 获取 goal_obs 的副本
        goal_obs = state[:, goal_index].clone()  # [batch_size, goal_size]

        for i in range(batch_size):
            goal_obs_ = goal_obs[i].clone()
            for j in range(dfa.node_num):
                goal_obs_[-(j + 1)] = 1
            out_states = out_states_batch[i]
            goal_obs_[out_states] = 0
            indices = torch.where(goal_obs_== 0)[0]
            if len(indices) == 0:
                continue  # 根据需求处理没有可选索引的情况
            # 确保 index[i] 的值在 [0, len(indices) - 1] 之间
            calculated_index = min(int(len(indices) * index[i].item()), len(indices) - 1)
            selected = indices[calculated_index].item()

            if selected < len(goal_index) - dfa.node_num:
                goal_obs[i, selected] = 1
            else:
                for j in range(dfa.node_num):
                    goal_obs[i, -(j + 1)] = 0
                goal_obs[i, selected] = 1

        return goal_obs.numpy().reshape(1, -1), selected

    def run_HAC(self, env, i_level, state, goal, is_subgoal_test, selected_goal=0):
        next_state = None
        done = None
        goal_transitions = []

        if i_level == 0:
            self.HAC[i_level].change_goal(goal)
            state['goal'] = goal
        # H attempts
        H = self.H_high if i_level > 0 else self.H_low

        if i_level > 0:
            self.HAC[0]._last_obs = self.HAC[0].env.reset()
            self.HAC[1]._last_obs = self.HAC[1].env.reset()

        step = 0
        for i in range(H):
            step += 1
            # if this is a subgoal test, then next/lower level goal has to be a subgoal test
            is_next_subgoal_test = is_subgoal_test
            #   <================ high level policy ================>
            if i_level > 0:
                action, buffer_action = self.HAC[i_level].collect_rollouts1(env.final_goal.reshape(1, -1), is_subgoal_test)
                goal_action, selected_goal = self.select_goal(state['obs'], action, env.goal_index[-env.dfa.node_num:], env.goal_index, env.dfa)

                # Pass subgoal to lower level
                next_state, done, info, step = self.run_HAC(env, i_level-1, copy.deepcopy(state), goal_action, is_next_subgoal_test, selected_goal)
                next_state['goal'] = env.final_goal.reshape(1, -1)
                done = self.HAC[i_level].collect_rollouts2(selected_goal, buffer_action, next_state, done, info, step)

                # for hindsight action transition
                # action = info[0]['goal_obs']
                
            #   <================ low level policy ================>
            else:
                # take action
                next_state, rew, done, info, buffer_action = self.HAC[i_level].collect_rollouts(goal, selected_goal, is_subgoal_test)
                next_state['goal'] = goal.reshape(1, -1)
                self.length += 1
                # this is for logging
                self.reward += rew
                self.timestep += 1

            #   <================ finish one step/transition ================>

            if self.HAC[i_level].num_timesteps > self.HAC[i_level].learning_starts and not is_next_subgoal_test:
                if i_level == 0:
                    self.HAC[i_level].train(batch_size=self.HAC[i_level].batch_size, gradient_steps=1)
                else:
                    self.HAC[i_level].train(batch_size=self.HAC[i_level].batch_size, gradient_steps=5)

            # check if goal is achieved
            if i_level > 0:
                goal_achieved = self.check_goal1(info[0]['goal_obs'], goal)
            else:
                goal_achieved = self.check_goal2(info[0]['goal_obs'], selected_goal)
                
            # copy for goal transition
            if i_level == 0:
                goal_transitions.append([state, buffer_action, next_state, 0.0, done, info])
            
            state = next_state
            
            if done or goal_achieved:
                break
        
        #   <================ finish H attempts ================>
        # hindsight goal transition
        # last transition reward and discount is 0
        if i_level == 0:
            goal_transitions[-1][3] = 1
            for transition in goal_transitions:
                # last state is goal for all transitions
                transition[0]['goal'] = info[0]['goal_obs'].reshape(1, -1)
                transition[2]['goal'] = info[0]['goal_obs'].reshape(1, -1)
                self.HAC[i_level].store_transition1(*transition)


        return next_state, done, info, step

        
        
