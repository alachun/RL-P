import pddlgym
import gymnasium as gym
from gymnasium.core import ActType, ObsType
import numpy as np
import pddlgym
from gymnasium.spaces import Discrete, MultiBinary, Box, Dict
from pddlgym.structs import Predicate, TypedEntity, Type, Literal, State
from stable_baselines3 import SAC, A2C
from datetime import datetime
from DFA import DFATransformer
from PredTree import PredTree
from anytree import Node, RenderTree
import time
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

class UpperEnv(gym.Env):
    def __init__(self, env_name, instance, dfa_text=None, accepting_state=None, sequence=3):
        # create env and dfa
        self.env = pddlgym.make(env_name)
        self.env.env.problems = [self.env.env.problems[instance]]
        self.current_obs = None
        self.horizon = 3000
        self.currentH = 0
        self.achieved_sub_goal_num = 0
        self.dfa_state = '0'
        self.dfa = DFATransformer(dfa_text, accepting_state)

        # get action tree and action space
        obs, _ = self.env.reset()
        self.env.action_space.sample(obs)
        self.action_tree = PredTree()
        self.action_tree.create_action_tree(self.env.action_space._all_ground_literals)
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # goal setting
        self.sequence = sequence
        low = np.zeros((len(obs.goal.literals) - self.sequence + self.dfa.node_num,), dtype=np.int8)
        high = np.ones((len(obs.goal.literals) - self.sequence + self.dfa.node_num,), dtype=np.int8)
        self.goal_space = Box(low=low, high=high, shape=(len(obs.goal.literals) - self.sequence + self.dfa.node_num,), dtype=np.int8)
        self.goal_obs = np.zeros((len(obs.goal.literals) - self.sequence + self.dfa.node_num,))
        self.non_dfa_goal_num = len(obs.goal.literals) - self.sequence
        self.final_goal = np.array([1] * self.non_dfa_goal_num + [0] * (self.dfa.node_num - 1) + [1], dtype=np.int8)
        # get object dict
        obj_dict = {}
        for object in self.env.env.problems[0].objects:
            key, category = str(object).split(":")
            if category not in obj_dict:
                obj_dict[category] = []
            obj_dict[category].append(key)

        # get obs tree and obs space
        pred_list = []
        objclass_dict = {}
        for pred in self.env.observation_space.predicates:
            pred_list.append(str(pred))
            objclass_dict[str(pred)] = []
            for var_type in pred.var_types:
                objclass_dict[str(pred)].append(str(var_type))
        self.obs_tree = PredTree()
        self.obs_tree.create_obs_tree(pred_list, objclass_dict, obj_dict)
        self.observation_space = Dict({"obs": Box(low=0, high=1, shape=(len(self.obs_tree.index_to_path) + self.dfa.node_num,),
                                         dtype=np.int8), "goal": Box(low=0, high=1, shape=(len(obs.goal.literals)-self.sequence+self.dfa.node_num,), dtype=np.int8)})

        # get goal index in obs
        self.goal_index = []
        for literal in obs.goal.literals[:-self.sequence]:
            index = self.obs_tree.get_index_by_path(self.get_tree_path(literal))
            self.goal_index.append(index)
        if self.dfa is not None:
            for i in range(self.dfa.node_num):
                self.goal_index.append(len(self.obs_tree.index_to_path) + i)

    def reset(self, **kwargs):
        obs, _ = self.env.reset()
        obs_ = np.zeros(self.observation_space['obs'].shape, dtype=np.int8)
        for literal in obs.literals:
            index = self.obs_tree.get_index_by_path(self.get_tree_path(literal))
            obs_[index] = 1
        self.current_obs = obs
        self.currentH = 0
        self.achieved_sub_goal_num = 0
        self.dfa_state = '0'
        obs_[len(self.obs_tree.index_to_path)] = 1
        if self.dfa is not None:
            self.dfa.reset()
        next_obs = {'obs': obs_, "goal": self.final_goal}
        return next_obs, {}

    def step(self, action: ActType):
        pass

    def get_tree_path(self, literal):
        path = ['root']
        path.append(str(literal.predicate))
        for var in literal.variables:
            obj, category = str(var).split(":")
            path.append(obj)
        return path
