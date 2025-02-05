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
from pddlgym.structs import Predicate, TypedEntity, Type, Literal, State


class LiteralStr:
    def __init__(self, predicate, variables):
        self.predicate = predicate
        self.variables = variables

    def __str__(self):
        v_str = ""
        for v in self.variables:
            v_str += (v + ",")
        return self.predicate + "(" + v_str[:-1] + ")"
class LowerEnv(gym.Env):
    def __init__(self, env_name, instance, dfa_text=None, accepting_state=None, action_space='box', sequence=3):
        # create env and dfa
        self.env = pddlgym.make(env_name)
        self.env.env.problems = [self.env.env.problems[instance]]
        self.problem_fname = self.env.env.problems[0].problem_fname.split('/')[-1].split('.')[0]
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
        self.max_single_action = 10
        if action_space == "box":
            self.action_space = Box(low=0, high=self.max_single_action, shape=(self.action_tree.root.height,), dtype=np.float32)
        else:
            self.action_space = Box(low=0, high=len(self.env.action_space._all_ground_literals)-1, shape=(1,), dtype=np.int32)
        self.action_space_type = action_space
        self.sequence = sequence

        # goal setting
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

        real_obj_dict = {}
        for object in self.env.env.problems[0].objects:
            key, category = str(object).split(":")
            if category not in real_obj_dict:
                real_obj_dict[category] = []
            real_obj_dict[category].append(object)

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
        self.obj_dict = real_obj_dict
        # get goal index in obs
        self.goal_index = []
        for literal in obs.goal.literals[:-self.sequence]:
            index = self.obs_tree.get_index_by_path(self.get_tree_path(literal))
            self.goal_index.append(index)
        self.dfa_prop_index = []
        for literal in obs.goal.literals[-self.sequence:]:
            index = self.obs_tree.get_index_by_path(self.get_tree_path(literal))
            self.dfa_prop_index.append(index)
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
        next_obs = {'obs': obs_, "goal": obs_[self.goal_index]}
        return next_obs, {}

    def dfa_trans(self, goal_obs, dfa_prop_obs):
        if self.sequence == 3:
            # if self.dfa is None then pass
            props = {'a': bool(goal_obs[0]), 'b': bool(goal_obs[1]), 'c': bool(goal_obs[2]),
                     'd': bool(dfa_prop_obs [0]), 'e': bool(dfa_prop_obs [1]), 'f': bool(dfa_prop_obs [2])}
        if self.sequence == 4:
            props = {'a': bool(goal_obs[0]), 'b': bool(goal_obs[1]), 'c': bool(goal_obs[2]), 'd': bool(goal_obs[3]),
                     'e': bool(dfa_prop_obs[0]), 'f': bool(dfa_prop_obs[1]), 'g': bool(dfa_prop_obs[2]), 'h': bool(dfa_prop_obs[3])}
        terminate, if_success, _, dfa_state = self.dfa.step(props)
        return terminate, if_success, dfa_state

    def step(self, action: ActType):
        if self.action_space_type == "box":
            # create smaller action tree
            valid_literals = self.env.action_space.all_ground_literals(self.current_obs)
            pred_list = []
            pred_objects = {}
            for literal in valid_literals:
                if str(literal.predicate) not in pred_list:
                    pred_list.append(str(literal.predicate))
                if str(literal.predicate) not in pred_objects:
                    pred_objects[str(literal.predicate)] = []
                pred_objects[str(literal.predicate)].append(literal)
            pred_index = min(int(action[0] / (self.max_single_action / len(pred_list))), len(pred_list)-1)
            action_tree = PredTree()
            action_tree.create_action_tree(pred_objects[pred_list[pred_index]])

            # get action (literal)
            current_node = action_tree.root.children[0]
            for act in action[1:-1]:
                if len(current_node.children) == 0:
                    break
                index = min(int(act / (self.max_single_action / len(current_node.children))), len(current_node.children)-1)
                current_node = current_node.children[index]
            index = current_node.index
            literal = pred_objects[pred_list[pred_index]][index]
        else:
            valid_literals = self.env.action_space.all_ground_literals(self.current_obs)
            index = min(int(action / self.action_space.high[0] * len(valid_literals)), len(valid_literals)-1)
            literal = list(valid_literals)[index]

        # perform step
        achieved_sub_goal_num = 0
        obs, reward, done1, done2, _ = self.env.step(literal)
        obs_ = np.zeros(self.observation_space['obs'].shape, dtype=np.int8)
        for literal in obs.literals:
            index = self.obs_tree.get_index_by_path(self.get_tree_path(literal))
            obs_[index] = 1
            if literal in obs.goal.literals[:-self.sequence]:
                achieved_sub_goal_num += 1
        # special for destroy load
        if str(literal.predicate) == "destroy_road":
            initial = literal.variables[0]
            final = literal.variables[1]
            road = literal.variables[2]
            for car in self.obj_dict['car']:
                literal_str = LiteralStr("at_car_road", [str(car), str(road)])
                index = self.obs_tree.get_index_by_path(self.get_tree_path(literal_str))
                if obs_[index] == 1:
                    obs_[index] = 0
                    not_at_car_load = Literal(predicate=Predicate(name="at_car_road", arity=2), variables=[car, road])
                    assert not_at_car_load in obs.literals
                    at_car_jun = Literal(predicate=Predicate(name="at_car_jun", arity=2), variables=[car, initial])
                    s = set(obs.literals)
                    s.remove(not_at_car_load)
                    s.add(at_car_jun)
                    s = frozenset(s)
                    obs.literals = s
                    self.env.env._state = obs

        # dfa trans
        goal_obs = obs_[self.goal_index]
        dfa_prop_obs = obs_[self.dfa_prop_index]
        dfa_terminate = False
        if_success = False
        if self.dfa is None:
            dfa_state = '0'
        if self.dfa is not None:
            dfa_terminate, if_success, dfa_state = self.dfa_trans(goal_obs, dfa_prop_obs)
            goal_obs[len(obs.goal.literals) - self.sequence + int(dfa_state)] = 1
            obs_[len(self.obs_tree.index_to_path) + int(dfa_state)] = 1

        self.current_obs = obs
        self.currentH += 1
        done = done1 or done2 or self.currentH >= self.horizon or dfa_terminate

        # reward shaping
        # reward += (achieved_sub_goal_num - self.achieved_sub_goal_num + int(dfa_state) - int(self.dfa_state))

        '''if dfa_terminate:
            reward = -10'''
        if (done1 or done2):
            reward = 1000

        self.achieved_sub_goal_num = achieved_sub_goal_num
        self.dfa_state = dfa_state

        next_obs = {'obs': obs_, "goal": goal_obs}
        '''print(self.currentH)
        print(next_obs, reward, done, done,)'''
        return next_obs, reward, done, done, {"goal_obs": goal_obs.reshape(1, -1)}

    def get_tree_path(self, literal):
        path = ['root']
        path.append(str(literal.predicate))
        for var in literal.variables:
            obj, category = str(var).split(":")
            path.append(obj)
        return path



if __name__ == '__main__':
    from dfa.dfa1 import dfa_text, accepting_state

    env = LowerEnv("PDDLEnvCitycar", 0, action_space='box', dfa_text=dfa_text, accepting_state=accepting_state)
    model = SAC("MultiInputPolicy", env, verbose=1,)
    model.learn(total_timesteps=1000000, log_interval=1)
