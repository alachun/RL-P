import math
import gym
import numpy as np
import pddlgym
from gym.spaces import Discrete,MultiBinary
from pddlgym.structs import Predicate,TypedEntity,Type,Literal,State
from stable_baselines3 import A2C

from myA2C import *
from datetime import datetime

'''citycar对应的pddlgym环境（四个连续时序+4f），用于纯强化学习训练和gym环境调用（计算规划解的执行效果）
在gym环境调用pddlgym使用时最好删去pddlgym中的所有动作屏蔽，因为动作数量很多，动作屏蔽会大量增加不必要的时间开销'''


class WrapperEnv_citycar(gym.Env):
    def __init__(self,place):
        super(WrapperEnv_citycar,self).__init__()
        self.env=pddlgym.make("PDDLEnvCitycar")
        state_sample=State(frozenset(self.env.env.problems[0].initial_state),
                           frozenset(self.env.env.problems[0].objects),self.env.env.problems[0].goal)
        state_sample=self.env.env._handle_derived_literals(state_sample)
        self.env.env.set_state(state_sample)
        self.env.env._action_space.reset_initial_state(state_sample)
        self.current_observation=self.env.env.get_state()#记录当前状态literal，实时更新
        self.actions=self.all_ground_literals()#理论上所有可行的动作
        self.action_space=Discrete(len(self.actions))#设置动作空间
        # self.object_num为字典类型，记录每个object类型的数量；self.object_index为列表，记录每个object类型开始的索引
        self.object_num,self.object_index=self.compute_object_alone_sum(self.env.env.problems[0].objects)
        self.graph_len=int(math.sqrt(self.object_num['junction']))#地图边长
        self.observation_init=True#新回合开始
        n=0
        self.predicate_index=[0]
        for key,value in self.env.env.problems[0].predicates.items():
            if value in self.env.env.problems[0].action_names:
                continue
            m = 1
            for predicate in value.var_types:
                m=m*self.object_num[str(predicate)]
            n+=m
            self.predicate_index.append(n)
        self.predicate_index.pop()#记录每类谓词开始的索引
        #same_line+diagonal+at_car_jun+at_car_road+starting+arrived+road_connect+clear+in_place+at_garage
        self.observation_space=MultiBinary(n)
        self.observation_shape_n=n
        self.flag_observation_init,self.flag_observation_not_init=self.compute_flag_observation(self.current_observation)
        self.observation_multibinary=self.observation(self.current_observation)
        self.action_mask=self.compute_valid_action(self.current_observation)
        # goal_index为字典，记录了原pddl文件子任务(car arrived)对应在observetion中的索引
        # car_arrived_jun为列表，表示原pddl文件子任务(car arrived)依次需要到达的junction路口
        # eg.[3,0,3,1]表示car0需arrived junction3-0, car1需arrived junction3-1
        self.goal_index,self.car_arrived_jun =self.compute_goal_index()
        self.place=place
        # 中途需要at_car_jun对应在observation中的索引
        self.place_index=self.compute_place_index(self.place)
        # 存储动作掩码，用于MyA2C中策略网络更新
        self.masks_buffer=dict()
        self.step_num=0#当前回合步数
        self.horizon=250#每回合最多步数
        self.garage_at_jun,self.car_start_garage=self.compute_init_value()
        self.f4_state=1#当前4f时序对应的DFA所处的状态节点，需实时更新

    # 理论上所有可行动作
    def all_ground_literals(self):
        act_sample=self.env.action_space.sample(self.current_observation)
        return self.env.action_space._all_ground_literals

    def compute_object_alone_sum(self,objects):
        object_num=dict()
        object_index=[]
        for object in objects:
            str_name=str(object)
            parts=str_name.split(':')
            object_index.append(parts[0])
            if parts[1] in object_num.keys():
                object_num[parts[1]]+=1
            else:
                object_num[parts[1]]=1
        return object_num,object_index

    def compute_flag_observation(self,obs):

        same_line = Predicate(name="same_line", arity=2)
        diagonal = Predicate(name="diagonal", arity=2)
        at_car_jun = Predicate(name="at_car_jun", arity=2)
        at_car_road = Predicate(name="at_car_road", arity=2)
        starting = Predicate(name="starting", arity=2)
        arrived = Predicate(name="arrived", arity=2)
        road_connect = Predicate(name="road_connect", arity=3)
        clear = Predicate(name="clear", arity=1)
        in_place = Predicate(name="in_place", arity=1)
        at_garage = Predicate(name="at_garage", arity=2)

        state = np.zeros(self.observation_shape_n, dtype=int)

        list_sample = list(obs[0])
        list_sample = list(sorted(list_sample))
        # objects中的顺序为:
        # car+garage+junction+road
        # same_line+diagonal+at_car_jun+at_car_road+starting+arrived+road_connect+clear+in_place+at_garage
        for literal in list_sample:
            if literal.predicate == same_line:# same_line不会发生改变
                x_name = str(literal.variables[0]).split(':')[0][8:].split('-')  # junction
                x1 = int(x_name[0])
                x2 = int(x_name[1])
                y_name = str(literal.variables[1]).split(':')[0][8:].split('-')  # junction
                y1 = int(y_name[0])
                y2 = int(y_name[1])
                state[self.predicate_index[0]+(x1*self.graph_len+x2)*self.object_num['junction']+y1*self.graph_len+y2]=1
            elif literal.predicate == diagonal:# diagonal不会发生改变
                x_name = str(literal.variables[0]).split(':')[0][8:].split('-')  # junction
                x1 = int(x_name[0])
                x2 = int(x_name[1])
                y_name = str(literal.variables[1]).split(':')[0][8:].split('-')  # junction
                y1 = int(y_name[0])
                y2 = int(y_name[1])
                state[self.predicate_index[1] + (x1 * self.graph_len + x2) * self.object_num['junction'] + y1 * self.graph_len + y2] = 1
            elif literal.predicate == at_garage: # at_garage不会发生改变
                x_name = str(literal.variables[0]).split(':')  # garage
                x = int(x_name[0][6:])
                y_name = str(literal.variables[1]).split(':')[0][8:].split('-')  # junction
                y1 = int(y_name[0])
                y2 = int(y_name[1])
                state[self.predicate_index[9]+x*self.object_num['junction']+y1*self.graph_len+y2]=1
        flag_observation_not_init=state.copy()

        for literal in list_sample:
            # same_line+diagonal+at_car_jun+at_car_road+starting+arrived+road_connect+clear+in_place+at_garage
            if literal.predicate == road_connect:  # 涉及到的谓词最多，放在第一个，减少判断次数
                x_name = str(literal.variables[0]).split(':')  # road
                x = int(x_name[0][4:])
                y_name = str(literal.variables[1]).split(':')[0][8:].split('-')  # junction
                y1 = int(y_name[0])
                y2 = int(y_name[1])
                z_name = str(literal.variables[2]).split(':')[0][8:].split('-')  # junction
                z1 = int(z_name[0])
                z2 = int(z_name[1])
                state[
                    self.predicate_index[6] + x * (self.object_num['junction'] ** 2) + (y2 + y1 * self.graph_len) *
                    self.object_num['junction'] + z2 + z1 * self.graph_len] = 1

            elif literal.predicate == same_line:
                continue  # same_line不会发生改变
            elif literal.predicate == diagonal:
                continue  # diagonal不会发生改变
            elif literal.predicate == at_garage:
                continue  # at_garage不会发生改变
            elif literal.predicate == at_car_jun:
                x_name = str(literal.variables[0]).split(':')  # car
                x = int(x_name[0][3:])
                y_name = str(literal.variables[1]).split(':')[0][8:].split('-')  # junction
                y1 = int(y_name[0])
                y2 = int(y_name[1])
                state[self.predicate_index[2] + x * self.object_num['junction'] + y2 + y1 * self.graph_len] = 1
            elif literal.predicate == at_car_road:
                x_name = str(literal.variables[0]).split(':')  # car
                x = int(x_name[0][3:])
                y_name = str(literal.variables[1]).split(':')  # road
                y = int(y_name[0][4:])
                state[self.predicate_index[3] + x * self.object_num['road'] + y] = 1
            elif literal.predicate == starting:
                x_name = str(literal.variables[0]).split(':')  # car
                x = int(x_name[0][3:])
                y_name = str(literal.variables[1]).split(':')  # garage
                y = int(y_name[0][6:])
                state[self.predicate_index[4] + x * self.object_num['garage'] + y] = 1
            elif literal.predicate == arrived:
                x_name = str(literal.variables[0]).split(':')  # car
                x = int(x_name[0][3:])
                y_name = str(literal.variables[1]).split(':')[0][8:].split('-')  # junction
                y1 = int(y_name[0])
                y2 = int(y_name[1])
                state[self.predicate_index[5] + x * self.object_num['junction'] + y1 * self.graph_len + y2] = 1
            elif literal.predicate == clear:
                x_name = str(literal.variables[0]).split(':')[0][8:].split('-')  # junction
                x1 = int(x_name[0])
                x2 = int(x_name[1])
                state[self.predicate_index[7] + x1 * self.graph_len + x2] = 1
            elif literal.predicate == in_place:
                x_name = str(literal.variables[0]).split(':')  # road
                x = int(x_name[0][4:])
                state[self.predicate_index[8] + x] = 1
        flag_observation_init=state.copy()
        return flag_observation_init,flag_observation_not_init

    def observation(self,obs):
        #literal转multibinary
        #same_line+diagonal+at_car_jun+at_car_road+starting+arrived+road_connect+clear+in_place+at_garage
        same_line=Predicate(name="same_line",arity=2)
        diagonal = Predicate(name="diagonal", arity=2)
        at_car_jun = Predicate(name="at_car_jun", arity=2)
        at_car_road = Predicate(name="at_car_road", arity=2)
        starting = Predicate(name="starting", arity=2)
        arrived= Predicate(name="arrived", arity=2)
        road_connect= Predicate(name="road_connect", arity=3)
        clear= Predicate(name="clear", arity=1)
        in_place= Predicate(name="in_place", arity=1)
        at_garage= Predicate(name="at_garage", arity=2)

        list_sample = list(obs[0])
        list_sample = list(sorted(list_sample))
        # objects中的顺序为:
        # car+garage+junction+road
        if self.observation_init==True:
            return self.flag_observation_init.copy()
        else:
            state=self.flag_observation_not_init.copy()
            for literal in list_sample:
                if literal.predicate == road_connect:  # 涉及到的谓词最多，放在第一个，减少判断次数
                    x_name = str(literal.variables[0]).split(':')  # road
                    x = int(x_name[0][4:])
                    y_name = str(literal.variables[1]).split(':')[0][8:].split('-')  # junction
                    y1 = int(y_name[0])
                    y2 = int(y_name[1])
                    z_name = str(literal.variables[2]).split(':')[0][8:].split('-')  # junction
                    z1 = int(z_name[0])
                    z2 = int(z_name[1])
                    state[
                        self.predicate_index[6] + x * (self.object_num['junction'] ** 2) + (y2 + y1 * self.graph_len) *
                        self.object_num['junction'] + z2 + z1 * self.graph_len] = 1

                elif literal.predicate == same_line:
                    continue  # same_line不会发生改变
                elif literal.predicate == diagonal:
                    continue  # diagonal不会发生改变
                elif literal.predicate == at_garage:
                    continue  # at_garage不会发生改变
                elif literal.predicate == at_car_jun:
                    x_name = str(literal.variables[0]).split(':')  # car
                    x = int(x_name[0][3:])
                    y_name = str(literal.variables[1]).split(':')[0][8:].split('-')  # junction
                    y1 = int(y_name[0])
                    y2 = int(y_name[1])
                    state[self.predicate_index[2] + x * self.object_num['junction'] + y2 + y1 * self.graph_len] = 1
                elif literal.predicate == at_car_road:
                    x_name = str(literal.variables[0]).split(':')  # car
                    x = int(x_name[0][3:])
                    y_name = str(literal.variables[1]).split(':')  # road
                    y = int(y_name[0][4:])
                    state[self.predicate_index[3] + x * self.object_num['road'] + y] = 1
                elif literal.predicate == starting:
                    x_name = str(literal.variables[0]).split(':')  # car
                    x = int(x_name[0][3:])
                    y_name = str(literal.variables[1]).split(':')  # garage
                    y = int(y_name[0][6:])
                    state[self.predicate_index[4] + x * self.object_num['garage'] + y] = 1
                elif literal.predicate == arrived:
                    x_name = str(literal.variables[0]).split(':')  # car
                    x = int(x_name[0][3:])
                    y_name = str(literal.variables[1]).split(':')[0][8:].split('-')  # junction
                    y1 = int(y_name[0])
                    y2 = int(y_name[1])
                    state[self.predicate_index[5] + x * self.object_num['junction'] + y1 * self.graph_len + y2] = 1
                elif literal.predicate == clear:
                    x_name = str(literal.variables[0]).split(':')[0][8:].split('-')  # junction
                    x1 = int(x_name[0])
                    x2 = int(x_name[1])
                    state[self.predicate_index[7] + x1 * self.graph_len + x2] = 1
                elif literal.predicate == in_place:
                    x_name = str(literal.variables[0]).split(':')  # road
                    x = int(x_name[0][4:])
                    state[self.predicate_index[8] + x] = 1
        return state


    def multibinary_to_literal_observation(self,obs):
        #same_line+diagonal+at_car_jun+at_car_road+starting+arrived+road_connect+clear+in_place+at_garage
        same_line = Predicate(name="same_line", arity=2)
        diagonal = Predicate(name="diagonal", arity=2)
        at_car_jun = Predicate(name="at_car_jun", arity=2)
        at_car_road = Predicate(name="at_car_road", arity=2)
        starting = Predicate(name="starting", arity=2)
        arrived = Predicate(name="arrived", arity=2)
        road_connect = Predicate(name="road_connect", arity=3)
        clear = Predicate(name="clear", arity=1)
        in_place = Predicate(name="in_place", arity=1)
        at_garage = Predicate(name="at_garage", arity=2)
        state=set()
        for i in range(len(obs)):
            if obs[i]==0:
                continue
            if i<self.predicate_index[1]:#same_line(junction,junction)
                x=(i-self.predicate_index[0])//(self.object_num['junction'])
                x1=x//self.graph_len
                x2=x%self.graph_len
                y = (i - self.predicate_index[0])%(self.object_num['junction'])
                y1 = y // self.graph_len
                y2 = y % self.graph_len
                x_name="junction"+str(x1)+"-"+str(x2)
                y_name = "junction" + str(y1) + "-" + str(y2)
                variables=[]
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                variables.append(self.env.env.problems[0].objects[self.object_index.index(y_name)])
                literal=Literal(predicate=same_line,variables=variables)
                state.add(literal)
            elif i<self.predicate_index[2]:#diagonal
                x = (i - self.predicate_index[1]) // (self.object_num['junction'])
                x1 = x // self.graph_len
                x2 = x % self.graph_len
                y = (i - self.predicate_index[1]) % (self.object_num['junction'])
                y1 = y // self.graph_len
                y2 = y % self.graph_len
                x_name = "junction" + str(x1) + "-" + str(x2)
                y_name = "junction" + str(y1) + "-" + str(y2)
                variables = []
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                variables.append(self.env.env.problems[0].objects[self.object_index.index(y_name)])
                literal = Literal(predicate=diagonal, variables=variables)
                state.add(literal)
            elif i<self.predicate_index[3]:#at_car_jun
                x = (i - self.predicate_index[2]) // (self.object_num['junction'])#car
                y = (i - self.predicate_index[2]) % (self.object_num['junction'])#junction
                y1 = y // self.graph_len
                y2 = y % self.graph_len
                x_name = "car" + str(x)
                y_name = "junction" + str(y1) + "-" + str(y2)
                variables = []
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                variables.append(self.env.env.problems[0].objects[self.object_index.index(y_name)])
                literal = Literal(predicate=at_car_jun, variables=variables)
                state.add(literal)
            elif i<self.predicate_index[4]:#at_car_road
                x = (i - self.predicate_index[3]) // (self.object_num['road'])  # car
                y = (i - self.predicate_index[3]) % (self.object_num['road'])  # road
                x_name = "car" + str(x)
                y_name = "road" + str(y)
                variables = []
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                variables.append(self.env.env.problems[0].objects[self.object_index.index(y_name)])
                literal = Literal(predicate=at_car_road, variables=variables)
                state.add(literal)
            elif i<self.predicate_index[5]:#starting
                x = (i - self.predicate_index[4]) // (self.object_num['garage'])  # car
                y = (i - self.predicate_index[4]) % (self.object_num['garage'])  # garage
                x_name = "car" + str(x)
                y_name = "garage" + str(y)
                variables = []
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                variables.append(self.env.env.problems[0].objects[self.object_index.index(y_name)])
                literal = Literal(predicate=starting, variables=variables)
                state.add(literal)
            elif i<self.predicate_index[6]:#arrived
                x = (i - self.predicate_index[5]) // (self.object_num['junction'])  # car
                y = (i - self.predicate_index[5]) % (self.object_num['junction'])  # junction
                y1 = y // self.graph_len
                y2 = y % self.graph_len
                x_name = "car" + str(x)
                y_name = "junction" + str(y1) + "-" + str(y2)
                variables = []
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                variables.append(self.env.env.problems[0].objects[self.object_index.index(y_name)])
                literal = Literal(predicate=arrived, variables=variables)
                state.add(literal)
            elif i<self.predicate_index[7]:#road_connect
                x = (i - self.predicate_index[6]) // (self.object_num['junction']**2)  # road
                y = (i - self.predicate_index[6]-x*(self.object_num['junction']**2))//(self.object_num['junction'])  # junction
                z = (i - self.predicate_index[6]-x*(self.object_num['junction']**2))%(self.object_num['junction']) # junction
                y1 = y // self.graph_len
                y2 = y % self.graph_len
                z1 = z // self.graph_len
                z2 = z % self.graph_len
                x_name = "road" + str(x)
                y_name = "junction" + str(y1) + "-" + str(y2)
                z_name = "junction" + str(z1) + "-" + str(z2)
                variables = []
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                variables.append(self.env.env.problems[0].objects[self.object_index.index(y_name)])
                variables.append(self.env.env.problems[0].objects[self.object_index.index(z_name)])
                literal = Literal(predicate=road_connect, variables=variables)
                state.add(literal)
            elif i<self.predicate_index[8]:#clear
                x =i - self.predicate_index[7]#junction
                x1 = x // self.graph_len
                x2 = x % self.graph_len
                x_name="junction" + str(x1) + "-" + str(x2)
                variables=[]
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                literal=Literal(predicate=clear,variables=variables)
                state.add(literal)
            elif i<self.predicate_index[9]:#in_place
                x = i - self.predicate_index[8]  #road
                x_name = "road" + str(x)
                variables = []
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                literal = Literal(predicate=in_place, variables=variables)
                state.add(literal)
            else:#at_garage
                x = (i - self.predicate_index[9]) // (self.object_num['junction'])  #garage
                y = (i - self.predicate_index[9]) % (self.object_num['junction'])  # junction
                y1 = y // self.graph_len
                y2 = y % self.graph_len
                x_name = "garage" + str(x)
                y_name = "junction" + str(y1) + "-" + str(y2)
                variables = []
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                variables.append(self.env.env.problems[0].objects[self.object_index.index(y_name)])
                literal = Literal(predicate=at_garage, variables=variables)
                state.add(literal)
        state_sample=State.with_literals(self.current_observation,state)
        return state_sample


    def compute_valid_action(self,obs):
        #传入的obs为literal_observation
        masks=[]
        valid_literals=self.env.action_space.all_ground_literals(obs)
        valid_literals=list(sorted(valid_literals))
        for action in self.actions:
            if action in valid_literals:
                masks.append(True)
            else:
                masks.append(False)
        return masks

    def action(self,act):
        #return self.actions[act]
        #传入的act为discrete
        if self.action_mask[act] is True:
            return self.actions[act]
        else:
            return False

    def reset(self,**kwargs):
        state=self.env.reset()
        self.current_observation=state[0]
        self.observation_init=True
        state=self.flag_observation_init.copy()
        self.observation_init=False
        self.observation_multibinary=state.copy()
        state_sample = State(frozenset(self.current_observation.literals),
                             frozenset(self.current_observation.objects), self.current_observation.goal)
        state_sample = self.env.env._handle_derived_literals(state_sample)
        self.env.env.set_state(state_sample)
        self.action_mask=self.compute_valid_action(self.current_observation)
        self.f4_state = 1
        self.step_num=0
        return state

    def step(self,action):
        action=self.action(action)
        if action==False:
            reward=0
            done=False
            debug_info={}
            obs=self.observation_multibinary
            print(action)
        else:
            obs,reward,done,_,debug_info=self.env.step(action)


            # 当动作为destroy_road时，其中的forall语句怎么处理？
            # pddlgym目前不能处理forall语句所以domain文件里面删除了forall语句单独处理
            if str(action.predicate)=="destroy_road":
                obs = self.observation(obs)
                self.observation_multibinary = obs
                road_name=str(action.variables[2])
                road_index=int(road_name.split(':')[0][4:])
                car_road_flag = False
                for i in range(self.object_num['car']):
                    #cari在这条路上
                    if self.observation_multibinary[self.predicate_index[3]+i*self.object_num['road']+road_index]==1:
                        car_road_flag = True
                        j_name = str(action.variables[0]).split(':')[0][8:].split('-')
                        j1 = int(j_name[0])
                        j2 = int(j_name[1])
                        # (not (at_car_road ?c1 ?r1))
                        self.observation_multibinary[self.predicate_index[3] + i * self.object_num['road'] + road_index]=0
                        # (at_car_jun ?c1 ?xy_initial)
                        self.observation_multibinary[self.predicate_index[2]+i*self.object_num['junction']+j1*self.graph_len+j2]=1
                self.current_observation=self.multibinary_to_literal_observation(self.observation_multibinary)
                if car_road_flag:
                    state_sample = State(frozenset(self.current_observation.literals),
                                         frozenset(self.current_observation.objects), self.current_observation.goal)
                    state_sample = self.env.env._handle_derived_literals(state_sample)
                    self.env.env.set_state(state_sample)
            else:
                self.current_observation = obs
                obs = self.observation(obs)
                self.observation_multibinary = obs

            self.action_mask=self.compute_valid_action(self.current_observation)
        #更新当前回合步数，超过阈值则结束
        self.step_num+=1
        if(self.step_num>self.horizon):
            done=True
        #判断当前状态是否处于自动机错误状态，如果处于则结束
        q2=self.check_q2(obs)
        if q2:
            done=True

        #判断第二个自动机f4到达哪个状态
        f4_state=self.compute_f4_state(self.f4_state,obs)
        self.f4_state=f4_state
        #判断回合结束时任务是否已解决
        if done==True:
            if  q2==False and self.compute_goal_complete(obs) and f4_state==5:
                reward=1000
            else:
                reward=-1000
        return obs,reward,done,debug_info

    def compute_goal_index(self):
        goal_index=dict()
        car_arrived_jun=[]
        for literal in self.current_observation[2].literals:
            name1=str(literal.variables[0]).split(':')[0]#car
            name2 = str(literal.variables[1]).split(':')[0][8:].split('-')#junction
            index1=int(name1[3:])
            index2_1 = int(name2[0])
            index2_2 = int(name2[1])
            index2=index2_1*self.graph_len+index2_2
            car_arrived_jun.append(index2_1)
            car_arrived_jun.append(index2_2)
            goal_index[index1]=self.predicate_index[5]+index1*self.object_num['junction']+index2
        return goal_index,car_arrived_jun

    def compute_goal_complete(self,obs):
        #判断goal中的所有目标是否已完成（忽视实现顺序/自动机状态）
        for index in self.goal_index.values():
            if obs[index]==0:
                return False
        return True

    def set_state(self,state):
        #跳过执行时也应该保存f4的状态
        self.observation_multibinary=state.copy()
        self.current_observation=self.multibinary_to_literal_observation(state)
        state_sample = State(frozenset(self.current_observation.literals),
                             frozenset(self.current_observation.objects), self.current_observation.goal)
        state_sample = self.env.env._handle_derived_literals(state_sample)
        self.env.env.set_state(state_sample)
        #self.action_mask = self.compute_valid_action(self.current_observation)


    def check_q2(self, obs):
        # 4个连续时序
        if obs[self.goal_index[0]] == 0 and obs[self.goal_index[1]] == 1:  # 未完成目标1的情况下完成目标2
            return True
        if obs[self.goal_index[0]] == 0 and obs[self.goal_index[2]] == 1:  # 未完成目标1的情况下完成目标3
            return True
        if obs[self.goal_index[0]] == 0 and obs[self.goal_index[3]] == 1:  # 未完成目标1的情况下完成目标4
            return True
        if obs[self.goal_index[1]] == 0 and obs[self.goal_index[2]] == 1:  # 未完成目标2的情况下完成目标3
            return True
        if obs[self.goal_index[1]] == 0 and obs[self.goal_index[3]] == 1:  # 未完成目标2的情况下完成目标4
            return True
        if obs[self.goal_index[2]] == 0 and obs[self.goal_index[3]] == 1:  # 未完成目标3的情况下完成目标4
            return True
        return False


    def compute_place_index(self,place):
        #place按顺序传入car不能到达/必须到达的地点坐标
        #eg.[2,1,2,2]:junction2-1， junction2-2
        place_index=dict()
        for i in range(0,len(place),2):
            place_index[i//2]=self.predicate_index[2]+(i//2)*self.object_num['junction']+place[i]*self.graph_len+place[i+1]
        return place_index

    def compute_init_value(self):
        # garage_at_jun(garage,junction)
        #car_start_garage(car,garage)
        #car_arrived_jun(car,junction)
        garage_at_jun=dict()
        car_start_garage=dict()
        car_arrived_jun=[]
        for i in range(self.predicate_index[9],self.observation_shape_n):
            if self.flag_observation_init[i]==1:
                garage=(i-self.predicate_index[9])//self.object_num['junction']
                jun=(i-self.predicate_index[9])%self.object_num['junction']
                garage_at_jun[garage]=[jun//self.graph_len,jun%self.graph_len]
        for i in range(self.predicate_index[4], self.predicate_index[5]):
            if self.flag_observation_init[i] == 1:
                car=(i-self.predicate_index[4])//self.object_num['garage']
                gar = (i - self.predicate_index[4]) % self.object_num['garage']
                car_start_garage[car]=gar
        return garage_at_jun,car_start_garage

    def compute_f4_state(self,f4_state,obs):
        #F(a & (F(b & (F c))))
        if f4_state==1:
            if obs[self.place_index[0]]==0:
                return 1
            elif obs[self.place_index[1]]==0:
                return 2
            elif obs[self.place_index[2]]==0:
                return 3
            elif obs[self.place_index[3]] == 0:
                return 4
            else:
                return 5
        elif f4_state==2:
            if obs[self.place_index[1]]==0:
                return 2
            elif obs[self.place_index[2]]==0:
                return 3
            elif obs[self.place_index[3]] == 0:
                return 4
            else:
                return 5
        elif f4_state==3:
            if obs[self.place_index[2]]==0:
                return 3
            elif obs[self.place_index[3]]==0:
                return 4
            else:
                return 5
        elif f4_state==4:
            if obs[self.place_index[3]]==0:
                return 4
            else:
                return 5
        else:
            return 5









if __name__=="__main__":
    start_time = datetime.now()
    env = WrapperEnv_citycar(place=[2,2,2,0,3,1,0,2])
    model = MyA2C('MlpPolicy', env=env, verbose=1,tensorboard_log="./grid-visit-all_tensorboard")  #
    model.learn(total_timesteps=20000, log_interval=100)
    end_time = datetime.now()
    print(end_time - start_time)
    model.save("p4-5-3-0-2")



























