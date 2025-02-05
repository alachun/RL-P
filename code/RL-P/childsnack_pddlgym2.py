import signal
import sys

import gym
import numpy as np
import pddlgym
from gym.spaces import Discrete,MultiBinary
from pddlgym.structs import Predicate,TypedEntity,Type,Literal,State
from stable_baselines3 import A2C
from myA2C import *
from datetime import datetime

'''childsnack双时序的pddlgym文件'''
class WrapperEnv_childsnack(gym.Env):
    def __init__(self):
        super(WrapperEnv_childsnack,self).__init__()
        self.env=pddlgym.make("PDDLEnvChild-snack")
        state_sample=State(frozenset(self.env.env.problems[0].initial_state),
                           frozenset(self.env.env.problems[0].objects),self.env.env.problems[0].goal)
        state_sample=self.env.env._handle_derived_literals(state_sample)
        self.env.env.set_state(state_sample)
        self.env.env._action_space.reset_initial_state(state_sample)
        self.current_observation=self.env.env.get_state()
        self.current_observation,_=self.env.reset()


        '''at = Predicate(name="at", arity=2)
        list_sample=list(self.current_observation[0])
        list_sample=list(sorted(list_sample))
        for literal in list_sample:
            if literal.predicate==at:
                x = self.env.env.problems[0].objects.index(literal.variables[0])
                pass'''

        self.actions=self.all_ground_literals()
        self.action_space=Discrete(len(self.actions))
        self.object_num,self.object_index=self.compute_object_alone_sum(self.env.env.problems[0].objects)

        #self.current_observation,_=self.env.reset()

        self.observation_init=True
        #self.observation_predicate=[]
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
        self.predicate_index.pop()
        #at_kitchen_bread+at_kitchen_content+at_kitchen_sandwich+no_gluten_bread+no_gluten_content
        # +ontray+no_gluten_sandwich+allergic_gluten+not_allergic_gluten+served+waiting+at+notexist
        self.observation_space=MultiBinary(n)
        self.observation_shape_n=n
        self.observation_multibinary=self.observation(self.current_observation)
        #state=self.multibinary_to_literal_observation(self.observation_multibinary)
        self.valid_action=self.compute_valid_action(self.current_observation)
        self.masks_buffer=dict()
        self.step_num=0
        self.serve_sum=0
        self.horizon=3000
        self.child_sum=self.object_num['child']




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

    def observation(self,obs):
        # literal_to_multibinary_observation
        # at_kitchen_bread+at_kitchen_content+at_kitchen_sandwich+no_gluten_bread+no_gluten_content
        # +ontray+no_gluten_sandwich+allergic_gluten+not_allergic_gluten+served+waiting+at+notexist
        at_kitchen_bread=Predicate(name="at_kitchen_bread",arity=1)
        at_kitchen_content = Predicate(name="at_kitchen_content", arity=1)
        at_kitchen_sandwich = Predicate(name="at_kitchen_sandwich", arity=1)
        no_gluten_bread = Predicate(name="no_gluten_bread", arity=1)
        no_gluten_content = Predicate(name="no_gluten_content", arity=1)
        ontray = Predicate(name="ontray", arity=2)
        no_gluten_sandwich = Predicate(name="no_gluten_sandwich", arity=1)
        allergic_gluten= Predicate(name="allergic_gluten", arity=1)
        not_allergic_gluten = Predicate(name="not_allergic_gluten", arity=1)
        served= Predicate(name="served", arity=1)
        waiting = Predicate(name="waiting", arity=2)
        at = Predicate(name="at", arity=2)
        notexist= Predicate(name="notexist", arity=1)

        state=np.zeros(self.observation_shape_n,dtype=int)

        list_sample = list(obs[0])
        list_sample = list(sorted(list_sample))
        # objects中的顺序为:
        # bread+child+content+sandwich+table+tray+kitchen
        for literal in list_sample:
            if literal.predicate == at_kitchen_bread:
                x_name = str(literal.variables[0]).split(':')  # bread
                x = int(x_name[0][5:])
                state[self.predicate_index[0]+x-1]=1
            elif literal.predicate == at_kitchen_content:
                x_name = str(literal.variables[0]).split(':')  # content
                x = int(x_name[0][7:])
                state[self.predicate_index[1]+x-1]=1
            elif literal.predicate == at_kitchen_sandwich:
                x_name = str(literal.variables[0]).split(':')  # sandw
                x = int(x_name[0][5:])
                state[self.predicate_index[2]+x-1]=1
            elif literal.predicate == no_gluten_bread:
                x_name = str(literal.variables[0]).split(':')  # bread
                x = int(x_name[0][5:])
                state[self.predicate_index[3]+x-1]=1
            elif literal.predicate == no_gluten_content:
                x_name = str(literal.variables[0]).split(':')  # content
                x = int(x_name[0][7:])
                state[self.predicate_index[4]+x-1]=1
            elif literal.predicate == ontray:
                x_name = str(literal.variables[0]).split(':')  # sandw
                x = int(x_name[0][5:])
                y_name = str(literal.variables[1]).split(':')  # tray
                y = int(y_name[0][4:])
                state[self.predicate_index[5]+(x-1)*self.object_num['tray']+y-1]=1
            elif literal.predicate == no_gluten_sandwich:
                x_name = str(literal.variables[0]).split(':')  # sandw
                x = int(x_name[0][5:])
                state[self.predicate_index[6]+x-1]=1
            elif literal.predicate == allergic_gluten:
                x_name=str(literal.variables[0]).split(':')#child
                x=int(x_name[0][5:])
                state[self.predicate_index[7]+x-1]=1
            elif literal.predicate == not_allergic_gluten:
                x_name = str(literal.variables[0]).split(':')  # child
                x = int(x_name[0][5:])
                state[self.predicate_index[8]+x-1]=1
            elif literal.predicate == served:
                x_name = str(literal.variables[0]).split(':')  # child
                x = int(x_name[0][5:])
                state[self.predicate_index[9]+x-1]=1
            elif literal.predicate == waiting:
                x_name = str(literal.variables[0]).split(':')  # child
                x = int(x_name[0][5:])
                y_name = str(literal.variables[1]).split(':')  # place
                if y_name[0][:5]=="table":
                    y = int(y_name[0][5:])
                else:
                    y=self.object_num['place']
                state[self.predicate_index[10]+(x-1)*self.object_num['place']+y-1]=1
            elif literal.predicate == at:
                x_name = str(literal.variables[0]).split(':')  # tray
                x = int(x_name[0][4:])
                y_name = str(literal.variables[1]).split(':')  # place
                if y_name[0][:5] == "table":
                    y = int(y_name[0][5:])
                else:
                    y = self.object_num['place']
                state[self.predicate_index[11]+(x-1)*self.object_num['place']+y-1]=1
            elif literal.predicate == notexist:
                x_name = str(literal.variables[0]).split(':')  # sandw
                x = int(x_name[0][5:])
                state[self.predicate_index[12]+x-1]=1

        return state


    def multibinary_to_literal_observation(self,obs):
        # at_kitchen_bread+at_kitchen_content+at_kitchen_sandwich+no_gluten_bread+no_gluten_content
        # +ontray+no_gluten_sandwich+allergic_gluten+not_allergic_gluten+served+waiting+at+notexist
        at_kitchen_bread = Predicate(name="at_kitchen_bread", arity=1)
        at_kitchen_content = Predicate(name="at_kitchen_content", arity=1)
        at_kitchen_sandwich = Predicate(name="at_kitchen_sandwich", arity=1)
        no_gluten_bread = Predicate(name="no_gluten_bread", arity=1)
        no_gluten_content = Predicate(name="no_gluten_content", arity=1)
        ontray = Predicate(name="ontray", arity=2)
        no_gluten_sandwich = Predicate(name="no_gluten_sandwich", arity=1)
        allergic_gluten = Predicate(name="allergic_gluten", arity=1)
        not_allergic_gluten = Predicate(name="not_allergic_gluten", arity=1)
        served = Predicate(name="served", arity=1)
        waiting = Predicate(name="waiting", arity=2)
        at = Predicate(name="at", arity=2)
        notexist = Predicate(name="notexist", arity=1)
        state=set()
        for i in range(len(obs)):
            if obs[i]==0:
                continue
            if i<self.predicate_index[1]:#at_kitchen_bread
                x=i-self.predicate_index[0]+1
                x_name="bread"+str(x)
                variables=[]
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                literal=Literal(predicate=at_kitchen_bread,variables=variables)
                state.add(literal)
            elif i<self.predicate_index[2]:#at_kitchen_content
                x=i-self.predicate_index[1]+1
                x_name="content"+str(x)
                variables=[]
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                literal=Literal(predicate=at_kitchen_content,variables=variables)
                state.add(literal)
            elif i<self.predicate_index[3]:#at_kitchen_sandwich
                x=i-self.predicate_index[2]+1
                x_name="sandw"+str(x)
                variables=[]
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                literal=Literal(predicate=at_kitchen_sandwich,variables=variables)
                state.add(literal)
            elif i<self.predicate_index[4]:#no_gluten_bread
                x=i-self.predicate_index[3]+1
                x_name="bread"+str(x)
                variables=[]
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                literal=Literal(predicate=no_gluten_bread,variables=variables)
                state.add(literal)
            elif i<self.predicate_index[5]:#no_gluten_content
                x=i-self.predicate_index[4]+1
                x_name="content"+str(x)
                variables=[]
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                literal=Literal(predicate=no_gluten_content,variables=variables)
                state.add(literal)
            elif i<self.predicate_index[6]:#ontray (sandw,tray)
                x=(i-self.predicate_index[5])//(self.object_num['tray'])+1
                x_name="sandw"+str(x)
                y= (i - self.predicate_index[5])%(self.object_num['tray']) + 1
                y_name = "tray" + str(y)
                variables=[]
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                variables.append(self.env.env.problems[0].objects[self.object_index.index(y_name)])
                literal=Literal(predicate=ontray,variables=variables)
                state.add(literal)
            elif i<self.predicate_index[7]:#no_gluten_sandwich
                x=i-self.predicate_index[6]+1
                x_name="sandw"+str(x)
                variables=[]
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                literal=Literal(predicate=no_gluten_sandwich,variables=variables)
                state.add(literal)
            elif i<self.predicate_index[8]:#allergic_gluten
                x=i-self.predicate_index[7]+1
                x_name="child"+str(x)
                variables=[]
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                literal=Literal(predicate=allergic_gluten,variables=variables)
                state.add(literal)
            elif i<self.predicate_index[9]:#not_allergic_gluten
                x=i-self.predicate_index[8]+1
                x_name="child"+str(x)
                variables=[]
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                literal=Literal(predicate=not_allergic_gluten,variables=variables)
                state.add(literal)
            elif i<self.predicate_index[10]:#served
                x=i-self.predicate_index[9]+1
                x_name="child"+str(x)
                variables=[]
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                literal=Literal(predicate=served,variables=variables)
                state.add(literal)
            elif i<self.predicate_index[11]:#waiting(child,place)
                x=(i-self.predicate_index[10])//(self.object_num['place'])+1
                x_name="child"+str(x)
                y= (i - self.predicate_index[10])%(self.object_num['place']) + 1
                if y==self.object_num['place']:
                    y_name="kitchen"
                else:
                    y_name = "table" + str(y)
                variables=[]
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                variables.append(self.env.env.problems[0].objects[self.object_index.index(y_name)])
                literal=Literal(predicate=waiting,variables=variables)
                state.add(literal)
            elif i<self.predicate_index[12]:#at(tray,place)
                x=(i-self.predicate_index[11])//(self.object_num['place'])+1
                x_name="tray"+str(x)
                y= (i - self.predicate_index[11])%(self.object_num['place']) + 1
                if y==self.object_num['place']:
                    y_name="kitchen"
                else:
                    y_name = "table" + str(y)
                variables=[]
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                variables.append(self.env.env.problems[0].objects[self.object_index.index(y_name)])
                literal=Literal(predicate=at,variables=variables)
                state.add(literal)
            else:#notexist
                x=i-self.predicate_index[12]+1
                x_name="sandw"+str(x)
                variables=[]
                variables.append(self.env.env.problems[0].objects[self.object_index.index(x_name)])
                literal=Literal(predicate=notexist,variables=variables)
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
        #传入的act为discrete
        if self.valid_action[act] is True:
            return self.actions[act]
        else:
            return False

    def reset(self,**kwargs):
        state=self.env.reset()
        self.current_observation=state[0]
        self.observation_init=True
        state=self.observation(state[0])
        self.observation_init=False
        self.observation_multibinary=state
        self.valid_action=self.compute_valid_action(self.current_observation)
        self.step_num=0
        self.serve_sum=0
        return state

    def step(self,action):
        action=self.action(action)
        if action==False:
            reward=0
            done=True
            debug_info={}
            obs=self.observation_multibinary
        else:
            obs,reward,done,_,debug_info=self.env.step(action)
            reward=0
            self.current_observation=obs
            obs=self.observation(obs)
            self.observation_multibinary = obs
            self.valid_action=self.compute_valid_action(self.current_observation)

        #如果新招待了孩子reward+1
        n=0
        for i in range(self.predicate_index[9],self.predicate_index[10]):
            if self.observation_multibinary[i]==1:
                n+=1
        if n>self.serve_sum:
            reward=1/self.child_sum
            self.serve_sum=n
        #更新当前回合步数，超过阈值则结束
        self.step_num+=1
        if(self.step_num>self.horizon):
            done=True
        #判断当前状态是否处于自动机错误状态，如果处于则结束
        q2=self.check_q2_2_2(obs)
        if q2:
            done=True
        #判断回合结束时任务是否已解决
        '''if done==True:
            if self.serve_sum>=self.object_num['child'] and q2==False:
                reward=1000
            else:
                reward=-1000'''



        #print(action, done, reward)

        return obs,reward,done,debug_info

    def check_q2_3(self,obs):
        if obs[self.predicate_index[9]]==0 and obs[self.predicate_index[9]+1]==1:#未招待child1的情况下招待child2
            return True
        if obs[self.predicate_index[9]+1]==0 and obs[self.predicate_index[9]+2]==1:#未招待child2的情况下招待child3
            return True
        if obs[self.predicate_index[9]]==0 and obs[self.predicate_index[9]+2]==1:#未招待child1的情况下招待child3
            return True
        return False

    def check_q2_2_2(self,obs):
        if obs[self.predicate_index[9]]==0 and obs[self.predicate_index[9]+1]==1:#未招待child1的情况下招待child2
            return True
        if obs[self.predicate_index[9]+2]==0 and obs[self.predicate_index[9]+3]==1:#未招待child3的情况下招待child4
            return True
        return False
























if __name__=="__main__":
    start_time = datetime.now()
    env = WrapperEnv_childsnack()
    model=MyA2C('MlpPolicy',env=env,verbose=1,tensorboard_log="./grid-visit-all_tensorboard/c15_2_4")#
    def timeout_handler(signum,frame):
        print("Time is over.")
        sys.exit(1)
    #设置定时器时间
    signal.signal(signal.SIGALRM,timeout_handler)
    signal.alarm(3900)#设置定时器
    #模拟程序运行：
    try:
        model.learn(total_timesteps=500000000, log_interval=100)
    except KeyboardInterrupt:
        print("程序被手动终止")
    finally:
        signal.alarm(0)#取消定时器

    #model.save("c10_1_3")
    end_time = datetime.now()
    print(end_time - start_time)


