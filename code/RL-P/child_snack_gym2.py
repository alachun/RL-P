import copy
from datetime import datetime
import gym
from unified_planning.shortcuts import *
import numpy as np
from datetime import timedelta
from kstar_planner import planners
from pathlib import Path
'''双时序+table 先a再b同时先c再d
使用kstar_planner作为规划器，其要求传入的为pddl文件'''

'''
示例
env=ChildsnackEnv(child_sum=15,bread_sum=15,content_sum=15,tray_sum=3,table_sum=3,sandw_sum=20,
                  no_gluten_bread_index=[3,14,6,13,2,9],
                  no_gluten_content_index=[8,2,13,5,1,4],
                  allergic_gluten_index=[1,10,5,7,8,9],
                  waiting_tables=[3,2,1,3,1,1,3,1,1,1,2,3,1,2,2],
                  served_order=[1,2,3,4],
                  viable_state=[3,4,5,6,7,8,9,10],
                  final_state=[10],
                  edges=[(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 2), (3, 2), (3, 3), (3, 5), (3, 6), (3, 7),
                         (4, 2), (4, 4), (4, 5), (4, 8), (4, 9), (5, 2), (5, 5), (5, 7), (5, 9), (5, 10), (6, 2),
                         (6, 6), (6, 7), (7, 2), (7, 7), (7, 10), (8, 2), (8, 8), (8, 9), (9, 2), (9, 9), (9, 10),
                         (10, 2), (10, 10)],
                  plan_sum=2)
'''

class ChildsnackEnv(gym.Env):
    def __init__(self, child_sum, bread_sum, content_sum, tray_sum, table_sum, sandw_sum, no_gluten_bread_index,
                 no_gluten_content_index, allergic_gluten_index, waiting_tables, served_order, viable_state,
                 final_state,edges,plan_sum):
        '''child_sum表示需要招待的孩子总数
        bread_sum表示拥有的bread总数
        content_sum表示拥有的content总数
        tray_sum表示拥有的盘子总数
        table_sum表示拥有的桌子总数
        sandw_sum表示初始化时的三明治总数
        no_gluten_bread_index表示不含麸质的面包的索引,为列表类型，其中1表示bread1不含麸质
        no_gluten_content_index表示不含麸质的content的索引,为列表类型，其中1表示content1不含麸质
        allergic_gluten_index表示对麸质过敏的孩子的索引，为列表类型，其中1表示child1对麸质过敏
        waiting_tables表示每个孩子在哪个餐桌等待，为列表类型，[1,1,3]表示child1,child2在table1等待,child3在table3等待
        served_order表示招待顺序，为列表类型;[1,8]表示先招待child1再招待child8
        viable_state表示自动机中可以到达的节点,为列表类型,不包含q1;eg:[3,4,5]
        final_state表示哪些状态是终态，为列表类型;eg:[5]
        edges表示DFA图中的所有边
        plan_sum表示规划器返回的规划解数量
        '''
        # 定义先后招待的child索引
        self.served_order = served_order
        # 定义dfa中可行状态节点
        self.viable_state = viable_state
        self.viable_state_node = len(viable_state)
        # 定义终态
        self.final_state = final_state

        # 定义环境的状态空间，其中observation_shape_size表示状态空间的总大小，下面给出了observation中存储谓词的顺序
        # at_kitchen_bread(bread_sum)+at_kitchen_content(content_sum)+at_kitchen_sandwich(sandw_sum)
        # +no_gluten_sandwich(sandw_sum)+ontray(sandw_sum*tray_sum)+served(child_sum)
        # +at(tray_sum*(table_sum+1))+notexist(sandw_sum)+viable_state(viable_state_node)
        self.observation_shape_size = bread_sum + content_sum + sandw_sum * 3 + child_sum + sandw_sum * tray_sum + tray_sum * (
                    table_sum + 1) + self.viable_state_node
        self.observation_space = gym.spaces.MultiBinary(self.observation_shape_size)
        # 定义状态空间中各个谓词开始的索引
        self.fluents_index = [0, bread_sum, bread_sum + content_sum, bread_sum + content_sum + sandw_sum,
                              bread_sum + content_sum + sandw_sum * 2,
                              bread_sum + content_sum + sandw_sum * 2 + sandw_sum * tray_sum,
                              bread_sum + content_sum + sandw_sum * 2 + sandw_sum * tray_sum + child_sum,
                              bread_sum + content_sum + sandw_sum * 2 + sandw_sum * tray_sum + child_sum + tray_sum * (
                                          table_sum + 1),
                              bread_sum + content_sum + sandw_sum * 3 + sandw_sum * tray_sum + child_sum + tray_sum * (
                                          table_sum + 1), ]

        # 定义环境的动作空间
        self.plan_sum=plan_sum
        self.action_space = gym.spaces.Discrete((child_sum + self.viable_state_node)*self.plan_sum)
        # reward_table用于存储历史规划解
        self.reward_table = dict()

        self.child_sum = child_sum
        self.bread_sum = bread_sum
        self.content_sum = content_sum
        self.sandw_sum = sandw_sum
        self.tray_sum = tray_sum
        self.table_sum = table_sum
        self.waiting_tables = waiting_tables
        self.no_gluten_bread_index = no_gluten_bread_index
        self.no_gluten_content_index = no_gluten_content_index
        self.allergic_gluten_index = allergic_gluten_index

        # 双时序的状态匹配字典
        '''键表示双时序中a,b,c,d分别对应的served child的值，其中1为真0为假；值表示当前DFA处于哪个状态节点'''
        self.cases = {
            "0000": 1,
            "0010": 3,
            "1000": 4,
            "1010": 5,
            "0011": 6,
            "1011": 7,
            "1100": 8,
            "1110": 9,
            "1111": 10,

        }
        # 计算DFA中每个状态的所有直接后继节点(不包括自身节点)
        self.succs = self.succ(edges)
        # 设置当前状态的动作掩码，会随着状态更新
        self.valid_action = [True] * self.action_space.n
        # 设置初始环境状态，后续会更新
        self.state = self.reset()
        # 存储动作掩码，用于MyA2C中策略网络更新
        self.masks_buffer=dict()
        # 新回合开始标志
        self.reset_flag = True
        # 规划器调用的总时间
        self.planner_time=timedelta()
        # pddl_content是problem文件的一部分，避免重复计算
        self.pddl_content = self.compute_pddl_content(self.bread_sum, self.content_sum, self.child_sum,
                                                      self.tray_sum, self.table_sum, self.sandw_sum,
                                                      self.no_gluten_bread_index, self.no_gluten_content_index,
                                                      self.allergic_gluten_index, self.waiting_tables)
        # 记录所有回合的cost，只在测试时有作用
        self.all_episode_cost = []
        # 当前回合的cost
        self.current_episode_cost = 0


    def step(self, action):
        '''actions_file.txt文件中记录了策略网络每次选择的动作[选择的子任务的索引，选择的规划解的索引]，
                以及每次调用规划器解决问题的时间'''
        action = [action // self.plan_sum, action % self.plan_sum]
        # action[0]为选择的子任务的索引
        # action[1]为选择的规划解的索引
        with open("actions_file.txt", "a") as file:
            if self.reset_flag == True:
                file.write(f"\n")
            file.write(f"[{action[0]}  {action[1]}]")
        self.reset_flag = False

        state_str = ''.join(self.state.astype(str))  # 将 state 转换为字符串
        action1_str = str(action[0])  # 将 action 转换为字符串
        action2_str=str(action[1])
        # 连接 state 和 action
        combined_str = state_str + action1_str+action2_str

        if combined_str in self.reward_table.keys():
            #更新state和mask
            self.state = self.reward_table[combined_str][0].copy()
            self.valid_action = self.reward_table[combined_str][3].copy()
            self.current_episode_cost += self.reward_table[combined_str][4]
            return self.reward_table[combined_str][0], self.reward_table[combined_str][1], \
            self.reward_table[combined_str][2], dict()

        # 更新pddl的problem文件
        pddl_content=self.pddl_content
        with open('problem.pddl','w') as file:
            file.write(pddl_content)

        at_kitchen_bread=[]
        at_kitchen_content=[]
        at_kitchen_sandwich=[]
        no_gluten_sandwich=[]
        on_tray=[]
        served=[]
        at=[]
        notexist=[]

        # 更改pddl中的部分谓词
        for i in range(self.observation_shape_size):
            if i < self.bread_sum:
                # at_kitchen_bread
                if self.state[i] == 1:
                    at_kitchen_bread.append(i+1)
            elif i < self.fluents_index[2]:
                # at_kitchen_content
                if self.state[i] == 1:
                    at_kitchen_content.append(i-self.fluents_index[1]+1)
            elif i < self.fluents_index[3]:
                # at_kitchen_sandwich
                if self.state[i] == 1:
                    at_kitchen_sandwich.append(i-self.fluents_index[2]+1)
            elif i < self.fluents_index[4]:
                # no_gluten_sandwich
                if self.state[i] == 1:
                    no_gluten_sandwich.append(i-self.fluents_index[3]+1)
            elif i < self.fluents_index[5]:
                # on_tray
                if self.state[i] == 1:
                    on_tray.append((i-self.fluents_index[4])//self.tray_sum+1) #sandwich
                    on_tray.append((i-self.fluents_index[4])%self.tray_sum+1)  #tray
            elif i < self.fluents_index[6]:
                # served
                if self.state[i] == 1:
                    served.append(i-self.fluents_index[5]+1)
            elif i < self.fluents_index[7]:
                # at
                if self.state[i]==1:
                    at.append((i - self.fluents_index[6]) // (self.table_sum + 1) + 1)  # tray
                    at.append((i - self.fluents_index[6]) % (self.table_sum + 1) + 1)  # place
            elif i<self.fluents_index[8]:
                # notexist
                if self.state[i]==1:
                    notexist.append(i-self.fluents_index[7]+1)

        #当前dfa状态
        first_dig = str(self.state[self.fluents_index[5] + self.served_order[0] - 1])
        second_dig = str(self.state[self.fluents_index[5] + self.served_order[1] - 1])
        third_dig = str(self.state[self.fluents_index[5] + self.served_order[2] - 1])
        fourth_dig = str(self.state[self.fluents_index[5] + self.served_order[3] - 1])
        match_str = first_dig + second_dig + third_dig + fourth_dig
        dfa_state = self.cases[match_str]

        #更改problem文件中的init和goal
        with open('problem.pddl', 'a') as file:
            file.write(f"\n")
            for i in at_kitchen_bread:
                file.write(f"     (at_kitchen_bread bread{i})\n")
            for i in at_kitchen_content:
                file.write(f"     (at_kitchen_content content{i})\n")
            for i in at_kitchen_sandwich:
                file.write(f"     (at_kitchen_sandwich sandw{i})\n")
            for i in no_gluten_sandwich:
                file.write(f"     (no_gluten_sandwich sandw{i})\n")
            for i in range(0,len(on_tray),2):
                file.write(f"     (ontray sandw{i} tray{on_tray[i+1]})\n")
            for i in served:
                file.write(f"     (served child{i})\n")
            for i in range(0,len(at),2):
                if at[i+1]==4:
                    file.write(f"     (at tray{at[i]} kitchen)\n")
                else:
                    file.write(f"     (at tray{at[i]} table{at[i+1]})\n")
            for i in notexist:
                file.write(f"     (notexist sandw{i})\n")
            file.write(f"     (q{dfa_state}))\n")
            if action[0]<self.child_sum:
                file.write(f"  (:goal(and(served child{action[0]+1}))))")
            else:
                file.write(f"  (:goal(and(q{action[0]-self.child_sum+3}))))")

        domain_file = Path("domain_2*2.pddl")#domain文件不会变化
        problem_file = Path("problem.pddl")
        heuristic = "ff()"#启发式算法
        start_time = datetime.now()

        # 调用kstar规划器求解
        plans = planners.plan_topk(domain_file=domain_file, problem_file=problem_file, number_of_plans_bound=self.plan_sum,search_heuristic=heuristic,timeout=180)  #
        end_time = datetime.now()
        solve_time = end_time - start_time
        self.planner_time += solve_time
        with open("actions_file.txt", "a") as file:
            file.write(f"{solve_time}   ")

        '''在依次计算执行每个plan的效果后state和mask都会改变，所以需要提前保存之前的值，方便正确计算下一个plan的执行效果'''
        state_pre=self.state.copy()#执行规划解前的状态
        mask_pre=copy.deepcopy(self.valid_action)# 执行规划解前的掩码
        j=0
        while j <len(plans['plans']):
            plan_len = plans['plans'][j]['cost']#规划解长度（包括trans）
            if plan_len==0:
                break
            self.state = state_pre.copy()
            for i in range(plan_len):
                if i % 2 == 0:#自动过滤trans
                    continue

                #测试的时候用于记录每个plan的具体动作，训练时注释掉
                '''with open("actions_file.txt", "a") as file:
                    file.write(f"{plans['plans'][j]['actions'][i]}   ")'''

                action_split = plans['plans'][j]['actions'][i].split()
                action_name = action_split[0]#动作名称
                parameters_tuple = action_split[1:]#动作参数
                if action_name == 'make_sandwich':
                    s_index = int(str(parameters_tuple[0])[5:]) - 1
                    b_index = int(str(parameters_tuple[1])[5:]) - 1
                    c_index = int(str(parameters_tuple[2])[7:]) - 1
                    # 更改at_kitchen_bread
                    self.state[b_index] = 0
                    # 更改at_kitchen_content
                    self.state[c_index + self.bread_sum] = 0
                    # 更改at_kitchen_sandwich
                    self.state[s_index + self.bread_sum + self.content_sum] = 1
                    # 更改notexist
                    self.state[self.fluents_index[7] + s_index] = 0
                elif action_name == 'make_sandwich_no_gluten':
                    s_index = int(str(parameters_tuple[0])[5:]) - 1
                    b_index = int(str(parameters_tuple[1])[5:]) - 1
                    c_index = int(str(parameters_tuple[2])[7:]) - 1
                    # 更改at_kitchen_bread
                    self.state[b_index] = 0
                    # 更改at_kitchen_content
                    self.state[c_index + self.bread_sum] = 0
                    # 更改at_kitchen_sandwich
                    self.state[s_index + self.bread_sum + self.content_sum] = 1
                    # 更改no_gluten_sandwich
                    self.state[self.fluents_index[3] + s_index] = 1
                    # 更改notexist
                    self.state[self.fluents_index[7] + s_index] = 0
                elif action_name == 'put_on_tray':
                    s_index = int(str(parameters_tuple[0])[5:]) - 1
                    tray_index = int(str(parameters_tuple[1])[4:]) - 1
                    # 更改at_kitchen_sandwich
                    self.state[s_index + self.fluents_index[2]] = 0
                    # 更改ontray
                    self.state[self.fluents_index[4] + s_index * self.tray_sum + tray_index] = 1
                elif action_name == 'serve_sandwich_no_gluten':
                    s_index = int(str(parameters_tuple[0])[5:]) - 1
                    c_index = int(str(parameters_tuple[1])[5:]) - 1
                    t_index = int(str(parameters_tuple[2])[4:]) - 1
                    # p_index=int(str(parameters_tuple[3])[5:]) - 1
                    # 更改ontray
                    self.state[self.fluents_index[4] + s_index * self.tray_sum + t_index] = 0
                    # 更改served
                    self.state[self.fluents_index[5] + c_index] = 1
                elif action_name == 'serve_sandwich':
                    s_index = int(str(parameters_tuple[0])[5:]) - 1
                    c_index = int(str(parameters_tuple[1])[5:]) - 1
                    t_index = int(str(parameters_tuple[2])[4:]) - 1
                    # 更改ontray
                    self.state[self.fluents_index[4] + s_index * self.tray_sum + t_index] = 0
                    # 更改served
                    self.state[self.fluents_index[5] + c_index] = 1
                elif action_name == 'move_tray':
                    t_index = int(str(parameters_tuple[0])[4:]) - 1
                    p1 = str(parameters_tuple[1])
                    p2 = str(parameters_tuple[2])
                    if p1 == 'kitchen':
                        p1_index = self.table_sum
                    else:
                        p1_index = int(p1[5:]) - 1
                    if p2 == 'kitchen':
                        p2_index = self.table_sum
                    else:
                        p2_index = int(p2[5:]) - 1
                    # 更改at(t,p1),at(t,p2)
                    self.state[self.fluents_index[6] + t_index * (self.table_sum + 1) + p1_index] = 0
                    self.state[self.fluents_index[6] + t_index * (self.table_sum + 1) + p2_index] = 1


            # 计算dfa状态以及q2状态的判定和返回值的设置
            self.state[self.fluents_index[8]:] = 0
            first_dig = str(self.state[self.fluents_index[5] + self.served_order[0] - 1])
            second_dig = str(self.state[self.fluents_index[5] + self.served_order[1] - 1])
            third_dig = str(self.state[self.fluents_index[5] + self.served_order[2] - 1])
            fourth_dig = str(self.state[self.fluents_index[5] + self.served_order[3] - 1])
            match_str = first_dig + second_dig + third_dig + fourth_dig

            reward = 1/(plan_len // 2)

            if match_str in self.cases.keys():#若当前dfa不处于错误状态
                dfa_state = self.cases[match_str]
                if dfa_state != 1:
                    self.state[self.fluents_index[8] + dfa_state - 3] = 1
                done = self.check_done(self.state)
                if done == True:#总目标已完成，给予一个比较大的正向奖励
                    reward += 1000
            else:  #DFA到达错误状态，回合结束，给予惩罚
                reward = -1000
                done = True

            # 更新动作掩码
            valid_action, temp = self.compute_valid_action_array(self.state, mask_pre.copy())
            if temp == True:
                done = True
                reward = -1000
            '''存储结果
                        键为当前状态+被选择的子任务索引+被选择的规划解索引的字符串
                        值为下一状态+奖励+回合结束标志+更新后的动作掩码+规划解长度/cost(childsnack问题中规划解长度和cost相等)
                        '''
            self.reward_table[state_str+action1_str+str(j)]= [self.state.copy(), reward, done, valid_action.copy(),plan_len // 2]
            j=j+1

        # 如果没有找到足够的规划解
        while j < self.plan_sum:
            '''#+action mask
            self.reward_table[state_str + action1_str + str(j)] = [state_pre.copy(), -1000, True, mask_pre.copy()]'''

            # 不加action mask（加入action mask也能用，就是多算了几步）
            # 选择的子任务为原pddl问题文件的目标
            if action[0] < self.child_sum:
                # 不加action mask可能会存在重复选择同一子任务的情况，此时规划解长度为0
                if state_pre[self.fluents_index[5] + action[0]] == 1:
                    self.reward_table[state_str + action1_str + str(j)] = [state_pre.copy(), 0, False, mask_pre.copy(),0]
                else:#无解的情况
                    self.reward_table[state_str + action1_str + str(j)] = [state_pre.copy(), -1000, True,
                                                                           mask_pre.copy(),0]
            # 选择的子任务为DFA状态节点
            else:
                first_dig = str(state_pre[self.fluents_index[5] + self.served_order[0] - 1])
                second_dig = str(state_pre[self.fluents_index[5] + self.served_order[1] - 1])
                third_dig = str(state_pre[self.fluents_index[5] + self.served_order[2] - 1])
                fourth_dig = str(state_pre[self.fluents_index[5] + self.served_order[3] - 1])
                match_str = first_dig + second_dig + third_dig+fourth_dig
                dfa_state = self.cases[match_str]#当前DFA所处状态节点
                if dfa_state == action[0] - self.child_sum + 3:#选择的子任务恰好是当前所处的状态节点，规划解长度为0
                    self.reward_table[state_str + action1_str + str(j)] = [state_pre.copy(), 0, False, mask_pre.copy(),0]
                else:#选择的子任务是其他状态节点导致的无解或超时
                    self.reward_table[state_str + action1_str + str(j)] = [state_pre.copy(), -1000, True,
                                                                           mask_pre.copy(),0]
            j = j + 1

        # 更新下一状态，动作掩码，奖励，当前回合累积cost，回合结束标志done
        self.state=self.reward_table[combined_str][0].copy()
        self.valid_action=self.reward_table[combined_str][3].copy()
        reward=self.reward_table[combined_str][1]
        done=self.reward_table[combined_str][2]
        self.current_episode_cost += self.reward_table[combined_str][4]
        # debug_info默认返回空字典
        debug_info = dict()
        '''if reward>100:#可用于记录当前回合是否成功
            with open("actions_file.txt", "a") as file:
                file.write(f" succ ")'''
        return self.state, reward, done, debug_info

    # 检查执行action后，当前状态done是否为True
    def check_done(self, state):
        # 所有child均被招待
        for i in range(self.child_sum):
            if state[self.fluents_index[5] + i] == 0:
                return False
        # 到达自动机终止状态
        for index, value in enumerate(self.viable_state):
            # state中的终态节点对应值为1
            if value in self.final_state and state[self.fluents_index[8] + index] == 1:
                return True
        return False

    # reset函数
    def reset(self, **kwargs):
        # 更新回合开始标志
        self.reset_flag = True
        # 初始化状态，默认全0
        state = np.zeros(self.observation_shape_size, dtype=int)
        # 初始状态at_kitchen_bread,at_kitchen_content为True
        replace_indices = np.arange(0, self.bread_sum + self.content_sum)
        state[replace_indices] = 1
        # 更改初始状态时at(tray,place)的值
        m = self.bread_sum + self.content_sum + self.sandw_sum * 2 + self.sandw_sum * self.tray_sum + self.child_sum
        for i in range(self.tray_sum):
            state[m + i * (self.table_sum + 1) + 3] = 1
        # 初始状态notexist都为True
        replace_indices = np.arange(m + self.tray_sum * (self.table_sum + 1),
                                    m + self.tray_sum * (self.table_sum + 1) + self.sandw_sum)
        state[replace_indices] = 1
        # 更新当前状态
        self.state = state
        # 更新当前状态可选子目标
        self.valid_action = [True] * self.action_space.n
        self.valid_action, actionmask_done = self.compute_valid_action_array(self.state, self.valid_action)
        # 更新当前回合累积cost
        self.current_episode_cost = 0
        return state

    # 计算各个状态的可选子目标
    def compute_valid_action_array(self,state,valid_action_array):
        done = False
        first_dig = str(self.state[self.fluents_index[5] + self.served_order[0] - 1])
        second_dig = str(self.state[self.fluents_index[5] + self.served_order[1] - 1])
        third_dig = str(self.state[self.fluents_index[5] + self.served_order[2] - 1])
        fourth_dig = str(self.state[self.fluents_index[5] + self.served_order[3] - 1])
        match_str = first_dig + second_dig + third_dig + fourth_dig
        if match_str in self.cases.keys():
            dfa_state = self.cases[match_str]#当前所处DFA状态
        else:
            return valid_action_array, done#DFA处于错误状态，结束
        for i, mask in enumerate(valid_action_array):
            subgoal_index = i // self.plan_sum#被选择的子任务索引
            # 存在新完成的子任务
            if (mask == True) and ((subgoal_index < self.child_sum and state[self.fluents_index[5] + subgoal_index] == 1)
                                   or (subgoal_index >= self.child_sum and state[self.fluents_index[8] + subgoal_index - self.child_sum] == 1)):
                valid_action_array[i] = False
            # DFA当前状态的直接后继（不包括自身）可被选择
            if subgoal_index >= self.child_sum:
                if self.viable_state[subgoal_index - self.child_sum] in self.succs[dfa_state] and (
                        subgoal_index - self.child_sum + 3) != dfa_state:
                    valid_action_array[i] = True
                else:
                    valid_action_array[i] = False
        return valid_action_array, done

    # 计算直接后继节点
    def succ(self,edges):
        succs = dict()
        for u, v in edges:
            if u not in succs:
                succs[u] = []
            succs[u].append(v)
        return succs

    # 计算部分保持不变problem文件
    def compute_pddl_content(self, bread_sum, content_sum, child_sum, tray_sum, table_sum, sandw_sum,
                             no_gluten_bread_index, no_gluten_content_index, allergic_gluten_index, waiting_tables):
        pddl_content = """(define (problem prob-snack)
    (:domain child-snack)
    (:objects\n"""
        pddl_content += f"    "
        for i in range(1, child_sum + 1):
            pddl_content += f"child{i} "
        pddl_content += f"- child\n"
        pddl_content += f"    "
        for i in range(1, bread_sum + 1):
            pddl_content += f"bread{i} "
        pddl_content += f"- bread-portion\n"
        pddl_content += f"    "
        for i in range(1, content_sum + 1):
            pddl_content += f"content{i} "
        pddl_content += f"- content-portion\n"
        pddl_content += f"    "
        for i in range(1, tray_sum + 1):
            pddl_content += f"tray{i} "
        pddl_content += f"- tray\n"
        pddl_content += f"    "
        for i in range(1, table_sum + 1):
            pddl_content += f"table{i} "
        pddl_content += f"- place\n"
        pddl_content += f"    "
        for i in range(1, sandw_sum + 1):
            pddl_content += f"sandw{i} "
        pddl_content += f"- sandwich\n)\n  (:init\n"
        for i in no_gluten_bread_index:
            pddl_content += f"     (no_gluten_bread bread{i})\n"
        for i in no_gluten_content_index:
            pddl_content += f"     (no_gluten_content content{i})\n"
        for i in range(1, child_sum + 1):
            if i in allergic_gluten_index:
                pddl_content += f"     (allergic_gluten child{i})\n"
            else:
                pddl_content += f"     (not_allergic_gluten child{i})\n"
        for i in range(1, child_sum + 1):
            pddl_content += f"     (waiting child{i} table{waiting_tables[i - 1]})\n"
        pddl_content += f"     (dfa)\n"

        return pddl_content



















