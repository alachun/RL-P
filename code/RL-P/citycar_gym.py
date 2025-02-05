from datetime import datetime
import gym
from unified_planning.shortcuts import *
from unified_planning.engines import PlanGenerationResultStatus
from datetime import timedelta
from citycar_pddlgym import WrapperEnv_citycar
from pddlgym.structs import Predicate,Literal
from citycar_pddl_problem import get_citycar_problem
'''输入的问题为problem文件'''
"三个时序+3F+table，同时将时序约束拆分成两个:三个连续时序（q1-q5）和3F(q6-q9)"
'''动作空间为Discrete  
同时，每次执行动作只调用被选中的规划器，节约时间
该gym实现还调用了citycar的pddlgym实现，主要调用其执行函数(step)求规划解的执行效果，也带来了一些需要注意的细节：
主要是状态和一些参数的同步问题
'''

'''示例
env=CitycarEnv(place=[3,2,2,0,0,1],
                  served_order=[1,2,3],
                  viable_state=[3,4,5,7,8,9],
                  final_state=[5,9],
                  edges=[(1,1),(1,2),(1,3),(2,2),(3,2),(3,3),(3,4),(4,2),(4,4),(4,5),(5,2),(5,5),
                         (6,6),(6,7),(6,8),(6,9),(7,7),(7,8),(7,9),(8,8),(8,9),(9,9)],
                  plan_sum=1)'''

class CitycarEnv(gym.Env):
    def __init__(self,place,served_order,viable_state,final_state,edges,plan_sum):
        '''
        place表示F(a & F(b & Fc))约束中要求小车先后经过的路口;
        eg:[3,2,2,0,0,1]表示先经过junction3-2，再经过junction2-0,然后经过junction0-1
        served_order表示小车到达顺序，为列表类型;[1,2,3]表示第一辆车(car0)先到达目标后，第二辆车(car1)再到达目标，接着第三辆车(car2)再到达目标
        viable_state表示自动机中可以到达的节点,为列表类型,不包含q1;eg:[3,4,5]
        final_state表示哪些状态是终态，为列表类型;eg:[5]
        edges表示DFA图中的所有边
        plan_sum表示规划器返回的规划解数量
        '''
        #调用citycar对应的pddlgym环境
        self.pddlgym_env=WrapperEnv_citycar(place)
        #定义小车到达的先后顺序
        self.served_order=served_order
        #定义dfa中可行状态节点
        self.viable_state=viable_state
        self.viable_state_node=len(viable_state)
        #定义终态
        self.final_state=final_state
        #定义环境的状态空间，其中observation_shape_size表示状态空间的总大小，
        self.observation_shape_size=self.pddlgym_env.observation_shape_n
        self.observation_space=gym.spaces.MultiBinary(self.observation_shape_size)
       #原pddl文件中的子任务数量
        self.pddl_goal_sum = len(self.pddlgym_env.goal_index)
        #规划器能返回的规划解数量
        self.plan_sum = plan_sum
        #定义环境的动作空间
        self.action_space = gym.spaces.Discrete((self.pddl_goal_sum + self.viable_state_node)*self.plan_sum)
        #plan buffer  存储历史规划解
        self.reward_table=dict()
        #设置当前状态的动作掩码，会随着状态更新
        self.valid_action=[True] * self.action_space.n
        #存储动作掩码，用于策略网络更新
        self.masks_buffer=dict()
        #三个连续时序的DFA状态匹配字典
        self.cases={
            "000":1,
            "100":3,
            "110":4,
            "111":5,
        }
        #直接后继节点
        self.succs=self.succ(edges)
        # 设置初始环境状态，后续会更新
        self.state = self.reset()
        #回合开始标志
        self.reset_flag=True
        #规划器调用总时间
        self.planner_time = timedelta()

        self.graph_len = self.pddlgym_env.graph_len#地图边长
        self.car_sum = self.pddlgym_env.object_num['car']#小车数量
        self.garage_sum = self.pddlgym_env.object_num['garage']#车库数量
        self.road_sum = self.pddlgym_env.object_num['road']#可以铺的路的数量
        self.jun_sum=self.pddlgym_env.object_num['junction']#路口数量
        #定义对应的problem文件，后续传给规划器
        self.problem=get_citycar_problem(graph_len=self.graph_len,
                                         car_sum=self.car_sum,
                                         garage_sum=self.garage_sum,
                                         road_sum=self.road_sum,
                                         garage_at_jun=self.pddlgym_env.garage_at_jun,
                                         car_start_garage=self.pddlgym_env.car_start_garage,
                                         car_arrived_jun=self.pddlgym_env.car_arrived_jun,
                                         place=place)
        self.action_predicates = self.recode_action_predicate()#定义了一些谓词
        self.all_episode_cost = []#所有回合的cost，只在模型测试时有用
        self.all_episode_len = []#所有回合的规划解长度，只在模型测试时有用
        self.current_episode_cost = 0#当前回合的cost
        self.current_episode_len = 0#当前回合的规划解长度

    def step(self, action):
        '''actions_file.txt文件中记录了策略网络每次选择的动作[选择的子任务的索引，选择的规划解的索引]，
                以及每次调用规划器解决问题的时间'''
        action=[action//self.plan_sum,action%self.plan_sum]
        # action[0]为选择的子任务的索引
        # action[1]为选择的规划解的索引
        with open("actions_file.txt", "a") as file:
            if self.reset_flag == True:
                file.write(f"\n")
            file.write(f"[{action[0]}  {action[1]}]")
        self.reset_flag = False
        state_str = ''.join(self.state.astype(str))  # 将 state 转换为字符串
        action1_str = str(action[0])  # 将 action 转换为字符串
        action2_str = str(action[1])
        # 连接 state 和 action
        combined_str = state_str + action1_str + action2_str

        if combined_str in self.reward_table.keys():
            # 更新state和mask
            self.state = self.reward_table[combined_str][0].copy()
            self.valid_action = self.reward_table[combined_str][3].copy()
            '''调用pddlgym会导致本gym环境和pddlgym环境不同步
            eg:gym环境执行到此处时即存在历史规划解直接调用其存储的结果，
            但是pddlgym环境还处于上一个子任务执行后的状态，必须强制使pddlgym环境状态与gym环境同步
            需要f3_state是因为pddlgym中更新f3对应的自动机状态时需要当前f3自动机的状态，所以需要一起同步'''
            self.pddlgym_env.set_state(self.state.copy())#重置pddlgym环境的state使之同步
            self.pddlgym_env.f3_state=self.reward_table[combined_str][4]#重置pddlgym环境的f3对应的dfa状态使之同步
            self.current_episode_cost += self.reward_table[combined_str][5]
            self.current_episode_len += self.reward_table[combined_str][6]
            return self.reward_table[combined_str][0], self.reward_table[combined_str][1], \
                self.reward_table[combined_str][2], dict()

        # 更改problem中一些谓词的值
        for i in range(self.pddlgym_env.predicate_index[2],self.pddlgym_env.predicate_index[9]):
            #只包含值会发生变化的predicate
            if i<self.pddlgym_env.predicate_index[3]:#at_car_jun
                x = (i - self.pddlgym_env.predicate_index[2]) // (self.pddlgym_env.object_num['junction'])#car
                y = (i - self.pddlgym_env.predicate_index[2]) % (self.pddlgym_env.object_num['junction'])#junction
                if self.state[i] == 1:
                    self.problem.set_initial_value((self.problem.fluents[2](self.problem.all_objects[self.jun_sum + x],
                                                                            self.problem.all_objects[y])), True)
                else:
                    self.problem.set_initial_value((self.problem.fluents[2](self.problem.all_objects[self.jun_sum + x],
                                                                            self.problem.all_objects[y])), False)


            elif i<self.pddlgym_env.predicate_index[4]:#at_car_road
                x = (i - self.pddlgym_env.predicate_index[3]) // (self.pddlgym_env.object_num['road'])  # car
                y = (i - self.pddlgym_env.predicate_index[3]) % (self.pddlgym_env.object_num['road'])  # road
                if self.state[i]==1:
                    self.problem.set_initial_value((self.problem.fluents[3](self.problem.all_objects[self.jun_sum + x],
                                                                            self.problem.all_objects[
                                                                                self.jun_sum + self.car_sum + self.garage_sum + y])),
                                                   True)
                else:
                    self.problem.set_initial_value((self.problem.fluents[3](self.problem.all_objects[self.jun_sum + x],
                                                                            self.problem.all_objects[
                                                                                self.jun_sum + self.car_sum + self.garage_sum + y])),
                                                   False)


            elif i<self.pddlgym_env.predicate_index[5]:#starting
                x = (i - self.pddlgym_env.predicate_index[4]) // (self.pddlgym_env.object_num['garage'])  # car
                y = (i - self.pddlgym_env.predicate_index[4]) % (self.pddlgym_env.object_num['garage'])  # garage
                if self.state[i]==1:
                    self.problem.set_initial_value((self.problem.fluents[4](self.problem.all_objects[self.jun_sum + x],
                                                                            self.problem.all_objects[
                                                                                self.jun_sum + self.car_sum + y])),
                                                   True)
                else:
                    self.problem.set_initial_value((self.problem.fluents[4](self.problem.all_objects[self.jun_sum + x],
                                                                            self.problem.all_objects[
                                                                                self.jun_sum + self.car_sum + y])),
                                                   False)
            elif i<self.pddlgym_env.predicate_index[6]:#arrived
                x = (i - self.pddlgym_env.predicate_index[5]) // (self.pddlgym_env.object_num['junction'])  # car
                y = (i - self.pddlgym_env.predicate_index[5]) % (self.pddlgym_env.object_num['junction'])  # junction
                if self.state[i]==1:
                    self.problem.set_initial_value((self.problem.fluents[5](self.problem.all_objects[self.jun_sum + x],
                                                                            self.problem.all_objects[y])), True)
                else:
                    self.problem.set_initial_value((self.problem.fluents[5](self.problem.all_objects[self.jun_sum + x],
                                                                            self.problem.all_objects[y])), False)
            elif i<self.pddlgym_env.predicate_index[7]:#road_connect
                x = (i - self.pddlgym_env.predicate_index[6]) // (self.pddlgym_env.object_num['junction']**2)  # road
                y = (i - self.pddlgym_env.predicate_index[6]-x*(self.pddlgym_env.object_num['junction']**2))//(self.pddlgym_env.object_num['junction'])  # junction
                z = (i - self.pddlgym_env.predicate_index[6]-x*(self.pddlgym_env.object_num['junction']**2))%(self.pddlgym_env.object_num['junction']) # junction
                if self.state[i]==1:
                    self.problem.set_initial_value((self.problem.fluents[6](
                        self.problem.all_objects[self.jun_sum + self.car_sum + self.garage_sum + x],
                        self.problem.all_objects[y], self.problem.all_objects[z])), True)
                else:
                    self.problem.set_initial_value((self.problem.fluents[6](
                        self.problem.all_objects[self.jun_sum + self.car_sum + self.garage_sum + x],
                        self.problem.all_objects[y], self.problem.all_objects[z])), False)
            elif i<self.pddlgym_env.predicate_index[8]:#clear
                x =i - self.pddlgym_env.predicate_index[7]#junction
                if self.state[i]==1:
                    self.problem.set_initial_value(self.problem.fluents[7](self.problem.all_objects[x]), True)
                else:
                    self.problem.set_initial_value(self.problem.fluents[7](self.problem.all_objects[x]), False)
            else :#in_place
                x = i - self.pddlgym_env.predicate_index[8]  #road
                if self.state[i]==1:
                    self.problem.set_initial_value(self.problem.fluents[8](
                        self.problem.all_objects[self.jun_sum + self.car_sum + self.garage_sum + x]), True)
                else:
                    self.problem.set_initial_value(self.problem.fluents[8](
                        self.problem.all_objects[self.jun_sum + self.car_sum + self.garage_sum + x]), False)

        # 当前dfa状态
        first_dig = str(self.state[self.pddlgym_env.goal_index[0]])
        second_dig = str(self.state[self.pddlgym_env.goal_index[1]])
        third_dig = str(self.state[self.pddlgym_env.goal_index[2]])
        match_str = first_dig + second_dig + third_dig
        dfa_state = self.cases[match_str]
        for index in range(10,15):#q1-q5
            if (index-9)==dfa_state:
                self.problem.set_initial_value(self.problem.fluents[index],True)
            else:
                self.problem.set_initial_value(self.problem.fluents[index], False)
        for index in range(15,19):#q6-q9
            if (index-14)==self.pddlgym_env.f3_state:
                self.problem.set_initial_value(self.problem.fluents[index], True)
            else:
                self.problem.set_initial_value(self.problem.fluents[index], False)

        # 添加pddl子目标
        self.problem.clear_goals()
        if action[0] < self.pddl_goal_sum:#原pddl文件子目标
            junction_index = (self.pddlgym_env.goal_index[action[0]] - self.pddlgym_env.predicate_index[5]) % \
                             self.pddlgym_env.object_num['junction']
            self.problem.add_goal(self.problem.fluents[5](self.problem.all_objects[self.jun_sum+action[0]],self.problem.all_objects[junction_index]))
        elif action[0] <self.pddl_goal_sum+3:#三个连续时序目标（q3,q4,q5）
            self.problem.add_goal(self.problem.fluents[12+action[0]-self.pddl_goal_sum])
        else:#f3自动机目标(q7,q8,q9)
            self.problem.add_goal(self.problem.fluents[action[0]-self.pddl_goal_sum+13])

        pv = PlanValidator(problem_kind=self.problem.kind)
        metric = self.problem.quality_metrics[0]
        params = []
        params.append({
            'fast_downward_search_time_limit': '180s',
            'fast_downward_search_config': 'let(hcea,cea(),lazy_greedy([hcea],preferred=[hcea]))'
        })
        params.append({
            'fast_downward_search_time_limit': '180s',
            'fast_downward_search_config': 'let(hcea,cea(),lazy(alt([type_based([g()]),single(hcea),single(hcea,pref_only=true)],boost=0),preferred=[hcea],reopen_closed=true,cost_type=plusone,bound=10000,verbosity=silent))'
        })
        params.append({
            'fast_downward_search_time_limit': '180s',
            'fast_downward_search_config': 'let(hcea,cea(),let(hcg,cg(),lazy_greedy([hcea,hcg],preferred=[hcea,hcg])))'
        })
        '''在依次计算执行每个plan的效果后state和mask都会改变，所以需要提前保存之前的值，方便正确计算下一个plan的执行效果'''
        state_pre = self.state.copy()#执行plan前的state
        mask_pre = self.valid_action.copy()#执行plan前的mask
        f3_pre = self.pddlgym_env.f3_state#执行plan前f3所处状态

        for plan_i in range(self.plan_sum):
            # 在实际训练时，如果每次都调用规划器求解出plan_sum个规划解，对于citycar问题来讲，时间开销过大，所以只求解被选中的那个规划解
            if plan_i!=action[1]:
                continue
            #fast-downward规划器
            planner = OneshotPlanner(name="fast-downward", params=params[plan_i])
            start_time = datetime.now()
            result = planner.solve(self.problem)#规划解
            end_time = datetime.now()
            solve_time = end_time - start_time
            self.planner_time += solve_time
            with open("actions_file.txt", "a") as file:
                file.write(f"{solve_time}   ")

            self.pddlgym_env.set_state(state_pre.copy())#重置pddlgym环境的state使之同步
            self.pddlgym_env.f3_state=f3_pre#重置pddlgym环境的f3对应的dfa状态使之同步
            if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING or result.status == PlanGenerationResultStatus.SOLVED_OPTIMALLY:
                pv_res = pv.validate(self.problem, result.plan)
                # 记录规划解长度(包括trans)
                plan_len = len(result.plan.actions)
                cost = pv_res.metric_evaluations[metric]
                obs = state_pre.copy()
                reward = 0
                done = False
                for i in range(plan_len):
                    if i % 2 == 0:
                        continue
                    '''#测试时用于记录具体的动作
                    with open("actions_file.txt", "a") as file:
                        file.write(f"{result.plan.actions[i]}   ")'''
                    action_split = str(result.plan.actions[i]).split('(')
                    action_name = action_split[0]
                    action_split = action_split[1].split(',')
                    action_split[-1] = action_split[-1][:-1]
                    variables = []

                    for action_index in range(len(action_split)):
                        if action_index != 0:
                            action_split[action_index] = action_split[action_index][1:]
                        variables.append(self.pddlgym_env.env.env.problems[0].objects[
                                             self.pddlgym_env.object_index.index(action_split[action_index])])
                    literal = Literal(predicate=self.action_predicates[action_name], variables=variables)
                    act = self.pddlgym_env.actions.index(literal)
                    # 每执行一次动作还需要判断dfa状态到达错误状态，若到达错误状态立即停止
                    #调用pddlgym的step函数
                    obs, reward, done, debug_info = self.pddlgym_env.step(act)
                    if done == True:
                        break
                state_after = obs.copy()
                valid_action, temp = self.compute_valid_action_array(state_after.copy(), mask_pre.copy())
                if temp == True:
                    done = True
                    reward = -1000
                if reward == 0:
                    if cost == 0:
                        reward = 1#防止出现1/0
                    else:
                        reward = 1 / cost
                '''存储结果
                            键为当前状态+被选择的子任务索引+被选择的规划解索引的字符串
                            值为下一状态+奖励+回合结束标志+更新后的动作掩码+执行动作之后f3对应的DFA处于哪个状态+
                            规划解cost+规划解长度(不算trans)
                            '''
                self.reward_table[state_str + action1_str + str(plan_i)] = [state_after.copy(), reward, done,
                                                                            valid_action.copy(),
                                                                            self.pddlgym_env.f3_state, cost,plan_len//2]




            else:#无解或超时
                self.reward_table[state_str + action1_str + str(plan_i)] = [state_pre.copy(), -1000, True,
                                                                            mask_pre.copy(),
                                                                            self.pddlgym_env.f3_state, 0,0]

        # 更新当前状态的state以及valid_action_array
        self.state = self.reward_table[combined_str][0].copy()
        self.valid_action = self.reward_table[combined_str][3].copy()
        reward = self.reward_table[combined_str][1]
        done = self.reward_table[combined_str][2]
        self.current_episode_cost += self.reward_table[combined_str][5]
        self.current_episode_len += self.reward_table[combined_str][6]
        # debug_info默认返回空字典
        debug_info = dict()
        #同步pddlgym中的state和f3_state
        self.pddlgym_env.set_state(self.state.copy())
        self.pddlgym_env.f3_state = self.reward_table[combined_str][4]
        '''if reward>100:#可用于记录该episode是否成功
            with open("actions_file.txt", "a") as file:
                file.write(f" succ  ")'''
        return self.state, reward, done, debug_info


    #reset函数
    def reset(self,**kwargs):
        self.reset_flag =True
        state= self.pddlgym_env.reset()
        #更新当前状态
        self.state=state
        #更新当前状态可选子目标
        self.valid_action=[True] * self.action_space.n
        self.valid_action,actionmask_temp=self.compute_valid_action_array(self.state,self.valid_action)
        self.current_episode_cost = 0
        self.current_episode_len = 0
        return state

    #计算各个状态的可选子目标
    def compute_valid_action_array(self,state,valid_action_array):
        done=False
        first_dig = str(state[self.pddlgym_env.goal_index[0]])
        second_dig = str(state[self.pddlgym_env.goal_index[1]])
        third_dig = str(state[self.pddlgym_env.goal_index[2]])
        match_str = first_dig + second_dig + third_dig
        if match_str in self.cases.keys():
            dfa_state = self.cases[match_str]#当前三个连续时序对应的自动机所处状态
        else:
            return valid_action_array,done#自动机到达错误状态q2
        for i,mask in enumerate(valid_action_array):
            subgoal_index=i//self.plan_sum#被选择的子任务索引
            #新完成的子任务
            if (mask==True) and ((subgoal_index<self.pddl_goal_sum and state[self.pddlgym_env.goal_index[subgoal_index]]==1)
                                 or (subgoal_index>=self.pddl_goal_sum and dfa_state==subgoal_index-self.pddl_goal_sum+3)):
                valid_action_array[i]=False
            #三个连续时序的自动机对应的时序目标（只有当前状态的直接后继不含当前状态可被选择）
            if (self.pddl_goal_sum<=subgoal_index) and (subgoal_index<self.pddl_goal_sum+3):
                if self.viable_state[subgoal_index-self.pddl_goal_sum] in self.succs[dfa_state] and (subgoal_index-self.pddl_goal_sum+3)!=dfa_state:
                    valid_action_array[i]=True
                else:
                    valid_action_array[i]=False
            # 3F的自动机对应的时序目标（只有当前状态的直接后继不含当前状态可被选择）
            if subgoal_index>=self.pddl_goal_sum+3:
                if self.viable_state[subgoal_index-self.pddl_goal_sum] in self.succs[self.pddlgym_env.f3_state+5] and (subgoal_index-self.pddl_goal_sum-1)!=self.pddlgym_env.f3_state:
                    valid_action_array[i]=True
                else:
                    valid_action_array[i]=False
        return valid_action_array,done

    #计算直接后继
    def succ(self,edges):
        succs = dict()
        for u, v in edges:
            if u not in succs:
                succs[u] = []
            succs[u].append(v)
        return succs

    def recode_action_predicate(self):
        #一些谓词
        move_car_in_road = Predicate(name="move_car_in_road", arity=4)
        move_car_out_road = Predicate(name="move_car_out_road", arity=4)
        car_arrived = Predicate(name="car_arrived", arity=2)
        car_start = Predicate(name="car_start", arity=3)
        build_diagonal_oneway = Predicate(name="build_diagonal_oneway", arity=3)
        build_straight_oneway = Predicate(name="build_straight_oneway", arity=3)
        destroy_road = Predicate(name="destroy_road", arity=3)
        predicates=dict()
        predicates["move_car_in_road"]=move_car_in_road
        predicates["move_car_out_road"] =move_car_out_road
        predicates["car_arrived"] = car_arrived
        predicates["car_start"] = car_start
        predicates["build_diagonal_oneway"] = build_diagonal_oneway
        predicates["build_straight_oneway"] = build_straight_oneway
        predicates["destroy_road"] = destroy_road
        return predicates
























