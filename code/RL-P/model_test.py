from stable_baselines3 import A2C
from child_snack_gym2 import ChildsnackEnv
from datetime import datetime
from myA2C import *
from citycar_gym2 import CitycarEnv
'''本文件用于模型测试，根据问题实例修改'''
all_start_time = datetime.now()
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
                  plan_sum=3)
model=MyA2C.load("c15_2_4.zip",env=env)
#用于存储每个回合是否成功
episode_success=[]
count=0
for i in range(100):
    obs=env.reset()
    done=False
    episode_len=0
    with open("actions_file.txt", "a") as file:
        file.write(f"\nepisode{i+1}")
    while(True):
        action,_state=model.predict(obs,deterministic=False)
        obs,reward,done,info=env.step(action)
        episode_len+=1
        if(episode_len>=250):
            reward=-1000
            done=True
        if(done):
            env.all_episode_cost.append(env.current_episode_cost)
            #env.all_episode_len.append(env.current_episode_len)
            with open("actions_file.txt", "a") as file:
                if reward>500:
                    file.write(f"succ\n")
                    count+=1
                    episode_success.append("succ")
                else:
                    file.write(f"fail\n")
                    episode_success.append("fail")
            break


all_end_time = datetime.now()
with open("actions_file.txt", "a") as file:
    file.write(f"all time:{all_end_time - all_start_time}\n")
    file.write(f"planner time:{env.planner_time}\n")
    file.write(f"success rate:{count/100}\n")
    file.write(f"episode is success?:{episode_success}\n")
    file.write(f"episode cost?:{env.all_episode_cost}\n")
    #file.write(f"episode len?:{env.all_episode_len}\n")
#all_len=0
all_cost=0
count=0
for i in range(100):
    if episode_success[i]=="succ":
        count+=1
        #all_len+=env.all_episode_len[i]
        all_cost += env.all_episode_cost[i]
with open("actions_file.txt", "a") as file:
    file.write(f"average cost:{all_cost/count}\n")
    #file.write(f"average len:{all_len/count}\n")





