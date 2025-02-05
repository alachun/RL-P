from citycar_gym2 import CitycarEnv
from stable_baselines3 import A2C
from child_snack_gym2 import ChildsnackEnv
from myA2C import *
from datetime import datetime
all_start_time = datetime.now()
'''调用对应的环境进行训练并保存训练好的模型'''
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
model =MyA2C('MlpPolicy',env=env,verbose=1)
#,tensorboard_log="./grid-visit-all_tensorboard"
model.learn(total_timesteps=60000,log_interval=100)
all_end_time = datetime.now()
print(all_end_time - all_start_time)
model.save("c15_2_4")
print(env.planner_time)








