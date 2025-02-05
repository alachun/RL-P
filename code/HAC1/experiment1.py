import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

def list_files(directory):
    file_paths = []  # 用于存储文件路径的列表
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths
dfa = 2
'''tasks = ['child-snack_pfile05', 'child-snack_pfile05-2', 'child-snack_pfile06-2', 'child-snack_pfile07-2',
'child-snack_pfile08', 'child-snack_pfile08-2', 'child-snack_pfile09', 'child-snack_pfile09-2', 'child-snack_pfile10',
'child-snack_pfile10-2']'''
tasks = ['child-snack_pfile05-2']
training_time = [[10, 10, 15, 20, 32, 32, 42, 35, 56, 55], [15, 13, 25, 26, 32, 46, 45, 51, 65, 75]]
def convert_time_to_minutes(time_str):
    h, m, s = map(int, time_str.replace('h', ' ').replace('min', ' ').replace('s', '').split())
    return h * 60 + m + s / 60

for task in tasks:
    rlp_log = 'log/PDDLEnvChild-snack/RL-P/' + task + '/dfa' + str(dfa)
    hac_log = 'log/PDDLEnvChild-snack/HAC/' + task + '/dfa' + str(dfa)
    a2c_log = 'log/PDDLEnvChild-snack/A2C/' + task + '/dfa' + str(dfa)
    rlp_nomask_log = 'log/PDDLEnvChild-snack/RL-P(no mask)/' + task + '/dfa' + str(dfa)
    file_paths1 = list_files(rlp_log)
    file_paths = list_files(hac_log)
    file_paths2 = list_files(a2c_log)
    file_paths3 = list_files(rlp_nomask_log)
    # 三个文件的路径

    # 初始化列表以存储每个文件的插值数据
    interpolated_rewards = []
    interpolated_lengths = []
    min_mean_length = []
    # 对每个文件的数据进行插值
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df['time_minutes'] = df['time'].apply(convert_time_to_minutes)

        # 筛选出mean_reward等于100的行
        filtered_df = df[df['mean_reward'] == 100]

        # 创建新的时间范围
        new_time_range = np.linspace(df['time_minutes'].min(), df['time_minutes'].max(), 1000)

        # 插值 mean_reward 和 mean_length
        interp_reward = np.interp(new_time_range, df['time_minutes'], df['mean_reward'])

        interpolated_rewards.append(interp_reward)

    # 将列表转换为 numpy 数组
    interpolated_rewards = np.array(interpolated_rewards)

    # 计算 mean_reward 和 mean_length 的平均值和标准差
    avg_reward = np.nanmean(interpolated_rewards, axis=0)
    std_reward = np.nanstd(interpolated_rewards, axis=0)

    interpolated_rewards1 = []
    interpolated_lengths1 = []
    for file_path in file_paths1:
        df = pd.read_csv(file_path)
        df['time_minutes'] = df['time'].apply(convert_time_to_minutes)

        # 筛选出mean_reward等于100的行
        filtered_df = df[df['mean_reward'] == 100]

        # 创建新的时间范围
        # new_time_range1 = np.linspace(df['time_minutes'].min(), df['time_minutes'].max(), 1000)

        # 插值 mean_reward 和 mean_length
        interp_reward = np.interp(new_time_range, df['time_minutes'], df['mean_reward'])

        interpolated_rewards1.append(interp_reward)

    # 将列表转换为 numpy 数组
    interpolated_rewards1 = np.array(interpolated_rewards1)

    # 计算 mean_reward 和 mean_length 的平均值和标准差
    avg_reward1 = np.nanmean(interpolated_rewards1, axis=0)
    std_reward1 = np.nanstd(interpolated_rewards1, axis=0)

    interpolated_rewards2 = []
    interpolated_lengths2 = []
    for file_path in file_paths2:
        df = pd.read_csv(file_path)
        df['time_minutes'] = df['time'].apply(convert_time_to_minutes)

        # 筛选出mean_reward等于100的行
        filtered_df = df[df['mean_reward'] == 100]

        # 创建新的时间范围
        # new_time_range1 = np.linspace(df['time_minutes'].min(), df['time_minutes'].max(), 1000)

        # 插值 mean_reward 和 mean_length
        interp_reward = np.interp(new_time_range, df['time_minutes'], df['mean_reward'])

        interpolated_rewards2.append(interp_reward)

    # 将列表转换为 numpy 数组
    interpolated_rewards2 = np.array(interpolated_rewards2)

    # 计算 mean_reward 和 mean_length 的平均值和标准差
    avg_reward2 = np.nanmean(interpolated_rewards2, axis=0)
    std_reward2 = np.nanstd(interpolated_rewards2, axis=0)

    interpolated_rewards3 = []
    interpolated_lengths3 = []
    for file_path in file_paths3:
        df = pd.read_csv(file_path)
        df['time_minutes'] = df['time'].apply(convert_time_to_minutes)

        # 筛选出mean_reward等于100的行
        filtered_df = df[df['mean_reward'] == 100]

        # 创建新的时间范围
        # new_time_range1 = np.linspace(df['time_minutes'].min(), df['time_minutes'].max(), 1000)

        # 插值 mean_reward 和 mean_length
        interp_reward = np.interp(new_time_range, df['time_minutes'], df['mean_reward'])

        interpolated_rewards3.append(interp_reward)

    # 将列表转换为 numpy 数组
    interpolated_rewards3 = np.array(interpolated_rewards3)

    # 计算 mean_reward 和 mean_length 的平均值和标准差
    avg_reward3 = np.nanmean(interpolated_rewards3, axis=0)
    std_reward3 = np.nanstd(interpolated_rewards3, axis=0)


    # 图 1：时间 vs 平均插值后的 mean_reward
    plt.figure(figsize=(10, 6))
    plt.tick_params(axis='x', labelsize=24,)
    plt.tick_params(axis='y', labelsize=24,)
    plt.plot(new_time_range, avg_reward1, label='RL-P')
    plt.fill_between(new_time_range, avg_reward1 - std_reward1, avg_reward1 + std_reward1, alpha=0.3)
    plt.plot(new_time_range, avg_reward3, label='RL-P(no mask)')
    plt.fill_between(new_time_range, avg_reward3 - std_reward3, avg_reward3 + std_reward3, alpha=0.3)
    plt.plot(new_time_range, avg_reward, label='HAC')
    plt.fill_between(new_time_range, avg_reward - std_reward, avg_reward + std_reward, alpha=0.3)
    plt.plot(new_time_range, avg_reward2, label='A2C')
    plt.fill_between(new_time_range, avg_reward2 - std_reward2, avg_reward2 + std_reward2, alpha=0.3)
    plt.xlabel('Time (minutes)', fontsize=26, fontweight='bold')
    # plt.ylabel('Mean reward', fontsize=26, fontweight='bold')
    plt.title(task.split('_')[1], fontsize=28, fontweight='bold', color='blue')
    # plt.legend(fontsize=24, loc='upper left')
    plt.xticks([1, 3, 6, 9, 12, 15])
    plt.ylim(0, None)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fig/' + task.split('_')[1] + "_dfa" + str(dfa))


