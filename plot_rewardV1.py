# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import csv
matplotlib.use('TkAgg')

with open(r'D:\MAAC_baseline\models\simple_spread\MAAC\run89\all_episode_reward.pickle', 'rb') as x:
    out = pickle.load(x)  # DDQN
# with open(r'D:\DQL_result\random_intru12+3_ATT_50k_randomOD_change7X7Centre_HaveReachOD_50_25_newIntruderColliCheck_reachableOD\GFG_intermediate.csv', 'r') as x:
#     sample_data_setZero = list(csv.reader(x, delimiter=","))  # DDQN  #"2" is set zero

sample_data = np.array(out)
# sample_data_setZero = np.array(sample_data_setZero[0])

ep_reward_arr = np.array([float(value) for value in sample_data])
# ep_reward_arr_setZero = np.array([float(value) for value in sample_data_setZero])

m = 100
n = len(ep_reward_arr) // m
# n_setZero = len(ep_reward_arr_setZero) // m
avg_reward_arr = np.mean(np.reshape(ep_reward_arr[: m * n], [n, m]), 1)
# avg_reward_arr_setZero = np.mean(np.reshape(ep_reward_arr_setZero[: m * n_setZero], [n_setZero, m]), 1)

# plot1 = plt.plot(avg_reward_arr, label='batch256')
# plot2 = plt.plot(avg_reward_arr_setZero, label='batch64')

# plot1 = plt.plot(avg_reward_arr, label='no gradient clip')
# plot2 = plt.plot(avg_reward_arr_setZero, label='gradient clip')

plot1 = plt.plot(avg_reward_arr, label='average_reward')
# plot2 = plt.plot(avg_reward_arr_setZero, label='3407')

plt.grid(linestyle='-.')
plt.xlabel('Episode')
plt.ylabel('Ave reward')
plt.legend()
plt.show()