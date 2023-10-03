# -*- coding: utf-8 -*-
"""
@Time    : 7/26/2023 9:51 AM
@Author  : Thu Ra
@FileName: 
@Description: 
@Package dependency:
"""
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
pre_fix = r'F:\githubClone\MAAC_baseline\models\simple_spread\MAAC\run188'
file_path = pre_fix + r'\all_action.pickle'
with open(file_path, 'rb') as handle:
    combine_reward = pickle.load(handle)

seperated_reward = [[] for _ in combine_reward[0][0][0]]
# First_term = []
# Second_term = []
# Third_term = []
sum_reward_last_agent = []
for per_ep_r in combine_reward:
    for per_step_r in per_ep_r:
        for per_agent_r in per_step_r:
            if per_agent_r is None:
                # print("None is found in a step of episode, skip it")
                # we are only saving the step reward, other term like goal reaching or crashing is not recorded here
                continue
            for individual_list_idx in range(len(seperated_reward)):
                seperated_reward[individual_list_idx].append(per_agent_r[individual_list_idx])
            sum_reward_last_agent.append(sum(per_agent_r))
            # seperated_reward[0].append()
            # seperated_reward[0].append()
            # seperated_reward[0].append()
            # seperated_reward[0].append()
            # CT_term.append(per_agent_r[0])
            # g_term.append(per_agent_r[1])
            # alive_term.append(per_agent_r[2])
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
fig, ax = plt.subplots(1, 1)

n_bins = 10
plot1 = plt.hist(seperated_reward[0], bins=n_bins)
plot2 = plt.hist(seperated_reward[1], bins=n_bins)
# plot1 = plt.boxplot(seperated_reward[-1])
# plot1 = plt.plot(seperated_reward[3], linestyle='-', label='goal_term')
# plot1 = plt.plot(sum_reward_last_agent, linestyle='-', label='overall')
# plot2 = plt.plot(g_term, linestyle='--', label='goal-term')
# plot3 = plt.plot(seperated_reward[0], linestyle='-.', label='alive-term')

plt.grid()
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()
print('done')