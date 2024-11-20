"""
Created on  Mar 1 2021
@author: wangmeng
"""
import os
import torch
import numpy as np
import pandas as pd

def Evaluate(env, agents, args, mode, agent_num):
    default_action = np.zeros(agent_num)
        
    agents_total_reward = [0 for _ in range(agent_num)]
    env.reset()
    obs_n, share_obs, reward_n, done_n, info_n, act_n, activate_agent_i, activate_to_act = env.step(default_action)

    while 1:
        action_n = np.array([-1 for i in range(agent_num)])
        for i, agent_i in enumerate(activate_agent_i):
            if activate_to_act[i]:
                with torch.no_grad():
                    if args.ps:
                        action, log_prob = agents[0].select_best_action(obs_n[agent_i])
                        action_n[agent_i] = action[0]
                    else:
                        action, log_prob = agents[agent_i].select_best_action(obs_n[agent_i])
                        action_n[agent_i] = action[0]
                    
        # last_info = info
        obs_n, share_obs, reward_n, done_n, info_n, act_n, activate_agent_i, activate_to_act = env.step(action_n)
        for i, agent_i in enumerate(activate_agent_i):
            if act_n[agent_i] == -1:
                pass
            else:
                agents_total_reward[agent_i] += reward_n[agent_i][0]
        ########### 若无启动的智能体，说明环境运行结束 ###########
        if env.agents_active == []: 
            break
        
    total_reward = 0
    for i in range(agent_num):
        ep_running_reward = agents_total_reward[i]
        total_reward += ep_running_reward
    
    if args.ps:
        dir = 'output/{}_{}_{}_PS'.format(args.sce_name, args.filename, mode)
    else:
        dir = 'output/{}_{}_{}'.format(args.sce_name, args.filename, mode)
    if not os.path.exists(dir):
        os.mkdir(dir)
    # CS
    columns = ['time']
    for i in range(env.num_cs):
        cs = 'CS{}'.format(i)
        columns.append(cs+'_waiting_num')
        columns.append(cs+'_charging_num')
        columns.append(cs+'_waiting_time')
    time_seq = env.time_memory
    df_cs = pd.DataFrame(columns=columns)
    for i in range(len(time_seq)):
        df_cs.loc[i, 'time'] = round(time_seq[i], 2)
        for j in range(env.num_cs):
            cs = 'CS{}'.format(j)
            df_cs.loc[i, cs+'_waiting_num'] = env.cs_waiting_cars_num_memory[i][j]
            df_cs.loc[i, cs+'_charging_num'] = env.cs_charging_cars_num_memory[i][j]
            df_cs.loc[i, cs+'_waiting_time'] = env.cs_waiting_time_memory[i][j]
    df_cs.to_csv(dir+'/CS.csv', index=False)
    # EV
    df_ev_g = pd.DataFrame(columns=['EV', 'Waiting_time', 'Charging_time', 'Total'])
    columns = ['time', 'distance', 'position', 'state', 'act_mode', 'SOC', 'action']
    if not os.path.exists(dir + '/EV'):
        os.mkdir(dir + '/EV')
    for j, ev in enumerate(env.agents):
        df_ev = pd.DataFrame(columns=columns)
        time_seq = ev.time_memory
        for i in range(len(time_seq)):
            df_ev.loc[i, 'time'] = round(time_seq[i], 2)
            df_ev.loc[i, 'distance'] = ev.trip_memory[i]
            df_ev.loc[i, 'position'] = ev.pos_memory[i]
            df_ev.loc[i, 'state'] = ev.state_memory[i]
            df_ev.loc[i, 'act_mode'] = int(ev.activity_memory[i])
            df_ev.loc[i, 'SOC'] = ev.SOC_memory[i]
            df_ev.loc[i, 'action'] = ev.action_choose_memory[i]
        df_ev.to_csv(dir + '/EV/EV{}.csv'.format(ev.id), index=False)

        df_ev_g.loc[j, 'EV'] = ev.id
        df_ev_g.loc[j, 'Waiting_time'] = ev.total_waiting
        df_ev_g.loc[j, 'Charging_time'] = ev.total_charging
        df_ev_g.loc[j, 'Total'] = ev.total_used_time
        df_ev_g.loc[j, 'Reward'] = ev.total_reward
    df_ev_g.to_csv(dir+'/EV_g.csv', index=False)
        
    print(total_reward)
    # env.render()
    env.close()

