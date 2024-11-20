'''
Author: CQZ
Date: 2024-04-11 22:51:15
Company: SEU
'''
'''
方案2：
    EV在CS选择充到某个目标电量，该值一定大于当前电力
    设置当目标电量大于1时，视为不充电
'''
import torch
import numpy as np
import time

def Train(envs, agents, writer, args, mode, agent_num):
    current_step = 0 # 总步数

    start_time = time.time()
    # agents_best_reward = [-10000 for _ in range(agent_num)] # 每个智能体的最佳奖励
    total_best_reward = -10000 # 最佳总奖励
    global_total_reward = 0 # 当前总奖励存档
    best_step = 0 # 最佳总奖励对应轮次
    log_interval = 10
    
    default_action = [np.zeros(agent_num) for _ in range(args.num_env)] # 默认动作
    # current_policy = []
    run_times = [0 for _ in range(args.num_env)] # 每个环境的运行次数

    for i_episode in range(1, args.num_update + 1):
        #^ 学习率递减
        if args.ps:
            lr = agents[0].lr_decay(i_episode)
        else:
            for agent in agents:
                lr = agent.lr_decay(i_episode)
        # expert_percent = 1 - current_step / args.demo_step
        writer.add_scalar("Global/lr", lr, i_episode-1)
        #* 环境初始化
        agents_total_reward = np.array([[0.0 for _ in range(agent_num)] for __ in range(args.num_env)])
        envs.reset()
        obs_n, share_obs, reward_n, done_n, info_n, act_n, activate_agent_i, activate_to_act = envs.step(default_action)
        buffer_times = np.array([[0.0 for _ in range(agent_num)] for __ in range(args.num_env)])
        ########### 采样循环 ###########
        while (buffer_times>=args.single_batch_size).sum() < agent_num * args.num_env:
            action_n = np.array([[-1 for _ in range(agent_num)] for __ in range(args.num_env)])
            #* 为已经激活的智能体选择动作，并记下当前的状态
            for e in range(args.num_env):
                for i, agent_i in enumerate(activate_agent_i[e]):
                    if activate_to_act[e][i]:
                        with torch.no_grad():
                            # Choose an action
                            if args.ps:
                                action, log_prob = agents[0].select_action(obs_n[e][agent_i])
                                if not args.ctde:
                                    share_obs_ = obs_n[e][agent_i].copy()
                                else:
                                    share_obs_ = share_obs[e].copy()
                                    one_hot = [0] * agent_num
                                    one_hot[agent_i] = 1
                                    share_obs_ =  np.concatenate([share_obs_, np.array(one_hot)])
                                # Push
                                agents[0].rolloutBuffer.push_last_state(
                                    state=obs_n[e][agent_i], 
                                    share_state=share_obs_, 
                                    action=action, 
                                    log_prob=log_prob,
                                    env_id=e, agent_id=agent_i
                                    )
                            else:
                                action, log_prob = agents[agent_i].select_action(obs_n[e][agent_i])
                                if not args.ctde:
                                    share_obs_ = obs_n[e][agent_i].copy()
                                else:
                                    share_obs_ = share_obs[e].copy()  
                                # Push
                                agents[agent_i].rolloutBuffer.push_last_state(
                                    state=obs_n[e][agent_i], 
                                    share_state=share_obs_, 
                                    action=action, 
                                    log_prob=log_prob,
                                    env_id=e
                                    )
                            action_n[e][agent_i] = action[0].copy()
            #* 环境运行，直到有智能体被激活
            # last_info = info
            obs_n, share_obs, reward_n, done_n, info_n, act_n, activate_agent_i, activate_to_act = envs.step(action_n)
            current_step += 1
            #* 将被激活的智能体当前状态作为上一次动作的结果保存
            for e in range(args.num_env):
                for i, agent_i in enumerate(activate_agent_i[e]):
                    agents_total_reward[e][agent_i] += reward_n[e][agent_i][0].copy()
                    if act_n[e][agent_i] != -1:
                        if args.ps:
                            if not args.ctde:
                                share_obs_ = obs_n[e][agent_i].copy()
                            else:
                                share_obs_ = share_obs[e].copy()
                                one_hot = [0] * agent_num
                                one_hot[agent_i] = 1
                                share_obs_ =  np.concatenate([share_obs_, np.array(one_hot)])
                            agents[0].rolloutBuffer.push(
                                reward=reward_n[e][agent_i],
                                next_state=obs_n[e][agent_i], 
                                next_share_state=share_obs_, 
                                done=done_n[e][agent_i],
                                env_id=e, agent_id=agent_i
                                )
                        else:
                            if not args.ctde:
                                share_obs_ = obs_n[e][agent_i].copy()
                            else:
                                share_obs_ = share_obs[e].copy()
                            agents[agent_i].rolloutBuffer.push(
                                reward=reward_n[e][agent_i],
                                next_state=obs_n[e][agent_i], 
                                next_share_state=share_obs_, 
                                done=done_n[e][agent_i],
                                env_id=e
                                )
                        buffer_times[e][agent_i] += 1
            #* 若没有可启动的智能体，说明环境运行结束，重启
            is_finished = envs.is_finished()
            if is_finished != []:
                # current_policy = env.get_policy().copy()
                obs_n_, share_obs_, reward_n_, done_n_, info_n_, act_n_, activate_agent_i_, activate_to_act_ = envs.reset_process(is_finished)
                for i, e in enumerate(is_finished):
                    obs_n[e] = obs_n_[i]
                    share_obs[e] = share_obs_[i]
                    reward_n[e] = reward_n_[i]
                    done_n[e] = done_n_[i]
                    info_n[e] = info_n_[i]
                    act_n[e] = act_n_[i]
                    activate_agent_i[e] = activate_agent_i_[i]
                    activate_to_act[e] = activate_to_act_[i]

                    # 计算该环境总奖励
                    total_reward = 0 
                    for i in range(agent_num):
                        total_reward += agents_total_reward[e][i]
                    # writer.add_scalar("Single_Env/reward_{}".format(e), total_reward, run_times[e])
                    writer.add_scalar("Single_Env/reward_{}".format(e), total_reward, i_episode)
                    # 统计总体奖励
                    if total_reward > total_best_reward:
                        total_best_reward = total_reward
                    # writer.add_scalar("Global/total_reward", total_reward, sum(run_times))
                    # writer.add_scalar("Global/total_best_reward", total_best_reward, sum(run_times))
                    writer.add_scalar("Global/total_reward", total_reward, i_episode)
                    writer.add_scalar("Global/total_best_reward", total_best_reward, i_episode)
                    best_step = i_episode

                    agents_total_reward[e] = np.array([0.0 for _ in range(agent_num)])
                    run_times[e] += 1
                    global_total_reward = total_reward

                # avg_length += 1
        # print(i_episode, i_episode)
        if i_episode % log_interval == 0:
            print('Episode {} \t Total reward: {:.3f} \t Total best reward: {:.3f}'.format(i_episode, global_total_reward, total_best_reward))
        
        # 更新网络
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        
        if args.ps:
            actor_loss, critic_loss, entropy_loss = agents[0].train()
            total_actor_loss += actor_loss
            total_critic_loss += critic_loss
            total_entropy_loss += entropy_loss
            writer.add_scalar("Loss/agent_ps_actor_loss", actor_loss, i_episode)
            writer.add_scalar("Loss/agent_ps_critic_loss", critic_loss, i_episode)
            writer.add_scalar("Loss/agent_ps_entropy_loss", entropy_loss, i_episode)
            if i_episode % args.save_freq == 0:
                agents[0].save("save/{}_{}_PS/agent_ps_{}".format(args.sce_name, args.filename, mode))
        else:
            for i, agent in enumerate(agents):
                actor_loss, critic_loss, entropy_loss = agent.train()
                total_actor_loss += actor_loss
                total_critic_loss += critic_loss
                total_entropy_loss += entropy_loss
                writer.add_scalar("Loss/agent_{}_actor_loss".format(i), actor_loss, i_episode)
                writer.add_scalar("Loss/agent_{}_critic_loss".format(i), critic_loss, i_episode)
                writer.add_scalar("Loss/agent_{}_entropy_loss".format(i), entropy_loss, i_episode)
                if i_episode % args.save_freq == 0:
                    agent.save("save/{}_{}/agent_{}_{}".format(args.sce_name, args.filename, i, mode))
                
        writer.add_scalar("Global_loss/actor_loss", total_actor_loss, i_episode)
        writer.add_scalar("Global_loss/critic_loss", total_critic_loss, i_episode)
        writer.add_scalar("Global_loss/entropy_loss", total_entropy_loss, i_episode)
        writer.add_scalar("Global/step_per_second", current_step / (time.time() - start_time), i_episode)
    envs.close()
    print("Running time: {}s".format(time.time() - start_time))
    return total_best_reward, best_step

