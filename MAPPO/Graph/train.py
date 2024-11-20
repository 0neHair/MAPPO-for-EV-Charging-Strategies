'''
方案2：
    EV在CS选择充到某个目标电量，该值一定大于当前电力
    设置当目标电量大于1时，视为不充电
    
    加入路径问题，采用图卷积网络处理路径信息
'''
import torch
import argparse
import gymnasium as gym
# import gym

import numpy as np
from torch.utils.tensorboard import SummaryWriter # type: ignore
import time
import os
from env.EV_Sce_Env import EV_Sce_Env
from env_wrappers import SubprocVecEnv, DummyVecEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sce_name", type=str, default="test_2")
    parser.add_argument("--filename", type=str, default="T1")
    parser.add_argument("--ctde", type=bool, default=True)
    parser.add_argument("--expert", type=bool, default=False)
    parser.add_argument("--randomize", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_env", type=int, default=4) # 环境数
    parser.add_argument("--num_update", type=int, default=1000) # 最大更新轮次
    parser.add_argument("--save_freq", type=int, default=50) # 保存频率

    parser.add_argument("--ps", type=bool, default=False) # parameter sharing

    parser.add_argument("--policy_arch", type=list, default=[32, 32, 32])
    parser.add_argument("--value_arch", type=list, default=[32, 32, 32])

    parser.add_argument("--lr", type=float, default=4e-4) # 学习率
    parser.add_argument("--gamma", type=float, default=0.95) # 折减因子
    parser.add_argument("--gae_lambda", type=float, default=0.95) # ？？？
    parser.add_argument("--k_epoch", type=int, default=15) # 策略更新次数
    parser.add_argument("--eps_clip", type=float, default=0.1) # 裁剪参数
    parser.add_argument("--max_grad_clip", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01) # 熵正则
    parser.add_argument("--single_batch_size", type=int, default=60) # 单个buffer数据量
    parser.add_argument("--num_mini_batch", type=int, default=1) # 小batcch数量
    arguments = parser.parse_args()
    return arguments

def train_policy(envs, agents, writer, args, mode, agent_num):
    current_step = 0 # 总步数

    start_time = time.time()
    # agents_best_reward = [-10000 for _ in range(agent_num)] # 每个智能体的最佳奖励
    total_best_reward = -10000 # 最佳总奖励
    global_total_reward = 0 # 当前总奖励存档
    best_step = 0 # 最佳总奖励对应轮次
    log_interval = 10
    
    default_caction = np.zeros([args.num_env, agent_num]) # 默认动作
    default_raction = np.zeros([args.num_env, agent_num]) # 默认动作
    default_action = (default_caction, default_raction)
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
        obs_n, obs_feature_n, obs_mask_n, \
            share_obs, global_cs_feature, \
                done_n, creward_n, rreward_n, cact_n, ract_n, \
                    activate_agent_ci, activate_to_cact, \
                        activate_agent_ri, activate_to_ract \
                            = envs.step(default_action)
        active_to_cpush = [[False for _ in range(agent_num)] for __ in range(args.num_env)]
        active_to_rpush = [[False for _ in range(agent_num)] for __ in range(args.num_env)]
        buffer_times = np.array([[0.0 for _ in range(agent_num)] for __ in range(args.num_env)])
        ########### 采样循环 ###########
        while (buffer_times>=args.single_batch_size).sum() < agent_num * args.num_env:
            caction_n = np.array([[-1 for _ in range(agent_num)] for __ in range(args.num_env)])
            raction_n = np.array([[-1 for _ in range(agent_num)] for __ in range(args.num_env)])
            #* 为已经激活的智能体选择动作，并记下当前的状态
            for e in range(args.num_env):
                # 决策充电
                for i, agent_i in enumerate(activate_agent_ci[e]):
                    if activate_to_cact[e][i]:
                        with torch.no_grad():
                            # Choose an action
                            if args.ps:
                                pass
                                # action, log_prob = agents[0].select_caction(obs_n[e][agent_i])
                                # if not args.ctde:
                                #     share_obs_ = obs_n[e][agent_i].copy()
                                # else:
                                #     share_obs_ = share_obs[e].copy()
                                #     # one_hot = [0] * agent_num
                                #     # one_hot[agent_i] = 1
                                #     # share_obs_ =  np.concatenate([share_obs_, np.array(one_hot)])
                                # # Push
                                # agents[0].rolloutBuffer.push_last_state(
                                #     state=obs_n[e][agent_i], 
                                #     share_state=share_obs_, 
                                #     action=action, 
                                #     log_prob=log_prob,
                                #     env_id=e, agent_id=agent_i
                                #     )
                            else:
                                caction, clog_prob = agents[agent_i].select_caction(obs_n[e][agent_i])
                                if not args.ctde:
                                    share_obs_ = obs_n[e][agent_i].copy()
                                else:
                                    share_obs_ = share_obs[e].copy()  
                                # Push
                                agents[agent_i].rolloutBuffer.cpush_last_state(
                                    state=obs_n[e][agent_i], 
                                    share_state=share_obs_, 
                                    action=caction, 
                                    log_prob=clog_prob,
                                    env_id=e
                                    )
                            active_to_cpush[e][agent_i] = True
                            caction_n[e][agent_i] = caction[0].copy()
                # 决策路径
                for i, agent_i in enumerate(activate_agent_ri[e]):
                    if activate_to_ract[e][i]:
                        with torch.no_grad():
                            # Choose an action
                            if args.ps:
                                pass
                                # action, log_prob = agents[0].select_raction(obs_n[e][agent_i])
                                # if not args.ctde:
                                #     share_obs_ = obs_n[e][agent_i].copy()
                                # else:
                                #     share_obs_ = share_obs[e].copy()
                                #     # one_hot = [0] * agent_num
                                #     # one_hot[agent_i] = 1
                                #     # share_obs_ =  np.concatenate([share_obs_, np.array(one_hot)])
                                # # Push
                                # agents[0].rolloutBuffer.push_last_state(
                                #     state=obs_n[e][agent_i], 
                                #     share_state=share_obs_, 
                                #     action=action, 
                                #     log_prob=log_prob,
                                #     env_id=e, agent_id=agent_i
                                #     )
                            else:
                                raction, rlog_prob = agents[agent_i].select_raction(
                                    obs_feature_n[e][agent_i], obs_mask_n[e][agent_i]
                                )
                                if not args.ctde:
                                    global_cs_feature_ = obs_feature_n[e][agent_i].copy()
                                else:
                                    global_cs_feature_ = global_cs_feature[e].copy()
                                # Push
                                agents[agent_i].rolloutBuffer.rpush_last_state(
                                    obs_feature=obs_feature_n[e][agent_i], 
                                    global_cs_feature=global_cs_feature_,
                                    action=raction, 
                                    action_mask=obs_mask_n[e][agent_i], 
                                    log_prob=rlog_prob,
                                    env_id=e
                                    )
                            active_to_rpush[e][agent_i] = True
                            raction_n[e][agent_i] = raction[0].copy()
            #* 环境运行，直到有智能体被激活
            # last_info = info
            obs_n, obs_feature_n, obs_mask_n, \
                share_obs, global_cs_feature, \
                    done_n, creward_n, rreward_n, cact_n, ract_n, \
                        activate_agent_ci, activate_to_cact, \
                            activate_agent_ri, activate_to_ract \
                                = envs.step((caction_n, raction_n)) # (caction_n, raction_n)
            current_step += 1
            #* 将被激活的智能体当前状态作为上一次动作的结果保存
            for e in range(args.num_env):
                for i, agent_i in enumerate(activate_agent_ci[e]):
                    # print("CR:", creward_n)
                    if active_to_cpush[e][agent_i]:
                        if args.ps:
                            pass
                            # if not args.ctde:
                            #     share_obs_ = obs_n[e][agent_i].copy()
                            # else:
                            #     share_obs_ = share_obs[e].copy()
                            #     # one_hot = [0] * agent_num
                            #     # one_hot[agent_i] = 1
                            #     # share_obs_ =  np.concatenate([share_obs_, np.array(one_hot)])
                            # agents[0].rolloutBuffer.push(
                            #     reward=reward_n[e][agent_i],
                            #     next_state=obs_n[e][agent_i], 
                            #     next_share_state=share_obs_, 
                            #     done=done_n[e][agent_i],
                            #     env_id=e, agent_id=agent_i
                            #     )
                        else:
                            if not args.ctde:
                                share_obs_ = obs_n[e][agent_i].copy()
                            else:
                                share_obs_ = share_obs[e].copy()
                            agents[agent_i].rolloutBuffer.cpush(
                                reward=creward_n[e][agent_i],
                                next_state=obs_n[e][agent_i], 
                                next_share_state=share_obs_, 
                                done=done_n[e][agent_i],
                                env_id=e
                                )
                for i, agent_i in enumerate(activate_agent_ri[e]):
                    # print("RR:", rreward_n)
                    agents_total_reward[e][agent_i] += rreward_n[e][agent_i][0].copy()
                    if active_to_rpush[e][agent_i]:
                        if args.ps:
                            pass
                            # if not args.ctde:
                            #     share_obs_ = obs_n[e][agent_i].copy()
                            # else:
                            #     share_obs_ = share_obs[e].copy()
                            #     # one_hot = [0] * agent_num
                            #     # one_hot[agent_i] = 1
                            #     # share_obs_ =  np.concatenate([share_obs_, np.array(one_hot)])
                            # agents[0].rolloutBuffer.push(
                            #     reward=reward_n[e][agent_i],
                            #     next_state=obs_n[e][agent_i], 
                            #     next_share_state=share_obs_, 
                            #     done=done_n[e][agent_i],
                            #     env_id=e, agent_id=agent_i
                            #     )
                        else:
                            if not args.ctde:
                                global_cs_feature_ = obs_feature_n[e][agent_i].copy()
                            else:
                                global_cs_feature_ = global_cs_feature[e].copy()
                            agents[agent_i].rolloutBuffer.rpush(
                                reward=rreward_n[e][agent_i],
                                next_obs_feature=obs_feature_n[e][agent_i], 
                                next_global_cs_feature=global_cs_feature_,
                                done=done_n[e][agent_i],
                                env_id=e
                                )
                            buffer_times[e][agent_i] += 1
            #* 若没有可启动的智能体，说明环境运行结束，重启
            is_finished = envs.is_finished()
            if is_finished != []:
                # current_policy = env.get_policy().copy()
                # print(agents_total_reward)
                obs_n_, obs_feature_n_, obs_mask_n_, \
                    share_obs_, global_cs_feature_, \
                        done_n_, creward_n_, rreward_n_, cact_n_, ract_n_, \
                            activate_agent_ci_, activate_to_cact_, \
                                activate_agent_ri_, activate_to_ract_ \
                                    = envs.reset_process(is_finished)
                for i, e in enumerate(is_finished):
                    # 计算该环境总奖励
                    total_reward = 0 
                    for j in range(agent_num):
                        total_reward += agents_total_reward[e][j]
                        writer.add_scalar("Single_Env/reward_{}_agent_{}".format(e, j), agents_total_reward[e][j], i_episode)
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

                    agents_total_reward[e] = np.array([0.0 for _ in range(agent_num)]).copy()
                    run_times[e] += 1
                    global_total_reward = total_reward
                    
                    # 重启环境
                    obs_n[e] = obs_n_[i]
                    share_obs[e] = share_obs_[i]
                    creward_n[e] = creward_n_[i]
                    cact_n[e] = cact_n_[i]
                    activate_agent_ci[e] = activate_agent_ci_[i]
                    activate_to_cact[e] = activate_to_cact_[i]
                    obs_feature_n[e] = obs_feature_n_[i]
                    obs_mask_n[e] = obs_mask_n_[i]
                    global_cs_feature[e] = global_cs_feature_[i]
                    rreward_n[e] = rreward_n_[i]
                    ract_n[e] = ract_n_[i]
                    activate_agent_ri[e] = activate_agent_ri_[i]
                    activate_to_ract[e] = activate_to_ract_[i]
                    done_n[e] = done_n_[i]
                    active_to_cpush[e] = [False for _ in range(agent_num)]
                    active_to_rpush[e] = [False for _ in range(agent_num)]
                # avg_length += 1
        # print(i_episode, i_episode)
        if i_episode % log_interval == 0:
            print('Episode {} \t Total reward: {:.3f} \t Total best reward: {:.3f}'.format(i_episode, global_total_reward, total_best_reward))
        
        # 更新网络
        total_actor_closs = 0
        total_critic_closs = 0
        total_entropy_closs = 0
        total_actor_rloss = 0
        total_critic_rloss = 0
        total_entropy_rloss = 0
        
        if args.ps:
            pass
            # actor_loss, critic_loss, entropy_loss = agents[0].train()
            # total_actor_loss += actor_loss
            # total_critic_loss += critic_loss
            # total_entropy_loss += entropy_loss
            # writer.add_scalar("Loss/agent_ps_actor_loss", actor_loss, i_episode)
            # writer.add_scalar("Loss/agent_ps_critic_loss", critic_loss, i_episode)
            # writer.add_scalar("Loss/agent_ps_entropy_loss", entropy_loss, i_episode)
            # if i_episode % args.save_freq == 0:
            #     agents[0].save("save/{}_{}/agent_ps_{}".format(args.sce_name, args.filename, mode))
        else:
            for i, agent in enumerate(agents):
                actor_closs, critic_closs, entropy_closs, \
                    actor_rloss, critic_rloss, entropy_rloss= agent.train()
                total_actor_closs += actor_closs
                total_critic_closs += critic_closs
                total_entropy_closs += entropy_closs
                total_actor_rloss += actor_rloss
                total_critic_rloss += critic_rloss
                total_entropy_rloss += entropy_rloss
                writer.add_scalar("Loss/agent_{}_actor_closs".format(i), actor_closs, i_episode)
                writer.add_scalar("Loss/agent_{}_critic_closs".format(i), critic_closs, i_episode)
                writer.add_scalar("Loss/agent_{}_entropy_closs".format(i), entropy_closs, i_episode)
                writer.add_scalar("Loss/agent_{}_actor_rloss".format(i), actor_rloss, i_episode)
                writer.add_scalar("Loss/agent_{}_critic_rloss".format(i), critic_rloss, i_episode)
                writer.add_scalar("Loss/agent_{}_entropy_rloss".format(i), entropy_rloss, i_episode)
                if i_episode % args.save_freq == 0:
                    agent.save("save/{}_{}/agent_{}_{}".format(args.sce_name, args.filename, i, mode))
                
        writer.add_scalar("Global_loss/actor_closs", total_actor_closs, i_episode)
        writer.add_scalar("Global_loss/critic_closs", total_critic_closs, i_episode)
        writer.add_scalar("Global_loss/entropy_closs", total_entropy_closs, i_episode)
        writer.add_scalar("Global_loss/actor_rloss", total_actor_rloss, i_episode)
        writer.add_scalar("Global_loss/critic_rloss", total_critic_rloss, i_episode)
        writer.add_scalar("Global_loss/entropy_rloss", total_entropy_rloss, i_episode)
        writer.add_scalar("Global/step_per_second", current_step / (time.time() - start_time), i_episode)
    envs.close()
    print("Running time: {}s".format(time.time() - start_time))
    return total_best_reward, best_step

def make_train_env(args, agent_num):
    def get_env_fn():
        def init_env():
            env = EV_Sce_Env(args.sce_name, args.seed, args.ps)
            return env
        return init_env
    if args.num_env == 1:
        return DummyVecEnv([get_env_fn()], agent_num)
    else:
        return SubprocVecEnv([get_env_fn() for _ in range(args.num_env)], agent_num)

def main():
    ############## Hyperparameters ##############
    args = parse_args()
    if args.randomize:
        seed = args.seed
        print("Random Seed: {}".format(seed))
        torch.manual_seed(seed)
        np.random.seed(seed)

    env = EV_Sce_Env(args.sce_name, args.seed, args.ps)
    caction_list = env.caction_list # 可选充电动作列表
    raction_list = env.raction_list # 可选路径动作列表
    
    agent_num = env.agent_num # 智能体数量
    num_cs = env.num_cs # 充电站数量
    
    args.train_times = int(args.single_batch_size * 2 // num_cs) # 单个环境运行次数
    args.batch_size = int(args.single_batch_size * args.num_env) # 总batch大小
    # args.batch_size = int(args.single_batch_size * args.num_env * agent_num) # 总batch大小

    args.mini_batch_size = int(args.batch_size // args.num_mini_batch)
    # args.train_steps = int(args.train_times * agent_num * (num_cs+1))
    # args.max_steps = int(args.train_steps * args.num_update)
    
    graph = env.map_adj
    caction_dim = env.caction_dim # type: ignore
    raction_dim = env.raction_dim # type: ignore
    raction_mask_dim = env.raction_dim # type: ignore
    raction_mask_shape = (raction_mask_dim, )
    state_dim = env.state_dim # type: ignore
    state_shape = (state_dim, )
    edge_index = env.edge_index # 原版地图
    obs_features_dim = env.obs_features_dim # 观测地图特征维度
    obs_features_shape = (graph.shape[0], obs_features_dim) # 观测地图特征尺寸
    mode = ''
    
    if args.ctde:
        print('Mode: MAPPO   Agent_num: {}   CS_num: {}   Env_num: {}'.format(agent_num, num_cs, args.num_env))
        mode = 'MAPPO'
        share_dim = env.share_dim
        share_shape = (share_dim, )
        global_features_dim = env.global_features_dim # 全局地图特征维度
        global_features_shape = (graph.shape[0], global_features_dim) # 全局地图特征尺寸
    else:
        print('Mode: IPPO   Agent_num: {}   CS_num: {}   Env_num: {}'.format(agent_num, num_cs, args.num_env))
        mode = 'IPPO'
        share_dim = env.state_dim
        share_shape = (state_dim, )
        global_features_dim = env.obs_features_dim
        global_features_shape = (graph.shape[0], obs_features_dim)
    #############################################

    envs = make_train_env(args, agent_num)
    writer = SummaryWriter(log_dir="logs/{}_{}_{}_{}".format(args.sce_name, args.filename, mode, int(time.time())))
    writer.add_text("HyperParameters", 
                    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))
    
    mappo = []
    if args.ps:
        pass
        # from ppo import PPOAgent
        # from ppo_ps import PPOPSAgent
        # from buffer import SharedRolloutBuffer
        
        # buffer = SharedRolloutBuffer(
        #         steps=args.single_batch_size, num_env=args.num_env,
        #         state_shape=state_shape, share_shape=share_shape, action_shape=(1, ), # type: ignore
        #         agent_num=agent_num,
        #         device=device
        #         )
        # ppo = PPOPSAgent(
        #     state_dim=state_dim, share_dim=share_dim, action_dim=action_dim, 
        #     action_list=action_list,
        #     buffer=buffer, device=device, agent_num=agent_num,
        #     args=args
        #     )
        # path = "save/{}_{}".format(args.sce_name, args.filename)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # # ppo.load("save/{}_{}/agent_{}".format(args.sce_name, args.filename, mode))
        # mappo.append(ppo)
    else:
        from ppo import PPOAgent
        from buffer import RolloutBuffer
        
        for i in range(agent_num):
            buffer = RolloutBuffer(
                steps=args.single_batch_size, num_env=args.num_env,
                state_shape=state_shape, share_shape=share_shape, caction_shape=(1, ), # type: ignore
                edge_index=edge_index,
                obs_features_shape=obs_features_shape, global_features_shape=global_features_shape, raction_shape=(1, ), # type: ignore
                raction_mask_shape=raction_mask_shape,
                device=device
                )
            ppo = PPOAgent(
                state_dim=state_dim, share_dim=share_dim, 
                caction_dim=caction_dim, caction_list=caction_list,
                obs_features_shape=obs_features_shape, global_features_shape=global_features_shape, 
                raction_dim=raction_dim, raction_list=raction_list,
                edge_index=edge_index, buffer=buffer, device=device, args=args
                )
            path = "save/{}_{}".format(args.sce_name, args.filename)
            if not os.path.exists(path):
                os.makedirs(path)
            # ppo.load("save/{}_{}/agent_{}_{}".format(args.sce_name, args.filename, i, mode))
            mappo.append(ppo)
        
    print("Random: {}   Learning rate: {}   Gamma: {}".format(args.randomize, args.lr, args.gamma))
    best_reward, best_step = train_policy(envs, mappo, writer, args, mode, agent_num)
    writer.close()
    print("best_reward:{}   best_step:{}".format(best_reward, best_step))
    print()

if __name__ == '__main__':
    main()
