'''
Author: CQZ
Date: 2024-12-01 11:00:41
Company: SEU
'''
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GA import GA_tqdm

from env.EV_Sce_Env import EV_Sce_Env
from run_env import run, x2action_seq

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sce_name", type=str, default="test_2")
    parser.add_argument("--filename", type=str, default="T1")
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--randomize", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)
    
    parser.add_argument("--size_pop", type=int, default=20)
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--prob_mut", type=float, default=0.05)
    
    arguments = parser.parse_args()
    return arguments

args = parse_args()
if args.randomize:
    seed = args.seed
    print("Random Seed: {}".format(seed))
    np.random.seed(seed)

env = EV_Sce_Env(sce_name=args.sce_name, seed=args.seed, ps=False)

ve_matrix = np.zeros((env.map_adj.shape[1], env.edge_index.shape[1]))
for e in range(env.edge_index.shape[1]):
    ve_matrix[env.edge_index[0][e], e] = -1
    ve_matrix[env.edge_index[1][e], e] = 1
b = np.zeros(ve_matrix.shape[0])
b[0] = -1
b[-1] = 1
# ve_matrix * x = b

caction_list = env.caction_list # 可选充电动作列表
cact_dim = ve_matrix.shape[0]
cat_max = len(caction_list)-1
ract_dim = ve_matrix.shape[1]
agent_num = env.agent_num # 智能体数量
num_cs = env.num_cs # 充电站数量

def obj_func(x):
    # print(x)
    x = np.array(x)
    edge_used = []
    point_used = []
    i = 0
    for _ in range(agent_num):
        point_used.append(x[i:i+cact_dim])
        i += cact_dim
    for _ in range(agent_num):
        edge_used.append(x[i:i+ract_dim])
        i += ract_dim
    
    rr = 0
    for i in range(agent_num):
        route_i = edge_used[i]
        rr_i = (~(np.dot(ve_matrix, route_i) == b)).sum().copy()
        rr += rr_i
    if rr != 0:
        # print("Failed")
        return -100 * agent_num

    caction_seq = []
    raction_seq = []
    for i in range(agent_num):
        cact_seq, ract_seq = x2action_seq(edge_used[i], point_used[i], ve_matrix, env.map_adj)
        caction_seq.append(cact_seq)
        raction_seq.append(ract_seq)

    reward = run(caction_seq, raction_seq, env, agent_num)
    return -reward

def route_cons(x):
    x = np.array(x)
    edge_used = []
    r = 0

    i = 0
    for _ in range(agent_num):
        i += cact_dim
    for _ in range(agent_num):
        edge_used.append(x[i:i+ract_dim])
        i += ract_dim
        
    for i in range(agent_num):
        route_i = edge_used[i]
        r_i = (~(np.dot(ve_matrix, route_i) == b)).sum().copy()
        r += r_i
    
    return r

constraint = [route_cons]
# point_used: 在每个节点的充电动作选择
# edge_used: 要用的路段编号

# v_charge = [[0, 0, 5, 0, 0]] * agent_num 在每个点上的动作编号，整数
# edge_seq = [[1, 1, 0, 0, 1, 0]] * agent_num 选择哪些路段，满足约束的话可以串起来形成路径，但生成解生成不出来
# section_p = [[点个数] * 点个数] * agent_num 每个点选择下一个点的概率，可以mask

if args.train:
    n_dim = cact_dim*agent_num+ract_dim*agent_num
    ga = GA_tqdm(
            func=obj_func, 
            n_dim=n_dim, 
            lb=[0] * n_dim, # type: ignore
            ub=[cat_max]*(cact_dim*agent_num) + [1]*(ract_dim*agent_num), # type: ignore
            size_pop=args.size_pop, # 种群数量
            max_iter=args.max_iter, # 迭代次数
            prob_mut=args.prob_mut, # 变异系数
            precision=1, # 精度，1则为整数
            constraint_ueq=constraint
        )

    best_x, best_y = ga.run()
    # print('Best_X: ', best_x)
    print('best reward: {:.5f} \t best average reward: {:.5f}'.format(best_y[0], best_y[0]/agent_num))
    act_pd = pd.DataFrame(best_x, columns=["action"])
    act_pd.to_csv('action/{}_{}.csv'.format(args.sce_name, args.filename), index=False)

    y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(y_history.index, y_history.values, '.', color='red')
    ax[0].set_ylim(0, 50)
    ax[1].plot(y_history.index, y_history.min(axis=1).cummin(), color='blue')
    ax[1].set_ylim(0, 50)
    plt.savefig('logs/{}_{}.png'.format(args.sce_name, args.filename), dpi=300)
else:
    act_pd = pd.read_csv('action/{}_{}.csv'.format(args.sce_name, args.filename))
    act = np.array(act_pd['action'])

    total_reward = obj_func(act)
    print(total_reward)

