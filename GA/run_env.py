'''
Author: CQZ
Date: 2024-12-02 15:38:02
Company: SEU
'''
import numpy as np

def x2action_seq(edge_used, point_used, ve_matrix, map_adj):
    ract_seq = []
    cact_seq = []
    
    current_point = 0
    final_point = map_adj.shape[1]-1
    while current_point != final_point:
        current_e = None
        for e in range(ve_matrix.shape[1]):
            if ve_matrix[current_point][e] == -1 and edge_used[e] == 1:
                current_e = e
        for v in range(ve_matrix.shape[0]):
            if ve_matrix[v][current_e] == 1:
                current_point = v
        ract_seq.append(current_point)
    if map_adj[0].sum() != 1:
        ract_seq.insert(0, 0)
    
    for a in ract_seq:
        cact_seq.append(point_used[a])
    ract_seq.pop(0)
    cact_seq.pop()
    
    return cact_seq, ract_seq

def run(caction_seq, raction_seq, env, agent_num):
    default_caction = np.zeros(agent_num) # 默认动作
    default_raction = np.zeros(agent_num) # 默认动作
    default_action = (default_caction, default_raction)
    
    # caction_seq = [
    #     [c * 节点数量] * 智能体数量
    # ]
    # raction_seq = [
    #     [r * 节点数量] * 智能体数量
    # ]
        
    agents_total_reward = [0 for _ in range(agent_num)]
    env.reset()
    obs_n, obs_mask_n, share_obs, done_n, \
        reward_n, cact_n, ract_n, \
            activate_agent_i, activate_to_act \
                    = env.step(default_action)
    for i, agent_i in enumerate(activate_agent_i):
        agents_total_reward[agent_i] += reward_n[agent_i][0].copy()
    while 1:
        caction_n = np.array([-1 for _ in range(agent_num)])
        raction_n = np.array([-1 for _ in range(agent_num)])
        for i, agent_i in enumerate(activate_agent_i):
            if activate_to_act[i]:
                caction_n[agent_i] = caction_seq[agent_i].pop(0)
                raction_n[agent_i] = raction_seq[agent_i].pop(0)
                if env.caction_list[caction_n[agent_i]] <= obs_n[agent_i][0]:
                    caction_n[agent_i] = 0

        obs_n, obs_mask_n, share_obs, done_n, \
            reward_n, cact_n, ract_n, \
                activate_agent_i, activate_to_act \
                    = env.step((caction_n, raction_n)) # (caction_n, raction_n)
        
        #* 将被激活的智能体当前状态作为上一次动作的结果保存
        for i, agent_i in enumerate(activate_agent_i):
            agents_total_reward[agent_i] += reward_n[agent_i][0].copy()

        if env.agents_active == []: 
            break
    
    # print(agents_total_reward)
    total_reward = 0
    for i in range(agent_num):
        total_reward += agents_total_reward[i]
    return total_reward

# if __name__ == "__main__":
#     env = EV_Sce_Env(sce_name=args.sce_name, seed=args.seed, ps=False)
#     ve_matrix = np.zeros((env.map_adj.shape[1], env.edge_index.shape[1]))
#     for e in range(env.edge_index.shape[1]):
#         ve_matrix[env.edge_index[0][e], e] = -1
#         ve_matrix[env.edge_index[1][e], e] = 1
#     caction_list = env.caction_list # 可选充电动作列表
#     cact_dim = ve_matrix.shape[0]
#     cat_max = len(caction_list)
#     ract_dim = ve_matrix.shape[1]
#     agent_num = env.agent_num # 智能体数量
#     num_cs = env.num_cs # 充电站数量