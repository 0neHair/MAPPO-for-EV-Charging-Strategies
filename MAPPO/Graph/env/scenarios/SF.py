'''
Author: CQZ
Date: 2024-03-31 18:02:01
Company: SEU
'''
import numpy as np
from env.EV_agent import EV_Agent

class Scenario(): # Sioux-Falls transportation system
    def __init__(self, frame, seed):
        # 设定随机数
        if seed is None:
            np.random.seed(0)
        else:
            np.random.seed(seed)
        # 仿真参数
        self.frame = frame
        
        # 图
        # self.route = [40, 40, 40, 40] # km CS之间距离
        self.edge_index = [
            [0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 8, 9, 9, 10, 9, 10, 11, 11, 12, 13, 13, 14, 14, 15, 15, 16, 17, 18, 19, 20, 21, 21, 22],
            [1, 2, 5, 3, 11, 10, 4, 5, 8, 7, 16, 6, 15, 9, 15, 16, 9, 14, 13, 10, 12, 19, 14, 22, 21, 18, 16, 17, 18, 23, 23, 20, 23, 23, 20, 19]
        ] # 边编号
        length = [60, 20, 20, 20, 40, 40, 20, 20, 20, 20, 20, 20, 20, 20, 20, 28, 20, 40, 40, 20, 80, 20, 20, 20, 20, 20, 20, 20, 20, 82, 40, 20, 20, 28, 20, 20]
        self.edge_attr = []
        for l in length:
            self.edge_attr.append([l]) # 边特征，即边长度
        self.map_adj = [
            [0 for _ in range(24)] for __ in range(24)
        ] # 邻接矩阵
        for i in range(len(self.edge_index[0])):
            self.map_adj[self.edge_index[0][i]][self.edge_index[1][i]] = 1
        # CS节点初始化特征
        # self.cs_charger_waiting_time = [
        #     [2.0, 2.0, 2.0], 
        #     [0.0, 0.0, 2.0], 
        #     [3.0, 3.0, 3.0]
        #     ] # h 每个CS每个充电桩等待时间
        self.cs_charger_waiting_time = [
            [0.0, 0.0] for _ in range(24)
            ] # h 每个CS每个充电桩等待时间
        self.cs_charger_min_id = [
            0 for _ in range(24)
            ] # h 每个CS每个充电桩最小时间的id
        self.cs_waiting_time = [0.0 for _ in range(24)] # h 每个CS最小等待时间

        # assert len(self.route) == len(self.cs_charger_waiting_time)+1, "Error in map"
        # self.power = [30, 30, 30] # 功率
        # 动作
        self.caction_list = [i/100 for i in range(0, 105, 5)] # 动作列表
        self.caction_list[0] = 1.5 # 动作列表
        self.raction_list = [0, 1]
        # 智能体
        self.agents = [] # 智能体列表
        total_vehicle_num = 20
        active_time_list = np.sort(np.round(np.random.uniform(0, 1, (total_vehicle_num)), 2))
        for i in range(0, len(active_time_list)):
            SOC_init = 0.5
            SOC_exp = 0.5
            agent = EV_Agent(
                id=i, frame=self.frame, 
                # route=self.route, 
                map_adj=self.map_adj,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                caction_list=self.caction_list, 
                raction_list=self.raction_list, 
                enter_time=active_time_list[i], 
                SOC_init=SOC_init, SOC_exp=SOC_exp, 
                SOC90_penalty=0, SOC20_penalty=0, 
                consume=0.15, speed=100, E_max=60
                )
            self.agents.append(agent)

        