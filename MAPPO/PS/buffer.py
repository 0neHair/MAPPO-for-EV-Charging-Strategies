'''
Author: CQZ
Date: 2024-09-17 16:11:38
Company: SEU
'''
import torch
import numpy as np


class RolloutBuffer(object):
    def __init__(
        self, 
        steps: int, num_env: int, 
        state_shape: tuple, share_shape: tuple, action_shape: tuple, 
        device
        ):
        self.steps = steps
        self.device = device

        self.state = np.zeros((steps, num_env) + state_shape, dtype=np.float32)
        self.share_state = np.zeros((steps, num_env) + share_shape, dtype=np.float32)
        self.action = np.zeros((steps, num_env) + action_shape, dtype=np.float32)
        self.log_prob = np.zeros((steps, num_env) + action_shape, dtype=np.float32)
        self.next_state = np.zeros((steps, num_env) + state_shape, dtype=np.float32)
        self.next_share_state = np.zeros((steps, num_env) + share_shape, dtype=np.float32)
        self.reward = np.zeros((steps, num_env), dtype=np.float32)
        self.done = np.zeros((steps, num_env), dtype=np.float32)

        self.ptr = [0 for _ in range(num_env)]

    def push(self, reward, next_state, next_share_state, done, env_id):
        ptr = self.ptr[env_id]
        self.reward[ptr][env_id] = reward
        self.next_state[ptr][env_id] = next_state
        self.next_share_state[ptr][env_id] = next_share_state
        self.done[ptr][env_id] = done

        self.ptr[env_id] = (ptr + 1) % self.steps

    def push_last_state(self, state, share_state, action, log_prob, env_id):
        ptr = self.ptr[env_id]
        self.state[ptr][env_id] = state
        self.share_state[ptr][env_id] = share_state 
        self.action[ptr][env_id] = action
        self.log_prob[ptr][env_id] = log_prob
    
    def pull(self):
        return (
            torch.tensor(self.state, dtype=torch.float32).to(self.device),
            torch.tensor(self.share_state, dtype=torch.float32).to(self.device),
            torch.tensor(self.action, dtype=torch.float32).to(self.device),
            torch.tensor(self.log_prob, dtype=torch.float32).to(self.device),
            torch.tensor(self.reward, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_state, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_share_state, dtype=torch.float32).to(self.device),
            torch.tensor(self.done, dtype=torch.float32).to(self.device)
        )

    # @property
    # def full(self):
    #     return self.ptr == 0

class SharedRolloutBuffer(object):
    def __init__(
        self, 
        steps: int, num_env: int, 
        state_shape: tuple, share_shape: tuple, action_shape: tuple, 
        agent_num: int,
        device
        ):
        self.steps = steps
        self.agent_num = agent_num
        self.device = device

        self.state = np.zeros((steps, num_env, agent_num) + state_shape, dtype=np.float32)
        self.share_state = np.zeros((steps, num_env, agent_num) + share_shape, dtype=np.float32)
        self.action = np.zeros((steps, num_env, agent_num) + action_shape, dtype=np.float32)
        self.log_prob = np.zeros((steps, num_env, agent_num) + action_shape, dtype=np.float32)
        self.next_state = np.zeros((steps, num_env, agent_num) + state_shape, dtype=np.float32)
        self.next_share_state = np.zeros((steps, num_env, agent_num) + share_shape, dtype=np.float32)
        self.reward = np.zeros((steps, num_env, agent_num), dtype=np.float32)
        self.done = np.zeros((steps, num_env, agent_num), dtype=np.float32)

        self.ptr = [0 for _ in range(num_env)]

    def push(self, reward, next_state, next_share_state, done, env_id, agent_id):
        ptr = self.ptr[env_id]
        self.reward[ptr][env_id][agent_id] = reward
        self.next_state[ptr][env_id][agent_id] = next_state
        self.next_share_state[ptr][env_id][agent_id] = next_share_state
        self.done[ptr][env_id][agent_id] = done

        self.ptr[env_id]  = (ptr + 1) % self.steps

    def push_last_state(self, state, share_state, action, log_prob, env_id, agent_id):
        ptr = self.ptr[env_id]
        self.state[ptr][env_id][agent_id] = state
        self.share_state[ptr][env_id][agent_id] = share_state 
        self.action[ptr][env_id][agent_id] = action
        self.log_prob[ptr][env_id][agent_id] = log_prob
    
    def pull(self):
        return (
            torch.tensor(self.state, dtype=torch.float32).to(self.device),
            torch.tensor(self.share_state, dtype=torch.float32).to(self.device),
            torch.tensor(self.action, dtype=torch.float32).to(self.device),
            torch.tensor(self.log_prob, dtype=torch.float32).to(self.device),
            torch.tensor(self.reward, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_state, dtype=torch.float32).to(self.device),
            torch.tensor(self.next_share_state, dtype=torch.float32).to(self.device),
            torch.tensor(self.done, dtype=torch.float32).to(self.device)
        )

    # @property
    # def full(self):
    #     return self.ptr == 0
