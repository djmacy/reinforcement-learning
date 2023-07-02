#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gymnasium as gym
import torch

class PreprocessEnv(gym.Wrapper):
    def __init__(self, env, seed=None):
        super().__init__(env)
        self.seed = seed
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    #Wrap env.reset so that it plays nicely with pytorch
    def reset(self):
        state = self.env.reset()
        return torch.from_numpy(state[0]).unsqueeze(dim = 0).float()
    
    #Wrap env.step to play nicely with pytorch
    def step(self, action):
        action = action.item()  # Converts the pytorch tensor 'action' that was passed in into an integer
        next_state, reward, terminated, truncated, info = self.env.step(action)
        # Now convert the np stuff from the step command into pytorch tensors
        next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float()
        reward = torch.tensor(reward).view(1,-1).float()   # the 'view' command nests the result in a tensor
        done = torch.tensor(terminated or truncated).view(1,-1)
        return next_state, reward, done, info