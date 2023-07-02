#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import copy
import gymnasium as gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PreprocessEnv import PreprocessEnv
from ReplayMemory import ReplayMemory
from torch import nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from typing import Callable
from IPython import display

import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

get_ipython().run_line_magic('matplotlib', 'notebook')


class Deep_Q_Class():
    
    def __init__(self, env, episodes, batchSize, gamma, alpha, epsilon):
        self.env = env
        self.seed = 42
        self.make_env(self.env)
        self.make_q_network()
        self.episodes = episodes
        self.batchSize = batchSize # 32 is default
        self.gamma = gamma #Discount Factor default is 0.99
        self.alpha = alpha #Learning Rate default is 0.001
        self.epsilon = epsilon #Randomness default is 0.05
        
        
    def policy(self, state):
        if torch.rand(1) < self.epsilon:
            return torch.randint(self.num_actions, (1,1))
        else:
            av = self.q_network(state).detach()
            return torch.argmax(av, dim=-1, keepdim = True)
        
    def deep_sarsa(self):
        optim = AdamW(self.q_network.parameters(), lr = self.alpha)   # AdamW is modification of SGD
        memory = ReplayMemory(capacity = 1000000)
        stats = {'MSE Loss': [], 'Returns': []}

        for episode in tqdm(range(1, self.episodes + 1)):      #tqdm tells us something about how many iteraitons are left.
            state = self.env.reset()
            done = False
            ep_return = 0.

            while not done:
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                memory.insert([state, action, reward, done, next_state])  # insert into memory buffer

                if memory.can_sample(self.batchSize):
                    state_b, action_b, reward_b, done_b, next_state_b = memory.sample(self.batchSize)

                    # compute q values of states
                    # Evaluate the network for each state in the batch under each action in the batch
                    qsa_b = self.q_network(state_b).gather(1, action_b)  
                    next_action_b = self.policy(next_state_b)  # next actions based on the next states under the policy.
                    next_qsa_b = self.target_q_network(next_state_b).gather(1, next_action_b)
                    target_b = reward_b + ~done_b * self.gamma * next_qsa_b

                    loss = F.mse_loss(qsa_b, target_b)

                    self.q_network.zero_grad()    # eliminate gradients previously computed.  Now compute new ones
                    loss.backward()          # Begin back-propagation process for computing gradients
                    optim.step()             # perform gradient descent stuff

                    stats['MSE Loss'].append(loss.item())
                state = next_state
                ep_return += reward.item()
            stats['Returns'].append(ep_return)

            if episode % 10 == 0:
                self.target_q_network.load_state_dict(self.q_network.state_dict())

        return stats
        
    def seed_everything(self, env: gym.Env, seed: int = 42) -> None:
        """
        Seeds all the sources of randomness so that experiments are reproducible.
        Args:
            env: the environment to be seeded.
            seed: an integer seed.
        Returns:
            None.
        """
   
        self.env.reset(seed = seed)
        self.env.action_space.seed(seed)
        self.env.observation_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch._set_deterministic(True) 
   
        
    def make_env(self, env, render_mode="rgb_array"):
        self.env = gym.make(env)
        self.seed_everything(self.env)
        self.env = PreprocessEnv(self.env, seed=self.seed)
        state_dims = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.state_dims = state_dims
        self.num_actions = num_actions
        print(f"MountainCar Env: State dimensions: {state_dims}, Number of actions: {num_actions}")
        
    def make_q_network(self):
        state_dim  = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        
        self.q_network = nn.Sequential(
        # Input to first hidden layer
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        # Input to second hidden layer
        nn.Linear(128, 64),
        nn.ReLU(),
        # Output to final layer
        nn.Linear(64, num_actions)
        )# Feed forward neural network

        self.target_q_network = copy.deepcopy(self.q_network)
        self.target_q_network.eval() #Now the new network won't change its parameters
       
        
    def plot_cost_to_go(self, xlabel=None, ylabel=None):
        highx, highy = self.env.observation_space.high
        lowx, lowy = self.env.observation_space.low
        X = torch.linspace(lowx, highx, 100)
        Y = torch.linspace(lowy, highy, 100)
        X, Y = torch.meshgrid(X, Y)

        q_net_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        Z = - self.q_network(q_net_input).max(dim=-1, keepdim=True)[0]
        Z = Z.reshape(100, 100).detach().numpy()
        X = X.numpy()
        Y = Y.numpy()

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='jet', linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel(xlabel, size=14)
        ax.set_ylabel(ylabel, size=14)
        ax.set_title("Estimated cost-to-go", size=18)
        plt.tight_layout()
        plt.show()

    def plot_max_q(env, xlabel=None, ylabel=None, action_labels=[]):
        highx, highy = env.observation_space.high
        lowx, lowy = env.observation_space.low
        X = torch.linspace(lowx, highx, 100)
        Y = torch.linspace(lowy, highy, 100)
        X, Y = torch.meshgrid(X, Y)
        q_net_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        Z = self.q_network(q_net_input).argmax(dim=-1, keepdim=True)
        Z = Z.reshape(100, 100).T.detach().numpy()
        values = np.unique(Z.ravel())
        values.sort()

        plt.figure(figsize=(5, 5))
        plt.xlabel(xlabel, size=14)
        plt.ylabel(ylabel, size=14)
        plt.title("Optimal action", size=18)

        # im = plt.imshow(Z, interpolation='none', cmap='jet')
        im = plt.imshow(Z, cmap='jet')
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, action_labels)]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.tight_layout()

    def plot_stats(self, stats):
        rows = len(stats)
        cols = 1

        fig, ax = plt.subplots(rows, cols, figsize=(12, 6))

        for i, key in enumerate(stats):
            vals = stats[key]
            vals = [np.mean(vals[i-10:i+10]) for i in range(10, len(vals)-10)]
            if len(stats) > 1:
                ax[i].plot(range(len(vals)), vals)
                ax[i].set_title(key, size=18)
            else:
                ax.plot(range(len(vals)), vals)
                ax.set_title(key, size=18)
        plt.tight_layout()
        plt.show()    

    def test_agent(self, episodes: int = 10) -> None:
        plt.figure(figsize=(8, 8))
        for episode in range(episodes):
            
            state = self.env.reset()
            done = False
            img = plt.imshow(self.env.render())
            while not done:
                p = self.policy(state)
                if isinstance(p, np.ndarray):
                    action = np.random.choice(p.shape[0], p=p)
                else:
                    action = p
                next_state, _, done, _ = self.env.step(action)
                img.set_data(self.env.render())
                plt.axis('off')
                display.display(plt.gcf())
                display.clear_output(wait=True)
                state = next_state

  
    

