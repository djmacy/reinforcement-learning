#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import random
import torch

class ReplayMemory:
    
    def __init__(self, capacity = 1000000):
        self.capacity = capacity
        self.memory = []
        self.position = 0   #Location where we'll insert the next transition
    
    # insert [s, a, r, s']
    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity   
    
    # sample [[s, a, r, s'], ...]
    def sample(self, batch_size):
        # return a batch of samples
        assert self.can_sample(batch_size)
        
        batch = random.sample(self.memory, batch_size)
        
        #Now, package the batch so that it's of the appropriate form.
        # [[s, a, r, s'], [s, a, r, s'], [s, a, r, s']] -> [[s, s, s],[a, a, a], [r, r, r], [s', s', s']]
        batch = zip(*batch)
        
        return [torch.cat(items) for items in batch]  # output tensor is N x D (elements by dimensions)
    
    # can_sample -> True/False  are there sufficient memories to sample from
    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10
    
    # __len__
    def __len__(self):
        return len(self.memory)

