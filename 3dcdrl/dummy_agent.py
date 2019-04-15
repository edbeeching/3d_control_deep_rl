#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 14:31:17 2018

@author: anonymous
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch.autograd import Variable
from collections import deque

class BaseAgent(object):
    def __init__(self, model, params):
        self.params = params
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if params.num_stack > 1:
            self.exp_size = params.num_stack
            self.short_term_memory = deque()

        self.mask = Tensor([[1.0]]).to(self.device)
        
        
    def get_action(self, observation, epsilon=0.0):
        if hasattr(self, 'short_term_memory'):
            observation = self._prepare_observation(observation)    
        observation = torch.from_numpy(observation).unsqueeze(0).to(self.device) 
        _, action, _, self.state = self.model.act(observation, self.state, self.mask,deterministic=True)     

        return action.cpu().data.numpy()[0,0]
        


    def get_action_value_and_probs(self, observation, epsilon=0.0, deterministic=True):
        if hasattr(self, 'short_term_memory'):
            observation = self._prepare_observation(observation)
        
        observation = torch.from_numpy(observation).unsqueeze(0).to(self.device)

        value, action, probs, self.state= self.model.get_action_value_and_probs(observation, self.state, self.mask ,deterministic=deterministic)
        
        return action.cpu().detach().numpy()[0,0], value.cpu().detach().numpy(), probs.cpu().detach().numpy()
        
    def reset(self):
        """
            reset the models hidden layer when starting a new rollout
        """
        if hasattr(self, 'short_term_memory'):
            self.short_term_memory = deque()
            

        self.state = torch.zeros(1, self.model.state_size).to(self.device)
            
      
        
     
    def _prepare_observation(self, observation):
        """
           As the network expects an input of n frames, we must store a small
           short term memory of frames. At input this is completely empty so 
           I pad with the firt observations 4 times, generally this is only used when the network
           is not recurrent
        """
        if len(self.short_term_memory) == 0 :
            for _ in range(self.exp_size):
                self.short_term_memory.append(observation)
            
        self.short_term_memory.popleft()
        self.short_term_memory.append(observation)
        
        return np.vstack(self.short_term_memory)
   
    
