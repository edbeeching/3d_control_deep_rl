#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 09:55:24 2019

@author: edward
"""
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

from torch.distributions import Categorical
import torch.nn.functional as F
from arguments import parse_a2c_args

class Lin_View(nn.Module):
	def __init__(self):
		super(Lin_View, self).__init__()
        
	def forward(self, x):
		return x.view(x.size()[0], -1)
    
        
class CNNPolicy(nn.Module):
    def __init__(self, input_shape, args):
        super(CNNPolicy, self).__init__()
        
        self.conv_head = nn.Sequential(nn.Conv2d(3, args.conv1_size, 8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(args.conv1_size, args.conv2_size, 4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(args.conv2_size, args.conv3_size, 3, stride=1),
                                        nn.ReLU(True))

        conv_input = torch.Tensor(torch.randn((1,) + input_shape))
        print(conv_input.size(), self.conv_head(conv_input).size(), self.conv_head(conv_input).size())
        self.conv_out_size = self.conv_head(conv_input).nelement()    
        self.hidden_size = args.hidden_size

        self.linear1 = nn.Linear(self.conv_out_size, self.hidden_size)
        self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)
        

        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.dist_linear = nn.Linear(self.hidden_size, args.num_actions)
        
        self.args = args
        self.train()
        self.apply(self.initialize_weights)
        self.apply_gain()
        
    def apply_gain(self):
        relu_gain = nn.init.calculate_gain('relu')
        for i in range(0, 6, 2):
            self.conv_head[i].weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)        
    
    
    def forward(self, obs, states, masks):
        x = self.conv_head(obs*(1.0/255.0))
        x = x.view(-1, self.conv_out_size)
        x = self.linear1(x)
        x = F.relu(x)          
        x = new_states = self.gru(x, states*masks.clone())
        
        values = self.critic_linear(x)
        log_probs = F.log_softmax(self.dist_linear(x), dim=1)

        dist = Categorical(logits=log_probs)  
        
        if self.training:
            actions = dist.sample()
        else:
            _, actions = dist.probs.max(1)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
    
        # state, actions, action log_probs, entropy, value estimates
        result = {'states': new_states,
                  'actions': actions,
                  'action_log_probs': action_log_probs,
                  'entropy': entropy,
                  'values': values}
        
        return result
        
    @staticmethod
    def initialize_weights(layer):
        if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
            nn.init.orthogonal_(layer.weight.data, gain=1)
            layer.bias.data.fill_(0)
        elif type(layer) == nn.GRUCell:
            nn.init.orthogonal_(layer.weight_ih, gain=1)
            nn.init.orthogonal_(layer.weight_hh, gain=1)
            layer.bias_ih.data.fill_(0)
            layer.bias_hh.data.fill_(0)
        else:
            pass
    
    
    
if __name__ == '__main__':
    
    args = parse_a2c_args()
    args.num_actions = 5
    shape = (3,64,112)
    model = CNNPolicy(shape, args)
    
    inp = {'observation': torch.randn(2,3,64,112)}
    state = torch.randn(2,128)
    mask = torch.ones(2,1)
    
    
    
    result = model(inp, state, mask)
    
    print('state', result['state'].size())
    print('actions', result['actions'].size())
    print('entropy', result['entropy'].size())
    print('action_log_probs', result['action_log_probs'].size())
    print('values', result['values'].size())
