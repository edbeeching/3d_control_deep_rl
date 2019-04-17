#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:24:42 2018

@author: edward

"""
import os
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import torch
import time
from doom_a2c.arguments import parse_game_args
from doom_a2c.models import CNNPolicy
from environments import DoomEnvironment
from doom_evaluation import BaseAgent

from plot_4_item import plot_4_item, plot_laby

def gen_data(agent, env, params, num_rollouts=1):
    
    with torch.no_grad():
        env.reset()
        agent.reset()
        
        rewards = []
        obss = []  
        actions = []
        action_probss = []
        values = []
        depth_preds = []
        states = []
        done = False
        obs = env.reset().astype(np.float32)
        
        xs = []
        ys = []
        thetas = []
        while not done:
            
            obss.append(obs)    
            state = agent.state.clone().detach().numpy()
            states.append(state)
            action, value, action_probs = agent.get_action_value_and_probs(obs, epsilon=0.0, deterministic=True)
            pdo = env.get_player_pos_delta_origin()
            x, y, theta = pdo[0], pdo[1], pdo[2]
            xs.append(x)
            ys.append(y)
            thetas.append(theta)
            
            # origin_x, origin_y = pdo[6], pdo[7]
            # dx, dy, dt = env.get_player_deltas()
            # print(xxs, yys, thetas, origin_x, origin_y, xxs - origin_x, yys-origin_y)
            # print(dx, dy, dt)
            
            if params.predict_depth:
                depths = agent.model.pred_depth(torch.from_numpy(obs).unsqueeze(0))
                depths = torch.max(depths, dim=1)[1]
                depths = depths.squeeze(0).numpy()
                
                depth_preds.append(depths)
            else:
                depths = np.zeros((4,8))
                depth_preds.append(depths)
                
            obs, reward, done, _ = env.step(action)
            obs = obs.astype(np.float32)
            rewards.append(reward)
            
            actions.append(actions)
            action_probss.append(action_probs)
            values.append(value) 

    sst = np.concatenate(states)

    
    # plt.figure()
    # for i,obs in enumerate(obss[0:32],1):
    #     plt.subplot(4,8,i)
    #     plt.title('Step {}'.format(i))
    #     plt.imshow(obs.transpose(1,2,0)/255.)
    #     plt.axis('off')
    return sst, obss, (xs, ys, thetas)
        

if __name__ == '__main__':
    params = parse_game_args()
    env = DoomEnvironment(params)

    print(env.num_actions)
    obs_shape = (3, params.screen_height, params.screen_width)

    actor_critic = CNNPolicy(obs_shape[0], obs_shape, params)
        
    assert params.model_checkpoint, 'No model checkpoint found'
    assert os.path.isfile(params.model_checkpoint), 'The model could not be loaded'
    # This lambda stuff is required otherwise it will try and load on GPU
    checkpoint = torch.load(params.model_checkpoint, map_location=lambda storage, loc: storage) 
    actor_critic.load_state_dict(checkpoint['model'])
    
    agent = BaseAgent(actor_critic, params)

    
    states, obs, pos = gen_data(agent, env, params, num_rollouts=100)

    data = states



    idxs = [len(d) for d in data]
    idxs = np.array([i for l in idxs for i in range(l)])
            
    
    data = np.concatenate(data)
    permute = np.random.permutation(len(data))
    perm_data = data[permute,:]
    perm_idxs = idxs[permute]
    
    
    num_points = 10000
    
    sub_data = perm_data[:num_points]
    sub_idxs = perm_idxs[:num_points]
    from sklearn.manifold import TSNE
    
    clf = TSNE()
    p0p1 = clf.fit_transform(sub_data)
    
    colors = ['r' if idx < 128 else 'b' if idx < 256 else 'g' for idx in sub_idxs ]
    
    plt.scatter(p0p1[:,0], p0p1[:,1], c=sub_idxs )
    plt.colorbar()
    
    
    
    
    
    
    
    
    
