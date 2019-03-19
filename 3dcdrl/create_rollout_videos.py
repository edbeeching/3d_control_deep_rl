#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 09:19:33 2018

@author: anonymous
"""

import os
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import torch
import time
from arguments import parse_game_args
from models import CNNPolicy
from environments import DoomEnvironment
from dummy_agent import BaseAgent

from moviepy.editor import ImageSequenceClip

def make_movie(agent, env, filename, params):
    with torch.no_grad():
        env.reset()
        agent.reset()
        
        obss = []  
        
        obs = env.reset().astype(np.float32)
        done = False
        while not done:
            obss.append(obs)   
            action, value, action_probs = agent.get_action_value_and_probs(obs, epsilon=0.0, deterministic=True)

            obs, reward, done, _ = env.step(action)
            obs = obs.astype(np.float32)
            
    observations = [o.transpose(1,2,0) for o in obss]
    clip = ImageSequenceClip(observations, fps=int(30/params.frame_skip))
    clip.write_videofile(filename)  

            
def evaluate_saved_model():  
    params = parse_game_args()
    
    env = DoomEnvironment(params, is_train=True)
    print(env.num_actions)
    obs_shape = (3, params.screen_height, params.screen_width)

    actor_critic = CNNPolicy(obs_shape[0], obs_shape, params)
    
    assert params.model_checkpoint, 'No model checkpoint found'
    assert os.path.isfile(params.model_checkpoint), 'The model could not be loaded'
    # This lambda stuff is required otherwise it will try and load on GPU
    checkpoint = torch.load(params.model_checkpoint, map_location=lambda storage, loc: storage) 
    actor_critic.load_state_dict(checkpoint['model'])
    
    base_filename = params.model_checkpoint.split('.')[0].split('/')[1]

    agent = BaseAgent(actor_critic, params)
 
    for i in range(params.num_mazes_test):
        env = DoomEnvironment(params, idx=i, is_train=True)
        movie_name = 'videos/{}_rollout_{:0004}.mp4'.format(base_filename, i)
        print('Creating movie {}'.format(movie_name))
        make_movie(agent, env, movie_name, params)
        
        
        
        
        
if __name__ == '__main__':
    
    evaluate_saved_model()
    
    
