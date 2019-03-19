#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:44:33 2019

@author: edward
"""
import numpy as np
from moviepy.editor import ImageSequenceClip

class Scorer():
    def __init__(self, num_envs, initial_obs, movie=True):
        self.best = [None, -100000] # obs, and best reward
        self.worst = [None, 100000] # obs, and worse reward
        self.trajectories = {}
        self.total_rewards = []
        self.total_times = []
        self.num_envs = num_envs
        self.movie = movie
        if self.movie:
            initial_obs = initial_obs.astype(np.uint8)           
        else:
            initial_obs = [None]*initial_obs.shape[0]
        
        for i in range(num_envs):
            self.trajectories[i] = [[initial_obs[i]], []]
            
    def update(self, obs, rewards, dones):   
        obs = obs.astype(np.uint8)   
        if self.movie:
            obs = obs.astype(np.uint8)           
        else:
            obs = [None]*obs.shape[0]
        
        
        for i in range(self.num_envs):
            if dones[i]:
                self.trajectories[i][1].append(rewards[i])
                accumulated_reward = sum(self.trajectories[i][1])
                self.total_rewards.append(accumulated_reward)
                self.total_times.append(len(self.trajectories[i][1]))
                
                if accumulated_reward > self.best[1]:
                    self.best[0] = self.trajectories[i][0]
                    self.best[1] = accumulated_reward
                    
                if accumulated_reward < self.worst[1]:
                    self.worst[0] = self.trajectories[i][0]
                    self.worst[1] = accumulated_reward   
             
                self.trajectories[i] = [[obs[i]], [0.0]]
  
            else:
                self.trajectories[i][0].append(obs[i])
                self.trajectories[i][1].append(rewards[i])
                
                
    def clear(self):
        self.trajectories = None
        
def write_movie(output_dir, observations, step, score, best_agent=True):    
    observations = [o.transpose(1,2,0) for o in observations]
    clip = ImageSequenceClip(observations, fps=int(30/4)) # assumes frame skip of 4

    clip.write_videofile('{}eval{:0004}_{:00005.0f}.mp4'.format(output_dir, step, score*100))