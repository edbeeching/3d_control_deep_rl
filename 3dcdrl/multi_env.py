#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 10:33:41 2019

@author: edward
"""
from itertools import islice

from collections import deque
import multiprocessing as mp
import numpy as np
from arguments import parse_game_args        
from doom_environment import DoomEnvironment


def pipe_worker(pipe, params, is_train, idx=0):
    env = DoomEnvironment(params, idx=idx, is_train=is_train, use_shaping=params.use_shaping)
    while True:
        action = pipe.recv()
        if action is None:
            break
        elif action == 'reset':
            pipe.send(env.reset())
        else:
            obs, reward, done, info = env.step(action)
            pipe.send((obs, reward, done, info))
            
def pipe_worker2(pipe, params, is_train, idx_range=[0]):
    envs_queue = deque()    
    for idx in idx_range:
        env = DoomEnvironment(params, idx=idx, is_train=is_train, use_shaping=params.use_shaping, fixed_scenario=True)        
        obs = env.reset()
        envs_queue.append((obs, env))
        
    obs, cur_env = envs_queue.pop()
    
    
    while True:
        action = pipe.recv()
        if action is None:
            break
        elif action == 'reset':
            pipe.send(env.reset())
        else:
            obs, reward, done, info = cur_env.step(action)
            
            if done:
                envs_queue.append((obs, cur_env))
                obs, cur_env = envs_queue.popleft()
                
            pipe.send((obs, reward, done, info))
            

            
def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())    


class MultiEnv(object):
    """
        Run many envs on different processes to speed up simulation.
        
        Here this is fixed to be 16 workers but this could be increased if more
        compute is available
    """
    def __init__(self, env_id, num_envs, params, is_train=True):
        self.parent_pipes, self.child_pipes = zip(*[mp.Pipe() for _ in range(num_envs)])
        self.workers = []
        
        if params.fixed_scenario:
            if is_train:
                num_scenarios = params.num_mazes_train
            else:                
                num_scenarios = params.num_mazes_test
            
            chunk_size = num_scenarios // num_envs
            print('scenarios, chunk size')
            print(num_scenarios, chunk_size)
            
            chunks = chunk(range(num_scenarios), chunk_size)
            
            for idx, (child_pipe, idx_range) in enumerate( zip(self.child_pipes, chunks)):
                process = mp.Process(target=pipe_worker2, args=(child_pipe, params, is_train, idx_range), daemon=True) # use daemon=true so jobs die when there is an exception in main thread
                self.workers.append(process)
                process.start()            
            
        else:
            for idx, child_pipe in enumerate( self.child_pipes):
                process = mp.Process(target=pipe_worker, args=(child_pipe, params, is_train, idx), daemon=True) # use daemon=true so jobs die when there is an exception in main thread
                self.workers.append(process)
                process.start()
            
        print('There are {} workers'.format(len(self.workers)))
    
        assert env_id == 'doom', 'Multiprocessing only implemented for doom envirnment'           
        # tmp_env = DoomEnvironment(params)
        if params.num_actions == 0:
            num_actions = 5 if params.limit_actions else 8
            params.num_actions = num_actions
        self.num_actions = params.num_actions
        
        self.obs_shape = (3, params.screen_height, params.screen_width)
        self.prep = False # Observations already in CxHxW order    

    def reset(self):
        new_obs = []
        for pipe in self.parent_pipes:
            pipe.send('reset')
            
        for pipe in self.parent_pipes:
            obs = pipe.recv()
            new_obs.append(self.prep_obs(obs))
            
        return np.stack(new_obs)
    
    def cancel(self):
        for pipe in self.parent_pipes:
            pipe.send(None)     
            
        for worker in self.workers:
            worker.join()
        print('workers cancelled')
        
                    
    def prep_obs(self, obs):
        if self.prep:
            return obs.transpose(2,0,1)
        else: 
            return obs
    
    def step(self, actions):
        new_obs = []
        rewards = []
        dones = []
        infos = []
 
        for action, pipe in zip(actions, self.parent_pipes):
            pipe.send(action)
        
        for pipe in self.parent_pipes:
            obs, reward, done, info = pipe.recv()
            new_obs.append(self.prep_obs(obs))
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return np.stack(new_obs), rewards, dones, infos                
       

if __name__ == '__main__':
    
    args = parse_game_args()
    args.scenario_dir = '../resources/scenarios/'
    

    mp_test_envs = MultiEnv(args.simulator, args.num_environments, 1, args)
    mp_test_envs.reset()
    