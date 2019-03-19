#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 09:54:38 2019

@author: edward
"""
import time, logging

import torch

from arguments import parse_a2c_args
from multi_env import MultiEnv
from models import CNNPolicy
from a2c_agent import A2CAgent
from utils import initialize_logging


def train():
    args = parse_a2c_args()
    output_dir = initialize_logging(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_updates = int(args.num_frames) // args.num_steps // args.num_environments
    # Create the train and test environments with Multiple processes
    train_envs = MultiEnv(args.simulator, args.num_environments, args, is_train=True)
    test_envs = MultiEnv(args.simulator, args.num_environments, args, is_train=False)
    
    obs_shape = train_envs.obs_shape
    
    # The agent's policy network and training algorithm A2C
    policy = CNNPolicy(obs_shape, args).to(device)
    agent = A2CAgent(policy, 
                     args.hidden_size,
                     value_weight=args.value_weight, 
                     entropy_weight=args.entropy_weight, 
                     num_steps=args.num_steps, 
                     num_parallel=args.num_parallel,
                     gamma=args.gamma,
                     lr=args.lr,
                     opt_alpha=args.opt_alpha,
                     opt_momentum=args.opt_momentum,
                     max_grad_norm=args.max_grad_norm)
    
    start_j = 0
    if args.reload_model:
        checkpoint_idx = args.reload_model.split(',')[1]
        checkpoint_filename = '{}models/checkpoint_{}.pth.tar'.format(output_dir, checkpoint_idx)        
        agent.load_model(checkpoint_filename)
        start_j = (int(checkpoint_idx) // args.num_steps // args.num_environments) + 1
        
    obs = train_envs.reset()
    start = time.time()
    
    for j in range(start_j, num_updates):
        if not args.skip_eval and j % args.eval_freq == 0:
            total_num_steps = (j + 1) * args.num_environments * args.num_steps
            mean_rewards, game_times = agent.evaluate(test_envs, j, total_num_steps)
            logging.info(mean_rewards)
            logging.info(game_times)
            
        for step in range(args.num_steps): 
            action = agent.get_action(obs, step)
            obs, reward, done, info = train_envs.step(action)
            agent.add_rewards_masks(reward, done, step)
            
        report = agent.update(obs)
        
        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_environments * args.num_steps
            save_num_steps = (start_j) * args.num_environments * args.num_steps
            FPS = int((total_num_steps - save_num_steps) / (end - start)),
            
            logging.info(report.format(j, total_num_steps, FPS))  
        
        if j % args.model_save_rate == 0:
            total_num_steps = (j + 1) * args.num_environments * args.num_steps
            agent.save_policy(total_num_steps, args, output_dir)
        
    # cancel the env processes    
    train_envs.cancel()
    test_envs.cancel()
    
    
if __name__ == '__main__':
    train()
    
       