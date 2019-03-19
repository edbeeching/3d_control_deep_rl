#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 09:55:14 2019

@author: edward
"""

import os

import numpy as np
import torch
from torch import nn

from scorer import Scorer
from scorer import write_movie


class A2CAgent():
    def __init__(self, 
                 policy, 
                 state_size,
                 value_weight=0.5, 
                 entropy_weight=0.001, 
                 num_steps=128, 
                 num_parallel=16,
                 gamma=0.99,
                 lr=7.5E-4,
                 opt_alpha=0.99,
                 opt_momentum=0.0,
                 max_grad_norm=0.5):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.policy = policy
        self.num_steps = num_steps
        self.num_parallel = num_parallel
            
        self.optimizer = torch.optim.RMSprop([p for p in self.policy.parameters() if p.requires_grad], 
                                              lr,
                                              alpha=opt_alpha,
                                              momentum=opt_momentum)
        self.value_weight = value_weight
        self.entropy_weight = entropy_weight
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        self.episode_rewards = torch.zeros(num_parallel, 1)
        self.final_rewards = torch.zeros(num_parallel, 1)
        
        # values, entropies and logprobs are requried for backprop so we store as list
        self.values = []
        self.entropies = []
        self.action_log_probs = []
        
        # actions, rewards and masks do not require backprop so can be stored in buffers on GPU
        self.rewards = torch.zeros(num_steps, num_parallel, 1).to(device)
        self.masks = torch.zeros(num_steps+1, num_parallel, 1).to(device)
        self.states = torch.zeros(num_parallel, state_size).to(device)

    
    def get_action(self, obs, step):
        obs = torch.from_numpy(obs).float().to(self.device)
        result = self.policy(obs, self.states, self.masks[step])
        
        # save value estimates, actions and log probs, ...
        self.states = result['states']
        self.values.append(result['values'])
        self.action_log_probs.append(result['action_log_probs'])
        self.entropies.append(result['entropy'])
        
        return result['actions'].detach().cpu().numpy()
    
    def add_rewards_masks(self, reward, done, step):
        reward = torch.from_numpy(np.expand_dims(np.array(reward), 1)).float()
        self.rewards[step].copy_(reward.to(self.device))

        masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])
        self.masks[step+1].copy_(masks.to(self.device))
        self.episode_rewards += reward
        self.final_rewards *= masks
        self.final_rewards += (1 - masks) * self.episode_rewards
        self.episode_rewards *= masks        
    
    def update(self, next_obs):
        next_obs = torch.from_numpy(next_obs).float().to(self.device)
        result = self.policy(next_obs, self.states.detach(), self.masks[-1])
        next_value = result['values'].detach()

        discounted_rewards = self.calculate_discounted_returns(next_value).squeeze(2)
        values = torch.stack(self.values).squeeze(2)
        action_log_probs = torch.stack(self.action_log_probs)
        entropies = torch.stack(self.entropies)
        # sanity check
                
        assert discounted_rewards.size() == values.size()
        assert discounted_rewards.size() == action_log_probs.size()
        assert discounted_rewards.size() == entropies.size()
        
        advantages = discounted_rewards.detach() - values
        value_loss = advantages.pow(2).mean()
        policy_loss = -(advantages.detach() * action_log_probs).mean()
        
        
        entropy_loss = entropies.mean()
        
        loss = ( policy_loss 
                + self.value_weight * value_loss 
                - self.entropy_weight * entropy_loss)
        
        self.optimizer.zero_grad()
        loss.backward()
    
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # reset buffers
        self.values = []
        self.action_log_probs = []
        self.entropies = []
        self.masks[0].copy_(self.masks[-1])
        self.states = self.states.detach()
        #self.masks = self.masks.detach()
        
        report = ("Updates {{}}, num timesteps {{}}, FPS {{}}, mean/median "
                  "reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, " 
                  "entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}").format(
                   self.final_rewards.mean(),
                   self.final_rewards.median(),
                   self.final_rewards.min(),
                   self.final_rewards.max(), entropy_loss.item(),
                   value_loss.item(), policy_loss.item())
                
        return report

    def calculate_discounted_returns(self, next_values):
        discounted_rewards = torch.zeros_like(self.rewards)
        R = next_values
        for i in reversed(range(discounted_rewards.size(0))):
            R = discounted_rewards[i] = self.rewards[i] + self.gamma*self.masks[i+1]*R
            
        return discounted_rewards
        
    def evaluate(self, test_envs, step, train_iters, num_games=128, movie_dir=''):
        self.policy.eval()
        with torch.no_grad():
            states = torch.zeros_like(self.states)
             
            games_played = 0
            obs = test_envs.reset()
            # add obs to scorer
            scorer = Scorer(self.num_parallel, obs, movie=movie_dir)
            obs = torch.from_numpy(obs).float().to(self.device)
            masks = torch.ones(self.num_parallel, 1).to(self.device)
      
            while games_played < num_games:
                result = self.policy(obs, states, masks)
                actions = result['actions'].detach().cpu().numpy()     
                states = result['states']
                obs, reward, done, info = test_envs.step(actions)
                # add obs, reward,  to scorer
                scorer.update(obs, reward, done)
        
                games_played += done.count(True) # done is true at end of a turn
    
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                masks = masks.to(self.device)
                obs = torch.from_numpy(obs).float().to(self.device)
    
        self.policy.train()
        
         # it is possible that this is larger that the total num games
        accumulated_rewards = sum(scorer.total_rewards[:num_games])
        best_obs, best_reward = scorer.best
        worst_obs, worst_reward = scorer.worst
        reward_list = scorer.total_rewards[:num_games]
        time_list = scorer.total_times[:num_games]
        scorer.clear()
        
        if movie_dir:
            write_movie(movie_dir, best_obs, step, best_reward)
            write_movie(movie_dir, worst_obs, step+1, worst_reward, best_agent=False)    
            
        mean_rewards = 'Step: {:0004}, Iter: {:000000008} Eval mean reward: {:0003.3f}'.format(step, train_iters, accumulated_rewards / num_games)
        game_times = 'Step: {:0004}, Game rewards: {}, Game times: {}'.format(step, reward_list, time_list)
        
        return mean_rewards, game_times
     
    def save_policy(self, total_num_steps, args, output_dir):
        checkpoint = {'args': args,
                      'model': self.policy.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        
        filepath = os.path.join(output_dir, 'models/')
        torch.save(checkpoint, '{}checkpoint_{:00000000010}.pth.tar'.format(filepath, total_num_steps))

    def load_model(self, checkpoint_filename):
        assert os.path.isfile(checkpoint_filename), 'The model could not be found {}'.format(checkpoint_filename)
        
        if self.device == 'cuda': # The checkpoint will try to load onto the GPU storage unless specified
            checkpoint = torch.load(checkpoint_filename)
        else:
            checkpoint = torch.load(checkpoint_filename, map_location=lambda storage, loc: storage)
            
        self.policy.load_state_dict(checkpoint['model'])   
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
    
if __name__ == '__main__':
    
    pass
    
    