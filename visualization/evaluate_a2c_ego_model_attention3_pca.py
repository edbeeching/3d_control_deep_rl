#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:24:42 2018

@author: edward

--num_steps 128 --log_interval 10 --eval_freq 100 --model_save_rate 1000 --eval_games 50 --num_frames 200000000 --gamma 0.95 --recurrent_policy --num_stack 1 --norm_obs --scenario_dir  resources/scenarios/ --scenario health_gathering_supreme_no_death_penalty.cfg --model_checkpoint saved_models/health_gathering_agent_checkpoint_0049154048.pth.tar

--num_steps 128 --log_interval 10 --eval_freq 100 --model_save_rate 1000 --eval_games 50 --num_frames 200000000 --gamma 0.95 --recurrent_policy --num_stack 1 --norm_obs --scenario_dir  resources/scenarios/ --scenario --scenario scenario_cw2.cfg --use_shaping --new_padding --no_reward_average --model_checkpoint ~/tmp/results/doom_rl/2018_09_24_14_41_43_628103/models/checkpoint_0094210048.pth.tar

"""
import os
from collections import deque
import joblib
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import torch
import time
from doom_a2c.arguments import parse_game_args
from doom_a2c.models import CNNPolicy, EgoMap0_Policy
from environments import DoomEnvironment
from doom_evaluation import BaseAgent

from plot_4_item import plot_4_item, plot_laby

def reshape(ego_read):
    _, c,h,w = ego_read.shape
    assert c == 16
    output = np.zeros((h*4+3, w*4+3))
    for k in range(16):
        i = k // 4
        j = k % 4
        
        output[i*h +i:(i+1)*h +i,j*w +j:(j+1)*w +j] = ego_read[0,k]
    
    for i in range(24,99,25):
        output[:,i] = 1
        output[i,:] = 1
    
    return output
        

def make_movie(agent, env, filename, params):
    
    pca = joblib.load('6item_pca_20190114.joblib')
    with torch.no_grad():
        env.reset()
        agent.reset()
        
        global obss, ego_reads, directions
        
        rewards = []
        attentions = []
        
        directions = []
        obss = []  
        actions = []
        action_probss = []
        values = []
        states = []
        ego_reads = []
        done = False
        obs = env.reset().astype(np.float32)
        
        xs = []
        ys = []
        k=0
        while not done:
            obss.append(obs)    
            state = agent.state.clone().detach().numpy()
            states.append(state)
            depths = env.get_ego_depth()
            if not params.new_padding:
                depths = depths[2:-2,2:-2]
            
            depths = torch.Tensor(depths).unsqueeze(0)
            pos_deltas_origins = env.get_player_pos_delta_origin()
            pos_deltas_origins = torch.from_numpy(np.array(pos_deltas_origins)).unsqueeze(0).float()
            action, value, action_probs = agent.get_action_value_and_probs(obs, epsilon=0.0, deterministic=True, 
                                                                           ego_depth=depths, pos_deltas_origins=pos_deltas_origins)
            pdo = env.get_player_pos_delta_origin()
            x, y, theta = pdo[0], pdo[1], pdo[2]
            xs.append(x)
            ys.append(y)
            
            # origin_x, origin_y = pdo[6], pdo[7]
            # dx, dy, dt = env.get_player_deltas()
            # print(xxs, yys, thetas, origin_x, origin_y, xxs - origin_x, yys-origin_y)
            # print(dx, dy, dt)
            xx, yy, theta = pos_deltas_origins[:,0], pos_deltas_origins[:,1], pos_deltas_origins[:,2]
            origin_x, origin_y = pos_deltas_origins[:,6], pos_deltas_origins[:,7]            
            ego_read = agent.model.ego_map.rotate_for_read(agent.ego_state, 
                                            xx,
                                            yy,
                                            theta,
                                            origin_x,
                                            origin_y).clone().data.numpy() 
            
            #ego_read_reshaped = reshape(ego_read)
            
            ego_read = ego_read.reshape(16,24*24).T
            ego_read_pca = pca.transform(ego_read).T.reshape(3,24,24)
            
            epca = []
            for i in range(3):
                
                epca.append(np.rot90(ego_read_pca[i]))
                
            ego_read_pca = np.hstack(epca)
                           
            ego_reads.append(ego_read_pca)
            attention = agent.model.ego_map.attention.detach().numpy()[0]
            attentions.append(attention)
            direction = agent.model.ego_map.weighted_positions.detach().numpy()[0]
            directions.append(direction)
            
            
            obs, reward, done, _ = env.step(action)
            obs = obs.astype(np.float32)
            
            rewards.append(reward)
            
            actions.append(actions)
            action_probss.append(action_probs)
            values.append(value) 
            k += 1
            
            # if k > 500:
            #     break       
        
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='',
                        comment='Example movie')
        writer = FFMpegWriter(fps=7.5, metadata=metadata)    
        global fig
        fig = plt.figure(figsize=(12,2))
        
        ax1 = plt.subplot2grid((2,12), (0,0), colspan=3, rowspan=2) # obs 
        ax2 = plt.subplot2grid((2,12), (0,3), colspan=6, rowspan=2) # ego_reads 
        ax3 = plt.subplot2grid((2,12), (0,9), colspan=2, rowspan=2) # ego_atts
        ax4 = plt.subplot2grid((2,12), (0,11), colspan=2, rowspan=2) # ego dir
        im1 = ax1.imshow(obss[0].transpose(1,2,0)/128.0, clim=[0.0,1.0])
        ax1.axis('off')
        im2 = ax2.imshow(ego_reads[0], clim=(-0.3,0.3), cmap='gray')
        ax2.axis('off')
        im3 = ax3.imshow(attentions[0], clim=(0,0.1), cmap='jet')
        ax3.axis('off')
        
        ax4.arrow(0,0,directions[0][1], directions[0][0])
        ax4.set_xlim([-1.0,1.0])
        ax4.set_ylim([-1.0,1.0])
        ax4.set_aspect(1.0)

        fig.tight_layout()
        #plt.savefig('figs/' + filename[:-4] + '{:003}.png'.format(0), dpi=300)
        #assert 0 
        with writer.saving(fig, filename, 100):
            for i,(o, ego_read, att) in enumerate(zip(obss, ego_reads, attentions)):
                
                im1.set_array(o.transpose(1,2,0)/128.0)
                im2.set_array(ego_read)
                im3.set_array(np.rot90(att))
                ax4.clear()
                ax4.arrow(0,0,directions[i][1], directions[i][0])
                ax4.set_xlim([-1.0,1.0])
                ax4.set_ylim([-1.0,1.0])
                
                plt.savefig('figs/' + filename[:-4] + '{:003}.png'.format(i), dpi=300)
                writer.grab_frame()  
                #plt.savefig('figs/' + filename[:-4] + '{:003}.png'.format(i), dpi=1200)
                
                

        #plt.savefig(filename[:-4]+'.pdf', dpi=1200)
        time.sleep(0.2)
        plt.close()

def evaluate_saved_model():
    params = parse_game_args()
    env = DoomEnvironment(params)

    print(env.num_actions)
    obs_shape = (3, params.screen_height, params.screen_width)

    actor_critic = EgoMap0_Policy(obs_shape[0], obs_shape, params)
        
    assert params.model_checkpoint, 'No model checkpoint found'
    assert os.path.isfile(params.model_checkpoint), 'The model could not be loaded'
    # This lambda stuff is required otherwise it will try and load on GPU
    checkpoint = torch.load(params.model_checkpoint, map_location=lambda storage, loc: storage) 
    actor_critic.load_state_dict(checkpoint['model'])
    
    agent = BaseAgent(actor_critic, params)
 
    for i in range(10):
        movie_name = 'sixitem_egomap_attention_tests_pca_{:0004}.mp4'.format(i)
        print('Creating movie {}'.format(movie_name))
        make_movie(agent, env, movie_name, params)

if __name__ == '__main__':
    evaluate_saved_model()
    # params = parse_game_args()
    # checkpoint = torch.load(params.model_checkpoint, map_location=lambda storage, loc: storage) 
    # state_dict = checkpoint['model']

    
    
    
    
    
    
    
    
    