#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 09:55:44 2019

@author: edward
"""
from math import isclose
import itertools
import random
from vizdoom import DoomGame, ScreenResolution, GameVariable, Button, AutomapMode, Mode, doom_fixed_to_double
import numpy as np
from cv2 import resize
import cv2
import math

class DoomEnvironment():
    """
        A wrapper class for the Doom Maze Environment
    """
    class PlayerInfo():
        """
            Small class to hold player position info etc
        
        """
        def __init__(self, x, y, theta):
            self.x = x
            self.y = y
            self.theta = theta # in radians
            self.starting_theta = theta
                
            self.dx, self.dy, self.dtheta = 0.0, 0.0, 0.0
            self.vx, self.vy, self.dv = 0.0, 0.0, 0.0
            self.origin_x = x
            self.origin_y = y
            
        def update(self, x, y, theta):         
            # recording of agents position and rotation during a rollout
            # We do some calculations in the agents reference frame which are not relavant 
            # for the moment but may be useful for future work
            self.dtheta = theta - self.theta
            self.theta = theta
            
            # the calculations below will fail if the agent has not moved
            if x == self.x and y == self.y:
                self.dx = 0
                self.dy = 0
                return
                
            # dx and dy are all in the agents current frame of reference
            world_dx = self.x - x   # swapped due to mismatch in world coord frame             
            world_dy = y - self.y
            
            # the hypotenus of the triangle between the agents previous and current position
            
            h = math.sqrt(world_dx**2 + world_dy**2)
            theta_tilda = math.atan2(world_dy, world_dx)
            theta_prime = math.pi - theta_tilda - theta
            # theta_prime = theta - theta_tilda this should be correct but the coordinate system in Doom in inverted
            
            self.dx = h*math.sin(theta_prime)
            self.dy = h*math.cos(theta_prime)
            # changes in x and y are all relative
            self.x = x
            self.y = y
            self.theta = theta
            
    
    def __init__(self, params, idx=0, is_train=True, get_extra_info=False, use_shaping=False, fixed_scenario=False):
        
        self.fixed_scenario = fixed_scenario
        self.is_train = is_train
        self.use_shaping = use_shaping
        self.game = self._create_game(params, idx, is_train, get_extra_info)
        self.screen_width = params.screen_width
        self.screen_height = params.screen_height       
        self.params = params

        self.resize = params.resize
        self.frame_skip = params.frame_skip
        #self.norm_obs = params.norm_obs
        
        self.action_map = self._gen_actions(self.game, params.limit_actions)
        self.num_actions = len(self.action_map)
        params.num_actions = self.num_actions
        self.player_info = self.PlayerInfo(
                self.game.get_game_variable(GameVariable.POSITION_X),
                self.game.get_game_variable(GameVariable.POSITION_Y),
                math.radians(self.game.get_game_variable(GameVariable.ANGLE)))
    
    def _create_game(self, params, idx, is_train, get_extra_info=False):
        game = DoomGame()

        VALID_SCENARIOS = ['my_way_home.cfg',
                           'health_gathering.cfg',
                           'health_gathering_supreme.cfg',
                           'health_gathering_supreme_no_death_penalty.cfg',
                           'deadly_corridor.cfg',
                           'defend_the_center.cfg',
                           'defend_the_line.cfg',  
                           'two_color_maze014.cfg',
                           'labyrinth_maze000.cfg',
                           'labyrinth_maze11_000.cfg']
 
        
        VALID_MULTI_SCENARIOS = ['maze_{:003}.cfg',
                                 'custom_scenario{:003}.cfg'
                                 'mino_maze{:003}.cfg',
                                 'labyrinth_maze{:003}.cfg',
                                 'two_item_maze{:003}.cfg',
                                 'six_item_maze{:003}.cfg',
                                 'four_item_maze{:003}.cfg',
                                 'eight_item_maze{:003}.cfg',
                                 'repeated_laby_maze{:003}.cfg',
                                 'two_color_maze{:003}.cfg',
                                 'custom_scenario{:003}.cfg']
        

        if params.scenario in VALID_SCENARIOS:
            game.load_config(params.scenario_dir + params.scenario)
        elif params.scenario in VALID_MULTI_SCENARIOS:
            assert params.multimaze
            if not is_train and params.test_scenario_dir:
                filename = params.test_scenario_dir + params.scenario.format(idx)
                #print('loading file', filename)
                game.load_config(filename)
            else:    
                if not is_train:
                    print('WARNING, LOADING TRAINING DATA FOR TESTING, THIS MAY NOT BE WHAT YOU INTENDED!')
                filename = params.scenario_dir + params.scenario.format(idx)
                #print('loading file', filename)
                game.load_config(filename)        
        else:
            assert 0 , 'Invalid environment {}'.format(params.scenario)
            
        if params.screen_size == '320X180':
            # TODO: Implement options for other resolutions
            game.set_screen_resolution(ScreenResolution.RES_320X180)
        else:
            assert 0 , 'Invalid screen_size {}'.format(params.screen_size)
        

        game.set_sound_enabled(False)
        #game.add_game_args("+vid_forcesurface 1")
        game.set_window_visible(params.show_window)

        if params.show_window:
            game.set_mode(Mode.SPECTATOR)
            game.add_game_args("+freelook 1")
        
        # Player variables for prediction of position etc
        game.add_available_game_variable(GameVariable.POSITION_X)
        game.add_available_game_variable(GameVariable.POSITION_Y)
        game.add_available_game_variable(GameVariable.POSITION_Z)        
        game.add_available_game_variable(GameVariable.VELOCITY_X)
        game.add_available_game_variable(GameVariable.VELOCITY_Y)
        game.add_available_game_variable(GameVariable.VELOCITY_Z)  
        game.add_available_game_variable(GameVariable.ANGLE)       
        game.add_available_game_variable(GameVariable.PITCH)       
        game.add_available_game_variable(GameVariable.ROLL)       
        
        if get_extra_info:
            game.set_labels_buffer_enabled(True)
            game.set_automap_buffer_enabled(True)
            game.set_automap_mode(AutomapMode.OBJECTS)
            game.set_automap_rotate(True)
            game.set_automap_render_textures(False)
            game.set_depth_buffer_enabled(True)
        
        
        game.init() 
        
        if GameVariable.HEALTH in game.get_available_game_variables():
            self.previous_health = game.get_game_variable(GameVariable.HEALTH) 
            
        if self.use_shaping:
            self.shaping_reward = doom_fixed_to_double(game.get_game_variable(GameVariable.USER1))
            
        if params.disable_head_bob:
            game.send_game_command('movebob 0.0')
            
        return game
    
    def _gen_actions(self, game, limit_action_space):
        buttons = game.get_available_buttons()
        if buttons == [Button.TURN_LEFT, Button.TURN_RIGHT, Button.MOVE_FORWARD, Button.MOVE_BACKWARD]:
            if limit_action_space:
                feasible_actions = [[True, False, False,  False], # Left
                                    [False, True, False, False],  # Right
                                    [False, False, True, False],  # Forward
                                    [True, False, True, False],   # Left + Forward
                                    [False, True, True, False]]   # Right + forward
            else:
                feasible_actions = [[True, False, False,  False], # Left
                                    [False, True, False, False],  # Right
                                    [False, False, True, False],  # Forward
                                    [False, False, False, True],  # Backward
                                    [True, False, True, False],   # Left + Forward
                                    [True, False, False, True],   # Left + Backward
                                    [False, True, True, False],   # Right + forward
                                    [False, True, False, True]]   # Right + backward  
                
        else:
            feasible_actions = [list(l) for l in itertools.product([True, False], repeat=len(buttons))]
            print('Size of action space:', len(feasible_actions), self.params.num_actions)
            message = 'Missmatch between size of actions and num_actions, try added num_actions={} to command line args'.format(len(feasible_actions))
            assert len(feasible_actions) == self.params.num_actions, message
            
        action_map = {i: act for i, act in enumerate(feasible_actions)}
        return action_map

    
    def reset(self):
        # if we are not in the efficent(but memory consuming) fixed scenario mode
        # we change the scenario with a 1/10 probability
        if ( self.params.multimaze and 
             not self.fixed_scenario and 
             random.randrange(0,10) == 0 ):

             if self.is_train:
                 idx = random.randrange(0, self.params.num_mazes_train)
                 #print('Creating new train maze with idx={}'.format(idx))
                 self.game = self._create_game(self.params, idx, self.is_train) 
             else:
                 idx = random.randrange(0, self.params.num_mazes_test) 
                 #print('Creating new test maze with idx={}'.format(idx))
                 self.game = self._create_game(self.params, idx, self.is_train) 
            
        self.game.new_episode()
        
        self.player_info = self.PlayerInfo(
            self.game.get_game_variable(GameVariable.POSITION_X),
            self.game.get_game_variable(GameVariable.POSITION_Y),
            math.radians(self.game.get_game_variable(GameVariable.ANGLE)))
        
        if GameVariable.HEALTH in self.game.get_available_game_variables():
            self.previous_health = self.game.get_game_variable(GameVariable.HEALTH)
            
        if self.use_shaping:
            self.shaping_reward = doom_fixed_to_double(self.game.get_game_variable(GameVariable.USER1))       
        
        return self.get_observation()
    
    def is_episode_finished(self):
        return self.game.is_episode_finished()
    
    def get_observation(self):
        state = self.game.get_state()
        observation = state.screen_buffer

        if self.resize:
            # TODO perhaps this can be sped up further
            # cv2 resize is 10x faster than skimage 1.37 ms -> 126 us 
            observation = resize(
                    observation.transpose(1,2,0), 
                    (self.screen_width, self.screen_height), cv2.INTER_AREA
                 ).transpose(2,0,1)
        return observation

    def make_action(self, action):
        """
            perform an action, includes an option to skip frames but repeat
            the same action.
            
        """     
        reward = self.game.make_action(self.action_map[action], self.frame_skip)
        
        # We shape rewards in health gathering to encourage collection of health packs
        if not self.use_shaping and self.is_train: 
            reward += self._check_health()
        
        # alternatively ViZDoom offers a shaping reward in some scenarios
        if self.use_shaping and self.is_train:
            current_shaping_reward = doom_fixed_to_double(self.game.get_game_variable(GameVariable.USER1))
            diff = current_shaping_reward - self.shaping_reward
            reward += diff

            self.shaping_reward += diff
            
        return reward
    
    def step(self, action):
        reward = self.make_action(action)
        done = self.is_episode_finished()
        if done:
            obs = self.reset()
        else:
            new_x = self.game.get_game_variable(GameVariable.POSITION_X)
            new_y = self.game.get_game_variable(GameVariable.POSITION_Y)
            new_theta = self.game.get_game_variable(GameVariable.ANGLE)
            self.player_info.update(new_x, new_y, math.radians(new_theta))
            
            obs = self.get_observation()
            
        return obs, reward, done, None
    
    
    def _check_health(self):
        """
            Modification to reward function in order to reward the act of finding a health pack
        
        """
        health_reward = 0.0
    
        if GameVariable.HEALTH not in self.game.get_available_game_variables():
            self.previous_health = self.game.get_game_variable(GameVariable.HEALTH)
            return health_reward
        
        if self.game.get_game_variable(GameVariable.HEALTH) > self.previous_health:
            #print('found healthkit')
            health_reward = 1.0
            
        self.previous_health = self.game.get_game_variable(GameVariable.HEALTH)
        return health_reward
    
    def get_total_reward(self):
        return self.game.get_total_reward()
    
    def get_player_position(self):
        return self.player_info.x, self.player_info.y, self.player_info.theta

    def get_player_deltas(self):
        return self.player_info.dx, self.player_info.dy, self.player_info.dtheta

    def get_player_origins(self):
        return self.player_info.origin_x, self.player_info.origin_y
    
    def get_player_pos_delta_origin(self):
        return (self.player_info.x, self.player_info.y, self.player_info.theta,
                self.player_info.dx, self.player_info.dy, self.player_info.dtheta,
                self.player_info.origin_x, self.player_info.origin_y)

if __name__  == '__main__':
    import matplotlib.pyplot as plt      
    import random
    from arguments import parse_game_args 
    params = parse_game_args()
    env = DoomEnvironment(params)
    
    


