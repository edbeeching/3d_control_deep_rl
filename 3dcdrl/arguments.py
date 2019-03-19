#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 09:55:58 2019

@author: edward
"""

import argparse
def parse_a2c_args():
    """ Defines the arguments used for both training and testing the network"""
    
    parser = argparse.ArgumentParser(description='Parameters')
    paa = parser.add_argument
    # =========================================================================
    #               Environment Parameters
    # =========================================================================
    paa('--simulator',          default="doom",                    help='The environment',  type=str)
    paa('--scenario',           default='health_gathering.cfg',    help='The scenario',  type=str)
    paa('--test_scenario',      default='',                        help='The scenario used for testing',  type=str)
    paa('--screen_size',        default='320X180',                 help='Size of Screen, width x height',  type=str)    
    paa('--screen_height',      default=64,                        help='Height of the screen',  type=int)
    paa('--screen_width',       default=112,                       help='Width of the screen',  type=int)
    paa('--num_environments',   default=16,                        help='the number of parallel enviroments',  type=int)
    paa('--limit_actions',      default=False,                     help='limited the size of the action space to F, L, R, F+L, F+R', action='store_true')
    paa('--scenario_dir',       default='../scenarios/',           help='location of scenarios',  type=str)    
    paa('--test_scenario_dir',  default='',                        help='location of game scenarios',  type=str)    
    paa('--show_window',        default=False,                     help='Show the game window',  type=bool)
    paa('--resize',             default=True,                      help='Use resize for decimation rather ran downsample',  type=bool)
    paa('--multimaze',          default=False,                     help='Are there multiple maze environments',  action='store_true')
    paa('--num_mazes_train',    default=16,                        help='the number of training mazes, only valid for multimaze', type=int )
    paa('--num_mazes_test',     default=16,                        help='the number of testing mazes, only valid for multimaze', type=int)
    paa('--disable_head_bob',   default=False,                     help='disable head bobbing', action='store_true')    
    paa('--use_shaping',        default=False,                     help='include shaping rewards', action='store_true')    
    paa('--fixed_scenario',     default=False,                     help='whether to stay on a fixed scenario',  action='store_true')    
    paa('--use_pipes',          default=False,                     help='use pipes instead of queue for the environment',  action='store_true')    
    paa('--num_actions',        default=0,                         help='size of action space', type=int)
    
    # =========================================================================
    #               Model Parameters
    # =========================================================================
    paa('--hidden_size',        default=128,    help='GRU hidden size', type=int) 
    paa('--reload_model',       default='',     help='directory and iter of model to load dir,iter', type=str)
    paa('--model_checkpoint',   default='',     help='the name of a specific model to evaluate, used when making videos', type=str)
    paa('--conv1_size',         default=16,     help='Number of filters in conv layer 1', type=int)
    paa('--conv2_size',         default=32,     help='Number of filters in conv layer 2', type=int)
    paa('--conv3_size',         default=16,     help='Number of filters in conv layer 3', type=int)
   
    # =========================================================================
    #               Training Parameters 
    # =========================================================================    
    paa('--learning_rate',      default=7e-4,       help='training learning rate', type=float)
    paa('--momentum',           default=0.0,        help='optimizer momentum', type=float)
    paa('--gamma',              default=0.99,       help='reward discount factor', type=float)
    paa('--frame_skip',         default=4,          help='number of frames to repeat last action', type=int)
    paa('--train_freq',         default=4,          help='how often the model is updated', type=int)
    paa('--train_report_freq',  default=100,        help='how often to report the train loss', type=int)
    paa('--max_iters',          default=5000000,    help='maximum number of training iterations', type=int)
    paa('--eval_freq',          default=1000,       help='how often the model is evaluated, in games', type=int)
    paa('--eval_games',         default=50,         help='the number of games the model is evaluated over', type=int)
    paa('--model_save_rate',    default=1000,       help='How often to save the model in iters', type=int)
    paa('--eps',                default=1e-5,       help='RMSprop optimizer epsilon (default: 1e-5)', type=float)
    paa('--alpha',              default=0.99,       help='RMSprop optimizer alpha (default: 0.99)', type=float)    
    paa('--use-gae',            default=False,      help='use generalized advantage estimation', action='store_true')
    paa('--tau',                default=0.95,       help='gae parameter (default: 0.95)', type=float)
    paa('--entropy_coef',       default=0.001,      help='entropy term coefficient (default: 0.01)', type=float)
    paa('--value_loss_coef',    default=0.5,        help='value loss coefficient (default: 0.5)', type=float)
    paa('--max_grad_norm',      default=0.5,        help='max norm of gradients (default: 0.5)', type=float)    
    paa('--num_steps',          default=128,        help='number of forward steps in A2C (default: 128)', type=int)
    paa('--num_stack',          default=1,          help='number of frames to stack (default: 1)', type=int)
    paa('--num_frames',         default=200000000,  help='total number of frames', type=int)          
    paa('--use_em_loss',        default=False,      help='Use the discrete EM loss, optimal transport for depth preds', action='store_true')
    paa('--skip_eval',          default=False,      help='skip the evaluation process', action='store_true')
    paa('--stoc_evals',         default=False,      help='evaluate stochastically', action='store_true')
    
    # =========================================================================
    #               Test Parameters 
    # =========================================================================        
    paa('--model_dir', default='', help='Location of model directory for test/train evaluation', type=str)
    
    # =========================================================================
    #               Logging Parameters 
    # =========================================================================
    paa('--out_dir',        default='/home/edward/',                  help='output directory for log files etc', type=str,)
    paa('--log_interval',   default=100,                    help='How often to log', type=int)
    paa('--job_id',         default=12345,                  help='the job queue id, useful of running on a cluster', type=int)
    paa('--test_name',      default='test_000.sh',          help='name of the test', type=str)
    paa('--use_visdom',     default=False,                  help='use visdom for live visualization', action='store_true')
    paa('--visdom_port',    default=8097,                   help='the port number visdom will use', type=int)
    paa('--visdom_ip',      default='http://10.0.0.1',      help='the ip address of the visdom server', type=str) 
    
    args = parser.parse_args()
    args.test_name = args.test_name[:-3]
    print(args)
    
    return args
          
    
if __name__ == '__main__':
    params = parse_a2c_args()
    print(params)
    print(params.action_size)
    import os
    print(os.listdir(params.scenario_dir))
    print(params.scenario)