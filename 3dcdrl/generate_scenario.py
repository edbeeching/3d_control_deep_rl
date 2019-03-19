#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 13:27:56 2018

@author: anonymous

"""
import argparse

from scenario_generation import generate_find_return
from scenario_generation import generate_two_item
from scenario_generation import generate_four_item
from scenario_generation import generate_six_item
from scenario_generation import generate_eight_item
from scenario_generation import generate_labyrinth
from scenario_generation import generate_two_color


def parse_args():    
    parser= argparse.ArgumentParser()    
    parser.add_argument('--scenario', type=str, default='labyrinth', help='the scenario to generate')
    parser.add_argument('--size', type=int, default=5, help='the size of the scenario')
    parser.add_argument('--difficulty', type=int, default=0, help='the difficulty, used for color scenario and k item')
    parser.add_argument('--num_train', type=int, default=1, help='number of training configurations')
    parser.add_argument('--num_test', type=int, default=0, help='number of testing confurations')
    parser.add_argument('--grid_size', type=float, default=128, help='the distrance between cells in the scenario')
    parser.add_argument('--scenario_dir', type=str, default='scenarios/custom_scenarios/examples/', help='the distrance between cells in the scenario')
    
    args = parser.parse_args()    
    return args


def gen_scenarios(func, args, kwargs={}):
    base_filepath = args.scenario_dir
    
    for stage, num in zip(['train/', 'test/'],
                          [args.num_train, args.num_test]):
        for m in range(num):
            filename = 'custom_scenario{:003}.wad'.format(m)
            func.create_maze(base_filepath + stage,
                             filename, 
                             size=args.size,
                             cell_size=args.grid_size, 
                             **kwargs)

    
def main():
    args = parse_args()
    
    message = """Generating {} training and {} testing scenarios in {}
                 for scenario {} with size {}, difficulty {} and grid size {}""".format(
                 args.num_train, args.num_test, args.scenario_dir, 
                 args.scenario,args.size, args.difficulty, args.grid_size)
                 
    print(message)
    
    kwargs = {}
    if args.scenario == 'k_item':
        difficulty_map = {2: generate_two_item,
                          4: generate_four_item,
                          6: generate_six_item,
                          8: generate_eight_item}
        
        if args.difficulty not in difficulty_map.keys():
            message = """The difficulty {} is not available yet for the k-item 
                         scenario, please raise an issue on our GitHub if you 
                         would like help implementing this scenario""".format(
                         args.difficulty)
            assert 0, message
        scenario_generator = difficulty_map[args.difficulty]
        
    elif args.scenario == 'find_return':
        scenario_generator = generate_find_return
        
    elif args.scenario == 'labyrinth':
        scenario_generator= generate_labyrinth
        
    elif args.scenario == 'two_color':
        scenario_generator = generate_two_color
        kwargs['keep_prob'] = args.difficulty
        
    else:
        message = """The scenario {} is not available yet., please raise an 
                     issue on our GitHub if you would like help implementing 
                     this scenario""".format(args.scenario)     
        assert 0, message   
    
    gen_scenarios(scenario_generator, args, kwargs=kwargs)
    




if __name__ == '__main__':    
    main()
    

