#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:35:10 2018

@author: anonymous
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
from .maze_generation2 import gen_maze

from omg import WAD
from .maze_functions import create_green_armor, create_red_armor, create_line_def, create_map_point, create_vertex, create_object
from .maze_functions import create_sector, create_side_def, create_spawn, gen_random_maze, create_red_pillar, create_green_pillar


def create_maze(base_filepath, filename, size, cell_size):
    # load the base file
    BASE_WAD = 'six_item_basefile.wad'
    BASE_CFG = 'six_item_basefile.cfg'
    
    wad = WAD('scenarios/basefiles/' + BASE_WAD)
    cfg_filename = '{}{}.cfg'.format(base_filepath,filename[:-4])
    shutil.copy('scenarios/basefiles/' + BASE_CFG, cfg_filename)
    
    if '/' in filename:
        wad_filename = filename.split('/')[-1]
    else:
        wad_filename = filename
    # change the maze name in .cfg file
    # Read in the file
    with open('scenarios/basefiles/' + BASE_CFG, 'r') as file:
      filedata = file.read()
    
    # Replace the target string
    filedata = filedata.replace(BASE_WAD, wad_filename)
    
    # Write the file out again
    with open(cfg_filename, 'w') as file:
      file.write(filedata)    
    
    
    verticies = []
    wall_cons = []
    wall_idx = 0
    map_point_idx = 10
    output_list = ['// Written by generate_mino_maze', 'namespace="zdoom";']

    # create the two map points
    
    
    xmin = -size//2 * cell_size
    xmax = xmin + size*cell_size
    ymin = -size//2 * cell_size
    ymax = ymin + size*cell_size
    
    item_start = 20;
    num_items = 6
    locations = []
    
    item_tids = [2018, 2019, 2012, 2013, 13, 5]
    colors = ['g','r', 'b', 'c', 'm', 'y']
    details = {}
    
    items = []
    
    for i in range(num_items):
        
        item_i = random.randint(0, size-1)
        item_j = random.randint(0, size-1)
        
        while (item_i, item_j) in locations:
            item_i = random.randint(0, size-1)
            item_j = random.randint(0, size-1)    
        
        locations.append((item_i, item_j))
        
    
    for loc, tid, idx, col in zip(locations, 
                                item_tids, 
                                range(item_start, item_start+num_items), 
                                colors):
        #print(loc, tid, idx, col)
        item_i, item_j = loc
        
        item_x = xmin + item_i*cell_size + cell_size/2
        item_y = ymin + item_j*cell_size + cell_size/2
        
        output_list += create_object(item_x, item_y, tid, idx)
        
        plt.scatter(item_x, item_y, c=col)
        items.append((item_x, item_y, col))
        
    spawn_i = random.randint(0, size-1)
    spawn_j = random.randint(0, size-1)    
        
    while (spawn_i, spawn_j) in locations:
        #print('retry')
        spawn_i = random.randint(0, size-1)
        spawn_j = random.randint(0, size-1) 
        
    spawn_x = xmin + spawn_i*cell_size + cell_size/2
    spawn_y = ymin + spawn_j*cell_size + cell_size/2
    
    output_list += create_spawn(spawn_x, spawn_y)
    plt.scatter(spawn_x, spawn_y, c='g', marker='s')
   
    map_point_idx += 1
    
    exterior, walls = gen_maze(size, cell_size, xmin=xmin, ymin=ymin, keep_prob=6)
    
    details['items'] = items
    details['spawn'] = (spawn_x, spawn_y)
    details['exterior'] = exterior[:-1]
    details['walls'] = walls
    with open(base_filepath+filename[:-4]+'.json', 'w') as f:
        json.dump(details, f)   
        
    verticies += exterior[:-1]
    
    for k in range(4):
        wall_cons.append((wall_idx + k, wall_idx + ((k +1)%4)))    
    wall_idx += 4    
    
    pad = 8
    
    for wall in walls:
        x0,y0,x1,y1 = wall

        if x0 == x1:
            verticies += [(x0-pad, y0), (x1+pad, y0),
                      (x1+pad, y1), (x0-pad, y1)]
        else:
            verticies += [(x0, y0-pad), (x1, y0-pad),
                      (x1, y1+pad), (x0, y1+pad)]           
        
        for k in range(4):
            wall_cons.append((wall_idx + k, wall_idx + ((k +1)%4)))
        wall_idx += 4          

    
    for vx, vy in verticies:
        output_list += create_vertex(vx, vy)
    
    for id1, id2 in wall_cons:
        output_list += create_line_def(id1,id2)
        output_list += create_side_def() 
    
    output_list += create_sector()
    
    ## iterate through list to create output text file
    output_string = ''
    for output in output_list:
        output_string += output + '\n'
        
    wad.data['TEXTMAP'].data = output_string.encode()
    wad.to_file(base_filepath +filename) 
    
    
    plt.savefig(base_filepath+filename[:-4]+'.jpg')
    plt.close()
    
if __name__ == '__main__':
    
    BASE_FILEPATH = 'resources/scenarios/custom_scenarios/six_item5/test/'
    NUM_MAZES = 64
    
    for m in range(NUM_MAZES):
        filename = 'six_item_maze{:003}.wad'.format(m)
        print('creating maze', filename)
    
        create_maze(BASE_FILEPATH, filename, size=5, cell_size=160)
    
    
    
    
    
    
    
    
    
    
    
    
    
    