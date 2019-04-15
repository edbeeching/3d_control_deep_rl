#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:35:10 2018

@author: anonymous
"""
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

import json


def create_maze(base_filepath, filename, size, cell_size):
    # load the base file
    BASE_WAD = 'labyrinth_basefile.wad'
    BASE_CFG = 'labyrinth_basefile.cfg'
    
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
    
    details = {}
    verticies = []
    wall_cons = []
    wall_idx = 0
    map_point_idx = 10
    output_list = ['// Written by generate_labyrinth_maze', 'namespace="zdoom";']

    # create the two map points
    
    xmin = -size//2 * cell_size
    ymin = -size//2 * cell_size
    
    green_spot_i = random.randint(0, size-1)
    green_spot_j = random.randint(0, size-1)
    
    start_spot_i = random.randint(0, size-1)
    start_spot_j = random.randint(0, size-1)
        
    
    while(start_spot_i, start_spot_j) == (green_spot_i, green_spot_j):
        start_spot_i = random.randint(0, size-1)
        start_spot_j = random.randint(0, size-1)        
    

    green_spot_x = xmin + green_spot_i*cell_size + cell_size/2
    green_spot_y = ymin + green_spot_j*cell_size + cell_size/2
    start_spot_x = xmin + start_spot_i*cell_size + cell_size/2
    start_spot_y = ymin + start_spot_j*cell_size + cell_size/2
    
    output_list += create_object(green_spot_x, green_spot_y, 2018, 21)    
    output_list += create_spawn(start_spot_x, start_spot_y)    
    
    #plt.subplot(1,2,2)
    plt.scatter(green_spot_x, green_spot_y, c='g')
    plt.scatter(start_spot_x, start_spot_y, c='k')
    
    details['start'] = (start_spot_x, start_spot_y)
    details['end'] = (green_spot_x, green_spot_y)
    
    map_point_idx += 1
    
    exterior, walls = gen_maze(size, cell_size, xmin=xmin, ymin=ymin, keep_prob=6)
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
    
    BASE_FILEPATH = 'resources/scenarios/custom_scenarios/examples/'
    
    for m in range(1):
        filename = 'labyrinth_maze{:003}.wad'.format(m)
        print('creating maze', filename)
        create_maze(BASE_FILEPATH, filename, size=9, cell_size=128)    
    
    
    # BASE_FILEPATH = 'resources/scenarios/custom_scenarios/labyrinth{}/{}/'
    # for size in [5,7,9,11,13]:
    #     for stage, num in [('train', 256), ('test', 64)]:
    #         path = BASE_FILEPATH.format(size, stage)
            
    #         for m in range(num):
    #             filename = 'labyrinth_maze{:003}.wad'.format(m)
    #             print('creating maze', filename)
    #             create_maze(path, filename, size=size, cell_size=128)
    
    
    
    
    
    
    
    
    
    
    
    
    
    