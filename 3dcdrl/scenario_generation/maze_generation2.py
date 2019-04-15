#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 14:11:25 2018

@author: edward
"""

import random
import matplotlib.pyplot as plt
import numpy as np

    
    

def gen_maze(size, cell_width, plot=True, xmin=0, ymin=0, keep_prob=5):
    range_start = 0 #(-size //2)
    range_end = size# //2
    
    walls = {}
    for i in range(range_start, range_end):
        for j in range(range_start, range_end):
            
            
            if i != range_end - 1 and random.randint(0,9) <keep_prob:
                # vertical
                start_x = i*cell_width + cell_width + xmin
                end_x = i*cell_width + cell_width + xmin
                start_y = j*cell_width +ymin
                end_y = j*cell_width + cell_width + ymin
                
                walls[(i,j,i+1,j)] = (start_x, start_y, end_x, end_y)
                #print('v',(i,j,i+1,j))
            # horizontal
            if j != range_end - 1 and random.randint(0,9) <keep_prob:
                start_x = i*cell_width  + xmin
                end_x = i*cell_width + cell_width + xmin
                start_y = j*cell_width + cell_width + ymin
                end_y = j*cell_width + cell_width + ymin
                
                walls[(i,j,i,j+1)] = (start_x, start_y, end_x, end_y)
                
           
               # print('h',(i,j,i,j+1))
            
            
            
            
    extents_x = [ range_start * cell_width + xmin, 
                  range_start * cell_width + xmin, 
                  range_end * cell_width + xmin, 
                  range_end * cell_width + xmin,
                  range_start  * cell_width + xmin]
    
    extents_y = [ range_start * cell_width + ymin, 
                  range_end * cell_width + ymin, 
                  range_end * cell_width + ymin, 
                  range_start * cell_width + ymin,
                  range_start * cell_width + ymin] 
    # if plot:
    #     plt.subplot(1,2,1)
    #     plt.plot(extents_x, extents_y, c='k')
         
    #     for indx, entry in walls.items():   
    #         x0,y0,x1,y1 = entry
            
    #         #plt.scatter([x0,x1],[y0,y1], c='r')
    #         plt.plot([x0,x1],[y0,y1], c='k')
        
       
      
    # create the neighbours dict  
    neighbours = {}
    for i in range(range_start, range_end):
        for j in range(range_start, range_end):
            neighbours[(i,j)] = [(i-1,j),
                                  (i+1, j),
                                   (i,j-1),
                                   (i,j+1)]
    
    def valid_neighbour(i,j):
        return i >= range_start and i < range_end and j >= range_start and j < range_end

    def walk(current_i, current_j):
        #print(current_i, current_j)
        visited.add((current_i, current_j))
        n = neighbours[(current_i, current_j)]
        random.shuffle(n)
        for (ni, nj) in n:
            if valid_neighbour(ni, nj) and (ni, nj) not in visited:
                if (current_i, current_j, ni, nj) in walls:
                    #print('wall1:',(current_i, current_j, ni, nj) )
                    #print(walls[(current_i, current_j, ni, nj)])
                    
                    del walls[(current_i, current_j, ni, nj)]
                if (ni, nj, current_i, current_j ) in walls:
                    #print('wall2:', (ni, nj, current_i, current_j ) )
                    #print(walls[(ni, nj, current_i, current_j )])
                    del walls[(ni, nj, current_i, current_j )]
                walk(ni, nj)
                
    visited = set()
    start_i = random.randint(range_start, range_end -1)
    start_j = random.randint(range_start, range_end -1)            
    walk(start_i, start_j)
    
            
    if plot:
        #plt.subplot(1,2,2)
        plt.plot(extents_x, extents_y, c='k')
         
        for indx, entry in walls.items():   
            x0,y0,x1,y1 = entry
            
            #plt.scatter([x0,x1],[y0,y1], c='r')
            plt.plot([x0,x1],[y0,y1], c='k')   
        
    walls = [w for w in walls.values()]
    exterior = [(x,y) for x,y in zip(extents_x, extents_y)]
    
    
    return exterior, walls
    

    
    
    
    




        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    


