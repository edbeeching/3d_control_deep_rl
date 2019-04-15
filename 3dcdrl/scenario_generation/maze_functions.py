#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 15:05:08 2018

@author: edward
"""
import random
from PIL import Image
import numpy as np


# Global Variables to keep track on indices, I should make this a class
thing_count, line_count, side_count, vertex_count, sector_count = 0,0,0,0,0
def create_vertex(x, y):
    global vertex_count
    output = 'vertex//#{} {{ x={:0.3f}; y={:0.3f}; }}'.format(vertex_count, x, y).split(' ')
    vertex_count += 1
    return output


  
    
def create_sector(tex_floor='FLAT1_1', tex_ceil='MFLR8_1', height_ceil=128):
    global sector_count
    output = 'sector//#{} {{ texturefloor="{}"; textureceiling="{}"; heightceiling={}; }}'.format(sector_count, tex_floor, tex_ceil, height_ceil).split(' ')
    sector_count += 1
    return output
    
def create_sector_toxic(tex_floor='NUKAGE1', tex_ceil='CEIL4_1', height_ceil=128):
    global sector_count
    output = 'sector//#{} {{ texturefloor="{}"; textureceiling="{}"; heightceiling={}; special=83; }}'.format(sector_count, tex_floor, tex_ceil, height_ceil).split(' ')
    sector_count += 1
    return output
    
def create_line_def(id1, id2):
    global line_count
    output = 'linedef//#{} {{ v1={}; v2={}; sidefront=1; blocking=true; }}'.format(line_count, id1, id2).split(' ')
    line_count += 1
    return output
    
def create_side_def(sector=0, tex='STONE2'):
    global side_count
    output = 'sidedef//#{} {{ sector={}; texturemiddle="{}"; }}'.format(side_count, sector, tex).split(' ')
    side_count += 1
    return output
    
def create_spawn(x,y):
    # requires two input coordinates
    global thing_count    
    output = 'thing//#{} {{ x={:0.3f}; y={:0.3f};'.format(thing_count, x,y).split(' ')
    output += 'type=1; coop=true; dm=true; single=true; skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; }'.split(' ')    
    thing_count +=1
    return output

def create_armour(x, y):
    global thing_count
    output = 'thing//#{} {{ x={:0.3f}; y={:0.3f};'.format(thing_count, x,y).split(' ')
    output += 'type=2018; id=222; coop=true; dm=true; single=true; skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; }'.split(' ')     
    thing_count += 1
    return output

def create_red_armor(x, y, invisible=False):
    global thing_count
    output = 'thing//#{} {{ x={:0.3f}; y={:0.3f};'.format(thing_count, x,y).split(' ')
    if invisible:
        output += 'type=2019; id=20; coop=true; dm=true; single=true; skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; invisible=true; }'.split(' ')         
    else:
        output += 'type=2019; id=20; coop=true; dm=true; single=true; skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; }'.split(' ')     
    thing_count += 1
    return output

def create_green_armor(x, y, invisible=False):
    global thing_count
    output = 'thing//#{} {{ x={:0.3f}; y={:0.3f};'.format(thing_count, x,y).split(' ')
    if invisible:
        output += 'type=2018; id=21; coop=true; dm=true; single=true; skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; invisible=true; }'.split(' ')     
    else:
        output += 'type=2018; id=21; coop=true; dm=true; single=true; skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; }'.split(' ')     
    thing_count += 1
    return output

def create_green_pillar(x, y):
    global thing_count
    output = 'thing//#{} {{ x={:0.3f}; y={:0.3f};'.format(thing_count, x,y).split(' ')
    output += 'type=30; id=31; coop=true; dm=true; single=true; skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; }'.split(' ')     
    thing_count += 1
    return output

def create_red_pillar(x, y):
    global thing_count
    output = 'thing//#{} {{ x={:0.3f}; y={:0.3f};'.format(thing_count, x,y).split(' ')
    output += 'type=32; id=30; coop=true; dm=true; single=true; skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; }'.split(' ')     
    thing_count += 1
    return output

def create_object(x, y, tid, idx):
    global thing_count
    output = 'thing//#{} {{ x={:0.3f}; y={:0.3f};'.format(thing_count, x,y).split(' ')
    output += 'type={}; id={}; coop=true; dm=true; single=true; skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; }}'.format(tid, idx).split(' ')     
    thing_count += 1
    return output
    
def create_map_point(x, y, idx=10):
    global thing_count
    output = 'thing//#{} {{ x={:0.3f}; y={:0.3f}; type=9001; id={};'.format(thing_count, x,y, idx).split(' ')
    output += 'coop=true; dm=true; single=true; skill1=true; skill2=true; skill3=true; skill4=true; skill5=true; }'.split(' ')     
    thing_count += 1
    return output 

def gen_random_maze(height, width):

    # Maze generator. source http://code.activestate.com/recipes/578356-random-maze-generator/

    mx = width; my = height # width and height of the maze
    maze = [[0 for x in range(mx)] for y in range(my)]
    dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
    # start the maze from a random cell
    cx = random.randint(0, mx - 1); cy = random.randint(0, my - 1)
    maze[cy][cx] = 1; stack = [(cx, cy, 0)] # stack element: (x, y, direction)
    
    while len(stack) > 0:
        (cx, cy, cd) = stack[-1]
        # to prevent zigzags:
        # if changed direction in the last move then cannot change again
        if len(stack) > 2:
            if cd != stack[-2][2]: dirRange = [cd]
            else: dirRange = range(4)
        else: dirRange = range(4)
    
        # find a new cell to add
        nlst = [] # list of available neighbors
        for i in dirRange:
            nx = cx + dx[i]; ny = cy + dy[i]
            if nx >= 0 and nx < mx and ny >= 0 and ny < my:
                if maze[ny][nx] == 0:
                    ctr = 0 # of occupied neighbors must be 1
                    for j in range(4):
                        ex = nx + dx[j]; ey = ny + dy[j]
                        if ex >= 0 and ex < mx and ey >= 0 and ey < my:
                            if maze[ey][ex] == 1: ctr += 1
                    if ctr == 1: nlst.append(i)
    
        # if 1 or more neighbors available then randomly select one and move
        if len(nlst) > 0:
            ir = nlst[random.randint(0, len(nlst) - 1)]
            cx += dx[ir]; cy += dy[ir]; maze[cy][cx] = 1
            stack.append((cx, cy, ir))
        else: stack.pop()
    maze_output = np.zeros((height, width))    
    # paint the maze
    for ky in range(height):
        for kx in range(width):
            maze_output[kx, ky] = maze[ky][kx]
    
    return maze_output

