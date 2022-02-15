#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import math
import matplotlib.pyplot as plt 
import numpy as np

EXPLORED_REGION = 1
OBSTACLE_REGION = 2
GOAL_REGION = 3
PATH = 4

class BasePathPlanner:
    
    def __init__(self):
        pass
    def plan_path(self):
        pass
    
def viz_path(grid, path):

    path_cost = -1
    plt.figure(1)
    if path.size != 0:
        for i in range(path.shape[1]-1):
            x = path[:,[i,i+1]]
            path_cost += np.linalg.norm(x[:,0]-x[:,1])
            x = x[[1,0],:]
            plt.plot(x[0,:],x[1,:],"r")
    
    plt.imshow(grid) # , cmap="gray")
    plt.show()
    plt.title("Path cost: %f"%(path_cost))


def add_circles(grid, circles_dims, X, Y, value):
    for obstacle in circles_dims:        
        obstacle_x, obstacle_y, obstacle_r = obstacle
        mask_x, mask_y = [], []        
        for x in range(obstacle_r*2):
            for y in range(obstacle_r*2):                
                x_cord, y_cord = obstacle_x + x - obstacle_r, obstacle_y + y - obstacle_r                
                is_in_obs = (x_cord-obstacle_x)**2 + (y_cord-obstacle_y)**2 <= obstacle_r**2
                is_in_grid = x_cord < X and y_cord < Y                
                if is_in_obs and is_in_grid:                    
                    mask_x.append(x_cord)
                    mask_y.append(y_cord)                    
        grid[mask_x,mask_y] = value
        
def add_rectangle(grid, rectangle_dims, X, Y, value):
    for obstacle in rectangle_dims:        
        obstacle_x, obstacle_y, obstacle_w, obstacle_h = obstacle
        mask_x, mask_y = [], []        
        for x in range(obstacle_h):
            for y in range(obstacle_w):                
                x_cord, y_cord = obstacle_x + x, obstacle_y + y
                is_in_grid = x_cord < X and y_cord < Y                
                if is_in_grid:
                    mask_x.append(x_cord)
                    mask_y.append(y_cord)                    
        grid[mask_x,mask_y] = value

def get_x_y_from_node(node, cols):
    x = node // cols
    y = node - x* cols
    return x, y

def get_node_from_x_y(x, y, N_cols):
    return x* N_cols + y

def get_neighbours(node, X, Y, grid):
    x, y = get_x_y_from_node(node, Y)
    neighbours = []
    for x_add, y_add in [(0,1),(0,-1),(1,0),(-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
        x_neigh = x_add + x
        y_neigh = y_add + y
        is_in_x_range = (x_neigh < X) and (x_neigh >= 0)
        is_in_y_range = (y_neigh < Y) and (y_neigh >= 0)
        if is_in_x_range and is_in_y_range and (grid[x_neigh, y_neigh] != OBSTACLE_REGION):
            neighbours.append(x_neigh*Y + y_neigh)
    return neighbours
    
def get_heuristic_cost(node,goal_node, cols):
    x, y = get_x_y_from_node(node, cols)
    x_g, y_g = get_x_y_from_node(goal_node, cols)
    return math.sqrt( (x-x_g)**2 + (y-y_g)**2 )

if __name__=="__main__":
    X, Y = 150, 150
    grid = np.zeros((X,Y))
    goal_region  = (X-20, Y-20, 10)
    X_goal, Y_goal, goal_R  = goal_region
    X_start, Y_start = 0, 0
    obstacle_dims = [(30,30,20), (60,80,30)]    
    add_circles(grid, obstacle_dims, X, Y, OBSTACLE_REGION)
    obstacle_dims = [(100,100, 10, 40),(100,100, 40, 10)]    
    add_rectangle(grid,obstacle_dims, X, Y, OBSTACLE_REGION)
    import PIL    
    img = PIL.Image.new("RGB",(X, Y))
    pixels = img.load()
    for x in range(X):
        for y in range(Y):
            if grid[x,y] == OBSTACLE_REGION:
                pixels[x,y] = (0,0,0)
            else:
                pixels[x,y] = (255,255,255)
    img.save("img1.ppm")
    
    
    
    
    
    
    
    