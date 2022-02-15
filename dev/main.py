#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 22:59:38 2019

@author: x
"""

from Astar import Astar
from RRT import RRT
from RRTstar import RRTstar
import numpy as np
from utils import add_circles, OBSTACLE_REGION, GOAL_REGION, BasePathPlanner, viz_path
import matplotlib.pyplot as plt 
import matplotlib 

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

X, Y = 150, 150
grid = np.zeros((X,Y))
goal_region  = (X-20, Y-20, 10)
X_goal, Y_goal, goal_R  = goal_region
X_start, Y_start = 0, 0
obstacle_dims = [(30,30,20), (60,80,30)]    
add_circles(grid, obstacle_dims, X, Y, OBSTACLE_REGION)
add_circles(grid, [goal_region], X, Y, GOAL_REGION)    

x_start = np.array([X_start,Y_start]).reshape(2,1)    

agents = [Astar, RRT, RRTstar]
colors = ["c","r","g"]

title_str = ""
plt.figure(1)

for color, planner in zip(colors, agents):
    agent = planner()
    path = agent.plan_path(grid, x_start, X, Y, X_goal, Y_goal)
    path_cost = -1
    if path.size != 0:
        for i in range(path.shape[1]-1):
            x = path[:,[i,i+1]]
            path_cost += np.linalg.norm(x[:,0]-x[:,1])
            x = x[[1,0],:]
            if i==path.shape[1]-2:
                plt.plot(x[0,:],x[1,:], color, label=planner.__name__) 
            else:
                plt.plot(x[0,:],x[1,:], color) 
    title_str += " %s Path cost: %0.2f "%(planner.__name__, path_cost)

plt.imshow(grid, cmap="gray")
plt.show()
plt.legend(loc="best")
plt.title(title_str)