#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:23:25 2019

@author: x
"""

import numpy as np
from utils import add_circles, OBSTACLE_REGION, GOAL_REGION, BasePathPlanner, viz_path
from utils import get_neighbours, get_heuristic_cost, get_x_y_from_node, get_node_from_x_y
import bisect

class Astar(BasePathPlanner):
    
    def __init__(self):
        pass
    
    def plan_path(self, grid, x_start, X, Y, X_goal, Y_goal):
        N = X*Y
        X_start, Y_start = x_start.reshape(-1).tolist()
        goal_node = get_node_from_x_y(X_goal, Y_goal, Y)
        start_node = get_node_from_x_y(X_start, Y_start, Y)
        past_cost = [N] * N
        past_cost[start_node] = 0
        est_total_cost = []
        closed_nodes = set()
        open_nodes = [start_node]
        parents = [None]*N
        transition_cost = 1
        num_iters = 0
        
        while len(open_nodes) > 0:            
            current = open_nodes.pop(0)
            x, y  = get_x_y_from_node(current, Y)
            if grid[x, y] == GOAL_REGION:
                return self._generate_path_to_goal(parents, current, Y)
            closed_nodes.add(current)
            for ngb in get_neighbours(current, X, Y, grid):                
                if ngb in closed_nodes:
                    continue
                tentative_past_cost = past_cost[current] + transition_cost
                if tentative_past_cost < past_cost[ngb]:
                    past_cost[ngb] = tentative_past_cost
                    parents[ngb] = current
                    est_total_cost_ngb = past_cost[ngb] + get_heuristic_cost(ngb, goal_node, Y)
                    i = bisect.bisect_left(est_total_cost, est_total_cost_ngb)
                    open_nodes.insert(i,ngb)
                    est_total_cost.insert(i,est_total_cost_ngb)
            num_iters += 1
        return np.array([])
        
    def _generate_path_to_goal(self, parents, current, cols):
        path = []
        while parents[current] is not None:
            x, y  = get_x_y_from_node(current, cols)
            if len(path) == 0:
                path = np.array([x, y])
            else:
                path = np.c_[np.array([x, y]), path]
            current = parents[current]
        x, y  = get_x_y_from_node(current, cols)
        path = np.c_[np.array([x, y]), path]
        return path 

if __name__ == "__main__":        
    X, Y = 150, 150
    grid = np.zeros((X,Y))
    goal_region  = (X-20, Y-20, 10)
    X_goal, Y_goal, goal_R  = goal_region
    X_start, Y_start = 0, 0
    obstacle_dims = [(30,30,20), (60,80,30)]    
    add_circles(grid, obstacle_dims, X, Y, OBSTACLE_REGION)
    add_circles(grid, [goal_region], X, Y, GOAL_REGION)    
    x_start = np.array([X_start,Y_start]).reshape(2,1)    
    agent = Astar()
    path = agent.plan_path(grid, x_start, X, Y, X_goal, Y_goal)
    viz_path(grid, path)