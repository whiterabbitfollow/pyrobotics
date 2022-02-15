#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:54:27 2019

@author: x
"""

import numpy as np
from utils import add_circles, OBSTACLE_REGION, GOAL_REGION, viz_path
from RRT import RRT
import time

class RRTstar(RRT):
    
    def __init__(self, max_tree_size= 500, max_local_planner_distance=5):
        self.MAX_T_SIZE = max_tree_size
        self.LOCAL_PLANNER_MAX_DISTANCE = max_local_planner_distance
        self.NEAREST_TOL = 5
    
    def _find_coll_free_nearest(self, x_new, path_cost, T, grid):
        indx, = np.where(np.linalg.norm(T-x_new.reshape(-1,1),axis=0)<=self.NEAREST_TOL) 
        i_min, cost_min = 0, 10000000
        is_collision_free_indx = []
        for i in indx:
            x_nearest = T[:,i].reshape(-1,1)
            is_coll_free = self._plan_locally(x_nearest, x_new, T, grid) is not None
            if is_coll_free:
                is_collision_free_indx.append(i)
                transition_cost = np.linalg.norm(x_nearest-x_new)
                total_cost = path_cost[0,i] + transition_cost
                if total_cost < cost_min:
                    cost_min = total_cost
                    i_min = i
        return cost_min, i_min, is_collision_free_indx
    
    def _replan_coll_free_nearest(self, coll_free_indxs, x_new, path_cost, edges, T, grid):
        i_new = len(edges) - 1
        for i in coll_free_indxs:
            x_nearest = T[:,i].reshape(-1,1)
            transition_cost = np.linalg.norm(x_nearest-x_new)
            old_total_cost = path_cost[0,i]
            check_new_total_cost = path_cost[0, i_new] + transition_cost
            if check_new_total_cost < old_total_cost:
                path_cost[0,i] = check_new_total_cost
                edges[i] = i_new
                
    def plan_path(self, grid, x_start, X, Y, X_goal, Y_goal):
        T = np.c_[x_start]
        path_cost = np.c_[0]
        edges = [0]
        time_s = time.time()
        max_time = 10
        goal_indx = []
        while T.shape[1] < self.MAX_T_SIZE and time.time()-time_s < max_time:
            x_sample = self._get_x_sample(X_goal, Y_goal, X, Y)
            i_nearest = np.argmin(np.linalg.norm(x_sample - T, axis=0))
            x_nearest = T[:,i_nearest].reshape(-1, 1)
            x_new = self._plan_locally(x_nearest, x_sample, T, grid)            
            if x_new is None:
                continue
            cost_min, i_min, coll_free_indxs = self._find_coll_free_nearest(x_new, path_cost, T, grid)
            edges.append(i_min)
            T = np.c_[T, x_new]
            path_cost = np.c_[path_cost, cost_min]
            self._replan_coll_free_nearest(coll_free_indxs, x_new, path_cost, edges, T, grid)
            if grid[x_new[0], x_new[1]] == GOAL_REGION:
                goal_indx.append(len(edges)-1)
        if len(goal_indx) == 0:
            return []
        else:
            i = np.argmin(path_cost[0, goal_indx])
            return self._generate_path_to_goal(edges, T, i_parent_start=goal_indx[i])
    
if __name__ == "__main__":    
    np.random.seed(1)
    X, Y = 150, 150
    grid = np.zeros((X,Y))
    goal_region  = (X-20, Y-20, 10)
    X_goal, Y_goal, goal_R  = goal_region
    X_start, Y_start = 0, 0
    obstacle_dims = [(30,30,20), (60,80,30)]
    add_circles(grid, obstacle_dims, X, Y, OBSTACLE_REGION)
    add_circles(grid, [goal_region], X, Y, GOAL_REGION)
    x_start = np.array([X_start,Y_start]).reshape(2,1)
    agent = RRTstar(500)
    path = agent.plan_path(grid, x_start, X, Y, X_goal, Y_goal)
    viz_path(grid, path)