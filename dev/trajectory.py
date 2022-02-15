#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:59:18 2019

@author: x
"""
import numpy as np

def trajectory_generate_via_points_cubic_spline(geo_path, Ts, Vs):
    dTs = np.diff(Ts)    
    B_j = geo_path[:,0:-1]
    Bd_j = Vs[:,0:-1]    
    B_jp1 = geo_path[:,1::]
    Bd_jp1 = Vs[:,1::]
    a0 = B_j
    a1 = Bd_j
    a2 = (3*B_jp1 - 3*B_j - 2*Bd_j*dTs - Bd_jp1*dTs)/np.power(dTs, 2)
    a3 = (2*B_j + (Bd_j + Bd_jp1)*dTs - 2*B_jp1)/np.power(dTs, 3)
    a = np.array([a0,a1,a2,a3])
    ts, confs = [], []
    for i_s in range(dTs.size):
        T_i = Ts[i_s]
        dT_i = dTs[i_s]
        t = np.arange(0, 1, 0.1) * dT_i
        a_i = a[:,:,i_s].T
        conf = a_i[:,0].reshape(-1,1) + a_i[:,1].reshape(-1,1)*t + a_i[:,2].reshape(-1,1)*np.square(t) + a_i[:,3].reshape(-1,1)*np.power(t,3)
        ts = np.r_[ts, T_i + t]
        if len(confs) ==0:
            confs = np.c_[conf]
        else:
            confs = np.c_[confs, conf]
    return ts, confs