# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 12:29:43 2021

calculates the autocorrelation time, averaged over l simulations.
@author: Duncan Miller s1844695
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random
from SHP_class import MCMC

N = 2000 #number of sweeps
therm_sweeps = 100 #number of disgarded sweeps
points = 150 #number of lattice points
m = 1
w = 1
lamda = 0.4


ac_avg = 0
err_list = []
l = 10
var = 0

for z in range(0,l):
    thermalised = MCMC.thermalisation(N,points,therm_sweeps,m,w,lamda)
    path_i = thermalised #set initial path
    avg_obs_sum = 0
    path_obs_list = []
    
    for i in range(0,N): #getting average of observable, loop for all paths
        path_i = MCMC.metropolis_sweep(path_i,m,w,lamda) #single sweep path update
        if i%1==0:
            avg_obs_path_i = MCMC.obs_calc_path(path_i) 
            path_obs_list.append(avg_obs_path_i)                                                #array containing the calculated observable for each dt
            avg_obs_sum += avg_obs_path_i
    expect_obs = avg_obs_sum/N
    result = MCMC.autocor_time(path_obs_list)
    err_list.append(result)
    ac_avg += result

ac_avg = ac_avg/l
print(ac_avg)

for y in range(0,l):
    var += (err_list[y]-ac_avg)**2
    
#MCMC.expect_val_50(path_obs_list)


err = math.sqrt(var/l) 
print('error = ',err)
     
