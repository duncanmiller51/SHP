# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 12:29:43 2021


true values for m = w = 1:
    <x^2> = 0.44721
    <x^4> = 0.60


@author: Duncan Miller s1844695
"""

import math
import numpy as np
from scipy.stats import chisquare
import matplotlib.pyplot as plt
from SHP_class import MCMC

N = 3000 #number of sweeps
therm_sweeps = 100 #number of disgarded sweeps
points = 150 #number of lattice points
m = 1
w = 1
lamda = 0.3
"""
N = 5000 #number of sweeps
therm_sweeps = 100 #number of disgarded sweeps
points = 150 #number of lattice points
m = 1
w = 0.5
lamda = 0
"""

thermalised = MCMC.thermalisation(N,points,therm_sweeps,m,w,lamda)
path_i = thermalised #set initial path
avg_obs_sum = 0
"""
plt.plot(path_i)
plt.title('thermalised plot')
plt.xlabel('dt')
plt.ylabel('x')
plt.show()
"""

path_obs_list = []
hist_list = []
p = 1 # measure every p paths


for i in range(0,N): #getting average of observable, loop for all paths
    path_i = MCMC.metropolis_sweep(path_i,m,w,lamda) #single sweep path update
    for j in range(0,len(path_i)):
        hist_list.append(path_i[j])
    if i%p == 0:
        avg_obs_path_i = MCMC.obs_calc_path(path_i) 
        path_obs_list.append(avg_obs_path_i)  #array containing the calculated observable for each dt
        avg_obs_sum += avg_obs_path_i
    """
    plt.plot(path_i)
    plt.title('single path plot')
    plt.xlabel('dt')
    plt.ylabel('x')
    plt.legend()
    plt.show()
    """ 
expect_obs = avg_obs_sum/N #remember to adjust for freq of measurements
print("expected observable:",expect_obs)

#print(expect_obs)
#print(path_obs_list)



#error analysis
B = 30
jack_var = MCMC.jackknife_var(B,path_obs_list,expect_obs)
print("error=",jack_var)
print("naive =", MCMC.naive_err(path_obs_list,expect_obs))

"""
#plotting psi
bins = 100
psi_list = []
max_ = max(hist_list)
min_ = min(hist_list)
i = -4 #appropriate values for harmonic pdf
i_list = []
binwidth = (max_- min_)/bins



while i < 4: 
    psi_list.append(MCMC.psi(m,w,points,i))
    i += 8/bins #smoothing, 8 as going from -4 to 4.
    i_list.append(i)
    

plt.axvline(0, color = 'y')
counts = plt.hist(hist_list, bins=np.arange(min_, max_ + binwidth, binwidth),density=True)[0]
plt.plot(i_list,psi_list, 'r')
plt.title('harmonic oscillator pdf')
plt.xlabel('x')
plt.ylabel('|psi(x)|^2')
plt.show()


chi_squared = MCMC.chi_squared(counts,psi_list,jack_var)
print(chi_squared)

MCMC.expect_val_50(path_obs_list) #anharmonic plot 
"""


     
