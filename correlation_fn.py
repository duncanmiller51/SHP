# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 17:07:55 2021

correlation function is for a single path, averaging over 1000 to get smooth behaviour

@author: dunca
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random
from SHP_class import MCMC


N = 1500 #number of sweeps
therm_sweeps = 100 #number of disgarded sweeps
points = 800 #number of lattice points
m = 1
w = 0.5
lamda = 1.1


tau = 6
B = 10

thermalised = MCMC.thermalisation(N,points,therm_sweeps,m,w,lamda)
path = MCMC.metropolis_sweep(thermalised,m,w,lamda)

G_list = []
m_eff = []
path_G_list = []
err_list = np.zeros((2*tau+1))
i_list = []
"""
for i in range(1,tau):
    G_list.append(MCMC.G(path,i))
    m_eff.append(MCMC.m_eff(path,i))
"""
G_array = np.zeros((N,2*tau+1))
#m_eff_array = np.zeros((N,tau+1))
#G_ac = [] #autocorrelation only

for j in range(0,N):
    for i in range(-tau,tau+1):
        G_array[j,i+tau]=MCMC.G(path,i)
        #if i>=0:
            #m_eff_array[j,i] = MCMC.m_eff(path,i)
    path = MCMC.metropolis_sweep(path,m,w,lamda)    
G_avg = G_array.mean(axis=0) #averaging over N paths
#m_eff_avg = m_eff_array.mean(axis=0)
    
for i in range(0,2*tau+1):     #taking average over many paths
    i_list.append(i-tau) #for plotting
    #G_ac.append(G_avg[i]) #autocorrelation only
#MCMC.autocor_time(G_ac) #autocorrelation only


for i in range(0,2*tau+1): #errors?
    for j in range(0,N):    
        path_G_list.append(G_array[j,i])
    #jack_var = MCMC.jackknife_var(B,path_G_list,G_avg[i])
    naive_err = MCMC.naive_err(path_G_list,G_avg[i])
    err_list[i]= naive_err * math.sqrt(10/N)
    path_G_list = []
    
#plt.plot(G_avg)
plt.errorbar(i_list[tau:2*tau],G_avg[tau:2*tau],yerr=err_list[tau:2*tau], ecolor='red')
plt.title('two point correlation function')
plt.xlabel('Delta_tau')
plt.ylabel('G(Delta tau)')
plt.show()
"""
plt.plot(m_eff_avg)
plt.title('m_eff')
plt.xlabel('Delta_tau')
plt.ylabel('m_eff')
plt.legend()
plt.show()
"""

for i in range(0,2*tau+1): #error propogation for ln plot
    err_prop = err_list[i]/G_avg[i]

plt.plot(i_list[tau:2*tau],np.log(G_avg[tau:2*tau]))
plt.errorbar(i_list[tau:2*tau],np.log(G_avg[tau:2*tau]),yerr=err_list[tau:2*tau], ecolor='red')
plt.title('correlation function ln plot')
plt.xlabel('Delta_tau')
plt.ylabel('ln(G)')
plt.show()


slope_1,cov_1 = np.polyfit(i_list[0:tau],np.log(G_avg[0:tau]),1,cov=True) #getting slope of graph
slope_2,cov_2 = np.polyfit(i_list[tau:2*tau],np.log(G_avg[tau:2*tau]),1,cov=True)

print(cov_1)

print(slope_1)
print(slope_2)
grad = (abs(slope_1)+abs(slope_2))/2
print("Gradient = ",grad[0])
print("error = ", np.sqrt(np.diag(cov_1))[0])






