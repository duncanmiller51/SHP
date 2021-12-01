# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 12:02:01 2021


@author: Duncan Miller s1844695
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random

class MCMC(object):
    
    def metropolis_sweep(path,m,w,lamda): #single sweep path update, path(i+1) from path(i)
        """
        single sweep path update, path(i+1) from path(i)
        
        input: path - 1D array containing path values
        output: path - updates path
        
        delta_x can be positive or negative for balance
        if update rejected, value stays the same
        """
        points = len(path)
        c = 0.8 #used in accept/reject
        accept = 0 #used to check acceptance ratio

        for i in range(0,np.size(path)):
            x = path[i]
            delta_x = random.uniform(-c, c)
            r = random.uniform(0,1)
            x_new = x + delta_x
            
            #harmonic
            
            s_old = 0.5*m*((path[(i+1)%points]-path[i])**2 +(path[i]-path[(i-1)%points])**2 + (w**2)*path[i]**2)
            s_new = 0.5*m*((path[(i+1)%points]-x_new)**2 +(x_new-path[(i-1)%points])**2 + (w**2)*x_new**2)
            """
            #anharmonic
            s_old = 0.5*m*((path[(i+1)%points]-path[i])**2 +(path[i]-path[(i-1)%points])**2) + lamda *(path[i]**2 -2)**2
            s_new = 0.5*m*((path[(i+1)%points]-x_new)**2 +(x_new-path[(i-1)%points])**2) + lamda *(x_new**2 -2)**2
            """
            Delta_S = s_new-s_old
            if Delta_S < 0 or r<math.exp(-Delta_S): #accept
                path[i] = x_new
                accept += 1
            #reject => value stays the same
        #print(accept/(points)) #check acceptance rate
        return path
    
    
    def thermalisation(N,points,therm_sweeps,m,w,lamda):
        """
        creates an array of random numbers (hot start) or zeros (cold start)
        runs metropolis for therm_sweeps and returns a path in equilibrium
        initial number therm_sweeps are discarded
        ocu
        N = number of sweeps, used for convergence plotting
        points = length of path
        therm_sweeps = no. of paths to discard        
        """
        initial = np.random.uniform(-1,1,points) #hot start
        #initial = np.ones(points) #cold start
        
        for i in range(0,therm_sweeps):
            initial = MCMC.metropolis_sweep(initial,m,w,lamda)
        return initial
  
    def obs(x):
        return x
    def s_obs(x,x_old,m,w,lamda):
        return 1/2*(m*((x - x_old)**2) + m*(w*x_old)**2) - 1/4 *lamda*x_old**4
    
           
    def obs_calc_path(path):
        """
        returns the average observable (set in MCMC.obs above) for a single path
        
        path: array
        """
        sum_obs_value = 0
        length = np.size(path)
        for i in range(0,length):
            obs_value = MCMC.obs(path[i]) #calculating obs(x)
            #obs_value = MCMC.s_obs(path[i],path[(i-1)%length],m,w) #for S observable
            sum_obs_value += obs_value
        return sum_obs_value/length
       
        
    def naive_err(path_obs_list,expect_obs):
        """
        calculates the error assuming independent points
        
        path_obs_list: list containing the average observable from each sweep
        expect_obs: the expectation value of an observable
        """
        sum_error = 0
        length = np.size(path_obs_list)
        for i in range(0,length):
            obs_value = MCMC.obs(path_obs_list[i])
            sum_error += (obs_value - expect_obs)**2
        return math.sqrt(sum_error/((length-1)*length))    
       
    def o_k(B,path_obs_list,k):
        """
        calculates B*o_k (eqn 57 westbroek)
        
        B:bin width as integer number of path updates
        path_obs_list: list containing average observable for each path
        k: integer (1,...,N_B) where N_B is number of blocks
        """
        o_k_sum = 0
        for i in range(0,B):
            o_k_sum += path_obs_list[(k-1)*B +i]
        o_k_sum = o_k_sum
        return o_k_sum
    
    def o_k_tilda(B,path_obs_list,k):
        N = len(path_obs_list)
        o_k_tilda_sum = 0
        o_k = MCMC.o_k(B,path_obs_list,k)
        for i in range(0,N):          
            o_k_tilda_sum += (path_obs_list[i])
        return 1/(N-B)*(o_k_tilda_sum - o_k) #o_k function is B*o_k   
         
    def jackknife_var(B,path_obs_list,expect_obs):
        """
        B: bin size
        path_obs_list: array containing the calculated observable for each path
        expect_obs = average observable over all paths      
        """
        N_B = int(len(path_obs_list)/B) #must be integer, floor
        jack_var_sum = 0
        for k in range(0,N_B):
            o_k_tilda = MCMC.o_k_tilda(B,path_obs_list,k)
            jack_var_sum += (o_k_tilda - expect_obs)**2
        jack_var = (N_B-1)/N_B * jack_var_sum
        return math.sqrt(jack_var)
    
    
    
    def E_calc(t_mc, path_obs_list,i,f):
        """
        t_mc: index of number of paths
        i: lower sum value
        f: upper sum value
        """
        #calculate E(O_i)
        N = len(path_obs_list)
        E_sum = 0
        for i in range(i,f):
            E_sum += path_obs_list[i]
        return E_sum/(N-t_mc)
        
    def autocor_fn(t_mc, path_obs_list,N): 
        """
        returns A_O(t_mc), the autocorrelation function for path t_mc
        note: unnormalised
        """
        
        a_sum = 0
        for i in range(0,(N-t_mc)):
            a_sum += (path_obs_list[i] - MCMC.E_calc(t_mc,path_obs_list,0,(N-t_mc)))*(path_obs_list[i+t_mc] - MCMC.E_calc(t_mc,path_obs_list,(N-t_mc),N))
        a_sum = (1/(N-t_mc-1))*a_sum  #when not needing to plot this can be streamlined by multiplying end sum instead of comps
        return a_sum
            
    def autocor_time(path_obs_list):
        """
        a_sum: 
        """
        N = len(path_obs_list)
        a_list = []
        a_0 = MCMC.autocor_fn(0, path_obs_list,N)
        #a_list.append(a_0)
        a_sum = 0
        
        
        for i in range(0,50): #N-1 as N-t_mc = 0 for final value, up to 40 to cut noise
            a_i = MCMC.autocor_fn(i, path_obs_list,N)/a_0
            #if a_i < 0:
            #    print("break, i = ",i)
            #    break
            #else:
            a_list.append(a_i)  #for plotting   
            a_sum += a_i
       
        ac_time = 0.5 + a_sum
        print("Autocorrelation time (integrated) = ",ac_time)
        
        #plotting autocorrelation function
        plt.plot(a_list)
        #plt.title('Autocorrelation function vs path number')
        plt.xlabel('t_mc')
        plt.ylabel('A(t_mc)/A(0)')
        plt.legend()
        plt.show()
        
        return ac_time

        
    def G(path,Delta_tau):
        """
        two point correlation function
        returns value for G
        """
        length = np.size(path)
        G_ = 0
        

        
        for i in range(0,length):
            G_ += path[i]*path[(i+Delta_tau)%length]
        G = 1/length * G_
        return G
        
          
    def m_eff(path,Delta_tau):
        """
        Calculates 1/xi == mass_eff
        """       
        m_eff = 1/2* math.log(MCMC.G(path,Delta_tau-1)/MCMC.G(path,Delta_tau+1))       
        return m_eff
        
        
    def psi(m,w,points,dt): #not working?
        psi = ((m*w/math.pi)**0.5)*math.exp(-m*w*dt**2)
        return psi
        
    def expect_val_50(path_obs_list):
        """
        plotting expectation value of every path in path_obs_list, used in anharmonic
        """
        i_list = []
        N = len(path_obs_list)

        for i in range(0,int(N)):
            i_list.append(i)
        
        plt.plot(i_list, path_obs_list)
        #plt.title('single path plot')
        plt.xlabel('dt')
        plt.ylabel('x')
        plt.show()
        
    def chi_squared(exp,theory,error):
        """
        calculates chi squared for an experimental (array) and theoretical (list)
        values. assuming constant error.
        """
        chi_squared = 0
        print('exp = ',len(exp))
        print('theory = ',len(theory))

        for i in range(0,len(theory)):
            chi_squared_ = (exp[i]-theory[i])**2
            chi_squared += chi_squared_/((error**2)*len(theory))
        return chi_squared
        
        
        
        
        
            