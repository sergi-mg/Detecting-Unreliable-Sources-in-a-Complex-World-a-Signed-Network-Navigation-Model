# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:33:23 2026

@author: ASergi Martínez Galindo
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numba import njit

#%%
from os.path import exists
from os import makedirs

#folder
directory="../data/time_evo/"
if not exists(directory):
    makedirs(directory)

#%%
#majority rule
N=1000
k=9
r=0.1 
N_i=1000
rule="mr"
name="time_evo_r_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
    +"_"+str(N_i)+".dat"
obs_1=np.loadtxt(directory+name)
name="time_evo_bfs_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
    +"_"+str(N_i)+".dat"
obs_2=np.loadtxt(directory+name)
name="time_evo_rw_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
    +"_"+str(N_i)+".dat"
obs_3=np.loadtxt(directory+name)

x = np.arange(N)
labels_y=["Distance","$q_{def}$","$q_{tot}$","Maximum Distance",\
          "Average Distance"]
cmap=plt.get_cmap("viridis")
for i in range(0,10,2):
    plt.figure()
    plt.errorbar(x, obs_1[:, i],yerr=obs_1[:, i+1],\
                  label="Random selection",c=cmap(0.5),\
                      linestyle="none",marker="o",ms=1)
    plt.plot(x, obs_1[:, i],c=cmap(0.6),linestyle="dotted",marker="o",ms=2)
    
    plt.errorbar(x, obs_2[:, i],yerr=obs_2[:, i+1],\
                  label="BFS",c=cmap(0.8),linestyle="none",marker="o",ms=1)
    plt.plot(x, obs_2[:, i],c=cmap(0.9),linestyle="dotted",marker="o",ms=2)
    
    plt.errorbar(x, obs_3[:, i],yerr=obs_3[:, i+1],\
                  label="Random walk",c=cmap(0),linestyle="none"\
                      ,marker="o",ms=1)
    plt.plot(x, obs_3[:, i],c=cmap(0.1),linestyle="dotted",marker="o",ms=2)
    plt.xscale("log")
    plt.xlabel("t")
    plt.ylabel(labels_y[int(i/2)])
    plt.legend()
    plt.show()
    
#%%
#random neighbour 
N_i=1000
rule="rn"
name="time_evo_r_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
    +"_"+str(N_i)+".dat"
obs_1=np.loadtxt(directory+name)
name="time_evo_bfs_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
    +"_"+str(N_i)+".dat"
obs_2=np.loadtxt(directory+name)
name="time_evo_rw_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
    +"_"+str(N_i)+".dat"
obs_3=np.loadtxt(directory+name)

x = np.arange(N)
labels_y=["Distance","$q_{def}$","$q_{tot}$","Maximum Distance",\
          "Average Distance"]
cmap=plt.get_cmap("viridis")
for i in range(0,10,2):
    plt.figure()
    plt.errorbar(x, obs_1[:, i],yerr=obs_1[:, i+1],\
                  label="Random selection",c=cmap(0.5),\
                      linestyle="none",marker="o",ms=1)
    plt.plot(x, obs_1[:, i],c=cmap(0.6),linestyle="dotted",marker="o",ms=2)
    
    plt.errorbar(x, obs_2[:, i],yerr=obs_2[:, i+1],\
                  label="BFS",c=cmap(0.8),linestyle="none",marker="o",ms=1)
    plt.plot(x, obs_2[:, i],c=cmap(0.9),linestyle="dotted",marker="o",ms=2)
    
    plt.errorbar(x, obs_3[:, i],yerr=obs_3[:, i+1],\
                  label="Random walk",c=cmap(0),linestyle="none"\
                      ,marker="o",ms=1)
    plt.plot(x, obs_3[:, i],c=cmap(0.1),linestyle="dotted",marker="o",ms=2)
    plt.xscale("log")
    plt.xlabel("t")
    plt.ylabel(labels_y[int(i/2)])
    plt.legend()
    plt.show()
    

