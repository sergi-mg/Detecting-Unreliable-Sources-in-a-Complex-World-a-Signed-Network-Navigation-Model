# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 11:33:23 2026

@author: Sergi Martínez Galindo
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numba import njit

#%%
from os.path import exists
from os import makedirs

#folder
directory="../data/time_evo_2/"
if not exists(directory):
    makedirs(directory)
    
directory_save="../images/time_evo_2/"
if not exists(directory_save):
    makedirs(directory_save)
    
    
#%%    
def create_plot_strategy(obs_1,obs_2,obs_3,directory_s,N,rule):

    x = np.arange(1,N)/N

    labels_y=["Distance","$q_{def}$","$q_{tot}$","Maximum Distance",\
              "Average Distance"]
        
    cmap=plt.get_cmap("viridis")
    
    name_s=rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))+"_"+str(N_i)+".pdf"
    
    for i in range(0,10,2):
        plt.figure()
        plt.errorbar(x, obs_1[1:, i],yerr=obs_1[1:, i+1],\
                      label="Random selection",c=cmap(0.5),\
                          linestyle="none",marker="o",ms=1)
        plt.plot(x, obs_1[1:, i],c=cmap(0.6),linestyle="dotted",marker="o",ms=2)
        
        plt.errorbar(x, obs_2[1:, i],yerr=obs_2[1:, i+1],\
                      label="BFS",c=cmap(0.8),linestyle="none",marker="o",ms=1)
        plt.plot(x, obs_2[1:, i],c=cmap(0.9),linestyle="dotted",marker="o",ms=2)
        
        plt.errorbar(x, obs_3[1:, i],yerr=obs_3[1:, i+1],\
                      label="Random walk",c=cmap(0),linestyle="none"\
                          ,marker="o",ms=1)
        plt.plot(x, obs_3[1:, i],c=cmap(0.1),linestyle="dotted",marker="o",ms=2)
        plt.xscale("log")
        plt.xlim(0.001,1)
        plt.xlabel("Fraction of defined nodes")
        plt.ylabel(labels_y[int(i/2)])
        plt.legend()
        if i==1 or i==2:
            plt.ylim([0,1])
        plt.grid(True, which='both')
        plt.savefig(directory_s+"pdf/"+labels_y[int(i/2)]+"_"+name_s+\
                    ".pdf",bbox_inches="tight")
        plt.savefig(directory_s+"png/"+labels_y[int(i/2)]+"_"+name_s+\
                    ".png",bbox_inches="tight")    
        plt.show()
        plt.close()
        
def create_plot_N(obs_1,obs_2,obs_3,directory_s,strategy,rule):


    labels_y=["Distance","$q_{def}$","$q_{tot}$","Maximum Distance",\
              "Average Distance"]
        
    cmap=plt.get_cmap("viridis")
    
    name_s=rule+"_"+strategy+"_"+str(k)+"_"+str(round(r,2))+"_"+str(N_i)+".pdf"
    
    for i in range(0,10,2):
        plt.figure()
        x = np.arange(1,100)/100
        plt.errorbar(x, obs_1[1:, i],yerr=obs_1[1:, i+1],\
                      label="$N=100$",c=cmap(0.5),\
                          linestyle="none",marker="o",ms=1)
        plt.plot(x, obs_1[1:, i],c=cmap(0.6),linestyle="dotted",marker="o",ms=2)
        
        x = np.arange(1,500)/500
        plt.errorbar(x, obs_2[1:, i],yerr=obs_2[1:, i+1],\
                      label="$N=500$",c=cmap(0.8),linestyle="none",marker="o",ms=1)
        plt.plot(x, obs_2[1:, i],c=cmap(0.9),linestyle="dotted",marker="o",ms=2)
        
        x = np.arange(1,1000)/1000
        plt.errorbar(x, obs_3[1:, i],yerr=obs_3[1:, i+1],\
                      label="$N=1000$",c=cmap(0),linestyle="none"\
                          ,marker="o",ms=1)
        plt.plot(x, obs_3[1:, i],c=cmap(0.1),linestyle="dotted",marker="o",ms=2)
        plt.xscale("log")
        plt.xlim(0.001,1)
        plt.xlabel("Fraction of defined nodes")
        plt.ylabel(labels_y[int(i/2)])
        plt.legend()
        if i==1 or i==2:
            plt.ylim([0,1])
        plt.grid(True, which='both')
        plt.savefig(directory_s+"pdf/"+labels_y[int(i/2)]+"_"+name_s+\
                    ".pdf",bbox_inches="tight")
        plt.savefig(directory_s+"png/"+labels_y[int(i/2)]+"_"+name_s+\
                    ".png",bbox_inches="tight")
        plt.show()
        plt.close()
        
def create_plot_rule(obs_1,obs_2,directory_s,N,strategy):

    x = np.arange(1,N)/N

    labels_y=["Distance","$q_{def}$","$q_{tot}$","Maximum Distance",\
              "Average Distance"]
        
    cmap=plt.get_cmap("viridis")
    
    name_s=strategy+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))+"_"+str(N_i)+".pdf"
    
    for i in range(0,10,2):
        plt.figure()
        plt.errorbar(x, obs_1[1:, i],yerr=obs_1[1:, i+1],\
                      label="Majority rule",c=cmap(0.5),\
                          linestyle="none",marker="o",ms=1)
        plt.plot(x, obs_1[1:, i],c=cmap(0.6),linestyle="dotted",marker="o",ms=2)
        
        plt.errorbar(x, obs_2[1:, i],yerr=obs_2[1:, i+1],\
                      label="Random Neighbour",c=cmap(0.8),linestyle="none",marker="o",ms=1)
        plt.plot(x, obs_2[1:, i],c=cmap(0.9),linestyle="dotted",marker="o",ms=2)
        
        plt.xscale("log")
        plt.xlim(0.001,1)
        plt.xlabel("Fraction of defined nodes")
        plt.ylabel(labels_y[int(i/2)])
        plt.legend()
        if i==1 or i==2:
            plt.ylim([0,1])
        plt.grid(True, which='both')
        plt.savefig(directory_s+"pdf/"+labels_y[int(i/2)]+"_"+name_s+\
                    ".pdf",bbox_inches="tight")
        plt.savefig(directory_s+"png/"+labels_y[int(i/2)]+"_"+name_s+\
                    ".png",bbox_inches="tight")
        plt.show()
        plt.close()

#%%
directory_save="../images/time_evo_2/strategies/"
if not exists(directory_save):
    makedirs(directory_save)
    
if not exists(directory_save+"pdf/"):
    makedirs(directory_save+"pdf/")
if not exists(directory_save+"png/"):
    makedirs(directory_save+"png/")
    
#Compare strategies

k=20
r=0.1 
N_i=1000
N_v=[100,500,1000]
rule_v=["mr","rn"]
  
for N in N_v:  
    for rule in rule_v:
    
        #reading the data
        
        name="time_evo_r_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
            +"_"+str(N_i)+".dat"
        obs_1=np.loadtxt(directory+name)
        name="time_evo_bfs_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
            +"_"+str(N_i)+".dat"
        obs_2=np.loadtxt(directory+name)
        name="time_evo_rw_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
            +"_"+str(N_i)+".dat"
        obs_3=np.loadtxt(directory+name)
        
        #plots
        create_plot_strategy(obs_1,obs_2,obs_3,directory_save,N,rule)
    


#%%
directory_save="../images/time_evo_2/rules/"
if not exists(directory_save):
    makedirs(directory_save)
    
if not exists(directory_save+"pdf/"):
    makedirs(directory_save+"pdf/")
if not exists(directory_save+"png/"):
    makedirs(directory_save+"png/")
    
#Compare rules

k=20
r=0.1 
N_i=1000
N_v=[100,500,1000]
s_v=["r","bfs","rw"]
  
for N in N_v:  
    for strategy in s_v:
    
        #reading the data
        
        name="time_evo_"+strategy+"_mr_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
            +"_"+str(N_i)+".dat"
        obs_1=np.loadtxt(directory+name)
        name="time_evo_"+strategy+"_rn_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
            +"_"+str(N_i)+".dat"
        obs_2=np.loadtxt(directory+name)
        
        #plots
        create_plot_rule(obs_1,obs_2,directory_save,N,strategy)
        
#%%
directory_save="../images/time_evo_2/N/"
if not exists(directory_save):
    makedirs(directory_save)
    
if not exists(directory_save+"pdf/"):
    makedirs(directory_save+"pdf/")
if not exists(directory_save+"png/"):
    makedirs(directory_save+"png/")
    
#Compare rules

k=20
r=0.1 
N_i=1000
rule_v=["mr","rn"]
s_v=["r","bfs","rw"]
  
for rule in rule_v:  
    for strategy in s_v:
    
        #reading the data
        
        name="time_evo_"+strategy+"_"+rule+"_100_"+str(k)+"_"+str(round(r,2))\
            +"_"+str(N_i)+".dat"
        obs_1=np.loadtxt(directory+name)
        name="time_evo_"+strategy+"_"+rule+"_500_"+str(k)+"_"+str(round(r,2))\
            +"_"+str(N_i)+".dat"
        obs_2=np.loadtxt(directory+name)
        name="time_evo_"+strategy+"_"+rule+"_1000_"+str(k)+"_"+str(round(r,2))\
            +"_"+str(N_i)+".dat"
        obs_3=np.loadtxt(directory+name)
        
        #plots
        create_plot_N(obs_1,obs_2,obs_3,directory_save,strategy,rule)