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
    
directory_save="../images/time_evo/"
if not exists(directory_save):
    makedirs(directory_save)
    
if not exists(directory_save+"pdf/"):
    makedirs(directory_save+"pdf/")
if not exists(directory_save+"png/"):
    makedirs(directory_save+"png/")
    
#%%    
def create_plot(obs_1,obs_2,obs_3,directory_s,N,rule):

    x = np.arange(N)

    labels_y=["Distance","$q_{def}$","$q_{tot}$","Maximum Distance",\
              "Average Distance"]
        
    cmap=plt.get_cmap("viridis")
    
    name_s=rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))+"_"+str(N_i)+".pdf"
    
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
        plt.savefig(directory_s+"pdf/"+labels_y[int(i/2)]+"_"+name_s+\
                    ".pdf",bbox_inches="tight")
        plt.savefig(directory_s+"png/"+labels_y[int(i/2)]+"_"+name_s+\
                    ".png",bbox_inches="tight")
        plt.grid(True)
        plt.show()
        plt.close()

#%%
#majority rule

N=100
k=21
r=0.1 
N_i=1000
rule="mr"

#reading the data
name="time_evo_r_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
    +"_"+str(N_i)+".dat"
obs_1_mr=np.loadtxt(directory+name)
name="time_evo_bfs_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
    +"_"+str(N_i)+".dat"
obs_2_mr=np.loadtxt(directory+name)
name="time_evo_rw_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
    +"_"+str(N_i)+".dat"
obs_3_mr=np.loadtxt(directory+name)

#plots
create_plot(obs_1_mr,obs_2_mr,obs_3_mr,directory_save,N,rule)

#%%
#random neighbour 
N_i=1000
rule="rn"
name="time_evo_r_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
    +"_"+str(N_i)+".dat"
obs_1_rn=np.loadtxt(directory+name)
name="time_evo_bfs_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
    +"_"+str(N_i)+".dat"
obs_2_rn=np.loadtxt(directory+name)
name="time_evo_rw_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
    +"_"+str(N_i)+".dat"
obs_3_rn=np.loadtxt(directory+name)

#plots
create_plot(obs_1_rn,obs_2_rn,obs_3_rn,directory_save,N,rule)

#%%
#comparisions
directory_s="../images/time_evo_comparisions/"
if not exists(directory_s):
    makedirs(directory_s)
if not exists(directory_s+"pdf/"):
    makedirs(directory_s+"pdf/")
if not exists(directory_s+"png/"):
    makedirs(directory_s+"png/")

#q_def
#majority rule
q_1_mr=obs_1_mr[:,2:4]
q_2_mr=obs_2_mr[:,2:4]
q_3_mr=obs_3_mr[:,2:4]

y12_mr=np.zeros_like(q_1_mr)
y12_mr[:,0]=np.abs(q_1_mr[:,0]-q_2_mr[:,0])
y12_mr[:,1]=np.abs(q_1_mr[:,1]+q_2_mr[:,1])

y13_mr=np.zeros_like(q_1_mr)
y13_mr[:,0]=np.abs(q_1_mr[:,0]-q_3_mr[:,0])
y13_mr[:,1]=np.abs(q_1_mr[:,1]+q_1_mr[:,1])

y23_mr=np.zeros_like(q_1_mr)
y23_mr[:,0]=np.abs(q_3_mr[:,0]-q_2_mr[:,0])
y23_mr[:,1]=np.abs(q_3_mr[:,1]+q_2_mr[:,1])


#random neighbour
q_1_rn=obs_1_rn[:,2:4]
q_2_rn=obs_2_rn[:,2:4]
q_3_rn=obs_3_rn[:,2:4]

y12_rn=np.zeros_like(q_1_rn)
y12_rn[:,0]=np.abs(q_1_rn[:,0]-q_2_rn[:,0])
y12_rn[:,1]=np.abs(q_1_rn[:,1]+q_2_rn[:,1])

y13_rn=np.zeros_like(q_1_rn)
y13_rn[:,0]=np.abs(q_1_rn[:,0]-q_3_rn[:,0])
y13_rn[:,1]=np.abs(q_1_rn[:,1]+q_1_rn[:,1])

y23_rn=np.zeros_like(q_1_rn)
y23_rn[:,0]=np.abs(q_3_rn[:,0]-q_2_rn[:,0])
y23_rn[:,1]=np.abs(q_3_rn[:,1]+q_2_rn[:,1])

cmap=plt.get_cmap("viridis")

x = np.arange(N)

plt.figure()
plt.errorbar(x, y12_mr[:,0],yerr=y12_mr[:,1],\
              label="RS-BFS",c=cmap(0.5),\
                  linestyle="none",marker="o",ms=1)
plt.plot(x, y12_mr[:,0],c=cmap(0.6),linestyle="dotted",marker="o",ms=2)

plt.errorbar(x, y13_mr[:,0],yerr=y13_mr[:,1],\
              label="RS-RW",c=cmap(0.8),\
                  linestyle="none",marker="o",ms=1)
plt.plot(x, y13_mr[:,0],c=cmap(0.9),linestyle="dotted",marker="o",ms=2)

plt.errorbar(x, y23_mr[:,0],yerr=y23_mr[:,1],\
              label="RW-BFS",c=cmap(0),\
                  linestyle="none",marker="o",ms=1)
plt.plot(x, y23_mr[:,0],c=cmap(0.1),linestyle="dotted",marker="o",ms=2)

plt.xscale("log")
plt.xlabel("t")
plt.ylabel("Difference on $q_{def}$")
plt.legend()
plt.savefig(directory_s+"pdf/q_mr"+str(k)+".pdf",bbox_inches="tight")
plt.savefig(directory_s+"png/q_mr"+str(k)+".png",bbox_inches="tight")
plt.grid(True)
plt.show()
plt.close()

plt.figure()
plt.errorbar(x, y12_rn[:,0],yerr=y12_rn[:,1],\
              label="RS-BFS",c=cmap(0.5),\
                  linestyle="none",marker="o",ms=1)
plt.plot(x, y12_rn[:,0],c=cmap(0.6),linestyle="dotted",marker="o",ms=2)

plt.errorbar(x, y13_rn[:,0],yerr=y13_rn[:,1],\
              label="RS-RW",c=cmap(0.8),\
                  linestyle="none",marker="o",ms=1)
plt.plot(x, y13_rn[:,0],c=cmap(0.9),linestyle="dotted",marker="o",ms=2)

plt.errorbar(x, y23_rn[:,0],yerr=y23_rn[:,1],\
              label="RW-BFS",c=cmap(0),\
                  linestyle="none",marker="o",ms=1)
plt.plot(x, y23_rn[:,0],c=cmap(0.1),linestyle="dotted",marker="o",ms=2)

plt.xscale("log")
plt.xlabel("t")
plt.ylabel("Difference on $q_{def}$")
plt.legend()
plt.savefig(directory_s+"pdf/q_rn"+str(k)+".pdf",bbox_inches="tight")
plt.savefig(directory_s+"png/q_rn"+str(k)+".png",bbox_inches="tight")
plt.grid(True)
plt.show()
plt.close()

#%%
#d
#majority rule
q_1_mr=obs_1_mr[:,:2]
q_2_mr=obs_2_mr[:,:2]
q_3_mr=obs_3_mr[:,:2]

y12_mr=np.zeros_like(q_1_mr)
y12_mr[:,0]=np.abs(q_1_mr[:,0]-q_2_mr[:,0])
y12_mr[:,1]=np.abs(q_1_mr[:,1]+q_2_mr[:,1])

y13_mr=np.zeros_like(q_1_mr)
y13_mr[:,0]=np.abs(q_1_mr[:,0]-q_3_mr[:,0])
y13_mr[:,1]=np.abs(q_1_mr[:,1]+q_1_mr[:,1])

y23_mr=np.zeros_like(q_1_mr)
y23_mr[:,0]=np.abs(q_3_mr[:,0]-q_2_mr[:,0])
y23_mr[:,1]=np.abs(q_3_mr[:,1]+q_2_mr[:,1])


#random neighbour
q_1_rn=obs_1_rn[:,:2]
q_2_rn=obs_2_rn[:,:2]
q_3_rn=obs_3_rn[:,:2]

y12_rn=np.zeros_like(q_1_rn)
y12_rn[:,0]=np.abs(q_1_rn[:,0]-q_2_rn[:,0])
y12_rn[:,1]=np.abs(q_1_rn[:,1]+q_2_rn[:,1])

y13_rn=np.zeros_like(q_1_rn)
y13_rn[:,0]=np.abs(q_1_rn[:,0]-q_3_rn[:,0])
y13_rn[:,1]=np.abs(q_1_rn[:,1]+q_1_rn[:,1])

y23_rn=np.zeros_like(q_1_rn)
y23_rn[:,0]=np.abs(q_3_rn[:,0]-q_2_rn[:,0])
y23_rn[:,1]=np.abs(q_3_rn[:,1]+q_2_rn[:,1])

cmap=plt.get_cmap("viridis")

x = np.arange(N)

plt.figure()
plt.errorbar(x, y12_mr[:,0],yerr=y12_mr[:,1],\
              label="RS-BFS",c=cmap(0.5),\
                  linestyle="none",marker="o",ms=1)
plt.plot(x, y12_mr[:,0],c=cmap(0.6),linestyle="dotted",marker="o",ms=2)

plt.errorbar(x, y13_mr[:,0],yerr=y13_mr[:,1],\
              label="RS-RW",c=cmap(0.8),\
                  linestyle="none",marker="o",ms=1)
plt.plot(x, y13_mr[:,0],c=cmap(0.9),linestyle="dotted",marker="o",ms=2)

plt.errorbar(x, y23_mr[:,0],yerr=y23_mr[:,1],\
              label="RW-BFS",c=cmap(0),\
                  linestyle="none",marker="o",ms=1)
plt.plot(x, y23_mr[:,0],c=cmap(0.1),linestyle="dotted",marker="o",ms=2)

plt.xscale("log")
plt.xlabel("t")
plt.ylabel("Difference on $d(t)$")
plt.legend()
plt.savefig(directory_s+"pdf/d_mr"+str(k)+".pdf",bbox_inches="tight")
plt.savefig(directory_s+"png/d_mr"+str(k)+".png",bbox_inches="tight")
plt.grid(True)
plt.show()
plt.close()

plt.figure()
plt.errorbar(x, y12_rn[:,0],yerr=y12_rn[:,1],\
              label="RS-BFS",c=cmap(0.5),\
                  linestyle="none",marker="o",ms=1)
plt.plot(x, y12_rn[:,0],c=cmap(0.6),linestyle="dotted",marker="o",ms=2)

plt.errorbar(x, y13_rn[:,0],yerr=y13_rn[:,1],\
              label="RS-RW",c=cmap(0.8),\
                  linestyle="none",marker="o",ms=1)
plt.plot(x, y13_rn[:,0],c=cmap(0.9),linestyle="dotted",marker="o",ms=2)

plt.errorbar(x, y23_rn[:,0],yerr=y23_rn[:,1],\
              label="RW-BFS",c=cmap(0),\
                  linestyle="none",marker="o",ms=1)
plt.plot(x, y23_rn[:,0],c=cmap(0.1),linestyle="dotted",marker="o",ms=2)

plt.xscale("log")
plt.xlabel("t")
plt.ylabel("Difference on $d(t)$")
plt.legend()
plt.savefig(directory_s+"pdf/d_rn"+str(k)+".pdf",bbox_inches="tight")
plt.savefig(directory_s+"png/d_rn"+str(k)+".png",bbox_inches="tight")
plt.grid(True)
plt.show()
plt.close()

