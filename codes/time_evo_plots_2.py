# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 11:36:24 2026

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
directory_save="../images/time_evo_2/minima/"
if not exists(directory_save):
    makedirs(directory_save)
    
if not exists(directory_save+"pdf/"):
    makedirs(directory_save+"pdf/")
if not exists(directory_save+"png/"):
    makedirs(directory_save+"png/")
    
#%%
#statistics only focusing on q_def(t)
def evo_matrix(rule,strategy,k_values,r_values,N,N_i):
    """Reads the files for k and r (1D arrays) indicated and returns
    the mean and the standard deviation of the accuracy q."""
    
    directory="../data/time_evo_2/"
    if not exists(directory):
        makedirs(directory)
    
    N_k=np.size(k_values)
    N_r=np.size(r_values)
    mean=np.zeros((N,N_k,N_r))
    error=np.zeros((N,N_k,N_r))
    for i in range(N_k):
        k=k_values[i]
        for j in range(N_r):
            r=r_values[j]
            
            #reading the file
            file_name="time_evo_"+strategy+"_"+rule+"_"+str(N)+"_"+str(k)\
                +"_"+str(round(r,2))+"_"+str(N_i)+".dat"
            results=np.loadtxt(fname=directory+file_name,dtype="float64")
            
            #we want to study q_def(t)
            mean[:,i,j]=results[:,2]
            error[:,i,j]=results[:,3]

    return mean,error

#evolution plot
def evo_plot(obs_1,obs_2,obs_3,N):

    x = np.arange(N)
        
    cmap=plt.get_cmap("viridis")
    
    
    plt.figure()
    i=0
    
    plt.errorbar(x, obs_1[:, i],yerr=obs_1[:, i+1],\
                  label="Random selection",c=cmap(0.5),\
                      linestyle="none",marker="o",ms=1)
    plt.plot(x, obs_1[:, i],c=cmap(0.6),linestyle="dotted",marker="o",ms=2)
    
    plt.errorbar(x, obs_2[:, i],yerr=obs_2[:, i+1],\
                  label="Ordered by distance",c=cmap(0.8),\
                  linestyle="none",marker="o",ms=1)
    plt.plot(x, obs_2[:, i],c=cmap(0.9),linestyle="dotted",marker="o",ms=2)
    
    plt.errorbar(x, obs_3[:, i],yerr=obs_3[:, i+1],\
                  label="Random walk",c=cmap(0),linestyle="none"\
                      ,marker="o",ms=1)
    plt.plot(x, obs_3[:, i],c=cmap(0.1),linestyle="dotted",marker="o",ms=2)
    
    #plt.xscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$\langle q_{def} \rangle$")
    plt.legend()
    plt.grid(True)
    plt.show()

#difference evolution plot
def diff_evo_plot(obs_1,obs_2,obs_3,N):

    x = np.arange(N)[1:]
        
    cmap=plt.get_cmap("viridis")
    
    
    plt.figure()
    i=0
    
    plt.plot(x, np.diff(obs_1[:, i]),c=cmap(0.6),linestyle="dotted",marker="o"\
             ,ms=2,label="Random selection")
    plt.plot(x, np.diff(obs_2[:, i]),c=cmap(0.9),linestyle="dotted",marker="o"\
             ,ms=2,label="Ordered by distance")
    plt.plot(x, np.diff(obs_3[:, i]),c=cmap(0.1),linestyle="dotted",marker="o"\
             ,ms=2,label="Random walk")
    
    plt.xscale("log")
    plt.xlabel("t")
    plt.ylabel(r"$d\langle q_{def} \rangle/dt$")
    plt.legend()
    plt.grid(True)
    plt.show()  

#time evo curve fitting to fiund the minima
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


def modelo_parabolico(x, a, b, c):
    return a * x**2 + b * x + c


def minimum_where(x, y, dy=None):
    
    popt, pcov = curve_fit(modelo_parabolico, x, y, sigma=dy, absolute_sigma=True)
    a, b, c = popt
    sigma_a = np.sqrt(pcov[0,0])

    # TEST 1: minimum value?
    if a <= 0:
        print("ERROR: there is not a minimum")
        return None, None

    # TEST 2: a!=0?
    #if a < 3 * sigma_a:
        #print("ERROR: not enough curvature")
        #return None, None

    xmin = -b / (2 * a)

    # TEST 3: minimum inside the data
    if xmin < min(x) or xmin > max(x):
        print(f"ERROR: Minimum out of range (x={xmin:.2f})")
        return None, None

    var_a = pcov[0, 0]
    var_b = pcov[1, 1]
    cov_ab = pcov[0, 1]

    deriv_a = b / (2 * a**2)
    deriv_b = -1 / (2 * a)

    var_xmin = (deriv_a**2 * var_a) + (deriv_b**2 * var_b) + (2 * deriv_a * deriv_b * cov_ab)
    dxmin = np.sqrt(var_xmin)
    
    if dxmin>N:
        print(f"ERROR: Uncertainty bigger rthan interval.")
        return  None, None
    
    return xmin, dxmin
    
    
    
#%%
k_values=np.array([20])
r_values=np.arange(0.01,0.501,0.01)
N=1000
strategy_list=["r","bfs","rw"]

q_def=np.zeros((N,k_values.size,r_values.size,len(strategy_list),2))
x=np.arange(N)

for j in range(len(strategy_list)):
    strategy=strategy_list[j]
    res=evo_matrix("mr", strategy, k_values, r_values, N, 1000)
    q_def[:,:,:,j,0]=res[0]
    q_def[:,:,:,j,1]=res[1]

#%%
t_min=np.zeros((k_values.size,r_values.size,len(strategy_list)))
dt_min=np.zeros_like(t_min)
for j in range(r_values.size):
    for k in range(len(strategy_list)):
        print(r_values[j],strategy_list[k])
        #smooth curve
        q=q_def[:,0,j,k,0]
        dq=q_def[:,0,j,k,1]
        
        q_smooth = savgol_filter(q, window_length=11, polyorder=2)
        idx_min = np.argmin(q_smooth)
        
        # region of interest
        a=0
        if idx_min<150:
            a=50
        elif idx_min<300:
            a=100
        else:
            a=150
        start = max(0, idx_min - a)
        end = min(len(x), idx_min + a + 1)
        
        
        x_roi = x[start:end]
        y_roi = q[start:end]
        
        dy_roi = dq[start:end]
        
        if np.isnan(t_min[0, j-1, k]):
            t_min[0,j,k],dt_min[0,j,k]=None,None
        else:
            t_min[0,j,k],dt_min[0,j,k]=minimum_where(x_roi,y_roi,dy_roi)

t_min*=1/N
dt_min*=1/N

    
#%%
plt.figure()
cmap=plt.get_cmap("viridis")
plt.errorbar(r_values,t_min[0,:,0],yerr=dt_min[0,:,0]\
             ,c=cmap(0.6),linestyle="dotted",marker="o"\
         ,ms=3,label="Random selection")
plt.errorbar(r_values,t_min[0,:,1],yerr=dt_min[0,:,1]\
             ,c=cmap(0.9),linestyle="dotted",marker="o"\
         ,ms=3,label="Ordered by distance")
plt.errorbar(r_values,t_min[0,:,2],yerr=dt_min[0,:,2]\
             ,c=cmap(0.1),linestyle="dotted",marker="o"\
         ,ms=3,label="Random walk")
    
plt.xlim([0,0.5])
plt.ylim([0,1])
#plt.yscale("log")
plt.xlabel("r")
plt.ylabel(r"Fraction of defined nodes when $\langle q_{def} \rangle$ is minimum")
plt.legend()
plt.grid(True, which='both')
plt.savefig(directory_save+"pdf/"+str(N)+"_minima.pdf",bbox_inches="tight")
plt.savefig(directory_save+"png/"+str(N)+"_minima.png",bbox_inches="tight")
plt.show()  
