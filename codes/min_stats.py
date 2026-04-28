# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 09:30:31 2026

@author: Sergi Martínez Galindo
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numba import njit
from os.path import exists
from os import makedirs
#%%
@njit
def statistics(x):
    """Returns the mean and its standard deviation of a 1D array x.
    You can get the uncertainty with 95 % of confidence with 1.96*sigma."""
    
    sumx=np.sum(x)
    sumx2=np.sum(x**2)
    N=np.size(x)
    xmed=sumx/N
    x2med=sumx2/N
    #distribution's statistics
    s2=(N*(x2med-xmed**2))/(N-1)
    s=s2**0.5
    #mean value's statistics
    var=s2/N
    if var<0:
        var=0
    sigma=(var)**0.5
    return xmed,sigma

@njit
def xifres(values,uncertainties,exp_max,exp_min):
    """Rounds the values and their uncertainties (both 1D arrays) with the
    adecuate number of significant figures. Values and uncertanties
    are 1D matrices and exp_max and exp_min are the power (x) of 10^x
    corresponding to the maximum and minimum order of the uncertainties,
    if you get 0's in the output try exppanding the exponent range."""
    
    values_c=np.zeros_like(values)
    uncertainties_c=np.zeros_like(uncertainties)
    for i in range(np.size(uncertainties)):
        for k in range(exp_max,exp_min,-1):
            if uncertainties[i]==0:
                values_c[i]=values[i]
                uncertainties_c[i]=uncertainties[i]
            if uncertainties[i]>=1.95*np.power(10.0, k):
                values_c[i]=round(values[i],-k)
                uncertainties_c[i]=round(uncertainties[i],-k)
                break
    return values_c,uncertainties_c

def t_r(N,k_p,r_list,N_i,rule,m):
    """Reads the data and averages over groups of m simulations, look for the 
    minimum of each averaged curve, and computes the average minimum.
    1<=m<=N_i"""
    
    #reading the data
    N_r=np.size(r_list)
    
    #folder
    directory="../data/time_evo_simulations/"
    
    t_min=np.zeros((N_r,2,3))
    q_min=np.zeros((N_r,2,3))
    
    strategies=["rs","obd","rw"]
    for k in range(3):
    
        #each simulation: array (N,5) d(t),q_def(t),q(t),d_max(t),<d>(t)
        
        for i in range(N_r):
            r=r_list[i]
            name=rule+"_"+str(N)+"_"+str(k_p)+"_"+str(round(r,2))\
                +"_"+str(N_i)+".npz"
            matrix=np.load(directory+name)
            
            data=matrix[strategies[k]].astype(np.float32)
            
            usable=(N_i//m)*m

            q_blocks=data[:,1,:usable].reshape(N, N_i//m, m)
            q_mean = np.mean(q_blocks, axis=2)
            
            q_m_r=np.min(q_mean, axis=0)
            t_m_r=np.argmin(q_mean, axis=0)/N
                
            t_min[i,0,k],t_min[i,1,k]=statistics(t_m_r)
            q_min[i,0,k],q_min[i,1,k]=statistics(q_m_r)
            
        t_min[:,0,k],t_min[:,1,k]=xifres(t_min[:,0,k],1.96*t_min[:,1,k],2,-10)
        q_min[:,0,k],q_min[:,1,k]=xifres(q_min[:,0,k],1.96*q_min[:,1,k],2,-10)
    
    return t_min,q_min

def grafics(t,q,r_values,m,N_i,N,k):
    """Saves two graphics (each one with a curve for each strategy): 
        - t_min(r)
        - q_min(r)"""
        
    #folder
    
    directory_save="../images/minima/t/"
    if not exists(directory_save):
        makedirs(directory_save)
        
    if not exists(directory_save+"pdf/"):
        makedirs(directory_save+"pdf/")
    if not exists(directory_save+"png/"):
        makedirs(directory_save+"png/")
        
    #plot
        
    cmap=plt.get_cmap("viridis") 
    plt.figure()
    plt.errorbar(r_values,t[:,0,0],xerr=0,yerr=t[:,1,0],linestyle="dashed",
                 marker="o",markersize=2,label="Random Selection",c=cmap(0.6))
    plt.errorbar(r_values,t[:,0,1],xerr=0,yerr=t[:,1,1],linestyle="dashed",
                 marker="o",markersize=2,label="Ordered by Distance",c=cmap(0.9))
    plt.errorbar(r_values,t[:,0,2],xerr=0,yerr=t[:,1,2],linestyle="dashed",
                 marker="o",markersize=2, label="Random Walk",c=cmap(0.1))

    plt.legend()
    plt.xlabel("r")
    plt.ylabel("$t_{min}$")
    plt.savefig(directory_save+"pdf/"+str(N)+"_"+str(k)+"_"+str(N_i)+"_"\
                +str(m)+"_t_minima.pdf",bbox_inches="tight")
    plt.savefig(directory_save+"png/"+str(N)+"_"+str(k)+"_"+str(N_i)+"_"\
                +str(m)+"_t_minima.png",bbox_inches="tight")
    plt.close()
        
    #folder
    
    directory_save="../images/minima/q/"
    if not exists(directory_save):
        makedirs(directory_save)
        
    if not exists(directory_save+"pdf/"):
        makedirs(directory_save+"pdf/")
    if not exists(directory_save+"png/"):
        makedirs(directory_save+"png/")
    
    #plot
    
    plt.figure()
    plt.errorbar(r_values,q[:,0,0],xerr=0,yerr=q[:,1,0],linestyle="dashed",
                 marker="o",markersize=2,label="Random Selection",c=cmap(0.6))
    plt.errorbar(r_values,q[:,0,1],xerr=0,yerr=q[:,1,1],linestyle="dashed",
                 marker="o",markersize=2,label="Ordered by Distance",c=cmap(0.9))
    plt.errorbar(r_values,q[:,0,2],xerr=0,yerr=q[:,1,2],linestyle="dashed",
                 marker="o",markersize=2, label="Random Walk",c=cmap(0.1))
    plt.legend()
    plt.xlabel("r")
    plt.ylabel("$q_{min}$")
    plt.savefig(directory_save+"pdf/"+str(N)+"_"+str(k)+"_"+str(N_i)+"_"\
                +str(m)+"_q_minima.pdf",bbox_inches="tight")
    plt.savefig(directory_save+"png/"+str(N)+"_"+str(k)+"_"+str(N_i)+"_"\
                +str(m)+"_q_minima.png",bbox_inches="tight")
    plt.close()
    
                
#%%
r_values=np.arange(0.01,0.5005,0.01)
k=20
N=1000
N_i=5000

t,q=t_r(N, k, r_values, N_i, "mr", 100)      

#%%  
cmap=plt.get_cmap("viridis") 
plt.figure()
plt.errorbar(r_values,t[:,0,0],xerr=0,yerr=t[:,1,0],linestyle="dashed",
             marker="o",markersize=2,label="Random Selection",c=cmap(0.6))
plt.errorbar(r_values,t[:,0,1],xerr=0,yerr=t[:,1,1],linestyle="dashed",
             marker="o",markersize=2,label="Ordered by Distance",c=cmap(0.9))
plt.errorbar(r_values,t[:,0,2],xerr=0,yerr=t[:,1,2],linestyle="dashed",
             marker="o",markersize=2, label="Random Walk",c=cmap(0.1))

plt.legend()
plt.xlabel("r")
plt.ylabel("$t_{min}$")
plt.show()

plt.figure()
plt.errorbar(r_values,q[:,0,0],xerr=0,yerr=q[:,1,0],linestyle="dashed",
             marker="o",markersize=2,label="Random Selection",c=cmap(0.6))
plt.errorbar(r_values,q[:,0,1],xerr=0,yerr=q[:,1,1],linestyle="dashed",
             marker="o",markersize=2,label="Ordered by Distance",c=cmap(0.9))
plt.errorbar(r_values,q[:,0,2],xerr=0,yerr=q[:,1,2],linestyle="dashed",
             marker="o",markersize=2, label="Random Walk",c=cmap(0.1))
plt.legend()
plt.xlabel("r")
plt.ylabel("$q_{min}$")
plt.show()



#%%
plt.figure()
m_max=2500
m=1
i=0
while m<m_max:
    t,q=t_r(N, k, r_values, N_i, "mr", m)  
    plt.errorbar(r_values,t[:,0,0],xerr=0,yerr=t[:,1,0],linestyle="dashed",
                 marker="o",markersize=2,label="m="+str(m),c=cmap(i/15))
    m*=2
    i+=1

plt.legend()
plt.xlabel("r")
plt.ylabel("$t_{min}$")
plt.show()