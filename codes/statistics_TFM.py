# -*- coding: utf-8 -*-
"""
Created on Sun Feb 15 18:36:04 2026

@author: Sergi Martínez Galindo
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numba import njit

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


def statistics_matrix(rule,k_values,r_values,N,N_i):
    """Reads the files for k and r (1D arrays) indicated and returns
    the mean and the standard deviation of the accuracy q."""
    
    directory="../data/"
    
    N_k=np.size(k_values)
    N_r=np.size(r_values)
    mean=np.zeros((N_k,N_r),dtype="float64")
    sigma=np.zeros((N_k,N_r),dtype="float64")
    for i in range(N_k):
        k=k_values[i]
        for j in range(N_r):
            r=r_values[j]
            #reading the file
            file_name=\
            rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))+"_"+str(N_i)+".dat"
            q=np.loadtxt(fname=directory+file_name,dtype="float64")
            delete=np.size(np.where(q<-1.5)[0])
            new_q=np.zeros(np.size(q)-delete)
            counter=0
            for l in range(np.size(q)):
                if q[l]>-1.5:
                    new_q[counter]=q[l]
                    counter+=1
            #mean and sigma calculus
            mean[i,j],sigma[i,j]=statistics(new_q)

    return mean,sigma

#%%
#majority rule 
k_values=np.array([9,21,36])
r_values=np.arange(0.05,0.51,0.05)
N=1000
rule="mr_rw_2"
A1=statistics_matrix(rule, k_values, r_values, N, 1000)


#%%
#majority rule
results_1=np.zeros((len(k_values),len(r_values)))
d_results_1=np.zeros((len(k_values),len(r_values)))
for i in range(len(k_values)):
    results_1[i,:],d_results_1[i,:]=xifres(A1[0][i,:],1.96*A1[1][i,:],1,-10)

# Plot the results
colors = ['b', 'g', 'r']
linestyles = ['--', '--', '--']

plt.figure(figsize=(8,5))

for j in range(len(k_values)):
    k=k_values[j]
    plt.errorbar(r_values,results_1[j,:],xerr=0.,yerr=d_results_1[j,:],
                 color=colors[j], linestyle=linestyles[j],
                          marker="o", label=f'k={k}')

plt.xlabel(r'$r$')
plt.ylabel(r'$\langle q \rangle$')
plt.title(rule+": N="+str(N))
plt.legend()
plt.grid(True)
plt.show()

#%%
#random neighbour 
k_values=np.array([9,21,36])
r_values=np.arange(0.05,0.51,0.05)
N=1000
rule="rn_rw_2"
A2=statistics_matrix(rule, k_values, r_values, N, 1000)

#%%    
#random neighbour
results_2=np.zeros((len(k_values),len(r_values)))
d_results_2=np.zeros((len(k_values),len(r_values)))
for i in range(len(k_values)):
    results_2[i,:],d_results_2[i,:]=xifres(A2[0][i,:],1.96*A2[1][i,:],1,-10)

#Plot the results
colors = ['b', 'g', 'r']
linestyles = ['--', '--', '--']

plt.figure(figsize=(8,5))

for j in range(len(k_values)):
    k=k_values[j]
    plt.errorbar(r_values,results_2[j,:],xerr=0.,yerr=d_results_2[j,:],
                 color=colors[j], linestyle="None",
                          marker="o", label=f'k={k}')
    x=np.arange(0.01,0.50001,0.01)
    plt.plot(x,N**(-2*x),label="Theoretical",color=colors[j])

plt.xlabel(r'$r$')
plt.ylabel(r'$\langle q \rangle$')
plt.title(rule+": N="+str(N))
plt.legend()
plt.grid(True)
plt.show()