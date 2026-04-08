# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:38:17 2026

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


def statistics_matrix(rule,k_values,r_values,N,N_i):
    """Reads the files for k and r (2D arrays) indicated and returns
    the mean and the standard deviation of the minima positions."""
    
    directory="../data/time_evo_minimum/"
    
    N_k=np.size(k_values)
    N_r=np.size(r_values)
    mean=np.zeros((N_k,N_r,3),dtype="float64")
    sigma=np.zeros((N_k,N_r,3),dtype="float64")
    for i in range(N_k):
        k=k_values[i]
        for j in range(N_r):
            r=r_values[j]
            #reading the file
            f_name="minimum_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
                +"_"+str(N_i)+".dat"
            results=np.loadtxt(fname=directory+f_name,dtype="float64")
            results*=(1./N)
            #mean and sigma calculus
            mean[i,j,0],sigma[i,j,0]=statistics(results[:,0])
            mean[i,j,1],sigma[i,j,1]=statistics(results[:,1])
            mean[i,j,2],sigma[i,j,2]=statistics(results[:,2])
            
    return mean,sigma

def uncertainty(mean,sigma,N_k,N_r):
    """Given a [N_k,N_r,3] array for the mean and the sigma 
    returns the mean value and its uncertainty 
    in its the same matrix shape."""
    
    t_min=np.zeros_like(mean)
    dt_min=np.zeros_like(sigma)
    
    for i in range(N_k):
        for j in range(3):
            t_min[i,:,j],dt_min[i,:,j]=xifres(mean[i,:,j],sigma[i,:,j]*1.96,5,-5)
        
    return t_min,dt_min

def box_plot(rule,index_k,k_values,r_values,N,N_i,index_strategy):
    """Reads the files for the indicated k and strategy and creates a box plot
    with the N_i values for each r.
    Strategies: Random Selection (0), Ordered by distance (1), 
                random walk (2)."""
    
    directory="../data/time_evo_minimum/"
    
    N_r=np.size(r_values)
    values=np.zeros((N_i,N_r),dtype="float64")
    k=k_values[index_k]
    for j in range(N_r):
        r=r_values[j]
        #reading the file
        f_name="minimum_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
            +"_"+str(N_i)+".dat"
        results=np.loadtxt(fname=directory+f_name,dtype="float64")
        results*=(1./N)
        values[:,j]=results[:,index_strategy]
     
    fig, ax = plt.subplots()

    ax.boxplot(
        values,
        positions=r_values,
        widths=0.005,
        showfliers=True,
        flierprops=dict(
        marker='x',        # tipo de marcador
        markersize=1,      # tamaño
        linestyle='none'   # sin líneas
    )
    )
    

    means = values.mean(axis=0)
    ax.scatter(r_values, means, color="red")
    
    ax.set_xlim(0, 0.501)
    ax.set_ylim(0,1)
    ticks = np.arange(0, 0.51, 0.1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.1f}" for t in ticks])
    
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$t_{min}$")
    
    plt.show()


def plot_minima(N,k_values,r_values,N_i):
    """Returns a plot for each k of the position of the minima as a 
    function of r. N and N_i are int and k_values and r_values 1D arrays."""
    
    N_k=np.size(k_values)
    N_r=np.size(r_values)
    
    #read the values
    m,s=statistics_matrix("mr",k_values,r_values,N,N_i)
    

    #calculate the ubcertainty
    t_min,dt_min=uncertainty(m, s, N_k, N_r)
    

    #saving the plot
    
    #folder
    directory_save="../images/minima/"
    if not exists(directory_save):
        makedirs(directory_save)
        
    if not exists(directory_save+"pdf/"):
        makedirs(directory_save+"pdf/")
    if not exists(directory_save+"png/"):
        makedirs(directory_save+"png/")
    

    for i in range(N_k):
        k=k_values[i]
        plt.figure()
        cmap=plt.get_cmap("viridis")
        plt.errorbar(r_values,t_min[i,:,0],yerr=dt_min[i,:,0]\
                     ,c=cmap(0.6),linestyle="dotted",marker="o"\
                 ,ms=3,label="Random selection")
        plt.errorbar(r_values,t_min[i,:,1],yerr=dt_min[i,:,1]\
                     ,c=cmap(0.9),linestyle="dotted",marker="o"\
                 ,ms=3,label="Ordered by distance")
        plt.errorbar(r_values,t_min[i,:,2],yerr=dt_min[i,:,2]\
                     ,c=cmap(0.1),linestyle="dotted",marker="o"\
                 ,ms=3,label="Random walk")
            
        plt.xlim([0,0.5])
        plt.ylim([0,1])
        #plt.yscale("log")
        plt.xlabel("r")
        plt.ylabel(r"Position of the minimum")
        plt.legend()
        plt.grid(True, which='both')
        plt.savefig(directory_save+"pdf/"+str(N)+"_"+str(k)+\
                    "_minima.pdf",bbox_inches="tight")
        plt.savefig(directory_save+"png/"+str(N)+"_"+str(k)+\
                    "_minima.png",bbox_inches="tight")
        plt.show()  
        plt.close()
    
    
#%%
#Values:
k_values=np.array([10,20,30,40,50])
r_values=np.arange(0.01,0.5005,0.01)
N_values=np.array([100,500,1000])
N_i=1000

for i in range(np.size(N_values)):
    plot_minima(N_values[i], k_values, r_values, N_i)

#%%
#Box plot
box_plot("mr", 1, k_values, r_values, 1000, N_i, 0)

#%%
@njit
def histogram(N,data,xmin,xmax,nbox):
    
    h=(xmax-xmin)/nbox
    valorsx=np.zeros((nbox+1),dtype="float64")
    xhis=np.zeros((nbox),dtype="float64")
    valorsx[-1]=xmax
    for i in range(nbox):
        valorsx[i]=xmin+i*h
        xhis[i]=xmin+i*h+h/2.
        
    nk=np.zeros((nbox),dtype="int32")
    for i in range(N):
        if data[i] >= xmin and data[i] <= xmax:
            nb=int(((data[i]-xmin)/h)+1)
            if (nb==nbox+1):
                nb=nb-1 
            nk[nb]+=1 
    
    vhis=np.zeros((nbox),dtype="float64")
    errhis=np.zeros((nbox),dtype="float64")
    for i in range(nbox):
        vhis[i]=nk[i]/(N*h)
        errhis[i]=(((nk[i]/N)*(1-nk[i]/N))**0.5)/(h*(N)**0.5)
        
    return vhis,errhis,xhis,h


N=1000

f_name="minimum_"+"mr"+"_"+str(N)+"_"+str(20)+"_"+str(round(0.1,2))\
    +"_"+str(N_i)+".dat"
    
directory="../data/time_evo_minimum/"

results=np.loadtxt(fname=directory+f_name,dtype="float64")
results*=(1./N)

data=results[:,0]

vhis,errhis,xhis,h=histogram(N_i, data, 0, 1, 30)
 
 
plt.figure(figsize=(7,5))

#Bars
plt.bar(xhis,vhis,width=h,color='lightblue',edgecolor='black',\
        align='center')

#Errorbars
plt.errorbar(xhis,vhis,yerr=errhis,fmt='k.',capsize=3,label="Experimental")

plt.xlabel("Minimum postion",fontsize=18)
plt.ylabel("Probability density",fontsize=18)


plt.legend(loc='upper center',bbox_to_anchor=(0.5, -0.15),fancybox=True,\
    shadow=True,ncol=2, fontsize=18)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.show()

statistics(data)

