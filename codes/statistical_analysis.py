# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:08:43 2026

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
    """Returns mean and its standard deviation of a 2D array (N, N_i),
    computed over simulations (axis=1)."""
    
    #mean and standard deviation
    sumx=np.sum(x,axis=1)
    sumx2=np.sum(x**2,axis=1)
    N=x.shape[1]
    xmed=sumx/N
    x2med=sumx2/N
    s2=(N*(x2med-xmed**2))/(N-1)
    s=s2**0.5
    
    #standard deviation of the mean (CLT)
    var=s2/N
    var[var<0]=0
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

def statistics_matrix(rule,k,r_values,N,N_i,strategy,index):
    """Reads the files for k and r (integer and 1D array) indicated and returns
    the mean and the standard deviation of the corresponding index associated
    variable (d(t),q_def(t),q(t),d_max(t),<d>(t)) as a function of time."""
    
    directory="../data/time_evo_simulations_biases/"
    
    N_r=np.size(r_values)
    results=np.zeros((N,N_r,2),dtype="float64")
    
    
    for j in range(N_r):
        r=r_values[j]
        #reading the file
        name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
            +"_"+str(N_i)+".npz"
        matrix=np.load(directory+name)
        data=matrix["r"].astype(np.float32) #shape (N,5,N_i)
        
        selected_data=data[:,index,:]
        
        #mean and sigma calculus
        results[:,j,0],results[:,j,1]=statistics(selected_data)

    return results

def final_accuracy_plot(results,r_values,N,N_i,k,rule,strategy,theory):
    """Saves a plot with <q>(r) for the corresponding rule and exploration
    strategy"""
    
    #folder
    
    directory_save="../images/biases/independent_final_state/"
    if not exists(directory_save):
        makedirs(directory_save)
        
    if not exists(directory_save+"pdf/"):
        makedirs(directory_save+"pdf/")
    if not exists(directory_save+"png/"):
        makedirs(directory_save+"png/")
        
    name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(N_i)+"_"
        
    #plot
        
    cmap=plt.get_cmap("viridis") 
    plt.figure()
    if theory==True:
        x=np.arange(0,0.501,0.005)
        plt.plot(x,N**(-2.*x),c="black",label="Theoretical")
        
        plt.errorbar(r_values,results[-1,:,0],xerr=0,yerr=results[-1,:,1],
                     linestyle="none",marker="o",markersize=2,c=cmap(0)
                     ,label="Simulations")
        plt.legend()
    else:
        plt.errorbar(r_values,results[-1,:,0],xerr=0,yerr=results[-1,:,1],
                     linestyle="none",marker="o",markersize=2,c=cmap(0))
    
    plt.ylim([0,1])
    plt.xlabel("$r$")
    plt.ylabel(r"$\langle q \rangle$")
    plt.savefig(directory_save+"pdf/"+name+".pdf",bbox_inches="tight")
    plt.savefig(directory_save+"png/"+name+".png",bbox_inches="tight")
    plt.close()
    
def biases_plots(results_list,r_values,N,N_i,k,rule,strategy,biases_list,theory):
    """Saves a plot with <q>(r) for the corresponding rule and exploration
    strategy"""
    
    N_b=len(biases_list)
    
    #folder
    
    directory_save="../images/biases/biases_effect/"
    if not exists(directory_save):
        makedirs(directory_save)
        
    if not exists(directory_save+"pdf/"):
        makedirs(directory_save+"pdf/")
    if not exists(directory_save+"png/"):
        makedirs(directory_save+"png/")
        
    name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(N_i)+"_"
        
    #plot
        
    cmap=plt.get_cmap("viridis") 
    plt.figure()
    
    if theory==True:
        x=np.arange(0,0.501,0.005)
        plt.plot(x,N**(-2.*x),c="black",label="Theoretical")
        
    for i in range(N_b):
        results=results_list[i]
        plt.errorbar(r_values,results[-1,:,0],xerr=0,yerr=results[-1,:,1],
                     linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
                     label=biases_list[i])
    
    plt.legend()
    plt.ylim([0,1])
    plt.xlabel("$r$")
    plt.ylabel(r"$\langle q \rangle$")
    plt.savefig(directory_save+"pdf/"+name+".pdf",bbox_inches="tight")
    plt.savefig(directory_save+"png/"+name+".png",bbox_inches="tight")
    plt.close()
    
#Histogram function

def histogram_program(N,N_i,k,r,rule,strategy,index):
    
    #reading the data file
    
    directory="../data/time_evo_simulations_biases/"
    
    name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
        +"_"+str(N_i)+".npz"
    matrix=np.load(directory+name)
    data=matrix["r"].astype(np.float32) #shape (N,5,N_i)
    
    #Plotting the histogram for the final state
    selected_data=data[-1,index,:]
    
    vhis,errhis,xhis,h=histogram(N_i,selected_data,0,1,int(np.sqrt(N_i)))
    
    plot_histogram(vhis,errhis,xhis,h,N,N_i,k,rule,strategy,r)
    

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
            nk[nb-1]+=1 
    
    vhis=np.zeros((nbox),dtype="float64")
    errhis=np.zeros((nbox),dtype="float64")
    for i in range(nbox):
        vhis[i]=nk[i]/(N*h)
        errhis[i]=(((nk[i]/N)*(1-nk[i]/N))**0.5)/(h*(N)**0.5)
        
    return vhis,errhis,xhis,h

def plot_histogram(vhis,errhis,xhis,h,N,N_i,k,rule,strategy,r):
    
    #folder
    directory_save="../images/biases/histograms/"
    if not exists(directory_save):
        makedirs(directory_save)
        
    if not exists(directory_save+"pdf/"):
        makedirs(directory_save+"pdf/")
    if not exists(directory_save+"png/"):
        makedirs(directory_save+"png/")
        
    name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(N_i)+"_"
    
    #plot
        
    cmap=plt.get_cmap("viridis") 
    plt.figure()
    
    #Bars
    plt.bar(xhis,vhis,width=h,color=cmap(0.75),edgecolor='black',\
            align='center')
    #Errorbars
    plt.errorbar(xhis,vhis,yerr=errhis,fmt='k.',capsize=3)

    plt.xlabel(r"$\langle q \rangle$")
    plt.ylabel("$p$")
    plt.savefig(directory_save+"pdf/"+name+".pdf",bbox_inches="tight")
    plt.savefig(directory_save+"png/"+name+".png",bbox_inches="tight")
    plt.close()
    

    
#%%
r_values=np.arange(0.01,0.5005,0.01)
k=20
N=1000
N_i=1000

#%%
#data reading (final accuracy)

#majority rule - random selection
mr_rs=statistics_matrix("mr", k, r_values, N, N_i, "rs", 2)
mr_rs_amb=statistics_matrix("mr_ambiguity", k, r_values, N, N_i, "rs", 2)
mr_rs_anc=statistics_matrix("mr_anchor", k, r_values, N, N_i, "rs", 2)

#majority rule - ordered by distance
mr_obd=statistics_matrix("mr", k, r_values, N, N_i, "obd", 2)
mr_obd_amb=statistics_matrix("mr_ambiguity", k, r_values, N, N_i, "obd", 2)
mr_obd_anc=statistics_matrix("mr_anchor", k, r_values, N, N_i, "obd", 2)

#random neighbour - random selection
rn_rs=statistics_matrix("rn", k, r_values, N, N_i, "rs", 2)
rn_rs_anc=statistics_matrix("rn_anchor", k, r_values, N, N_i, "rs", 2)

#random neighbour - ordered by distance
rn_obd=statistics_matrix("rn", k, r_values, N, N_i, "obd", 2)
rn_obd_anc=statistics_matrix("rn_anchor", k, r_values, N, N_i, "obd", 2)

#%%

#majority rule - random selection

mr_rs_list=[mr_rs,mr_rs_amb,mr_rs_anc]
mr_rs_bias=["Without bias", "Ambiguity effect" ,"Anchoring"]

biases_plots(mr_rs_list, r_values, N, N_i, k, "mr", "rs", mr_rs_bias,False)

#majority rule - ordered by distance

mr_obd_list=[mr_obd,mr_obd_amb,mr_obd_anc]
mr_obd_bias=["Without bias", "Ambiguity effect" ,"Anchoring"]

biases_plots(mr_obd_list, r_values, N, N_i, k, "mr", "obd", mr_obd_bias,False)

#random neighbour - random selection

rn_rs_list=[rn_rs,rn_rs_anc]
rn_rs_bias=["Without bias","Anchoring"]

biases_plots(rn_rs_list, r_values, N, N_i, k, "rn", "rs", rn_rs_bias,True)

#random neighbour - ordered by distance

rn_obd_list=[rn_obd,rn_obd_anc]
rn_obd_bias=["Without bias" ,"Anchoring"]

biases_plots(rn_obd_list, r_values, N, N_i, k, "rn", "obd", rn_obd_bias,True)

#%%
#individual plots
final_accuracy_plot(mr_rs, r_values, N, N_i, k, "mr", "rs",False)
final_accuracy_plot(mr_rs_amb, r_values, N, N_i, k, "mr_amb", "rs",False)
final_accuracy_plot(mr_rs_anc, r_values, N, N_i, k, "mr_anchor", "rs",False)
final_accuracy_plot(mr_obd, r_values, N, N_i, k, "mr", "obd",False)
final_accuracy_plot(mr_obd_amb, r_values, N, N_i, k, "mr_amb", "obd",False)
final_accuracy_plot(mr_obd_anc, r_values, N, N_i, k, "mr_anchor", "obd",False)
final_accuracy_plot(rn_rs, r_values, N, N_i, k, "rn", "rs",True)
final_accuracy_plot(rn_rs_anc, r_values, N, N_i, k, "rn_anchor", "rs",True)
final_accuracy_plot(rn_obd, r_values, N, N_i, k, "rn", "obd",True)
final_accuracy_plot(rn_obd_anc, r_values, N, N_i, k, "rn_anchor", "obd",True)