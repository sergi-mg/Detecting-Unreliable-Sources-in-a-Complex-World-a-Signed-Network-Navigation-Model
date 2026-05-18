# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:08:43 2026

@author: Sergi Martínez Galindo
"""

"""
#######################################
IMPORTANT INFORMATION BEFORE EXECUTING
#######################################
    
This program provides the code to analyse the results obtained with different 
exploration strategies, node definition heuristic rules and biases.
It is important to use the corresponding strings to identify each case:

- Rule strings
    1. Majority rule
        A. No bias: mr
        B. Anchoring bias: mr_anchor
        C. Ambiguity bias: mr_ambiguity
        D. Primacy linear: mr_primacy_linear
        E. Primacy exponential: mr_primacy_exp
        F. Primacy power law: mr_primacy_power
        G. Rececny linear: mr_recency_linear
        H. Rececny exponential: mr_recency_exp
        I. Rececny power law: mr_recency_power
        
    2. Random neighbour
        A. No bias: rn
        B. Anchoring bias: rn_anchor
        C. Primacy linear: rn_primacy_linear
        D. Primacy exponential: rn_primacy_exp
        E. Primacy power law: rn_primacy_power
        F. Rececny linear: rn_recency_linear
        G. Rececny exponential: rn_recency_exp
        H. Rececny power law: rn_recency_power
        
- Strategy strings:
    1. Random selection: rs
    2. Ordered by distance: obd
    2. Random walk: rw (note that this strategy can only be used with No bias)
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
    """Returns mean and its uncertainty of a 2D array (N, N_i),
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
    unc=1.96*sigma
    return xmed,unc

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

def statistics_matrix(rule,k,r_values,N,N_i,strategy,index,M=0):
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
        if M!=0:
            name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
                +"_"+str(N_i)+"_"+str(M)+".npz"
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
        plt.legend(fontsize=18)
    else:
        plt.errorbar(r_values,results[-1,:,0],xerr=0,yerr=results[-1,:,1],
                     linestyle="none",marker="o",markersize=2,c=cmap(0))
    
    plt.ylim([0,1])
    plt.xlabel("$r$",fontsize=18)
    plt.ylabel(r"$\langle q \rangle$",fontsize=18)
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
    
    plt.legend(fontsize=18)
    plt.ylim([0,1])
    plt.xlabel("$r$",fontsize=18)
    plt.ylabel(r"$\langle q \rangle$",fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(directory_save+"pdf/"+name+".pdf",bbox_inches="tight")
    plt.savefig(directory_save+"png/"+name+".png",bbox_inches="tight")
    plt.close()
    
def temporal_evo(results_list,r_values,r_index,N,N_i,k,rule,
                 biases_list,ylabel,var):
    """Saves a plot with <q>(r) for the corresponding rule and exploration
    strategy"""
    
    N_b=len(biases_list)
    r=r_values[r_index]
    
    #folder
    
    directory_save="../images/biases/temporal_evolution/"
    if not exists(directory_save):
        makedirs(directory_save)
        
    if not exists(directory_save+"pdf/"):
        makedirs(directory_save+"pdf/")
    if not exists(directory_save+"png/"):
        makedirs(directory_save+"png/")
        
    name=var+"_"+rule+"_"+str(N)+"_"+str(k)+"_"+str(N_i)\
        +"_"+str(round(r,2))+"_"
        
    #plot
        
    cmap=plt.get_cmap("viridis") 
    plt.figure()
        
    for i in range(N_b):
        results=results_list[i]
        plt.errorbar(np.arange(0,N)/N,results[:,r_index,0],
                     xerr=0,yerr=results[:,r_index,1],
                     linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
                     label=biases_list[i])
    
    plt.legend(fontsize=18)
    #plt.ylim([0,1])
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.xlabel("$t$",fontsize=18)
    plt.ylabel(ylabel,fontsize=18)
    plt.savefig(directory_save+"pdf/"+name+".pdf",bbox_inches="tight")
    plt.savefig(directory_save+"png/"+name+".png",bbox_inches="tight")
    plt.close()
    
#Histogram function

def histogram_program(N,N_i,k,r,rule,strategy,index):    
    """Creates a histogram for the specified noise (r), expected number
    of conncetions (k), rule and strategy for the variable corresponding
    to index (d(t),q_def(t),q(t),d_max(t),<d>(t))."""
    
    #reading the data file
    
    directory="../data/time_evo_simulations_biases/"
    
    name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
        +"_"+str(N_i)+".npz"
    matrix=np.load(directory+name)
    data=matrix["r"].astype(np.float32) #shape (N,5,N_i)
    
    #Plotting the histogram for the final state
    selected_data=data[-1,index,:]
    
    x_min=selected_data.min()
    x_max=selected_data.max()
    mean=selected_data.mean()
    
    vhis,errhis,xhis,h=histogram(N_i,selected_data,x_min-0.05,x_max+0.05,
                                 int(np.sqrt(N_i)))
    
    plot_histogram(vhis,errhis,xhis,h,N,N_i,k,rule,strategy,r,mean)
    

@njit
def histogram(N,data,xmin,xmax,nbox):
    """Returns the histogram of data (N-sized vector) with nbox boxes between
    xmin and xmax. Outputs: vhis,errhis,xhis,h
        - vhis: values of each box.
        - errhis: uncertainty associated to the values.
        - xhis: position of the boxes.
        - h: box width."""
    
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

def plot_histogram(vhis,errhis,xhis,h,N,N_i,k,rule,strategy,r,mean):
    """Plots the histogram along the mean value."""
    
    #folder
    directory_save="../images/biases/histograms/"
    if not exists(directory_save):
        makedirs(directory_save)
        
    if not exists(directory_save+"pdf/"):
        makedirs(directory_save+"pdf/")
    if not exists(directory_save+"png/"):
        makedirs(directory_save+"png/")
        
    name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(N_i)+"_"+str(r)+"_"
    
    #plot
        
    cmap=plt.get_cmap("viridis") 
    plt.figure()
    
    #Bars
    plt.bar(xhis,vhis,width=h,color=cmap(0),edgecolor='black',\
            align='center',alpha=0.7,label="Histogram")
    #Errorbars
    plt.errorbar(xhis,vhis,yerr=errhis,fmt='k.',capsize=3)
    
    # Línea horizontal de la media
    plt.axvline(mean, color=cmap(0.67), linestyle='--',
                label=f'Average value={mean:.2f}')

    plt.legend(fontsize=18)
    #plt.xlim([-1,1])
    plt.xlabel(r"$\langle q \rangle$",fontsize=18)
    plt.ylabel("$p$",fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(directory_save+"pdf/"+name+".pdf",bbox_inches="tight")
    plt.savefig(directory_save+"png/"+name+".png",bbox_inches="tight")
    plt.close()
    

def box_plot(rule,k,r_values,N,N_i,strategy,index):
    """Reads the files for the indicated k and strategy and creates a box plot
    with the N_i values for each r.
    Strategies: Random Selection (0), Ordered by distance (1), 
                random walk (2)."""
    
    directory="../data/time_evo_minimum/"
    
    directory="../data/time_evo_simulations_biases/"
    
    N_r=np.size(r_values)
    results=np.zeros((N_i,N_r),dtype="float64")
    
    
    for j in range(N_r):
        r=r_values[j]
        #reading the file
        name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
            +"_"+str(N_i)+".npz"
        matrix=np.load(directory+name)
        data=matrix["r"].astype(np.float32) #shape (N,5,N_i)
        
        selected_data=data[-1,index,:]
        
        results[:,j]=selected_data[:]
        
    #folder
    
    directory_save="../images/biases/box_plots/"
    if not exists(directory_save):
        makedirs(directory_save)
        
    if not exists(directory_save+"pdf/"):
        makedirs(directory_save+"pdf/")
    if not exists(directory_save+"png/"):
        makedirs(directory_save+"png/")
    
    name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(N_i)+"_"
     
    fig, ax = plt.subplots()
    
    cmap=plt.get_cmap("viridis")

    ax.boxplot(
        results,
        positions=r_values,
        widths=0.005,
        showfliers=True,
        flierprops=dict(
        marker='x',        # tipo de marcador
        markersize=1,      # tamaño
        linestyle='none'   # sin líneas
    ), medianprops=dict(
        color=cmap(0.67),      # color de la mediana
        linewidth=2       # grosor opcional
    )
    )
    
     
    means = results.mean(axis=0)
    ax.plot(r_values,means,linestyle='none',marker='o',markersize=2,
            color=cmap(0))
    
    from matplotlib.lines import Line2D
    legend_elements = [
    Line2D([0], [0], color=cmap(0.67), lw=2, label='Median value'),
    Line2D([0], [0], marker='o', color='w', label='Mean value',
           markerfacecolor=cmap(0), markersize=5)
    ]
    
    ax.legend(handles=legend_elements)
    
    ax.set_xlim(0, 0.501)
    ax.set_ylim(-1,1)
    ticks = np.arange(0, 0.51, 0.1)
    ax.set_xticks(ticks)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xticklabels([f"{t:.1f}" for t in ticks])
    
    ax.set_xlabel(r"$r$",fontsize=18)
    ax.set_ylabel(r"$\langle q \rangle$",fontsize=18)
    
    plt.savefig(directory_save+"pdf/"+name+".pdf",bbox_inches="tight")
    plt.savefig(directory_save+"png/"+name+".png",bbox_inches="tight")
    plt.show()
    plt.close()
    
    
    
#%%
#parameter's definition
r_values=np.arange(0.0,0.5005,0.01)
k=20
N=1000
N_i=1000

#%%
#data reading 

#index: 0-d(t),1-q_def(t),2-q(t),3-d_max(t),4-<d>(t)

#Results for the global accuracy q

#majority rule - random selection
mr_rs=statistics_matrix("mr", k, r_values, N, N_i, "rs", 2)
mr_rs_amb=statistics_matrix("mr_ambiguity", k, r_values, N, N_i, "rs", 2)
mr_rs_anc=statistics_matrix("mr_anchor", k, r_values, N, N_i, "rs", 2)
mr_rs_p=statistics_matrix("mr_primacy_linear", k, r_values, N, N_i, "rs", 2)
mr_rs_r=statistics_matrix("mr_recency_linear", k, r_values, N, N_i, "rs", 2)

#majority rule - ordered by distance
mr_obd=statistics_matrix("mr", k, r_values, N, N_i, "obd", 2)
mr_obd_amb=statistics_matrix("mr_ambiguity", k, r_values, N, N_i, "obd", 2)
mr_obd_anc=statistics_matrix("mr_anchor", k, r_values, N, N_i, "obd", 2)
mr_obd_p=statistics_matrix("mr_primacy_linear", k, r_values, N, N_i, "obd", 2)
mr_obd_r=statistics_matrix("mr_recency_linear", k, r_values, N, N_i, "obd", 2)

#random neighbour - random selection
rn_rs=statistics_matrix("rn", k, r_values, N, N_i, "rs", 2)
rn_rs_anc=statistics_matrix("rn_anchor", k, r_values, N, N_i, "rs", 2)
rn_rs_p=statistics_matrix("rn_primacy_linear", k, r_values, N, N_i, "rs", 2)
rn_rs_r=statistics_matrix("rn_recency_linear", k, r_values, N, N_i, "rs", 2)

#random neighbour - ordered by distance
rn_obd=statistics_matrix("rn", k, r_values, N, N_i, "obd", 2)
rn_obd_anc=statistics_matrix("rn_anchor", k, r_values, N, N_i, "obd", 2)
rn_obd_p=statistics_matrix("rn_primacy_linear", k, r_values, N, N_i, "obd", 2)
rn_obd_r=statistics_matrix("rn_recency_linear", k, r_values, N, N_i, "obd", 2)

#percentage threshold dependance in the ambiguity effect
obd_ambiguity=[mr_obd]
rs_ambiguity=[mr_rs]
for i in range(4):
    M_i=0.6+i*0.1
    A=statistics_matrix("mr_ambiguity", k, r_values, N, N_i, "rs", 2, M=M_i)
    rs_ambiguity.append(A)
    B=statistics_matrix("mr_ambiguity", k, r_values, N, N_i, "obd", 2,M=M_i)
    obd_ambiguity.append(B)

#%%
#Comparison of the biases' effects

#majority rule - random selection

mr_rs_list=[mr_rs,mr_rs_amb,mr_rs_anc,mr_rs_p,mr_rs_r]
mr_rs_bias=["Without bias", "Ambiguity effect" ,"Anchoring",\
            "Primacy bias", "Recency bias"]

biases_plots(mr_rs_list, r_values, N, N_i, k, "mr", "rs", mr_rs_bias,False)

#majority rule - ordered by distance

mr_obd_list=[mr_obd,mr_obd_amb,mr_obd_anc,mr_obd_p,mr_obd_r]
mr_obd_bias=["Without bias", "Ambiguity effect" ,"Anchoring",\
            "Primacy bias", "Recency bias"]

biases_plots(mr_obd_list, r_values, N, N_i, k, "mr", "obd", mr_obd_bias,False)

#random neighbour - random selection

rn_rs_list=[rn_rs,rn_rs_anc,rn_rs_p,rn_rs_r]
rn_rs_bias=["Without bias","Anchoring",\
            "Primacy bias", "Recency bias"]

biases_plots(rn_rs_list, r_values, N, N_i, k, "rn", "rs", rn_rs_bias,True)

#random neighbour - ordered by distance

rn_obd_list=[rn_obd,rn_obd_anc,rn_obd_p,rn_obd_r]
rn_obd_bias=["Without bias","Anchoring",\
            "Primacy bias", "Recency bias"]

biases_plots(rn_obd_list, r_values, N, N_i, k, "rn", "obd", rn_obd_bias,True)

#percentage threshold dependance in the ambiguity effect
names=["Without bias","$M=60\%$","$M=70\%$","$M=80\%$","$M=90\%$"]
#obd
biases_plots(obd_ambiguity, r_values, N, N_i, k, "mr", "obd_amb", names,False)
#random selection
biases_plots(rs_ambiguity, r_values, N, N_i, k, "mr", "rs_amb", names,False)


#%%
#box plots
#majority rule - random selection
box_plot("mr", k, r_values, N, N_i, "rs", 2)
box_plot("mr_ambiguity", k, r_values, N, N_i, "rs", 2)
box_plot("mr_anchor", k, r_values, N, N_i, "rs", 2)
box_plot("mr_primacy_linear", k, r_values, N, N_i, "rs", 2)
box_plot("mr_recency_linear", k, r_values, N, N_i, "rs", 2)

#majority rule - ordered by distance
box_plot("mr", k, r_values, N, N_i, "obd", 2)
box_plot("mr_ambiguity", k, r_values, N, N_i, "obd", 2)
box_plot("mr_anchor", k, r_values, N, N_i, "obd", 2)
box_plot("mr_primacy_linear", k, r_values, N, N_i, "obd", 2)
box_plot("mr_recency_linear", k, r_values, N, N_i, "obd", 2)

#random neighbour - random selection
box_plot("rn", k, r_values, N, N_i, "rs", 2)
box_plot("rn_anchor", k, r_values, N, N_i, "rs", 2)
box_plot("rn_primacy_linear", k, r_values, N, N_i, "rs", 2)
box_plot("rn_recency_linear", k, r_values, N, N_i, "rs", 2)

#random neighbour - ordered by distance
box_plot("rn", k, r_values, N, N_i, "obd", 2)
box_plot("rn_anchor", k, r_values, N, N_i, "obd", 2)
box_plot("rn_primacy_linear", k, r_values, N, N_i, "obd", 2)
box_plot("rn_recency_linear", k, r_values, N, N_i, "obd", 2)

#%%
#histograms
r_hist=[0.05,0.25,0.45]
for r in r_hist:
    
    #majority rule - random selection
    histogram_program(N,N_i,k,r,"mr","rs",2)
    histogram_program(N,N_i,k,r,"mr_anchor","rs",2)
    histogram_program(N,N_i,k,r,"mr_ambiguity","rs",2)
    histogram_program(N,N_i,k,r,"mr_primacy_linear","rs",2)
    histogram_program(N,N_i,k,r,"mr_recency_linear","rs",2)
    
    #majority rule - ordered by distance
    histogram_program(N,N_i,k,r,"mr","obd",2)
    histogram_program(N,N_i,k,r,"mr_anchor","obd",2)
    histogram_program(N,N_i,k,r,"mr_ambiguity","obd",2)
    histogram_program(N,N_i,k,r,"mr_primacy_linear","obd",2)
    histogram_program(N,N_i,k,r,"mr_recency_linear","obd",2)
    
    #random neighbour - random selection
    histogram_program(N,N_i,k,r,"rn","rs",2)
    histogram_program(N,N_i,k,r,"rn_anchor","rs",2)
    histogram_program(N,N_i,k,r,"rn_primacy_linear","rs",2)
    histogram_program(N,N_i,k,r,"rn_recency_linear","rs",2)
    
    #random neighbour - ordered by distance
    histogram_program(N,N_i,k,r,"rn","obd",2)
    histogram_program(N,N_i,k,r,"rn_anchor","obd",2)
    histogram_program(N,N_i,k,r,"rn_primacy_linear","obd",2)
    histogram_program(N,N_i,k,r,"rn_recency_linear","obd",2)
    
#%%
#temporal evolution
#data
mr_obd_d=statistics_matrix("mr", k, r_values, N, N_i, "obd", 0)
mr_obd_q=statistics_matrix("mr", k, r_values, N, N_i, "obd", 1)
mr_rs_d=statistics_matrix("mr", k, r_values, N, N_i, "rs", 0)
mr_rs_q=statistics_matrix("mr", k, r_values, N, N_i, "rs", 1)

rn_obd_d=statistics_matrix("rn", k, r_values, N, N_i, "obd", 0)
rn_obd_q=statistics_matrix("rn", k, r_values, N, N_i, "obd", 1)
rn_rs_d=statistics_matrix("rn", k, r_values, N, N_i, "rs", 0)
rn_rs_q=statistics_matrix("rn", k, r_values, N, N_i, "rs", 1)

#%%
#plots

strategies=["Random Selection","Ordered by distance"]

y_label=r"$\langle d \rangle$"
d_list=[mr_rs_d,mr_obd_d]
temporal_evo(d_list,r_values,10,N,N_i,k,"mr",strategies,y_label,"d")

d_list=[rn_rs_d,rn_obd_d]
temporal_evo(d_list,r_values,10,N,N_i,k,"rn",strategies,y_label,"d")

y_label=r"$\langle q_{def} \rangle$"
q_list=[mr_rs_q,mr_obd_q]
temporal_evo(q_list,r_values,10,N,N_i,k,"mr",strategies,y_label,"q")

q_list=[rn_rs_q,rn_obd_q]
temporal_evo(q_list,r_values,10,N,N_i,k,"rn",strategies,y_label,"q")
#%%
"""
#######################################
IMPORTANT INFORMATION BEFORE EXECUTING
#######################################
    
This program provides the code to analyse the results obtained with different 
exploration strategies, node definition heuristic rules and biases.
It is important to use the corresponding strings to identify each case:

- Rule strings
    1. Majority rule
        A. No bias: mr
        B. Anchoring bias: mr_anchor
        C. Ambiguity bias: mr_ambiguity
        D. Primacy linear: mr_primacy_linear
        E. Primacy exponential: mr_primacy_exp
        F. Primacy power law: mr_primacy_power
        G. Rececny linear: mr_recency_linear
        H. Rececny exponential: mr_recency_exp
        I. Rececny power law: mr_recency_power
        
    2. Random neighbour
        A. No bias: rn
        B. Anchoring bias: rn_anchor
        C. Primacy linear: rn_primacy_linear
        D. Primacy exponential: rn_primacy_exp
        E. Primacy power law: rn_primacy_power
        F. Rececny linear: rn_recency_linear
        G. Rececny exponential: rn_recency_exp
        H. Rececny power law: rn_recency_power
        
- Strategy strings:
    1. Random selection: rs
    2. Ordered by distance: obd
    2. Random walk: rw (note that this strategy can only be used with No bias)
"""