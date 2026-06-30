# -*- coding: utf-8 -*-
"""
@author: Sergi Martínez Galindo
"""

"""
#######################################
IMPORTANT INFORMATION BEFORE EXECUTING
#######################################
    
This program provides the code to analyse the results obtained with different 
exploration strategies, node definition heuristic rules and biases.
It is important to use the corresponding strings to identify each case,
along with the heuristic rule function indicated in parentheses:

- Rule strings
    1. Majority rule
    
        A. No bias: mr (update_majority)
        B. Anchoring bias: mr_anchor (update_majority_anchor)
        C. Ambiguity bias: mr_ambiguity (update_majority_ambiguity) 
         - additional paramter M -
        D. Primacy linear: mr_primacy_linear (update_majority_weighted)
        E. Rececny linear: mr_recency_linear (update_majority_weighted)

        
    2. Random neighbour
        A. No bias: rn (update_rn)
        B. Anchoring bias: rn_anchor (update_rn_anchor)
        C. Primacy linear: rn_primacy_linear (update_rn_weighted)
        D. Rececny linear: rn_recency_linear (update_rn_weighted)
        
- Strategy strings:
    1. Random selection: rs
    2. Ordered by distance: obd
    
- Network topotlogy (without biases):
    - add to mr o rn:
        A. _WS (Watts-Strogatz) - additional parameter p_r -  
        B. _BA (Barabasi-Albert) - additional parameter c_BA -  
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numba import njit
from os.path import exists
from os import makedirs
import seaborn as sns


#%%
@njit
def statistics(x):
    """Returns mean and its uncertainty of a 2D array (N, N_i),
    computed over N_i simulations (axis=1). Returns the mean value
    and its uncertainty (95%)."""
    
    #mean and standard deviation
    sumx=np.sum(x,axis=1)
    sumx2=np.sum(x**2,axis=1)
    N_i=x.shape[1]
    xmed=sumx/N_i
    x2med=sumx2/N_i
    s2=(N_i*(x2med-xmed**2))/(N_i-1)
    s=s2**0.5
    
    #standard deviation of the mean (CLT)
    var=s2/N_i
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


def statistics_matrix(rule,k,r_values,N,N_i,strategy,index,c_BA="Random",
                      M=0,p_r=-1):
    """Reads the files for k and r (integer and 1D array) indicated and returns
    the mean and the standard deviation of the corresponding index associated
    variable (d(t),q_def(t),q(t),d_max(t),<d>(t)) as a function of time. Input:
        - N: number of nodes.
        - N_i: number of simulations.
        - k: expected number of connections per node.
        - r_values: 1D array (N_r)
        - rule: str of the rule.
        - strategy: str of the strategy used.
        - index: integer indicating the variable.
        - c_BA: "Max", "Min", "Random", only for Ba, indicates the criteria to 
        select the initial node.
        - p_r: rewiring probability, only for Watts-Strogatz
        - M: float, indicates the percentage of agreeing neighbours
        needed to trust a node. (only for ambiguity bias if not ignored)"""
    
    directory="../data/time_evo_simulations_biases/"
    
    N_r=np.size(r_values)
    results=np.zeros((N,N_r,2),dtype="float64")
    
    
    for j in range(N_r):
        r=r_values[j]
        #reading the file
        if M!=0:
            name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
                +"_"+str(N_i)+"_"+str(round(M,2))+".npz"
        elif p_r!=-1:
            name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
                +"_"+str(N_i)+"_"+str(round(p_r,5))+".npz"
        elif c_BA!="Random":
            name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
                +"_"+str(N_i)+"_"+c_BA+".npz"
        else:
            name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
                +"_"+str(N_i)+".npz"
                
        matrix=np.load(directory+name)
        data=matrix["r"].astype(np.float32) #shape (N,5,N_i)
        
        selected_data=data[:,index,:]
        
        #mean and sigma calculus
        results[:,j,0],results[:,j,1]=statistics(selected_data)

    return results

#individual plot
def final_accuracy_plot(results,r_values,N,N_i,k,rule,strategy,theory):
    """Saves a plot with <q>(r) for the corresponding rule and exploration
    strategy. Inputs:
        - results: numpy array (N,N_r,2).
        - r_values: 1D array (N_r)
        - N: number of nodes.
        - N_i: number of simulations.
        - k: expected number of connections per node.
        - rule: str of the rule.
        - strategy: str of the strategy used.
        - theory: boolean indicating if a theoretical line (random neighbour)
            is needed.
        """
    
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
        plt.legend(fontsize=18,loc='lower center',bbox_to_anchor=(0.5, 1.02),ncol=2
)
    else:
        plt.errorbar(r_values,results[-1,:,0],xerr=0,yerr=results[-1,:,1],
                     linestyle="none",marker="o",markersize=2,c=cmap(0))
    
    plt.ylim([0,1])
    plt.xlabel("$r$",fontsize=18)
    plt.ylabel(r"$\langle q \rangle$",fontsize=18)
    plt.savefig(directory_save+"pdf/"+name+".pdf",bbox_inches="tight")
    plt.savefig(directory_save+"png/"+name+".png",bbox_inches="tight")
    plt.close()


#more than one curve (not necessarly biases)    
def biases_plots(results_list,r_values,N,N_i,k,rule,strategy,biases_list,theory):
    """Saves a plot with <q>(r) for the corresponding rule and exploration
    strategy. Plots a curve for each array in the list. Inputs:
        - results_list: list of numpy arrays (N,N_r,2).
        - r_values: 1D array (N_r)
        - N: number of nodes.
        - N_i: number of simulations.
        - k: expected number of connections per node.
        - rule: str of the rule.
        - strategy: str of the strategy used.
        - biases_list: strings with the name of each case for the legend.
        - theory: boolean indicating if a theoretical line (random neighbour)
            is needed.
        """
    
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
    
    plt.legend(fontsize=18,loc='lower center',bbox_to_anchor=(0.5, 1.02),ncol=2
)
    plt.ylim([-0.05,1.05])
    plt.xlim([-0.015,0.515])
    plt.xlabel("$r$",fontsize=18)
    plt.ylabel(r"$\langle q \rangle$",fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(directory_save+"pdf/"+name+".pdf",bbox_inches="tight")
    plt.savefig(directory_save+"png/"+name+".png",bbox_inches="tight")
    plt.close()
 
#temporal evolution   
def temporal_evo(results_list,r_values,r_index,N,N_i,k,rule,
                 biases_list,ylabel,var):
    """Saves a plot with the time evolution for the indicated r_value. Inputs:
        - results_list: list of numpy arrays (N,N_r,2).
        - r_values: 1D array (N_r)
        - r_index: indicates the position of the r value to plot.
        - N: number of nodes.
        - N_i: number of simulations.
        - k: expected number of connections per node.
        - rule: str of the rule.
        - strategy: str of the strategy used.
        - biases_list: strings with the name of each case for the legend.
        - ylabel: str for the ylabel.
        - var: str for the name of the file, variable."""
    
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
        x=np.arange(0,N)/N
        y=results[:,r_index,0]
        dy=results[:,r_index,1]
        #plt.errorbar(np.arange(0,N)/N,results[:,r_index,0],
        #             xerr=0,yerr=results[:,r_index,1],
        #             linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
        #             label=biases_list[i])
        plt.plot(x,y,linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
                     label=biases_list[i])
        plt.fill_between(x, y-dy, y+dy, alpha=0.5, color=cmap(i/N_b))
    
    plt.legend(fontsize=18,loc='lower center',bbox_to_anchor=(0.5, 1.02),ncol=2
)
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
    to index (d(t),q_def(t),q(t),d_max(t),<d>(t)). Input:
        - N: number of nodes.
        - N_i: number of simulations.
        - k: expected number of connections per node.
        - r: noise [0,0.5]
        - rule: str of the rule.
        - strategy: str of the strategy used.
        - index: integer indicating the variable."""
    
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
    """Plots the histogram along the mean value cand saves it. Input:
        - vhis: values of each box.
        - errhis: uncertainty associated to the values.
        - xhis: position of the boxes.
        - h: box width.
        - N: number of nodes.
        - N_i: number of simulations.
        - k: expected number of connections per node.
        - rule: str of the rule.
        - strategy: str of the strategy used.
        - r: noise [0,0.5]
        - mean: array with the mean value."""
    
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

    plt.legend(fontsize=18,loc='lower center',bbox_to_anchor=(0.5, 1.02),ncol=2
               )
    #plt.xlim([-1,1])
    plt.xlabel(r"$\langle q \rangle$",fontsize=18)
    plt.ylabel("$p$",fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(directory_save+"pdf/"+name+".pdf",bbox_inches="tight")
    plt.savefig(directory_save+"png/"+name+".png",bbox_inches="tight")
    plt.close()
    

def box_plot(rule,k,r_values,N,N_i,strategy,index,c_BA="Random",M=0,p_r=-1):
    """Reads the files for the indicated k and strategy and creates a box plot
    with the N_i values for each r. Input:
        - N: number of nodes.
        - N_i: number of simulations.
        - k: expected number of connections per node.
        - r_values: 1D array (N_r)
        - rule: str of the rule.
        - strategy: str of the strategy used.
        - index: integer indicating the variable.
        - c_BA: "Max", "Min", "Random", only for Ba, indicates the criteria to 
        select the initial node.
        - p_r: rewiring probability, only for Watts-Strogatz
        - M: float, indicates the percentage of agreeing neighbours
        needed to trust a node. (only for ambiguity bias if not ignored)"""
    
    directory="../data/time_evo_minimum/"
    
    directory="../data/time_evo_simulations_biases/"
    
    N_r=np.size(r_values)
    results=np.zeros((N_i,N_r),dtype="float64")
    
    
    for j in range(N_r):
        r=r_values[j]
        #reading the file
        if M!=0:
            name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
                +"_"+str(N_i)+"_"+str(round(M,2))+".npz"
        elif p_r!=-1:
            name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
                +"_"+str(N_i)+"_"+str(round(p_r,5))+".npz"
        elif c_BA!="Random":
            name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
                +"_"+str(N_i)+"_"+c_BA+".npz"
        else:
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
        
    if M!=0:
        name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"\
            +"_"+str(N_i)+"_"+str(round(M,2))+"_"
    elif p_r!=-1:
        name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"\
            +"_"+str(N_i)+"_"+str(round(p_r,5))+"_"
    elif c_BA!="Random":
        name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"\
            +"_"+str(N_i)+"_"+c_BA
    else:
        name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"\
            +"_"+str(N_i)+"_"
    
     
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
        linewidth=2      # grosor opcional
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
    
    ax.set_xlim(0, 0.501)
    ax.set_ylim(-1,1)
    ticks = np.arange(0, 0.51, 0.1)
    ax.set_xticks(ticks)
    ax.tick_params(axis='both', labelsize=16)
    ax.set_xticklabels([f"{t:.1f}" for t in ticks])
    
    ax.set_xlabel(r"$r$",fontsize=18)
    ax.set_ylabel(r"$\langle q \rangle$",fontsize=18)
    
    plt.legend(handles=legend_elements,fontsize=18,loc='lower center',
               bbox_to_anchor=(0.5, 1.02),ncol=2)
    
    plt.savefig(directory_save+"pdf/"+name+".pdf",bbox_inches="tight")
    plt.savefig(directory_save+"png/"+name+".png",bbox_inches="tight")
    plt.show()
    plt.close()
    
def heatmap(results_list,r_values,name):
    """Creates the heatmapo for the Watts-strogatz networks. Inputs:
        - r_values: 1D array (N_r)
        - results_list: list of numpy arrays (N,N_r,2).
        - name: str, saving file name (without .pdf or .png)"""
    
    directory_save="../images/biases/heatmaps/"
    if not exists(directory_save):
        makedirs(directory_save)
        
    if not exists(directory_save+"pdf/"):
        makedirs(directory_save+"pdf/")
    if not exists(directory_save+"png/"):
        makedirs(directory_save+"png/")
    
    q=[]
    for results in results_list:
        q.append(results[-1,:,0])
        
    M = np.array(q)

    fig, ax = plt.subplots()
    
    im = ax.imshow(
        M,
        cmap="viridis",
        origin="lower",
        aspect="auto"
    )
    
    ticks_x = np.linspace(0, 0.5, 6)
    ax.set_xticks(np.linspace(0, M.shape[1] - 1, 6))
    ax.set_xticklabels([f"{t:.1f}" for t in ticks_x])
    
    y_values = np.logspace(-4, 0, M.shape[0])
    
    ax.set_yticks(np.linspace(0, M.shape[0]-1, len(y_values)))
    ax.set_yticklabels([f"{p:.1e}" for p in y_values])
    
    ax.set_xlabel(r"$r$", fontsize=18)
    ax.set_ylabel(r"$p_r$", fontsize=18)
    
    ax.tick_params(axis='both', labelsize=12)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\langle q \rangle$",fontsize=18)
    
    plt.savefig(directory_save+"pdf/"+name+".pdf", bbox_inches="tight")
    plt.savefig(directory_save+"png/"+name+".png", bbox_inches="tight")
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
    
#Watts-Strogatz
#p_r=0.001
mr_rs_WS_1=statistics_matrix("mr_WS", k, r_values, N, N_i, "rs", 2, p_r=0.001)
mr_obd_WS_1=statistics_matrix("mr_WS", k, r_values, N, N_i, "obd", 2, p_r=0.001)
rn_rs_WS_1=statistics_matrix("rn_WS", k, r_values, N, N_i, "rs", 2, p_r=0.001)
rn_obd_WS_1=statistics_matrix("rn_WS", k, r_values, N, N_i, "obd", 2, p_r=0.001)
#p_r=0.01
mr_rs_WS_2=statistics_matrix("mr_WS", k, r_values, N, N_i, "rs", 2, p_r=0.01)
mr_obd_WS_2=statistics_matrix("mr_WS", k, r_values, N, N_i, "obd", 2, p_r=0.01)
rn_rs_WS_2=statistics_matrix("rn_WS", k, r_values, N, N_i, "rs", 2, p_r=0.01)
rn_obd_WS_2=statistics_matrix("rn_WS", k, r_values, N, N_i, "obd", 2, p_r=0.01)

#Barabasi-Albert
mr_rs_BA=statistics_matrix("mr_BA", k, r_values, N, N_i, "rs", 2)
mr_obd_BA=statistics_matrix("mr_BA", k, r_values, N, N_i, "obd", 2)
rn_rs_BA=statistics_matrix("rn_BA", k, r_values, N, N_i, "rs", 2)
rn_obd_BA=statistics_matrix("rn_BA", k, r_values, N, N_i, "obd", 2)

#k_variability
k_val=[10,20,30,40,50]
obd_mr_k=[]
rs_mr_k=[]
obd_rn_k=[]
rs_rn_k=[]
for k in k_val:
    A=statistics_matrix("mr", k, r_values, N, N_i, "rs", 2)
    rs_mr_k.append(A)
    B=statistics_matrix("mr", k, r_values, N, N_i, "obd", 2)
    obd_mr_k.append(B)
    C=statistics_matrix("rn", k, r_values, N, N_i, "rs", 2)
    rs_rn_k.append(C)
    D=statistics_matrix("rn", k, r_values, N, N_i, "obd", 2)
    obd_rn_k.append(D)

k=20 #return to original value
#N_variability
N_val=[100,500]
obd_mr_N=[]
rs_mr_N=[]
obd_rn_N=[]
rs_rn_N=[]

for N in N_val:
    A=statistics_matrix("mr", k, r_values, N, N_i, "rs", 2)
    rs_mr_N.append(A)
    B=statistics_matrix("mr", k, r_values, N, N_i, "obd", 2)
    obd_mr_N.append(B)
    C=statistics_matrix("rn", k, r_values, N, N_i, "rs", 2)
    rs_rn_N.append(C)
    D=statistics_matrix("rn", k, r_values, N, N_i, "obd", 2)
    obd_rn_N.append(D)
    
obd_mr_N.append(mr_obd)
rs_mr_N.append(mr_rs)
obd_rn_N.append(rn_obd)
rs_rn_N.append(rn_rs)

N=1000 #return to original value

#%%
#p_r dependence - WS
r_values=np.arange(0.,0.5001,0.05)
p_r_values=np.concatenate(([0], np.logspace(-4, 0, 10)))
k=20
N=1000
N_i=1000
obd_mr_WS_pr=[]
rs_mr_WS_pr=[]
obd_rn_WS_pr=[]
rs_rn_WS_pr=[]
for p_r in p_r_values:
    #Watts-Strogatz
    A=statistics_matrix("mr_WS", k, r_values, N, N_i, "rs", 2, p_r=p_r)
    rs_mr_WS_pr.append(A)
    B=statistics_matrix("mr_WS", k, r_values, N, N_i, "obd", 2, p_r=p_r)
    obd_mr_WS_pr.append(B)
    C=statistics_matrix("rn_WS", k, r_values, N, N_i, "rs", 2, p_r=p_r)
    rs_rn_WS_pr.append(C)
    D=statistics_matrix("rn_WS", k, r_values, N, N_i, "obd", 2, p_r=p_r)
    obd_rn_WS_pr.append(D)

r_values=np.arange(0.,0.5001,0.01) #return to original value
#%%   
#initial node degree order dependence - BA
BA_list=["Random","Min","Max"]
obd_mr_BA_k=[mr_obd_BA]
rs_mr_BA_k=[mr_rs_BA]
obd_rn_BA_k=[rn_obd_BA]
rs_rn_BA_k=[rn_rs_BA]
for i in range(1,3):
    A=statistics_matrix("mr_BA", k, r_values, N, N_i, "rs", 2, c_BA=BA_list[i])
    rs_mr_BA_k.append(A)
    B=statistics_matrix("mr_BA", k, r_values, N, N_i, "obd", 2, c_BA=BA_list[i])
    obd_mr_BA_k.append(B)
    C=statistics_matrix("rn_BA", k, r_values, N, N_i, "rs", 2, c_BA=BA_list[i])
    rs_rn_BA_k.append(C)
    D=statistics_matrix("rn_BA", k, r_values, N, N_i, "obd", 2, c_BA=BA_list[i])
    obd_rn_BA_k.append(D)

#%%
#heatmap

r_values_WS=np.arange(0.,0.5001,0.05)
heatmap(obd_mr_WS_pr[1:],r_values_WS,"heatmap_obd_mr")
heatmap(rs_mr_WS_pr[1:],r_values_WS,"heatmap_rs_mr")
heatmap(obd_rn_WS_pr[1:],r_values_WS,"heatmap_obd_rn")
heatmap(rs_rn_WS_pr[1:],r_values_WS,"heatmap_rs_rn")

#%%
#Comparison of different cases

#compare rules
mr_rs_list=[mr_rs,rn_rs]
mr_rs_bias=["Majority Rule", "Random Neighbour"]

biases_plots(mr_rs_list, r_values, N, N_i, k, "rules", "rs", mr_rs_bias,False)

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

network=["ER", "WS, $p_r=0.001$", "WS, $p_r=0.01$", "BA" ]
#network_topology
network_obd_mr=[mr_obd,mr_obd_WS_1,mr_obd_WS_2,mr_obd_BA]
biases_plots(network_obd_mr, r_values, N, N_i, k, "mr", "obd_network", network,False)

network_obd_rn=[rn_obd,rn_obd_WS_1,rn_obd_WS_2,rn_obd_BA]
biases_plots(network_obd_rn, r_values, N, N_i, k, "rn", "obd_network", network,False)

network_rs_mr=[mr_rs,mr_rs_WS_1,mr_rs_WS_2,mr_rs_BA]
biases_plots(network_rs_mr, r_values, N, N_i, k, "mr", "rs_network", network,False)

network_rs_rn=[rn_rs,rn_rs_WS_1,rn_rs_WS_2,rn_rs_BA]
biases_plots(network_rs_rn, r_values, N, N_i, k, "rn", "rs_network", network,False)

#k dependancy
k_list=[r"$\langle k \rangle=10$",r"$\langle k \rangle=20$",
        r"$\langle k \rangle=30$",r"$\langle k \rangle=40$",
        r"$\langle k \rangle=50$"]

biases_plots(obd_mr_k, r_values, N, N_i, k, "mr", "obd_k", k_list ,False)
biases_plots(rs_mr_k, r_values, N, N_i, k, "mr", "rs_k", k_list ,False)
biases_plots(obd_rn_k, r_values, N, N_i, k, "rn", "obd_k", k_list ,False)
biases_plots(rs_rn_k, r_values, N, N_i, k, "rn", "rs_k", k_list ,False)

#N dependancy
N_list=["$N=100$","$N=500$","$N=1000$"]

biases_plots(obd_mr_N, r_values, N, N_i, k, "mr", "obd_N", N_list ,False)
biases_plots(rs_mr_N, r_values, N, N_i, k, "mr", "rs_N", N_list ,False)
biases_plots(obd_rn_N, r_values, N, N_i, k, "rn", "obd_N", N_list ,False)
biases_plots(rs_rn_N, r_values, N, N_i, k, "rn", "rs_N", N_list ,False)


r_values=np.arange(0.,0.5001,0.05)
#pr dependency - WS
p_r_list=["$p_r=$"+str(round(i,5)) for i in p_r_values]
biases_plots(obd_mr_WS_pr, r_values, N, N_i, k, "mr", "obd_WS_pr", p_r_list ,False)
biases_plots(rs_mr_WS_pr, r_values, N, N_i, k, "mr", "rs_WS_pr", p_r_list ,False)
biases_plots(obd_rn_WS_pr, r_values, N, N_i, k, "rn", "obd_WS_pr", p_r_list ,False)
biases_plots(rs_rn_WS_pr, r_values, N, N_i, k, "rn", "rs_WS_pr", p_r_list ,False)

r_values=np.arange(0.,0.5001,0.01)
#order degree initial node dependency
biases_plots(obd_mr_BA_k, r_values, N, N_i, k, "mr", "obd_BA", BA_list ,False)
biases_plots(rs_mr_BA_k, r_values, N, N_i, k, "mr", "rs_BA", BA_list ,False)
biases_plots(obd_rn_BA_k, r_values, N, N_i, k, "rn", "obd_BA", BA_list ,False)
biases_plots(rs_rn_BA_k, r_values, N, N_i, k, "rn", "rs_BA", BA_list ,False)


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

#Watts-Strogatz
#p_r=0.001
box_plot("mr_WS", k, r_values, N, N_i, "rs", 2, p_r=0.001)
box_plot("mr_WS", k, r_values, N, N_i, "obd", 2, p_r=0.001)
box_plot("rn_WS", k, r_values, N, N_i, "rs", 2, p_r=0.001)
box_plot("rn_WS", k, r_values, N, N_i, "obd", 2, p_r=0.001)
#p_r=0.01
box_plot("mr_WS", k, r_values, N, N_i, "rs", 2, p_r=0.01)
box_plot("mr_WS", k, r_values, N, N_i, "obd", 2, p_r=0.01)
box_plot("rn_WS", k, r_values, N, N_i, "rs", 2, p_r=0.01)
box_plot("rn_WS", k, r_values, N, N_i, "obd", 2, p_r=0.01)


#Barabasi-Albert - Random
box_plot("mr_BA", k, r_values, N, N_i, "rs", 2)
box_plot("mr_BA", k, r_values, N, N_i, "obd", 2)
box_plot("rn_BA", k, r_values, N, N_i, "rs", 2)
box_plot("rn_BA", k, r_values, N, N_i, "obd", 2)


#Barabasi-Albert - Max
box_plot("mr_BA", k, r_values, N, N_i, "rs", 2, c_BA="Max")
box_plot("mr_BA", k, r_values, N, N_i, "obd", 2, c_BA="Max")
box_plot("rn_BA", k, r_values, N, N_i, "rs", 2, c_BA="Max")
box_plot("rn_BA", k, r_values, N, N_i, "obd", 2, c_BA="Max")

#Barabasi-Albert - Min
box_plot("mr_BA", k, r_values, N, N_i, "rs", 2, c_BA="Min")
box_plot("mr_BA", k, r_values, N, N_i, "obd", 2, c_BA="Min")
box_plot("rn_BA", k, r_values, N, N_i, "rs", 2, c_BA="Min")
box_plot("rn_BA", k, r_values, N, N_i, "obd", 2, c_BA="Min")

#%%
#histograms
r_hist=[0.1,0.25,0.4]
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
#Erdos-Renyi
mr_obd_d=statistics_matrix("mr", k, r_values, N, N_i, "obd", 0)
mr_obd_q=statistics_matrix("mr", k, r_values, N, N_i, "obd", 1)
mr_rs_d=statistics_matrix("mr", k, r_values, N, N_i, "rs", 0)
mr_rs_q=statistics_matrix("mr", k, r_values, N, N_i, "rs", 1)

rn_obd_d=statistics_matrix("rn", k, r_values, N, N_i, "obd", 0)
rn_obd_q=statistics_matrix("rn", k, r_values, N, N_i, "obd", 1)
rn_rs_d=statistics_matrix("rn", k, r_values, N, N_i, "rs", 0)
rn_rs_q=statistics_matrix("rn", k, r_values, N, N_i, "rs", 1)


#p_r dependence - WS
r_values=np.arange(0.,0.5001,0.05)
p_r_values=np.concatenate(([0], np.logspace(-4, 0, 10)))
k=20
N=1000
N_i=1000

obd_mr_WS_pr_d=[]
rs_mr_WS_pr_d=[]
obd_rn_WS_pr_d=[]
rs_rn_WS_pr_d=[]

obd_mr_WS_pr_q=[]
rs_mr_WS_pr_q=[]
obd_rn_WS_pr_q=[]
rs_rn_WS_pr_q=[]
for p_r in p_r_values:
    #Watts-Strogatz
    A=statistics_matrix("mr_WS", k, r_values, N, N_i, "rs", 0, p_r=p_r)
    rs_mr_WS_pr_d.append(A)
    A2=statistics_matrix("mr_WS", k, r_values, N, N_i, "rs", 1, p_r=p_r)
    rs_mr_WS_pr_q.append(A2)
    
    B=statistics_matrix("mr_WS", k, r_values, N, N_i, "obd", 0, p_r=p_r)
    obd_mr_WS_pr_d.append(B)
    B2=statistics_matrix("mr_WS", k, r_values, N, N_i, "obd", 1, p_r=p_r)
    obd_mr_WS_pr_q.append(B2)
    
    C=statistics_matrix("rn_WS", k, r_values, N, N_i, "rs", 0, p_r=p_r)
    rs_rn_WS_pr_d.append(C)
    C2=statistics_matrix("rn_WS", k, r_values, N, N_i, "rs", 1, p_r=p_r)
    rs_rn_WS_pr_q.append(C2)
    
    D=statistics_matrix("rn_WS", k, r_values, N, N_i, "obd", 0, p_r=p_r)
    obd_rn_WS_pr_d.append(D)
    D2=statistics_matrix("rn_WS", k, r_values, N, N_i, "obd", 1, p_r=p_r)
    obd_rn_WS_pr_q.append(D2)
 
    
#initial node degree order dependence - BA
r_values=np.arange(0.,0.5001,0.01)
BA_list=["Random","Min","Max"]
obd_mr_BA_k=[mr_obd_BA]
rs_mr_BA_k=[mr_rs_BA]
obd_rn_BA_k=[rn_obd_BA]
rs_rn_BA_k=[rn_rs_BA]


obd_mr_BA_k_d=[]
rs_mr_BA_k_d=[]
obd_rn_BA_k_d=[]
rs_rn_BA_k_d=[]

obd_mr_BA_k_q=[]
rs_mr_BA_k_q=[]
obd_rn_BA_k_q=[]
rs_rn_BA_k_q=[]
for i in range(3):
    #Watts-Strogatz
    A=statistics_matrix("mr_BA", k, r_values, N, N_i, "rs", 0, c_BA=BA_list[i])
    rs_mr_BA_k_d.append(A)
    A2=statistics_matrix("mr_BA", k, r_values, N, N_i, "rs", 1, c_BA=BA_list[i])
    rs_mr_BA_k_q.append(A2)
    
    B=statistics_matrix("mr_BA", k, r_values, N, N_i, "obd", 0, c_BA=BA_list[i])
    obd_mr_BA_k_d.append(B)
    B2=statistics_matrix("mr_BA", k, r_values, N, N_i, "obd", 1, c_BA=BA_list[i])
    obd_mr_BA_k_q.append(B2)
    
    C=statistics_matrix("rn_BA", k, r_values, N, N_i, "rs", 0, c_BA=BA_list[i])
    rs_rn_BA_k_d.append(C)
    C2=statistics_matrix("rn_BA", k, r_values, N, N_i, "rs", 1, c_BA=BA_list[i])
    rs_rn_BA_k_q.append(C2)
    
    D=statistics_matrix("rn_BA", k, r_values, N, N_i, "obd", 0, c_BA=BA_list[i])
    obd_rn_BA_k_d.append(D)
    D2=statistics_matrix("rn_BA", k, r_values, N, N_i, "obd", 1, c_BA=BA_list[i])
    obd_rn_BA_k_q.append(D2)

    
    
    
#Watts-Strogatz

#p_r=0.001
mr_rs_WS_1_d=statistics_matrix("mr_WS", k, r_values, N, N_i, "rs", 0, p_r=0.001)
mr_rs_WS_1_q=statistics_matrix("mr_WS", k, r_values, N, N_i, "rs", 1, p_r=0.001)
mr_obd_WS_1_d=statistics_matrix("mr_WS", k, r_values, N, N_i, "obd", 0, p_r=0.001)
mr_obd_WS_1_q=statistics_matrix("mr_WS", k, r_values, N, N_i, "obd", 1, p_r=0.001)

rn_rs_WS_1_d=statistics_matrix("rn_WS", k, r_values, N, N_i, "rs", 0, p_r=0.001)
rn_rs_WS_1_q=statistics_matrix("rn_WS", k, r_values, N, N_i, "rs", 1, p_r=0.001)
rn_obd_WS_1_d=statistics_matrix("rn_WS", k, r_values, N, N_i, "obd", 0, p_r=0.001)
rn_obd_WS_1_q=statistics_matrix("rn_WS", k, r_values, N, N_i, "obd", 1, p_r=0.001)

#p_r=0.01
mr_rs_WS_2_d=statistics_matrix("mr_WS", k, r_values, N, N_i, "rs", 0, p_r=0.01)
mr_rs_WS_2_q=statistics_matrix("mr_WS", k, r_values, N, N_i, "rs", 1, p_r=0.01)
mr_obd_WS_2_d=statistics_matrix("mr_WS", k, r_values, N, N_i, "obd", 0, p_r=0.01)
mr_obd_WS_2_q=statistics_matrix("mr_WS", k, r_values, N, N_i, "obd", 1, p_r=0.01)

rn_rs_WS_2_d=statistics_matrix("rn_WS", k, r_values, N, N_i, "rs", 0, p_r=0.01)
rn_rs_WS_2_q=statistics_matrix("rn_WS", k, r_values, N, N_i, "rs", 1, p_r=0.01)
rn_obd_WS_2_d=statistics_matrix("rn_WS", k, r_values, N, N_i, "obd", 0, p_r=0.01)
rn_obd_WS_2_q=statistics_matrix("rn_WS", k, r_values, N, N_i, "obd", 1, p_r=0.01)


#Barabasi-Albert
mr_rs_BA_d=statistics_matrix("mr_BA", k, r_values, N, N_i, "rs", 0)
mr_rs_BA_q=statistics_matrix("mr_BA", k, r_values, N, N_i, "rs", 1)
mr_obd_BA_d=statistics_matrix("mr_BA", k, r_values, N, N_i, "obd", 0)
mr_obd_BA_q=statistics_matrix("mr_BA", k, r_values, N, N_i, "obd", 1)

rn_rs_BA_d=statistics_matrix("rn_BA", k, r_values, N, N_i, "rs", 0)
rn_rs_BA_q=statistics_matrix("rn_BA", k, r_values, N, N_i, "rs", 1)
rn_obd_BA_d=statistics_matrix("rn_BA", k, r_values, N, N_i, "obd", 0)
rn_obd_BA_q=statistics_matrix("rn_BA", k, r_values, N, N_i, "obd", 1)



#k_dependency

k_val=[10,20,30,40,50]
obd_mr_k_d=[]
rs_mr_k_d=[]
obd_rn_k_d=[]
rs_rn_k_d=[]
obd_mr_k_q=[]
rs_mr_k_q=[]
obd_rn_k_q=[]
rs_rn_k_q=[]
for k_i in k_val:
    A=statistics_matrix("mr", k_i, r_values, N, N_i, "rs", 0)
    rs_mr_k_d.append(A)
    A2=statistics_matrix("mr", k_i, r_values, N, N_i, "rs", 1)
    rs_mr_k_q.append(A2)
    
    B=statistics_matrix("mr", k_i, r_values, N, N_i, "obd", 0)
    obd_mr_k_d.append(B)
    B2=statistics_matrix("mr", k_i, r_values, N, N_i, "obd", 1)
    obd_mr_k_q.append(B2)
    
    C=statistics_matrix("rn", k_i, r_values, N, N_i, "rs", 0)
    rs_rn_k_d.append(C)
    C2=statistics_matrix("rn", k_i, r_values, N, N_i, "rs", 1)
    rs_rn_k_q.append(C2)
    
    D=statistics_matrix("rn", k_i, r_values, N, N_i, "obd", 0)
    obd_rn_k_d.append(D)
    D2=statistics_matrix("rn", k_i, r_values, N, N_i, "obd", 1)
    obd_rn_k_q.append(D2)

#%%
#plots


#comparing strategies on Erdos-Renyi

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

#comparing networks' topologies with the same strategy
network=["ER", "WS, $p_r=0.001$", "WS, $p_r=0.01$", "BA" ]

y_label=r"$\langle d \rangle$"
d_list=[mr_rs_d,mr_rs_WS_1_d,mr_rs_WS_2_d,mr_rs_BA_d]
temporal_evo(d_list,r_values,10,N,N_i,k,"mr_rs",network,y_label,"d")

d_list=[mr_obd_d,mr_obd_WS_1_d,mr_obd_WS_2_d,mr_obd_BA_d]
temporal_evo(d_list,r_values,10,N,N_i,k,"mr_obd",network,y_label,"d")

d_list=[rn_rs_d,rn_rs_WS_1_d,rn_rs_WS_2_d,rn_rs_BA_d]
temporal_evo(d_list,r_values,10,N,N_i,k,"rn_rs",network,y_label,"d")

d_list=[rn_obd_d,rn_obd_WS_1_d,rn_obd_WS_2_d,rn_obd_BA_d]
temporal_evo(d_list,r_values,10,N,N_i,k,"rn_obd",network,y_label,"d")

y_label=r"$\langle q_{def} \rangle$"
q_list=[mr_rs_q,mr_rs_WS_1_q,mr_rs_WS_2_q,mr_rs_BA_q]
temporal_evo(q_list,r_values,10,N,N_i,k,"mr_rs",network,y_label,"q")

q_list=[mr_obd_q,mr_obd_WS_1_q,mr_obd_WS_2_q,mr_obd_BA_q]
temporal_evo(q_list,r_values,10,N,N_i,k,"mr_obd",network,y_label,"q")

q_list=[rn_rs_q,rn_rs_WS_1_q,rn_rs_WS_2_q,rn_rs_BA_q]
temporal_evo(q_list,r_values,10,N,N_i,k,"rn_rs",network,y_label,"q")

q_list=[rn_obd_q,rn_obd_WS_1_q,rn_obd_WS_2_q,rn_obd_BA_q]
temporal_evo(q_list,r_values,10,N,N_i,k,"rn_obd",network,y_label,"q")


# comparing BA
BA_list=["Random","Min","Max"]

y_label=r"$\langle d \rangle$"
d_list=rs_mr_BA_k_d
temporal_evo(d_list,r_values,10,N,N_i,k,"mr_rs_BA",BA_list,y_label,"d")

d_list=obd_mr_BA_k_d
temporal_evo(d_list,r_values,10,N,N_i,k,"mr_obd_BA",BA_list,y_label,"d")

d_list=rs_rn_BA_k_d
temporal_evo(d_list,r_values,10,N,N_i,k,"rn_rs_BA",BA_list,y_label,"d")

d_list=obd_rn_BA_k_d
temporal_evo(d_list,r_values,10,N,N_i,k,"rn_obd_BA",BA_list,y_label,"d")

y_label=r"$\langle q_{def} \rangle$"
q_list=rs_mr_BA_k_q
temporal_evo(q_list,r_values,10,N,N_i,k,"mr_rs_BA",BA_list,y_label,"q")

q_list=obd_mr_BA_k_q
temporal_evo(q_list,r_values,10,N,N_i,k,"mr_obd_BA",BA_list,y_label,"q")

q_list=rs_rn_BA_k_q
temporal_evo(q_list,r_values,10,N,N_i,k,"rn_rs_BA",BA_list,y_label,"q")

q_list=obd_rn_BA_k_q
temporal_evo(q_list,r_values,10,N,N_i,k,"rn_obd_BA",BA_list,y_label,"q")


r_values=np.arange(0.,0.5001,0.05)
# comparing WS
p_r_list=["$p_r=$"+str(round(i,5)) for i in p_r_values]

y_label=r"$\langle d \rangle$"
d_list=rs_mr_WS_pr_d
temporal_evo(d_list,r_values,2,N,N_i,k,"mr_rs_WS",p_r_list,y_label,"d")

d_list=obd_mr_WS_pr_d
temporal_evo(d_list,r_values,2,N,N_i,k,"mr_obd_WS",p_r_list,y_label,"d")

d_list=rs_rn_WS_pr_d
temporal_evo(d_list,r_values,2,N,N_i,k,"rn_rs_WS",p_r_list,y_label,"d")

d_list=obd_rn_WS_pr_d
temporal_evo(d_list,r_values,2,N,N_i,k,"rn_obd_WS",p_r_list,y_label,"d")

y_label=r"$\langle q_{def} \rangle$"
q_list=rs_mr_WS_pr_q
temporal_evo(q_list,r_values,2,N,N_i,k,"mr_rs_WS",p_r_list,y_label,"q")

q_list=obd_mr_WS_pr_q
temporal_evo(q_list,r_values,2,N,N_i,k,"mr_obd_WS",p_r_list,y_label,"q")

q_list=rs_rn_WS_pr_q
temporal_evo(q_list,r_values,2,N,N_i,k,"rn_rs_WS",p_r_list,y_label,"q")

q_list=obd_rn_WS_pr_q
temporal_evo(q_list,r_values,2,N,N_i,k,"rn_obd_WS",p_r_list,y_label,"q")


r_values=np.arange(0.,0.5001,0.01)
#comparing k on ER
k_list=[r"$\langle k \rangle=10$",r"$\langle k \rangle=20$",
        r"$\langle k \rangle=30$",r"$\langle k \rangle=40$",
        r"$\langle k \rangle=50$"]

y_label=r"$\langle d \rangle$"

temporal_evo(rs_mr_k_d,r_values,10,N,N_i,k,"mr_rs_k",k_list,y_label,"d")

temporal_evo(obd_mr_k_d,r_values,10,N,N_i,k,"mr_obd_k",k_list,y_label,"d")

temporal_evo(rs_rn_k_d,r_values,10,N,N_i,k,"rn_rs_k",k_list,y_label,"d")

temporal_evo(obd_rn_k_d,r_values,10,N,N_i,k,"rn_obd_k",k_list,y_label,"d")

y_label=r"$\langle q_{def} \rangle$"

temporal_evo(rs_mr_k_q,r_values,10,N,N_i,k,"mr_rs_k",k_list,y_label,"q")

temporal_evo(obd_mr_k_q,r_values,10,N,N_i,k,"mr_obd_k",k_list,y_label,"q")

temporal_evo(rs_rn_k_q,r_values,10,N,N_i,k,"rn_rs_k",k_list,y_label,"q")

temporal_evo(obd_rn_k_q,r_values,10,N,N_i,k,"rn_obd_k",k_list,y_label,"q")


#%%
"""
#######################################
IMPORTANT INFORMATION BEFORE EXECUTING
#######################################
    
This program provides the code to analyse the results obtained with different 
exploration strategies, node definition heuristic rules and biases.
It is important to use the corresponding strings to identify each case,
along with the heuristic rule function indicated in parentheses:

- Rule strings
    1. Majority rule
    
        A. No bias: mr (update_majority)
        B. Anchoring bias: mr_anchor (update_majority_anchor)
        C. Ambiguity bias: mr_ambiguity (update_majority_ambiguity) 
         - additional paramter M -
        D. Primacy linear: mr_primacy_linear (update_majority_weighted)
        E. Rececny linear: mr_recency_linear (update_majority_weighted)

        
    2. Random neighbour
        A. No bias: rn (update_rn)
        B. Anchoring bias: rn_anchor (update_rn_anchor)
        C. Primacy linear: rn_primacy_linear (update_rn_weighted)
        D. Rececny linear: rn_recency_linear (update_rn_weighted)
        
- Strategy strings:
    1. Random selection: rs
    2. Ordered by distance: obd
    
- Network topotlogy (without biases):
    - add to mr o rn:
        A. _WS (Watts-Strogatz) - additional parameter p_r -  
        B. _BA (Barabasi-Albert) - additional parameter c_BA -  
"""
