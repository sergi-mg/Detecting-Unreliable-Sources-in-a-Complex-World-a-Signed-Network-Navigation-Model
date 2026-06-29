# -*- coding: utf-8 -*-
"""
@author: Sergi Martínez Galindo
"""

"""
#######################################
IMPORTANT INFORMATION BEFORE EXECUTING
#######################################
    
This program provides the code to create the plots used in the memory with 
different exploration strategies, node definition heuristic rules and biases.
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

def raw_data(rule,k,r_values,N,N_i,strategy,index,c_BA="Random",M=0,p_r=-1):
    """Reads the files for the indicated k and strategy and returns the
    raw data (N_i,N_r) of the final state. Input:
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
        
    return results

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
        - index: integer indicating the variable.
    Outputs: vhis,errhis,xhis,h
        - vhis: values of each box.
        - errhis: uncertainty associated to the values.
        - xhis: position of the boxes.
        - h: box width."""
    
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
    
    vhis,errhis,xhis,h=histogram(N_i,selected_data,x_min,x_max,
                                 int(np.sqrt(N_i)))
    
    return vhis,errhis,xhis,h,mean

#%%
#parameter's definition
r_values=np.arange(0.0,0.5005,0.01)
k=20
N=1000
N_i=1000

directory_save="../images/final_plots/pdf/"
if not exists(directory_save):
    makedirs(directory_save)
    
#%%
#------------------------------------------------------------------------------
#Fig 2
#------------------------------------------------------------------------------

mr_rs=statistics_matrix("mr", k, r_values, N, N_i, "rs", 2)
rn_rs=statistics_matrix("rn", k, r_values, N, N_i, "rs", 2)

mr_obd=statistics_matrix("mr", k, r_values, N, N_i, "obd", 2)
rn_obd=statistics_matrix("rn", k, r_values, N, N_i, "obd", 2)


#%%

from scipy.special import gammaln
mr_rs_list=[rn_rs,mr_rs]
mr_obd_list=[rn_obd,mr_obd]
biases_list=["Random Neighbour", "Majority"]
N_b=len(biases_list)
results_list_1=mr_rs_list
results_list_2=mr_obd_list
r_l=[results_list_1,results_list_2]


N=1000
x=np.arange(0,0.501,0.005)
y_t=np.exp(gammaln(N+1-2*x)-gammaln(N+1)-gammaln(2-2*x))
cmap=plt.get_cmap("viridis") 
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
fig.subplots_adjust(wspace=0.2)
for j in range(2):
    results_list=r_l[j]
    
    ax[j].plot(x,y_t,c="black",ls="--",
               label="Theoretical Random Neighbour")
    for i in range(N_b):
        results=results_list[i]
        ax[j].errorbar(r_values,results[-1,:,0],xerr=0,yerr=results[-1,:,1],
                     linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
                     label=biases_list[i])
        ax[j].set_xlabel(r"$r$", fontsize=18)
        
        ax[j].tick_params(axis='both', labelsize=16)
        #ax[j].set_ylim([-0.05,1.05])
        #ax[j].set_xlim([-0.015,0.515])
        
        
        if j==0:
            ax[j].set_title("(a) Random Selection",y=-0.3, fontsize=20)
            ax[j].set_ylabel(r"$\langle q \rangle$", fontsize=18)
        else:
            ax[j].set_title("(b) Ordered by Distance",y=-0.3, fontsize=20)
        

handles, labels = ax[0].get_legend_handles_labels()

fig.legend(handles, labels,
           fontsize=18,
           loc='lower center',
           bbox_to_anchor=(0.5, .95),
           ncol=3)

            

name="ER_q"
plt.savefig(directory_save+name+".pdf",bbox_inches="tight")
plt.show()
    

#%%
#------------------------------------------------------------------------------
#Fig 3
#------------------------------------------------------------------------------

#Boxplots ER

#data
obd_data=raw_data("mr", k, r_values, N, N_i, "obd", 2)
rs_data=raw_data("mr", k, r_values, N, N_i, "rs", 2)


#%%
#plot
data=[rs_data,obd_data]
cmap = plt.get_cmap("viridis")
#create the subplot
fig, ax = plt.subplots(1, 2, figsize=(16,5))
fig.subplots_adjust(wspace=0.2)
for i in range(2):
    results=data[i]
    ax[i].boxplot(
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
    ax[i].plot(r_values,means,linestyle='none',marker='o',markersize=2,
            color=cmap(0))
    
    ax[i].set_xlim(0, 0.5)
    ax[i].set_ylim(-1,1)
    ticks = np.arange(0, 0.51, 0.1)
    ax[i].set_xticks(ticks)
    ax[i].set_yticks(np.arange(-1, 1.01, 0.5))
    ax[i].tick_params(axis='both', labelsize=16)
    ax[i].set_xticklabels([f"{t:.1f}" for t in ticks])
    
    
    ax[i].set_xlabel(r"$r$",fontsize=18)
    if i==0:
        ax[i].set_ylabel(r"$\langle q \rangle$",fontsize=18)
        ax[i].set_title("(a) Random Selection",y=-0.3, fontsize=20)
    else:
        ax[i].set_title("(b) Ordered by Distance",y=-0.3, fontsize=20)


from matplotlib.lines import Line2D


legend_elements = [
Line2D([0], [0], color=cmap(0.67), lw=2, label='Median value'),
Line2D([0], [0], marker='o', color='w', label='Mean value',
       markerfacecolor=cmap(0), markersize=5)
]

# Leyenda global
fig.legend(handles=legend_elements,fontsize=18,loc='lower center',
           bbox_to_anchor=(0.5, .95),ncol=2)
name="ER_strategies_boxplot"
plt.savefig(directory_save+name+".pdf",bbox_inches="tight")
plt.show()

#%%
#------------------------------------------------------------------------------
#Fig 4
#------------------------------------------------------------------------------

#Histogrames Erdos-Renyi
r_l=[0.1,0.25,0.4]
st=["rs","obd"]
label=["Random Selection", "Ordered by Distance"]
cmap=plt.get_cmap("viridis") 
fig, ax = plt.subplots(1, 3, figsize=(24, 5))
fig.subplots_adjust(wspace=0.2)
cmap=plt.get_cmap("viridis") 
color=[cmap(i/2) for i in range(2)]
ylim=[12,3,10]
for i in range(3):
    r=r_l[i]
    for j in range(2):
        strategy=st[j]
        vhis,errhis,xhis,h,mean=histogram_program(N,N_i,k,r,"mr",strategy,2)
        #plot
            
        #Bars
        ax[i].bar(xhis,vhis,width=h,color=color[j],\
                align='center',alpha=0.3)
        #ax[i].fill_between(xhis,0,vhis,color=color[j],alpha=0.2)
        #Errorbars
        ax[i].errorbar(xhis,vhis,yerr=errhis,capsize=3,color=color[j],
                       label=label[j],linestyle="none",marker="o",ms=2)
        
        # Línea horizontal de la media
        ax[i].axvline(mean, color=color[j], linestyle='--')
    
        #plt.xlim([-1,1])
        ax[i].set_xlabel(r"$q$", fontsize=18)
        ax[i].tick_params(axis='both', labelsize=16)
        ax[i].set_xlim([-1,1])
        ax[i].set_ylim([0,ylim[i]])
        
        if i==0:
            ax[i].set_title("(a) $r=0.10$",y=-0.3, fontsize=20)
            ax[i].set_ylabel(r"$p$", fontsize=18)
        elif i==1:
            ax[i].set_title("(b) $r=0.25$",y=-0.3, fontsize=20)
        else:
            ax[i].set_title("(c) $r=0.40$",y=-0.3, fontsize=20)
        
handles, labels = ax[1].get_legend_handles_labels()

fig.legend(handles, labels,
           fontsize=18,
           loc='lower center',
           bbox_to_anchor=(0.5, .95),
           ncol=5)

name="ER_strategies_hist"
plt.savefig(directory_save+name+".pdf",bbox_inches="tight")
plt.show()


#%%
#------------------------------------------------------------------------------
#Fig 5
#------------------------------------------------------------------------------

#Temporal evolution Erdos-Renyi strategies
#Erdos-Renyi

mr_obd_d=statistics_matrix("mr", k, r_values, N, N_i, "obd", 0)
mr_obd_q=statistics_matrix("mr", k, r_values, N, N_i, "obd", 1)
mr_rs_d=statistics_matrix("mr", k, r_values, N, N_i, "rs", 0)
mr_rs_q=statistics_matrix("mr", k, r_values, N, N_i, "rs", 1)

#%%
r_index_l=[10,10,25,40]

biases_list=["Random Selection","Ordered by distance"]

y_label=[r"$\langle d \rangle$",r"$\langle q_{def} \rangle$",
         r"$\langle q_{def} \rangle$",r"$\langle q_{def} \rangle$"]

results_list_1=[mr_rs_d,mr_obd_d]
results_list_2=[mr_rs_q,mr_obd_q]
r_l=[results_list_1,results_list_2,results_list_2,results_list_2]
N_b=len(biases_list)
ymin=[0.3,0.03,0.001]
ymax=[1,1,1]

#plot
titles=["(a) $r=0.10$","(b) $r=0.10$","(c) $r=0.25$","(d) $r=0.40$"]
cmap=plt.get_cmap("viridis") 
fig, ax = plt.subplots(1, 4, figsize=(28, 3.5))
fig.subplots_adjust(wspace=0.4)
for j in range(4):
    results_list=r_l[j]
    r_index=r_index_l[j]
    r=r_values[r_index]
    title_j=titles[j]
    for i in range(N_b):
        results=results_list[i]
        x=np.arange(1,N)/N
        y=results[1:,r_index,0]
        dy=results[1:,r_index,1]
        ax[j].plot(x,y,linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
                     label=biases_list[i])
        ax[j].fill_between(x, y-dy, y+dy, alpha=0.5, color=cmap(i/N_b))
        ax[j].set_xscale("log")
        ax[j].set_yscale("log")
        ax[j].tick_params(axis='both', labelsize=16)
        ax[j].grid(True, which="both", linestyle="--", alpha=0.4)
        ax[j].set_xlabel("$t/N$",fontsize=18)
        ax[j].set_ylabel(y_label[j],fontsize=18)
        ax[j].set_title(title_j,y=-0.4, fontsize=20)
        if j>0:
            ax[j].set_ylim([ymin[j-1],ymax[j-1]])


        

handles, labels = ax[1].get_legend_handles_labels()

fig.legend(handles, labels,
           fontsize=18,
           loc='lower center',
           bbox_to_anchor=(0.5, .95),
           ncol=2,markerscale=3)

name="ER_strategies_time_evo"
plt.savefig(directory_save+name+".pdf",bbox_inches="tight")
plt.show()

#%%
#------------------------------------------------------------------------------
#Fig 6
#------------------------------------------------------------------------------

#N_variability
N_val=[100,500,1000]
obd_mr_N=[]
rs_mr_N=[]


for N in N_val:
    A=statistics_matrix("mr", k, r_values, N, N_i, "rs", 2)
    rs_mr_N.append(A)
    B=statistics_matrix("mr", k, r_values, N, N_i, "obd", 2)
    obd_mr_N.append(B)


#%%
biases_list=["$N=100$","$N=500$","$N=1000$"]
N_b=len(biases_list)
results_list_1=rs_mr_N
results_list_2=obd_mr_N
r_l=[results_list_1,results_list_2]

cmap=plt.get_cmap("viridis") 
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
fig.subplots_adjust(wspace=0.2)

for j in range(2):
    results_list=r_l[j]
    for i in range(N_b):
        results=results_list[i]
        ax[j].errorbar(r_values,results[-1,:,0],xerr=0,yerr=results[-1,:,1],
                     linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
                     label=biases_list[i])
        ax[j].set_xlabel(r"$r$", fontsize=18)
        ax[j].tick_params(axis='both', labelsize=16)
        ax[j].set_ylim([-0.05,1.05])
        ax[j].set_xlim([-0.015,0.515])
        
        if j==0:
            ax[j].set_title("(a) Random Selection",y=-0.3, fontsize=20)
            ax[j].set_ylabel(r"$\langle q \rangle$", fontsize=18)
        else:
            ax[j].set_title("(b) Ordered by Distance",y=-0.3, fontsize=20)
        

handles, labels = ax[1].get_legend_handles_labels()

fig.legend(handles, labels,
           fontsize=18,
           loc='lower center',
           bbox_to_anchor=(0.5, .95),
           ncol=5)

            

name="ER_N_q"
plt.savefig(directory_save+name+".pdf",bbox_inches="tight")
plt.show()

#%%
#------------------------------------------------------------------------------
#Fig 7
#------------------------------------------------------------------------------

N=1000

k_val=[10,20,30,40,50]


obd_mr_k=[]
rs_mr_k=[]

for k_i in k_val:
    A2=statistics_matrix("mr", k_i, r_values, N, N_i, "rs", 2)
    rs_mr_k.append(A2)
    
    B2=statistics_matrix("mr", k_i, r_values, N, N_i, "obd", 2)
    obd_mr_k.append(B2)
    
#%%
biases_list=[r"$\langle k \rangle=10$",r"$\langle k \rangle=20$",
        r"$\langle k \rangle=30$",r"$\langle k \rangle=40$",
        r"$\langle k \rangle=50$"]

y_label=[r"$\langle d \rangle$",r"$\langle q_{def} \rangle$"]

N_b=len(biases_list)
results_list_1=rs_mr_k
results_list_2=obd_mr_k
r_l=[results_list_1,results_list_2]
#plot
    
cmap=plt.get_cmap("viridis") 
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
fig.subplots_adjust(wspace=0.2)

for j in range(2):
    results_list=r_l[j]
    for i in range(N_b):
        results=results_list[i]
        ax[j].errorbar(r_values,results[-1,:,0],xerr=0,yerr=results[-1,:,1],
                     linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
                     label=biases_list[i])
        ax[j].set_xlabel(r"$r$", fontsize=18)
        ax[j].tick_params(axis='both', labelsize=16)
        ax[j].set_ylim([-0.05,1.05])
        ax[j].set_xlim([-0.015,0.515])
        
        if j==0:
            ax[j].set_title("(a) Random Selection",y=-0.3, fontsize=20)
            ax[j].set_ylabel(r"$\langle q \rangle$", fontsize=18)
        else:
            ax[j].set_title("(b) Ordered by Distance",y=-0.3, fontsize=20)
        

handles, labels = ax[1].get_legend_handles_labels()

fig.legend(handles, labels,
           fontsize=18,
           loc='lower center',
           bbox_to_anchor=(0.5, .95),
           ncol=5)
            

name="ER_k_q"
plt.savefig(directory_save+name+".pdf",bbox_inches="tight")
plt.show()

#%%
#------------------------------------------------------------------------------
#Fig 8
#------------------------------------------------------------------------------

#Time evolution - k_dependency
k_val=[10,20,30,40,50]
obd_mr_k_d=[]
rs_mr_k_d=[]

obd_mr_k_q=[]
rs_mr_k_q=[]

for k_i in k_val:
    A=statistics_matrix("mr", k_i, r_values, N, N_i, "rs", 0)
    rs_mr_k_d.append(A)
    A2=statistics_matrix("mr", k_i, r_values, N, N_i, "rs", 1)
    rs_mr_k_q.append(A2)
    
    B=statistics_matrix("mr", k_i, r_values, N, N_i, "obd", 0)
    obd_mr_k_d.append(B)
    B2=statistics_matrix("mr", k_i, r_values, N, N_i, "obd", 1)
    obd_mr_k_q.append(B2)
    
#%%
#OBD
r_index=10

biases_list=[r"$\langle k \rangle=10$",r"$\langle k \rangle=20$",
        r"$\langle k \rangle=30$",r"$\langle k \rangle=40$",
        r"$\langle k \rangle=50$"]

y_label=[r"$\langle d \rangle$",r"$\langle q_{def} \rangle$"]

results_list_1=obd_mr_k_d
results_list_2=obd_mr_k_q
r_l=[results_list_1,results_list_2]
N_b=len(biases_list)
r=r_values[r_index]

#plot
    
cmap=plt.get_cmap("viridis") 
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
fig.subplots_adjust(wspace=0.3)
for j in range(2):
    results_list=r_l[j]
    for i in range(N_b):
        results=results_list[i]
        x=np.arange(1,N)/N
        y=results[1:,r_index,0]
        dy=results[1:,r_index,1]
        ax[j].plot(x,y,linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
                     label=biases_list[i])
        ax[j].fill_between(x, y-dy, y+dy, alpha=0.5, color=cmap(i/N_b))
        ax[j].set_xscale("log")
        ax[j].set_yscale("log")
        ax[j].tick_params(axis='both', labelsize=16)
        ax[j].grid(True, which="both", linestyle="--", alpha=0.4)
        ax[j].set_xlabel("$t/N$",fontsize=18)
        ax[j].set_ylabel(y_label[j],fontsize=18)
"""        if j==0:
            ax[j].set_title("(b1)",y=-0.3, fontsize=20)
        else:
            ax[j].set_title("(b2)",y=-0.3, fontsize=20)"""
        

handles, labels = ax[1].get_legend_handles_labels()
fig.suptitle("(b) Ordered by Distance",y=-0.15, fontsize=20)

"""fig.legend(handles, labels,
           fontsize=18,
           loc='lower center',
           bbox_to_anchor=(0.5, .95),
           ncol=5)"""

name="ER_k_obd_evo"
plt.savefig(directory_save+name+".pdf",bbox_inches="tight")
plt.show()

#%%
#RS
r_index=10

biases_list=[r"$\langle k \rangle=10$",r"$\langle k \rangle=20$",
        r"$\langle k \rangle=30$",r"$\langle k \rangle=40$",
        r"$\langle k \rangle=50$"]

y_label=[r"$\langle d \rangle$",r"$\langle q_{def} \rangle$"]

results_list_1=rs_mr_k_d
results_list_2=rs_mr_k_q
r_l=[results_list_1,results_list_2]
N_b=len(biases_list)
r=r_values[r_index]

#plot
    
cmap=plt.get_cmap("viridis") 
#fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
fig.subplots_adjust(wspace=0.3)
for j in range(2):
    results_list=r_l[j]
    for i in range(N_b):
        results=results_list[i]
        x=np.arange(1,N)/N
        y=results[1:,r_index,0]
        dy=results[1:,r_index,1]
        ax[j].plot(x,y,linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
                     label=biases_list[i])
        ax[j].fill_between(x, y-dy, y+dy, alpha=0.5, color=cmap(i/N_b))
        ax[j].set_xscale("log")
        ax[j].set_yscale("log")
        ax[j].tick_params(axis='both', labelsize=16)
        ax[j].grid(True, which="both", linestyle="--", alpha=0.4)
        ax[j].set_xlabel("$t/N$",fontsize=18)
        ax[j].set_ylabel(y_label[j],fontsize=18)
        """if j==0:
            ax[j].set_title("(a1)",y=-0.3, fontsize=20)
        else:
            ax[j].set_title("(a2)",y=-0.3, fontsize=20)"""
        

handles, labels = ax[1].get_legend_handles_labels()

fig.legend(handles, labels,
           fontsize=18,
           loc='lower center',
           bbox_to_anchor=(0.5, .95),
           ncol=5,markerscale=3)
            
fig.suptitle("(a) Random Selection",y=-0.15, fontsize=20)


name="ER_k_rs_evo"
plt.savefig(directory_save+name+".pdf",bbox_inches="tight")
plt.show()



#%%
#------------------------------------------------------------------------------
#Fig 9
#------------------------------------------------------------------------------

#Biases
mr_rs=statistics_matrix("mr", k, r_values, N, N_i, "rs", 2)
mr_rs_amb=statistics_matrix("mr_ambiguity", k, r_values, N, N_i, "rs", 2)
mr_rs_anc=statistics_matrix("mr_anchor", k, r_values, N, N_i, "rs", 2)
mr_rs_p=statistics_matrix("mr_primacy_linear", k, r_values, N, N_i, "rs", 2)
mr_rs_r=statistics_matrix("mr_recency_linear", k, r_values, N, N_i, "rs", 2)
mr_obd=statistics_matrix("mr", k, r_values, N, N_i, "obd", 2)
mr_obd_amb=statistics_matrix("mr_ambiguity", k, r_values, N, N_i, "obd", 2)
mr_obd_anc=statistics_matrix("mr_anchor", k, r_values, N, N_i, "obd", 2)
mr_obd_p=statistics_matrix("mr_primacy_linear", k, r_values, N, N_i, "obd", 2)
mr_obd_r=statistics_matrix("mr_recency_linear", k, r_values, N, N_i, "obd", 2)

#%%
mr_rs_list=[mr_rs,mr_rs_amb,mr_rs_anc,mr_rs_p,mr_rs_r]
mr_obd_list=[mr_obd,mr_obd_amb,mr_obd_anc,mr_obd_p,mr_obd_r]
biases_list=["Without bias", "Ambiguity effect ($M=0.75$)" ,"Anchoring",\
            "Primacy bias", "Recency bias"]
N_b=len(biases_list)
results_list_1=mr_rs_list
results_list_2=mr_obd_list
r_l=[results_list_1,results_list_2]

cmap=plt.get_cmap("viridis") 
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
fig.subplots_adjust(wspace=0.2)

for j in range(2):
    results_list=r_l[j]
    for i in range(N_b):
        results=results_list[i]
        ax[j].errorbar(r_values,results[-1,:,0],xerr=0,yerr=results[-1,:,1],
                     linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
                     label=biases_list[i])
        ax[j].set_xlabel(r"$r$", fontsize=18)
        ax[j].tick_params(axis='both', labelsize=16)
        ax[j].set_ylim([-0.05,1.05])
        ax[j].set_xlim([-0.015,0.515])
        
        if j==0:
            ax[j].set_title("(a) Random Selection",y=-0.3, fontsize=20)
            ax[j].set_ylabel(r"$\langle q \rangle$", fontsize=18)
        else:
            ax[j].set_title("(b) Ordered by Distance",y=-0.3, fontsize=20)
        

handles, labels = ax[1].get_legend_handles_labels()

fig.legend(handles, labels,
           fontsize=18,
           loc='lower center',
           bbox_to_anchor=(0.5, .95),
           ncol=3)

            

name="ER_biases_q"
plt.savefig(directory_save+name+".pdf",bbox_inches="tight")
plt.show()

#%%
#------------------------------------------------------------------------------
#Fig 10
#------------------------------------------------------------------------------
#Network topology
r_values = np.arange(0., 0.5001, 0.01)
mr_rs=statistics_matrix("mr", k, r_values, N, N_i, "rs", 2)
mr_rs_WS=statistics_matrix("mr_WS", k, r_values, N, N_i, "rs", 2, p_r=0.01)
mr_rs_BA=statistics_matrix("mr_BA", k, r_values, N, N_i, "rs", 2)

mr_obd=statistics_matrix("mr", k, r_values, N, N_i, "obd", 2)
mr_obd_WS=statistics_matrix("mr_WS", k, r_values, N, N_i, "obd", 2, p_r=0.01)
mr_obd_BA=statistics_matrix("mr_BA", k, r_values, N, N_i, "obd", 2)


#%%
r_values = np.arange(0., 0.5001, 0.01)
mr_rs_list=[mr_rs,mr_rs_WS,mr_rs_BA]
mr_obd_list=[mr_obd,mr_obd_WS,mr_obd_BA]
biases_list=["ER", "WS ($p_r=0.01$)", "BA (Random Initial Node)"]
N_b=len(biases_list)
results_list_1=mr_rs_list
results_list_2=mr_obd_list
r_l=[results_list_1,results_list_2]

cmap=plt.get_cmap("viridis") 
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
fig.subplots_adjust(wspace=0.2)

for j in range(2):
    results_list=r_l[j]
    for i in range(N_b):
        results=results_list[i]
        ax[j].errorbar(r_values,results[-1,:,0],xerr=0,yerr=results[-1,:,1],
                     linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
                     label=biases_list[i])
        ax[j].set_xlabel(r"$r$", fontsize=18)
        ax[j].tick_params(axis='both', labelsize=16)
        ax[j].set_ylim([-0.05,1.05])
        ax[j].set_xlim([-0.015,0.515])
        
        if j==0:
            ax[j].set_title("(a) Random Selection",y=-0.3, fontsize=20)
            ax[j].set_ylabel(r"$\langle q \rangle$", fontsize=18)
        else:
            ax[j].set_title("(b) Ordered by Distance",y=-0.3, fontsize=20)
        

handles, labels = ax[1].get_legend_handles_labels()

fig.legend(handles, labels,
           fontsize=18,
           loc='lower center',
           bbox_to_anchor=(0.5, .95),
           ncol=3)

            

name="ER_topo_q"
plt.savefig(directory_save+name+".pdf",bbox_inches="tight")
plt.show()
#%%
#------------------------------------------------------------------------------
#Fig 11
#------------------------------------------------------------------------------
#Barabasi Albert - Boxplot
r_values=np.arange(0.,0.5001,0.01)
obd_data=raw_data("mr_BA", k, r_values, N, N_i, "obd", 2)
obd_data_M=raw_data("mr_BA", k, r_values, N, N_i, "obd", 2, c_BA="Max")
#%%
data=[obd_data, obd_data_M]
#plot
cmap = plt.get_cmap("viridis")
#create the subplot
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
fig.subplots_adjust(wspace=0.2)
for i in range(2):
    results=data[i]
    ax[i].boxplot(
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
    ax[i].plot(r_values,means,linestyle='none',marker='o',markersize=2,
            color=cmap(0))
    
    ax[i].set_xlim(0, 0.5)
    ax[i].set_ylim(-1,1)
    ticks = np.arange(0, 0.51, 0.1)
    ax[i].set_xticks(ticks)
    ax[i].set_yticks(np.arange(-1, 1.01, 0.5))
    ax[i].tick_params(axis='both', labelsize=16)
    ax[i].set_xticklabels([f"{t:.1f}" for t in ticks])
    
    
    ax[i].set_xlabel(r"$r$",fontsize=18)
    if i==0:
        ax[i].set_ylabel(r"$\langle q \rangle$",fontsize=18)
        ax[i].set_title("(a) Initial Node: Random ",y=-0.3, fontsize=20)
    else:
        ax[i].set_title("(b) Initial Node: Highest Degree",y=-0.3, fontsize=20)


from matplotlib.lines import Line2D


legend_elements = [
Line2D([0], [0], color=cmap(0.67), lw=2, label='Median value'),
Line2D([0], [0], marker='o', color='w', label='Mean value',
       markerfacecolor=cmap(0), markersize=5)
]

# Leyenda global
fig.legend(handles=legend_elements,fontsize=18,loc='lower center',
           bbox_to_anchor=(0.5, .95),ncol=2)
name="BA_boxplot"
plt.savefig(directory_save+name+".pdf",bbox_inches="tight")
plt.show()


#%%
#------------------------------------------------------------------------------
#Fig 12
#------------------------------------------------------------------------------
#Barabasi Albert - Time evo
BA_rand_obd_q=statistics_matrix("mr_BA", k, r_values, N, N_i, "obd", 1)
BA_max_obd_q=statistics_matrix("mr_BA", k, r_values, N, N_i, "obd", 1, c_BA="Max")

BA_rand_obd_d=statistics_matrix("mr_BA", k, r_values, N, N_i, "obd", 0)
BA_max_obd_d=statistics_matrix("mr_BA", k, r_values, N, N_i, "obd", 0, c_BA="Max")
#%%
r_index_l=[10,10,25,40]

biases_list=["Initial Node: Random","Initial Node: Highest Degree"]

y_label=[r"$\langle d \rangle$",r"$\langle q_{def} \rangle$",
         r"$\langle q_{def} \rangle$",r"$\langle q_{def} \rangle$"]

results_list_1=[BA_rand_obd_d,BA_max_obd_d]
results_list_2=[BA_rand_obd_q,BA_max_obd_q]
r_l=[results_list_1,results_list_2,results_list_2,results_list_2]
N_b=len(biases_list)
ymin=[0.6,0.03,0.001]
ymax=[1,1,1]

#plot
titles=["$r=0.10$","$r=0.10$","$r=0.25$","r=0.40"]
titles=["(a)","(b)"]
cmap=plt.get_cmap("viridis") 
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
fig.subplots_adjust(wspace=0.3)
for j in range(2):
    results_list=r_l[j]
    r_index=r_index_l[j]
    r=r_values[r_index]
    title_j=titles[j]
    for i in range(N_b):
        results=results_list[i]
        x=np.arange(1,N)/N
        y=results[1:,r_index,0]
        dy=results[1:,r_index,1]
        ax[j].plot(x,y,linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
                     label=biases_list[i])
        ax[j].fill_between(x, y-dy, y+dy, alpha=0.5, color=cmap(i/N_b))
        ax[j].set_xscale("log")
        ax[j].set_yscale("log")
        ax[j].tick_params(axis='both', labelsize=16)
        ax[j].grid(True, which="both", linestyle="--", alpha=0.4)
        ax[j].set_xlabel("$t/N$",fontsize=18)
        ax[j].set_ylabel(y_label[j],fontsize=18)
        ax[j].set_title(title_j,y=-0.3, fontsize=20)
        if j>0:
            ax[j].set_ylim([ymin[j-1],ymax[j-1]])


        

handles, labels = ax[1].get_legend_handles_labels()

fig.legend(handles, labels,
           fontsize=18,
           loc='lower center',
           bbox_to_anchor=(0.5, .95),
           ncol=2,markerscale=3)

name="BA_time_evo"
plt.savefig(directory_save+name+".pdf",bbox_inches="tight")
plt.show()

#%%
#Watts-Strogatz
#p_r dependence - WS
r_values=np.arange(0.,0.5001,0.05)
p_r_values=np.concatenate(([0], np.logspace(-4, 0, 10)))
k=20
N=1000
N_i=1000

obd_mr_WS=[]
obd_mr_WS_pr_d=[]
obd_mr_WS_pr_q=[]

for p_r in p_r_values:
    #Watts-Strogatz
    
    B=statistics_matrix("mr_WS", k, r_values, N, N_i, "obd", 0, p_r=p_r)
    obd_mr_WS_pr_d.append(B)
    B2=statistics_matrix("mr_WS", k, r_values, N, N_i, "obd", 1, p_r=p_r)
    obd_mr_WS_pr_q.append(B2)
    B3=statistics_matrix("mr_WS", k, r_values, N, N_i, "obd", 2, p_r=p_r)
    obd_mr_WS.append(B3)
    
#%%
#------------------------------------------------------------------------------
#Fig 13
#------------------------------------------------------------------------------
import matplotlib as mpl
r_values = np.arange(0., 0.5001, 0.05)
p_r_values = np.concatenate(([0], np.logspace(-4, 0, 10)))


results_list = obd_mr_WS
N_b = len(p_r_values)

cmap = plt.get_cmap("viridis")
norm = mpl.colors.LogNorm(
    vmin=p_r_values[1],
    vmax=p_r_values[-1]
)

fig, ax = plt.subplots(figsize=(8, 5))
fig.subplots_adjust(wspace=0.2)

for i in range(N_b):

    results = results_list[i]

    if p_r_values[i] == 0:
        color = "black"
        label = r"$p_r=0$"
    else:
        color = cmap(norm(p_r_values[i]))
        label = None

    ax.errorbar(r_values,results[-1, :, 0],xerr=0, yerr=results[-1, :, 1],
                linestyle="dashed",marker="o",markersize=3,c=color,label=label)

ax.set_xlabel(r"$r$", fontsize=18)
ax.set_ylabel(r"$\langle q \rangle$", fontsize=18)

ax.tick_params(axis='both', labelsize=16)

ax.set_ylim([-0.05, 1.05])
ax.set_xlim([-0.015, 0.515])
ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

# Colorbar
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r"$p_r$", fontsize=18)
cbar.ax.tick_params(labelsize=14)

ax.legend(fontsize=16)

name = "WS_q"
plt.savefig(directory_save + name + ".pdf", bbox_inches="tight")
plt.show()


#%%
#------------------------------------------------------------------------------
#Fig 14
#------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

y_label = [r"$\langle d \rangle$", r"$\langle q_{def} \rangle$"]

r_values = np.arange(0., 0.5001, 0.05)
p_r_values = np.concatenate(([0], np.logspace(-4, 0, 10)))

r_index = 2

N_b = len(p_r_values)
results_list_1 = [obd_mr_WS_pr_d, obd_mr_WS_pr_q]

cmap = plt.get_cmap("viridis")
norm = mpl.colors.LogNorm(vmin=p_r_values[1],vmax=p_r_values[-1])

fig, ax = plt.subplots(1, 2, figsize=(16, 5))
fig.subplots_adjust(wspace=0.3)

mask=[0,2,4,6,7,8,9,10]
for j in range(2):

    results_list = results_list_1[j]

    for i in mask:

        results = results_list[i]

        x = np.arange(1, N)/N
        y = results[1:, r_index, 0]
        dy = results[1:, r_index, 1]

        if p_r_values[i] == 0:
            color = "black"
        else:
            color = cmap(norm(p_r_values[i]))


        ax[j].fill_between(x,y-dy,y+dy,
            color=color,alpha=0.3,zorder=2*i)


        ax[j].plot(x,y,linestyle="-",
            marker="o",markersize=2,c=color,zorder=2*i+1)

    ax[j].set_xscale("log")
    ax[j].set_yscale("log")
    
    if 0 in mask:
        ax[j].plot([],[],color="black",
            marker="o",linestyle="none",label=r"$p_r=0$")

        ax[j].legend(fontsize=14,loc="best",frameon=True)

    ax[j].tick_params(axis='both', labelsize=16)
    ax[j].grid(True, which="both", linestyle="--", alpha=0.4)

    ax[j].set_xlabel("$t/N$", fontsize=18)
    ax[j].set_ylabel(y_label[j], fontsize=18)

    if j == 0:
        ax[j].set_title("(a)", y=-0.3, fontsize=20)
    else:
        ax[j].set_title("(b)", y=-0.3, fontsize=20)


# colorbar
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = fig.colorbar(
    sm,
    ax=ax,
    pad=0.02
)

cbar.set_label(r"$p_r$", fontsize=18)
cbar.ax.tick_params(labelsize=14)


name = "WS_obd_evo"
plt.savefig(directory_save + name + ".pdf", bbox_inches="tight")
plt.show()

#%%
#------------------------------------------------------------------------------
#Fig 15
#------------------------------------------------------------------------------
#M dependency
#percentage threshold dependance in the ambiguity effect
mr_obd=statistics_matrix("mr", k, r_values, N, N_i, "obd", 2)
mr_rs=statistics_matrix("mr", k, r_values, N, N_i, "rs", 2)
r_values=np.arange(0.0,0.5005,0.01)
obd_ambiguity=[mr_obd]
rs_ambiguity=[mr_rs]
for i in range(4):
    M_i=0.6+i*0.1
    A=statistics_matrix("mr_ambiguity", k, r_values, N, N_i, "rs", 2, M=M_i)
    rs_ambiguity.append(A)
    B=statistics_matrix("mr_ambiguity", k, r_values, N, N_i, "obd", 2,M=M_i)
    obd_ambiguity.append(B)
    
#%%
mr_rs_list=rs_ambiguity
mr_obd_list=obd_ambiguity
biases_list=["Without bias", "$M=0.6$" ,"$M=0.7$",\
            "$M=0.8$", "$M=0.9$"]
N_b=len(biases_list)
results_list_1=mr_rs_list
results_list_2=mr_obd_list
r_l=[results_list_1,results_list_2]

cmap=plt.get_cmap("viridis") 
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
fig.subplots_adjust(wspace=0.2)

for j in range(2):
    results_list=r_l[j]
    for i in range(N_b):
        results=results_list[i]
        ax[j].errorbar(r_values,results[-1,:,0],xerr=0,yerr=results[-1,:,1],
                     linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
                     label=biases_list[i])
        ax[j].set_xlabel(r"$r$", fontsize=18)
        ax[j].tick_params(axis='both', labelsize=16)
        ax[j].set_ylim([-0.05,1.05])
        ax[j].set_xlim([-0.015,0.515])
        
        if j==0:
            ax[j].set_title("(a) Random Selection",y=-0.3, fontsize=20)
            ax[j].set_ylabel(r"$\langle q \rangle$", fontsize=18)
        else:
            ax[j].set_title("(b) Ordered by Distance",y=-0.3, fontsize=20)
        

handles, labels = ax[1].get_legend_handles_labels()

fig.legend(handles, labels,
           fontsize=18,
           loc='lower center',
           bbox_to_anchor=(0.5, .95),
           ncol=5)

            

name="ER_biases_M"
plt.savefig(directory_save+name+".pdf",bbox_inches="tight")
plt.show()

#%%
#------------------------------------------------------------------------------
#Fig 16
#------------------------------------------------------------------------------
#Ordering bias - Time evo

r_values=np.arange(0.0,0.5005,0.01)
mr_rs_d=statistics_matrix("mr", k, r_values, N, N_i, "rs", 0)
mr_rs_q=statistics_matrix("mr", k, r_values, N, N_i, "rs", 1)
mr_rs_p_d=statistics_matrix("mr_primacy_linear", k, r_values, N, N_i, "rs", 0)
mr_rs_p_q=statistics_matrix("mr_primacy_linear", k, r_values, N, N_i, "rs", 1)
mr_rs_r_d=statistics_matrix("mr_recency_linear", k, r_values, N, N_i, "rs", 0)
mr_rs_r_q=statistics_matrix("mr_recency_linear", k, r_values, N, N_i, "rs", 1)


#%%
biases_list=["No bias","Primacy", "Recency"]

y_label=[r"$\langle d \rangle$",r"$\langle q_{def} \rangle$",
         r"$\langle q_{def} \rangle$",r"$\langle q_{def} \rangle$"]

results_list_1=[mr_rs_d,mr_rs_p_d,mr_rs_r_d]
results_list_2=[mr_rs_q,mr_rs_p_q,mr_rs_r_q]
r_l=[results_list_1,results_list_2,results_list_2,results_list_2]
N_b=len(biases_list)
ymin=[0.3,0.03,0.001]
ymax=[1,1,1]

#plot
titles=["$r=0.1$","$r=0.1$","$r=0.25$","r=0.4"]
titles=["(a)","(b)"]
cmap=plt.get_cmap("viridis") 
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
fig.subplots_adjust(wspace=0.3)
r_index_l=[10,10]
for j in range(2):
    results_list=r_l[j]
    r_index=r_index_l[j]
    r=r_values[r_index]
    title_j=titles[j]
    for i in range(N_b):
        results=results_list[i]
        x=np.arange(1,N)/N
        y=results[1:,r_index,0]
        dy=results[1:,r_index,1]
        ax[j].plot(x,y,linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
                     label=biases_list[i])
        ax[j].fill_between(x, y-dy, y+dy, alpha=0.5, color=cmap(i/N_b))
        ax[j].set_xscale("log")
        ax[j].set_yscale("log")
        ax[j].tick_params(axis='both', labelsize=16)
        ax[j].grid(True, which="both", linestyle="--", alpha=0.4)
        ax[j].set_xlabel("$t/N$",fontsize=18)
        ax[j].set_ylabel(y_label[j],fontsize=18)
        #ax[j].set_title(title_j,y=-0.3, fontsize=20)
        if j>0:
            ax[j].set_ylim([ymin[j-1],ymax[j-1]])


fig.suptitle("(a) Random Selection",y=-0.1, fontsize=20)        

handles, labels = ax[1].get_legend_handles_labels()

fig.legend(handles, labels,
           fontsize=18,
           loc='lower center',
           bbox_to_anchor=(0.5, .95),
           ncol=3,markerscale=3)

name="bias_time_evo_rs"
plt.savefig(directory_save+name+".pdf",bbox_inches="tight")
plt.show()

#%%
mr_obd_d=statistics_matrix("mr", k, r_values, N, N_i, "obd", 0)
mr_obd_q=statistics_matrix("mr", k, r_values, N, N_i, "obd", 1)
mr_obd_p_d=statistics_matrix("mr_primacy_linear", k, r_values, N, N_i, "obd", 0)
mr_obd_p_q=statistics_matrix("mr_primacy_linear", k, r_values, N, N_i, "obd", 1)
mr_obd_r_d=statistics_matrix("mr_recency_linear", k, r_values, N, N_i, "obd", 0)
mr_obd_r_q=statistics_matrix("mr_recency_linear", k, r_values, N, N_i, "obd", 1)


#%%


biases_list=["No bias","Primacy", "Recency"]

y_label=[r"$\langle d \rangle$",r"$\langle q_{def} \rangle$",
         r"$\langle q_{def} \rangle$",r"$\langle q_{def} \rangle$"]

results_list_1=[mr_obd_d,mr_obd_p_d,mr_obd_r_d]
results_list_2=[mr_obd_q,mr_obd_p_q,mr_obd_r_q]
r_l=[results_list_1,results_list_2,results_list_2,results_list_2]
N_b=len(biases_list)
ymin=[0.5,0.03,0.001]
ymax=[1,1,1]

#plot
titles=["$r=0.10$","$r=0.10$","$r=0.25$","r=0.40"]
titles=["(a)","(b)"]
cmap=plt.get_cmap("viridis") 
fig, ax = plt.subplots(1, 2, figsize=(12.8, 4.8))
fig.subplots_adjust(wspace=0.3)
r_index_l=[10,10]
for j in range(2):
    results_list=r_l[j]
    r_index=r_index_l[j]
    r=r_values[r_index]
    title_j=titles[j]
    for i in range(N_b):
        results=results_list[i]
        x=np.arange(1,N)/N
        y=results[1:,r_index,0]
        dy=results[1:,r_index,1]
        ax[j].plot(x,y,linestyle="none",marker="o",markersize=2,c=cmap(i/N_b),
                     label=biases_list[i])
        ax[j].fill_between(x, y-dy, y+dy, alpha=0.5, color=cmap(i/N_b))
        ax[j].set_xscale("log")
        ax[j].set_yscale("log")
        ax[j].tick_params(axis='both', labelsize=16)
        ax[j].grid(True, which="both", linestyle="--", alpha=0.4)
        ax[j].set_xlabel("$t/N$",fontsize=18)
        ax[j].set_ylabel(y_label[j],fontsize=18)
        #ax[j].set_title(title_j,y=-0.3, fontsize=20)
        if j>0:
            ax[j].set_ylim([ymin[j-1],ymax[j-1]])


fig.suptitle("(b) Ordered by Distance",y=-0.1, fontsize=20)       

handles, labels = ax[1].get_legend_handles_labels()


name="bias_time_evo_obd"
plt.savefig(directory_save+name+".pdf",bbox_inches="tight")
plt.show()



