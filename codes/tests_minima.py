# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 12:22:03 2026

@author: Sergi Martínez Galindo
"""
#%%
#libraries 
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numba import njit
import networkx as nx

#%%
#Exploring the network
@njit    
def accuracy(s_t,o,N):
    """Returns the accuracy. Inputs:
        - s_t: 1D array with the true values of the nodes.
        - o: shape (N,2) array with each component being (0,0),(1,0) or (0,1), 
            where (p+,p-), defined by the observer.
        - N: number of nodes."""
        
    s_o=o[:,0]-o[:,1]
    q=np.sum(s_o*s_t)/N 
    
    return q

@njit
def update_majority(J_o,o,neighbours,k,i_s):
    """Returns the updated o matrix using the majority rule.
    o must be a shape (N,2) array with each component from 1 to N being (0,0),
    (1,0) or (0,1). Inputs:
        - i_s: integer from 0 to N-1. 
        - J_o: int32 NxN array which contains the signed connections.
        - neighbours: 1D array with the defined neighbours of i_s node. 
        - k: number of defined neighbours of i_s node."""
    #previous definitions
    s_o=o[:,0]-o[:,1]
    #we define the opinion considering all defined neighbours
    new_value=np.sign(np.sum(s_o[neighbours]*J_o[i_s,neighbours]))
    #if there is a draw we select one randomly
    if new_value==0:
        new_value=1-2*np.random.randint(0,2) 
    #we update the results
    # update
    if new_value==1:
        o[i_s,0]=1
        o[i_s,1]=0
    else:
        o[i_s,0]=0
        o[i_s,1]=1

    return o

@njit
def update_rn(J_o,o,neighbours,k,i_s):
    """Returns the updated o matrix using the random neighbour rule.
    o must be a shape (N,2) array with each component from 1 to N being (0,0),
    (1,0) or (0,1). Inputs:
        - i_s: integer from 0 to N-1. 
        - J_o: int32 NxN array which contains the signed connections.
        - neighbours: 1D array with the defined neighbours of i_s node. 
        - k: number of defined neighbours of i_s node."""
    #previous definitions
    s_o=o[:,0]-o[:,1]
    #we choose a neighbour
    c_n=neighbours[np.random.randint(k)]
    #update
    new_value=J_o[i_s,c_n]*s_o[c_n]
    if new_value==1:
        o[i_s,0]=1
        o[i_s,1]=0
    else:
        o[i_s,0]=0
        o[i_s,1]=1

    return o

@njit
def explore_nw_rw(o,J_o,update_rule,n_a,n_n,def_nodes,modify,d,s):
    """Explores the network and returns the final o matrix. Inputs:
        - o: shape (N,2) array with each component being (0,0),(1,0) or (0,1),
            since here it acts as the initial condition, only one non-zero
            component. 
        - J_o: int32 NxN array which contains the signed connections.
        - update_rule: function.
        - n_a: shape (N,k_max) array comtaineing the neihbours of each
            node, if k_i<k_max the rest of the values are -1. 
        - n_n: 1D array containin the number of neighbours (k_i) of each node.
        - def_nodes: 1D boolean array where True corresponts to defined.
        - modify: boolean indicating if a defined node can be modified.
        - d: array containing the distance to node 0.
        - s: 1D int32 array with values +1,-1 of the truth network.
        Outputs: 
            -observables: array (N,5) d(t),q_def(t),q(t),d_max(t),<d>(t)"""

    #number of subjects
    N=o.shape[0]
    #opinion definition
    o_new=o.copy()
    #previous node
    i_p=0
    
    #time evolutions
    observables=np.zeros((N,5),dtype="float64")
    #d(t),q_def(t),q(t),d_max(t),<d>(t)
    observables[0,:]=np.array([0.,1.,1./N,0.,0.])
    
    def_n=def_nodes.copy()
    
    iterations=0
    i=-1
    while np.sum(def_n)!=N:
        iterations+=1
        if iterations>N**2:
            iterations=-10**7
            print("More iterations than expected, the loop does not stop")
        #node selected
        i_s=n_a[i_p,np.random.randint(n_n[i_p])]
        
        if def_n[i_s]==True and modify==False:
            None
        else:
            #defined neighbours of the selected node
            k_s=n_n[i_s]
            n_s=np.zeros(k_s,dtype=np.int32)
            n_s[:]=n_a[i_s,:k_s]
            k=np.sum(def_n[n_s])
            d_n=np.zeros(k,dtype=np.int32)
            counter=0
            for j in n_s:
                if def_n[j]==True:
                    d_n[counter]=j
                    counter+=1
            
            #update of opinions
            o_new=update_rule(J_o,o_new,d_n,k,i_s)
            i+=1
            
            #observables: d(t),q_def(t),q(t),d_max(t),<d>(t)
            observables[i+1,0]=d[i_s]*1.
            s_o_i=o_new[i_s,0]-o_new[i_s,1]
            observables[i+1,1]=observables[i,1]*(i+1)/(i+2)+s_o_i*s[i_s]/(i+2)
            observables[i+1,2]=accuracy(s,o_new,N)
            observables[i+1,3] = max(observables[i,3], d[i_s])
            observables[i+1,4]=observables[i,4]*(i)/(i+1)+d[i_s]/(i+1)
        
        #update of tracking variables
        def_n[i_s]=True
        i_p=i_s
        

    return observables

@njit
def explore_nw_bfs(o,J_o,update_rule,n_a,n_n,def_nodes,order,d,s):
    """Explores the network and returns the final o matrix. Inputs:
        - o: shape (N,2) array with each component being (0,0),(1,0) or (0,1),
            since here it acts as the initial condition, only one non-zero
            component. 
        - J_o: int32 NxN array which contains the signed connections.
        - update_rule: function.
        - n_a: shape (N,k_max) array comtaineing the neihbours of each
            node, if k_i<k_max the rest of the values are -1. 
        - n_n: 1D array containin the number of neighbours (k_i) of each node
        - def_nodes: 1D boolean array where True corresponts to defined. 
        - order: 1D array containing the order to follow (BFS).
        - d: array containing the distance to node 0.
        - s: 1D int32 array with values +1,-1 of the truth network.
        Outputs: 
            -observables: array (N,5) d(t),q_def(t),q(t),d_max(t),<d>(t)"""

    #number of subjects
    N=o.shape[0]
    #opinion definition
    o_new=o.copy()
    def_n=def_nodes.copy()
    
    #time evolutions
    observables=np.zeros((N,5),dtype="float64")
    #d(t),q_def(t),q(t),d_max(t),<d>(t)
    observables[0,:]=np.array([0.,1.,1./N,0.,0.])
    
    for i in range(N-1):
        
        #node selected
        i_s=order[i+1]
        
        #defined neighbours of the selected node
        k_s=n_n[i_s]
        n_s=np.zeros(k_s,dtype=np.int32)
        n_s[:]=n_a[i_s,:k_s]
        k=np.sum(def_n[n_s])
        d_n=np.zeros(k,dtype=np.int32)
        counter=0
        for j in n_s:
            if def_n[j]==True:
                d_n[counter]=j
                counter+=1
        
        #update of opinions
        o_new=update_rule(J_o,o_new,d_n,k,i_s)
        
        #update of tracking variables
        def_n[i_s]=True
        
        #observables: d(t),q_def(t),q(t),d_max(t),<d>(t)
        observables[i+1,0]=d[i_s]*1.
        s_o_i=o_new[i_s,0]-o_new[i_s,1]
        observables[i+1,1]=observables[i,1]*(i+1)/(i+2)+s_o_i*s[i_s]/(i+2)
        observables[i+1,2]=accuracy(s,o_new,N)
        observables[i+1,3] = max(observables[i,3], d[i_s])
        observables[i+1,4]=observables[i,4]*(i)/(i+1)+d[i_s]/(i+1) 
    return observables
        

@njit
def explore_nw_r(o,J_o,update_rule,n_a,n_n,e_n,def_nodes,d,s):
    """Inputs:
        - o: shape (N,2) array with each component being (0,0),(1,0) or (0,1),
            since here it acts as the initial condition, only one non-zero
            component. 
        - J_o: int32 NxN array which contains the signed connections.
        - update_rule: function.
        - n_a: shape (N,k_max) array comtaineing the neihbours of each
            node, if k_i<k_max the rest of the values are -1. 
        - n_n: 1D array containin the number of neighbours (k_i) of each node.
        - e_n: 1D boolean array where True corresponds to being eligible.
        - def_nodes: 1D boolean array where True corresponts to defined.
        - d: array containing the distance to node 0.
        - s: 1D int32 array with values +1,-1 of the truth network.
    Outputs: 
        -observables: array (N,5) d(t),q_def(t),q(t),d_max(t),<d>(t)"""
        
    #number of subjects
    N=o.shape[0]

    #time evolutions
    observables=np.zeros((N,5),dtype="float64")
    #d(t),q_def(t),q(t),d_max(t),<d>(t)
    observables[0,:]=np.array([0.,1.,1./N,0.,0.])
    
    #opinion definition
    o_new=o.copy()
    e_nodes=e_n.copy()
    def_n=def_nodes.copy()
    
    
    for i in range(N-1):
        
        #node selected
        N_e=np.sum(e_nodes)
        eligible_array=np.zeros(N_e,dtype=np.int32)
        counter=0
        for j in range(N):
            if e_nodes[j]==True:
                eligible_array[counter]=j
                counter+=1
        i_s=eligible_array[np.random.randint(N_e)]
        
        #defined neighbours of the selected node
        k_s=n_n[i_s]
        n_s=np.zeros(k_s,dtype=np.int32)
        n_s[:]=n_a[i_s,:k_s]
        k=np.sum(def_n[n_s])
        d_n=np.zeros(k,dtype=np.int32)
        counter=0
        for j in n_s:
            if def_n[j]==True:
                d_n[counter]=j
                counter+=1
        
        #update of opinions
        o_new=update_rule(J_o,o_new,d_n,k,i_s)
        
        #update of tracking variables
        e_nodes[i_s]=False
        def_n[i_s]=True
        for j in n_s:
            if def_n[j]==False:
                e_nodes[j]=True
        
        #observables: d(t),q_def(t),q(t),d_max(t),<d>(t)
        observables[i+1,0]=d[i_s]*1.
        s_o_i=o_new[i_s,0]-o_new[i_s,1]
        observables[i+1,1]=observables[i,1]*(i+1)/(i+2)+s_o_i*s[i_s]/(i+2)
        observables[i+1,2]=accuracy(s,o_new,N)
        observables[i+1,3] = max(observables[i,3], d[i_s])
        observables[i+1,4]=observables[i,4]*(i)/(i+1)+d[i_s]/(i+1)
        
    return observables

def ground_truth_network(N,k):
    """Generates the ground truth network.
    Inputs:
        - N: number of nodes.
        - k: expected number of edges per node.
    Outputs: 
        - s: 1D int32 array with values +1,-1.
        - J: NxN array with the signed connections.
        - neighbours_array: shape (N,k_max) array comtaineing the neihbours
            of each node, if k_i<k_max the rest of the values are -1.
        - num_neighbours: 1D array containin the number of neighbours 
            of each node.
        - isolated: boolean indicating if there are isolated clusters.
        - G: network."""
        
    #nodes' values
    s=1-2*np.random.randint(0,2,N)
    
    #netwrok creation
    isolated=True
    n=0
    p=k/(N-1)
    
    #we check that there is only one cluster
    while isolated==True:
        #connections of the network
        G=nx.erdos_renyi_graph(N,p)
        if nx.is_connected(G):           
            isolated=False
        n+=1
    #conncetion matrix
    C=nx.to_numpy_array(G, dtype=int)
    
    #signed matrix
    M=np.outer(s,s)
    J=M*C 
    
    #list of neighbours
    neighbours=[[] for _ in range(N)]
    for i, j in G.edges():
        neighbours[i].append(j)
        neighbours[j].append(i) 
        
    #maximum number of neighbours
    kmax=max(len(nbrs) for nbrs in neighbours)
    
    #neighbour's array
    neighbours_array=np.full((N, kmax), -1, dtype=np.int32)
    num_neighbours=np.zeros(N, dtype=np.int32)
    for i in range(N):
        deg=len(neighbours[i])
        num_neighbours[i]=deg
        neighbours_array[i, :deg]=neighbours[i]
        
    return s,J,neighbours_array,num_neighbours,G

def observer(s,J,r,n_a,n_n):
    """Generates the obersver variables.
    Inputs:
        - s: 1D int32 array with values +1,-1.
        - J: NxN array with the signed connections.
        - r: noise [0,0.5]
        - n_a: shape (N,k_max) array comtaineing the neihbours
            of each node, if k_i<k_max the rest of the values are -1.
        - n_n: 1D array containin the number of neighbours 
            of each node.
    Outputs: 
        - o: shape (N,2) array with each component being (0,0),(1,0) or (0,1), 
            where (p+,p-).
        - J_o: NxN array with the signed connections affected by the noise
            with probability r.
        - defined: 1D boolean array where True corresponds to defined.
        - eligible: 1D boolean array where True corresponds to being eligible."""
        
    N=len(s)
    
    #effect of noise
    J_o=J.copy()
    u_i,u_j=np.triu_indices(N,k=1) #indices upper diagonal
    
    #modify with probability r
    change=np.random.rand(len(u_i))<r
    J_o[u_i[change],u_j[change]]*=-1
    
    #symmetric matrix
    J_o[u_j,u_i]=J_o[u_i,u_j]
    
    #o matrix
    o=np.zeros((N,2),dtype=int)
    o[0,:]=(s[0]==np.array([1,-1])).astype(int)
    
    #defined nodes
    defined=np.full((N),False,dtype=np.bool_)
    defined[0]=True
    
    #eligible nodes
    eligible=np.full((N),False,dtype=np.bool_)
    eligible[n_a[0][:n_n[0]]]=True
    
    
    return o,J_o,defined,eligible




def exploration(N,k,r,update_rule,GTN,observer_info):
    """Excutes the exploration of the network for the 3
    different exploration rules.
    Inputs:
        - N: number of nodes.
        - k: expected number of connections per node.
        - r: noise [0,0.5]
        - update_rule: function.
        - GTN: tupple (s,J,n_a,n_n,G)
        -observer_info: tupple (o,J_o,def_nodes,eli_nodes)
    Outputs: 
        - observables (1,2,3): time dependant observables arrays (N,5) - 
        d(t),q_def(t),q(t),d_max(t),<d>(t).
        """

    #ground truth network
    s,J,n_a,n_n,G=GTN
    #observer
    o,J_o,def_nodes,eli_nodes=observer_info
    
    #distances
    d_a = np.full(N, -1, dtype=np.int32)
    for node, d in nx.single_source_shortest_path_length(G, 0).items():
        d_a[node] = d
    
    #results
    observables_1=np.zeros((N,5))
    observables_2=np.zeros((N,5))
    observables_3=np.zeros((N,5))
    
    
    #random selection
    #exploration
    observables_1[:,:]=explore_nw_r(o,J_o,update_rule,n_a,n_n,eli_nodes,\
                             def_nodes,d_a,s)
        
        
    #ordered
    #exploration
    #order
    
    order=np.empty(N,dtype=np.int32)
    pos=0
    
    max_d=np.max(d_a)
    
    for d in range(max_d+1):
        layer=np.where(d_a==d)[0]
        np.random.shuffle(layer)
        order[pos:pos+len(layer)]=layer
        pos+=len(layer)
        
    #simulation
    observables_2[:,:]=explore_nw_bfs(o,J_o,update_rule,n_a,n_n,def_nodes,\
                                 order,d_a,s)
        
    
    #random walk no modify
    #exploration
    observables_3[:,:]=explore_nw_rw(o,J_o,update_rule,n_a,n_n,def_nodes,\
                                False,d_a,s)
    
    return observables_1,observables_2,observables_3

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

#%%

#averaging over different networks + saving the t_min for statistics
def main_program_3(N,k,r,update_rule,N_i,rule):
    """Executes realization N_i times and save the data at folder data,
    saves the time evolution and all the minima.
    Inputs:
        - N: number of nodes.
        - k: expected number of connections per node.
        - r: noise [0,0.5]
        - update_rule: function.
        - N_i: number of executions to average.
        - rule: str of the rule to save the data file."""
    
    
    results_r=np.zeros((N,5,N_i))
    results_rw=np.zeros((N,5,N_i))
    results_bfs=np.zeros((N,5,N_i))
    
    results_r_f=np.zeros((N,10))
    results_rw_f=np.zeros((N,10))
    results_bfs_f=np.zeros((N,10))
    
    results=np.zeros((N_i,3))
    
    for i in range(N_i):
        np.random.seed(i+10)
        GTN=ground_truth_network(N, k)
        s,J,n_a,n_n,G=GTN
        observer_info=observer(s, J, r, n_a, n_n)
        obs_1,obs_2,obs_3=exploration(N, k, r, update_rule, GTN, observer_info)
        results_r[:,:,i]=obs_1[:,:]
        results_bfs[:,:,i]=obs_2[:,:]
        results_rw[:,:,i]=obs_3[:,:]
        
        #search for the positions of the minima
        results[i,0]=np.argmin(obs_1[:,1])
        results[i,1]=np.argmin(obs_2[:,1])
        results[i,2]=np.argmin(obs_3[:,1])
        
    for i in range(0,10,2):
        for j in range(N):
            a=statistics(results_r[j,int(i/2),:])
            results_r_f[j,i]=a[0]
            results_r_f[j,i+1]=a[1]*1.96
            a=statistics(results_bfs[j,int(i/2),:])
            results_bfs_f[j,i]=a[0]
            results_bfs_f[j,i+1]=a[1]*1.96
            a=statistics(results_rw[j,int(i/2),:])
            results_rw_f[j,i]=a[0]
            results_rw_f[j,i+1]=a[1]*1.96
    
    return results,results_r_f,results_bfs_f,results_rw_f,\
        results_r,results_bfs,results_rw


#%%
N=1000
k=20
r=0.45
N_i=1000
minima,rs,obd,rw,rs_s,obd_s,rw_s=main_program_3(N,k,r,update_majority,N_i,"mr")
minima=minima/1000
rs=rs[:,2:4]
obd=obd[:,2:4]
rw=rw[:,2:4]
rs_s=rs_s[:,1]
obd_s=obd_s[:,1]
rw_s=rw_s[:,1]

strategies=[rs_s,obd_s,rw_s]
strategies_a=[rs,obd,rw]
strategy_name=["Random Selection", "Ordered by Distance", "Random Walk"]
t=np.arange(0,1000,1)/1000


#%%
from os.path import exists
from os import makedirs

#folder
directory_save="../images/tests_minima/"
if not exists(directory_save):
    makedirs(directory_save)

s=0
mean=np.mean(minima,axis=0)

for strategy in strategies:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for i in range(N_i):
        axs[0].plot(t[1:], strategy[1:, i])
    
    
    axs[0].set_xscale("log")
    axs[0].set_ylabel("$q_{def}$")
    axs[0].set_xlabel("Fraction of defined nodes")
    
    axs[1].errorbar(t[1:],strategies_a[s][1:,0],xerr=0,yerr=strategies_a[s][1:,1],
                    linestyle="none")
    axs[1].plot(t[1:],strategies_a[s][1:,0],c="black",label="Averaged curve")
    axs[1].axvline(mean[s], color='red', linestyle='--', linewidth=2, 
                   label=r"Averaged $t_{min}$")
    axs[1].set_ylabel(r"$\langle q_{def} \rangle$")
    axs[1].set_xlabel("Fraction of defined nodes")
    axs[1].set_xscale("log")
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()
    
    plt.savefig(directory_save+str(N)+"_"+str(k)+"_"+"_"+str(r)+"_"\
               +strategy_name[s]+"_minima_1000it.png",bbox_inches="tight")
        
    s+=1
