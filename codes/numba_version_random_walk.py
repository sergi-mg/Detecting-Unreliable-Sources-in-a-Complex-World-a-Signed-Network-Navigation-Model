# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 11:38:17 2026

Author: Sergi Martínez Galindo

Random walk to explore the network, updating or not the defined nodes.
"""
#%%
#libraries 
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numba import njit
import networkx as nx

#%%
#functions
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
def explore_nw(o,J_o,update_rule,n_a,n_n,def_nodes,modify):
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
        - modify: boolean indicating if a defined node can be modified."""

    #number of subjects
    N=o.shape[0]
    #opinion definition
    o_new=o.copy()
    #previous node
    i_p=0
    
    iterations=0
    while np.sum(def_nodes)!=N:
        iterations+=1
        if iterations>N**2:
            iterations=-10**7
            print("Too much iterations")
        #node selected
        i_s=n_a[i_p,np.random.randint(n_n[i_p])]
        
        if def_nodes[i_s]==True and modify==False:
            None
        else:
            #defined neighbours of the selected node
            k_s=n_n[i_s]
            n_s=np.zeros(k_s,dtype=np.int32)
            n_s[:]=n_a[i_s,:k_s]
            k=np.sum(def_nodes[n_s])
            d_n=np.zeros(k,dtype=np.int32)
            counter=0
            for j in n_s:
                if def_nodes[j]==True:
                    d_n[counter]=j
                    counter+=1
            
            #update of opinions
            o_new=update_rule(J_o,o_new,d_n,k,i_s)
        
        #update of tracking variables
        def_nodes[i_s]=True
        i_p=i_s
        
    return o_new
     
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
        - isolated: boolean indicating if there are isolated clusters."""
    
    #nodes' values
    s=1-2*np.random.randint(0,2,N)
    
    #network creation
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
        if n>1000:
            break
        
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
        
    # maximum number of neighbours
    kmax=max(len(nbrs) for nbrs in neighbours)
    
    #neighbour's array
    neighbours_array=np.full((N, kmax), -1, dtype=np.int32)
    num_neighbours=np.zeros(N, dtype=np.int32)
    for i in range(N):
        deg=len(neighbours[i])
        num_neighbours[i]=deg
        neighbours_array[i, :deg]=neighbours[i]
        
    return s,J,neighbours_array,num_neighbours,isolated
    
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
        - defined: 1D boolean array where True corresponds to defined."""
        
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
    
    
    
    return o,J_o,defined

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
    
def realization(N,k,r,update_rule,modify):
    """Excutes the creation and posterior exploration of the network.
    Inputs:
        - N: number of nodes.
        - k: expected number of connections per node.
        - r: noise [0,0.5]
        - update_rule: function.
        - modify: boolean indicating if the defined nodes can be modified.
    Outputs: 
        - q: float, accuracy.
        - iso: boolean indicating if there are isolated clusters.
        """
        
    #accuracy
    q=0
    #defect value must change in order to to know when there is no dynamics
    #and when q=0 is the real result, pending to modify (recalculate vaues?)
    
    #ground truth network
    s,J,n_a,n_n,iso=ground_truth_network(N, k)
    
    if iso==False:
        #observer
        o,J_o,def_nodes=observer(s,J,r,n_a,n_n)
        
        #exploration
        o_final=explore_nw(o,J_o,update_rule,n_a,n_n,def_nodes,modify)
        
        #accuracy
        q=accuracy(s,o_final,N)
        
    return q,iso
    
        
def data_generator(N,k,r,update_rule,N_i,rule,modify):
    """Executes realization N_i times and save the data at folder data.
    Inputs:
        - N: number of nodes.
        - k: expected number of connections per node.
        - r: noise [0,0.5]
        - update_rule: function.
        - N_i: number of executions to average.
        - rule: str of the rule to save the data file.
        - modify: boolean indicating if the defined nodes can be modified."""
        
    from os.path import exists
    from os import makedirs
    
    #folder
    directory="../data/"
    if not exists(directory):
        makedirs(directory)
        
    #file name
    name=rule+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))+"_"+str(N_i)+".dat"
    
    #executions
    data=np.zeros(N_i)
    n=0
    for i in range(N_i):
        q,iso=realization(N,k,r,update_rule,modify)
        if iso==False:
            data[n]=q
            n+=1
            
    #save the data
    np.savetxt(directory+name, data)
    
    return None

#%%
#only onde update per node
k_l = [9, 21, 36]
r_list = np.arange(0.05, 0.51, 0.05)
N=500
for j in range(len(k_l)):
    k=k_l[j]
    print(k)
    for i in range(10):
        r=i/20+0.05
        #data_generator(N, k, r, update_majority, 1000, "mr_rw", False)
        data_generator(N, k, r, update_rn, 1000, "rn_rw", False)
        

        
        
#%%
#each node updated every time that is visited
k_l = [9, 21, 36]
r_list = np.arange(0.05, 0.51, 0.05)
N=500
for j in range(len(k_l)):
    k=k_l[j]
    print(k)
    for i in range(10):
        r=i/20+0.05
        #data_generator(N, k, r, update_majority, 1000, "mr_rw", True)
        data_generator(N, k, r, update_rn, 1000, "rn_rw_2", True)