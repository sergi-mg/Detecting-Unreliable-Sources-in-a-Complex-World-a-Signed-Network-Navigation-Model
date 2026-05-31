# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:42:06 2026

@author: Sergi Martínez Galindo
"""

"""
#######################################
IMPORTANT INFORMATION BEFORE EXECUTING
#######################################
    
This program provides the code to simulate the system with different 
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
    
- Network topotlogy (without biases):
    add _WS (Watts-Strogatz) or _BA (Barabasi-Albert) to mr o rn
"""


#%%
#libraries 
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numba import njit
import networkx as nx

#%%
@njit
def prim_lin(t):
    N=1000
    return (N-t)/N 

@njit
def rec_lin(t):
    N=1000
    return (t+1)/N 

#%%
@njit    
def accuracy(s_t,s_o,N):
    """Returns the accuracy. Inputs:
        - s_t: 1D array with the true values of the nodes.
        - s_o: 1D array with the values determined by the observer.
        - N: number of nodes."""
        
    q=np.sum(s_o*s_t)/N 
    
    return q

@njit
def update_majority(links,s_o,n,n_d,k,k_d,i_s,M,w):
    """Returns the updated s_o vector using the majority rule.
    s_o must be a shape (N) array with each component from 1 to N being
    -1, 0 or +1. Inputs:
        - links: (N,k_max) array containing the links affected by noise, in 
            the same order as the n_a matrix.
        - s_o: 1D (N) array containing the observer's opinion of each node.
        - n: 1D (k) array with the neighbours (index) of i_s node. 
        - k: number of neighbours of i_s node.
        - n_d: 1D array (k_d) with the defined neighbours of i_s node. 
        - k_d: number of defined neighbours of i_s node.
        - i_s: integer from 0 to N-1 indicating the selected node.
        - M: float, not used here.
        - w: 1D array (N) containing the weights of the nodes as a function
        of the discovery time, not used here.
        """
    
    #we define the opinion considering all defined neighbours
    new_value=np.sign(np.sum(s_o[n]*links[i_s,:k]))
    
    #if there is a draw we select one randomly
    if new_value==0:
        new_value=1-2*np.random.randint(0,2) 
    #we update the results
    # update
    s_o[i_s]=new_value

    return s_o

@njit
def update_majority_weighted(links,s_o,n,n_d,k,k_d,i_s,M,w):
    """Returns the updated s_o vector using the majority rule (weighted).
    s_o must be a shape (N) array with each component from 1 to N being
    -1, 0 or +1. Inputs:
        - links: (N,k_max) array containing the links affected by noise, in 
            the same order as the n_a matrix.
        - s_o: 1D (N) array containing the observer's opinion of each node.
        - n: 1D (k) array with the neighbours (index) of i_s node. 
        - k: number of neighbours of i_s node.
        - n_d: 1D array (k_d) with the defined neighbours of i_s node. 
        - k_d: number of defined neighbours of i_s node.
        - i_s: integer from 0 to N-1 indicating the selected node.
        - M: float, not used here.
        - w: 1D array (N) containing the weights of the nodes as a function
        of the discovery time.
        """
    
    #we define the opinion considering all defined neighbours
    new_value=np.sign(np.sum(s_o[n]*links[i_s,:k]*w[n]))
    
    #if there is a draw we select one randomly
    if new_value==0:
        new_value=1-2*np.random.randint(0,2) 
    #we update the results
    # update
    s_o[i_s]=new_value

    return s_o

@njit
def update_majority_anchor(links,s_o,n,n_d,k,k_d,i_s,M,w):
    """Returns the updated s_o vector using the majority rule (anchoring bias).
    s_o must be a shape (N) array with each component from 1 to N being
    -1, 0 or +1. Inputs:
        - links: (N,k_max) array containing the links affected by noise, in 
            the same order as the n_a matrix.
        - s_o: 1D (N) array containing the observer's opinion of each node.
        - n: 1D (k) array with the neighbours (index) of i_s node. 
        - k: number of neighbours of i_s node.
        - n_d: 1D array (k_d) with the defined neighbours of i_s node. 
        - k_d: number of defined neighbours of i_s node.
        - i_s: integer from 0 to N-1 indicating the selected node.
        - M: float, not used here.
        - w: 1D array (N) containing the weights of the nodes as a function
        of the discovery time, not used here.
        """
    
    #is [0] a neighbour
    for idx in range(k):
        if n[idx]==0:
            s_o[i_s]=s_o[0]*links[i_s,idx]
            return s_o
        
    #if 0 is not a neighbour:
    #we define the opinion considering all defined neighbours
    new_value=np.sign(np.sum(s_o[n]*links[i_s,:k]))
    
    #if there is a draw we select one randomly
    if new_value==0:
        new_value=1-2*np.random.randint(0,2) 

    # update
    s_o[i_s]=new_value

    return s_o

@njit
def update_majority_ambiguity(links,s_o,n,n_d,k,k_d,i_s,M,w):
    """Returns the updated s_o vector using the majority rule (ambiguity bias).
    s_o must be a shape (N) array with each component from 1 to N being
    -1, 0 or +1. Inputs:
        - links: (N,k_max) array containing the links affected by noise, in 
            the same order as the n_a matrix.
        - s_o: 1D (N) array containing the observer's opinion of each node.
        - n: 1D (k) array with the neighbours (index) of i_s node. 
        - k: number of neighbours of i_s node.
        - n_d: 1D array (k_d) with the defined neighbours of i_s node. 
        - k_d: number of defined neighbours of i_s node.
        - i_s: integer from 0 to N-1 indicating the selected node.
        - M: float, indicates the percentage of agreeing neighbours
        needed to trust a node.
        - w: 1D array (N) containing the weights of the nodes as a function
        of the discovery time, not used here.
        """
    
    positive=0
    negative=0
    for idx in range(k):
        node=n[idx]
        val=s_o[node]*links[i_s,idx]
        if val==1:
            positive+=1
        elif val==-1:
            negative+=1
    if positive/(positive+negative)>M:
        s_o[i_s]=1
    else:
        s_o[i_s]=-1

    return s_o


@njit
def update_rn(links,s_o,n,n_d,k,k_d,i_s,M,w):
    """Returns the updated o matrix using the random neighbour rule.
    s_o must be a shape (N) array with each component from 1 to N being
    -1, 0 or +1. Inputs:
        - links: (N,k_max) array containing the links affected by noise, in 
            the same order as the n_a matrix.
        - s_o: 1D (N) array containing the observer's opinion of each node.
        - n: 1D (k) array with the neighbours (index) of i_s node. 
        - k: number of neighbours of i_s node.
        - n_d: 1D array (k_d) with the defined neighbours of i_s node. 
        - k_d: number of defined neighbours of i_s node.
        - i_s: integer from 0 to N-1 indicating the selected node.
        - M: float, not used here.
        - w: 1D array (N) containing the weights of the nodes as a function
        of the discovery time, not used here.
        """
            
    #we choose a neighbour
    c_n=n_d[np.random.randint(k_d)]
    
    for idx in range(k):
        if n[idx] == c_n:
            break
    #update
    s_o[i_s]=links[i_s,idx]*s_o[c_n]

    return s_o

@njit
def update_rn_weighted(links,s_o,n,n_d,k,k_d,i_s,M,w):
    """Returns the updated o matrix using the random neighbour rule (weighted).
    s_o must be a shape (N) array with each component from 1 to N being
    -1, 0 or +1. Inputs:
        - links: (N,k_max) array containing the links affected by noise, in 
            the same order as the n_a matrix.
        - s_o: 1D (N) array containing the observer's opinion of each node.
        - n: 1D (k) array with the neighbours (index) of i_s node. 
        - k: number of neighbours of i_s node.
        - n_d: 1D array (k_d) with the defined neighbours of i_s node. 
        - k_d: number of defined neighbours of i_s node.
        - i_s: integer from 0 to N-1 indicating the selected node.
        - M: float, not used here.
        - w: 1D array (N) containing the weights of the nodes as a function
        of the discovery time, not used here.
        """
            
    #we choose a neighbour
    #weights
    w_d_n = w[n_d].copy()
    #random selection
    a1=np.random.rand()*np.sum(w_d_n)
    c_n=n_d[-1]
    cumulative=0
    for j in range(k_d):
            cumulative+=w_d_n[j]
            if a1<cumulative:
                c_n=n_d[j]
                break
    
            
    for idx in range(k):
        if n[idx] == c_n:
            break
    #update
    s_o[i_s]=links[i_s,idx]*s_o[c_n]

    return s_o

@njit
def update_rn_anchor(links,s_o,n,n_d,k,k_d,i_s,M,w):
    """Returns the updated o matrix using the random neighbour rule (anchoring
    bias). s_o must be a shape (N) array with each component from 1 to N being
    -1, 0 or +1. Inputs:
        - links: (N,k_max) array containing the links affected by noise, in 
            the same order as the n_a matrix.
        - s_o: 1D (N) array containing the observer's opinion of each node.
        - n: 1D (k) array with the neighbours (index) of i_s node. 
        - k: number of neighbours of i_s node.
        - n_d: 1D array (k_d) with the defined neighbours of i_s node. 
        - k_d: number of defined neighbours of i_s node.
        - i_s: integer from 0 to N-1 indicating the selected node.
        - M: float, not used here.
        - w: 1D array (N) containing the weights of the nodes as a function
        of the discovery time, not used here.
        """
        
    #is [0] a neighbour
    for idx in range(k):
        if n[idx]==0:
            s_o[i_s]=s_o[0]*links[i_s,idx]
            return s_o
        
    #we choose a neighbour
    c_n=n_d[np.random.randint(k_d)]
    
    for idx in range(k):
        if n[idx] == c_n:
            break
    #update
    s_o[i_s]=links[i_s,idx]*s_o[c_n]


    return s_o



@njit
def explore_nw_obd(s_o,links_o,update_rule,n_a,n_n,def_nodes,order,d,s
                   ,M,weight,w_b):
    """Explores the network using an ordered by distance strategy and returns
    the system's evolution. Inputs:
       - s_o: 1D (N) array containing the observer's opinion of each node.
            Since here it acts as the initial condition, only one non-zero
            component. 
        - links_o: (N,k_max) array containing the links affected by noise, in 
            the same order as the n_a matrix.
        - n_a: shape (N,k_max) array containing the neihbours of each
            node, if k_i<k_max the rest of the values are -1. 
        - n_n: 1D array containing the number of neighbours (k_i) of each node.
        - def_nodes: 1D boolean array where True corresponts to defined.
        - modify: boolean indicating if a defined node can be modified.
        - d: array containing the distance to node 0.
        - s: 1D int32 array with values +1,-1 of the truth network.
        - M: float, indicates the percentage of agreeing neighbours
        needed to trust a node. (only for ambiguity bias if not ignored)
        - weight: function of one variable, used to calculate the wieght of 
        each node (only used for ordering effects)
        - w_b: Boolean indicating if there is an ordering affect to apply.
        Outputs: 
            -observables: array (N,5) d(t),q_def(t),q(t),d_max(t),<d>(t)"""

    #number of subjects
    N=s_o.shape[0]
    #opinion definition
    s_o_new=s_o.copy()
    def_n=def_nodes.copy()
    
    #time evolutions
    observables=np.zeros((N,5),dtype="float64")
    #d(t),q_def(t),q(t),d_max(t),<d>(t)
    observables[0,:]=np.array([0.,1.,1./N,0.,0.])
    
    w=np.zeros(N)
    if w_b==True:
        w[0]=weight(0)
    
    for i in range(N-1):
        
        #node selected
        i_s=order[i+1]
        
        #defined neighbours of the selected node
        k=n_n[i_s]
        n=np.zeros(k,dtype=np.int32)
        n[:]=n_a[i_s,:k]
        k_d=np.sum(def_n[n])
        n_d=np.zeros(k_d,dtype=np.int32)
        counter=0
        for j in n:
            if def_n[j]==True:
                n_d[counter]=j
                counter+=1
        
        #update of opinions n,n_d,k,k_d
        s_o_new=update_rule(links_o,s_o_new,n,n_d,k,k_d,i_s,M,w)
        if w_b==True:
            w[i_s]=weight(i+1)
        
        #update of tracking variables
        def_n[i_s]=True
        
        
        #observables: d(t),q_def(t),q(t),d_max(t),<d>(t)
        observables[i+1,0]=d[i_s]*1.
        s_o_i=s_o_new[i_s]
        observables[i+1,1]=observables[i,1]*(i+1)/(i+2)+s_o_i*s[i_s]/(i+2)
        observables[i+1,2]=accuracy(s,s_o_new,N)
        observables[i+1,3] = max(observables[i,3], d[i_s])
        observables[i+1,4]=observables[i,4]*(i)/(i+1)+d[i_s]/(i+1)
        
    return observables

@njit
def explore_nw_r(s_o,links_o,update_rule,n_a,n_n,e_n,def_nodes,d,s
                 ,M,weight,w_b):
    """"Explores the network using the random selection strategy and returns
    the system's evolution. Inputs:
       - s_o: 1D (N) array containing the observer's opinion of each node.
            Since here it acts as the initial condition, only one non-zero
            component. 
        - links_o: (N,k_max) array containing the links affected by noise, in 
            the same order as the n_a matrix.
        - n_a: shape (N,k_max) array containing the neihbours of each
            node, if k_i<k_max the rest of the values are -1. 
        - n_n: 1D array containing the number of neighbours (k_i) of each node.
        - def_nodes: 1D boolean array where True corresponts to defined.
        - modify: boolean indicating if a defined node can be modified.
        - d: array containing the distance to node 0.
        - s: 1D int32 array with values +1,-1 of the truth network.
        - M: float, indicates the percentage of agreeing neighbours
        needed to trust a node. (only for ambiguity bias if not ignored)
        - weight: function of one variable, used to calculate the wieght of 
        each node (only used for ordering effects)
        - w_b: Boolean indicating if there is an ordering affect to apply.
        Outputs: 
            -observables: array (N,5) d(t),q_def(t),q(t),d_max(t),<d>(t)"""
        
    #number of subjects
    N=s_o.shape[0]

    #time evolutions
    observables=np.zeros((N,5),dtype="float64")
    #d(t),q_def(t),q(t),d_max(t),<d>(t)
    observables[0,:]=np.array([0.,1.,1./N,0.,0.])
    
    #opinion definition
    s_o_new=s_o.copy()
    e_nodes=e_n.copy()
    def_n=def_nodes.copy()
    
    w=np.zeros(N)
    if w_b==True:
        w[0]=weight(0)
    
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
        k=n_n[i_s]
        n=np.zeros(k,dtype=np.int32)
        n[:]=n_a[i_s,:k]
        k_d=np.sum(def_n[n])
        n_d=np.zeros(k_d,dtype=np.int32)
        counter=0
        for j in n:
            if def_n[j]==True:
                n_d[counter]=j
                counter+=1
        
        #update of opinions n,n_d,k,k_d
        s_o_new=update_rule(links_o,s_o_new,n,n_d,k,k_d,i_s,M,w)
        if w_b==True:
            w[i_s]=weight(i+1) 
        
        #update of tracking variables
        e_nodes[i_s]=False
        def_n[i_s]=True
        for j in n:
            if def_n[j]==False:
                e_nodes[j]=True
        
        
        #observables: d(t),q_def(t),q(t),d_max(t),<d>(t)
        observables[i+1,0]=d[i_s]*1.
        s_o_i=s_o_new[i_s]
        observables[i+1,1]=observables[i,1]*(i+1)/(i+2)+s_o_i*s[i_s]/(i+2)
        observables[i+1,2]=accuracy(s,s_o_new,N)
        observables[i+1,3] = max(observables[i,3], d[i_s])
        observables[i+1,4]=observables[i,4]*(i)/(i+1)+d[i_s]/(i+1)
        
    return observables


def ground_truth_network_BA(N,k,p_rewiring):
    """Generates the ground truth network: Barabasi-Albert.
    Inputs:
        - N: number of nodes.
        - k: expected number of edges per node (must be even).
        - p_rewiring: probability of rewiring
    Outputs: 
        - s: 1D int32 array with values +1,-1.
        - links: shape (N,k_max) array containing the sign of the links given
            by si*sj, in the same order as neighbours_array
        - neighbours_array: shape (N,k_max) array containing the neihbours
            of each node, if k_i<k_max the rest of the values are -1.
        - num_neighbours: 1D array containing the number of neighbours 
            of each node.
        - G: network."""
        
    #nodes' values
    s=1-2*np.random.randint(0,2,N)
    
    if k % 2 != 0:
        raise ValueError("k must be even since m=k/2\
                         model implementation.")
    
    m=k//2
        
    G=nx.barabasi_albert_graph(N, m)
    
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
        
    #links array
    links=np.zeros_like(neighbours_array,dtype=np.int32)
    for i in range(N):
        idx = neighbours_array[i, :num_neighbours[i]]
        links[i, :num_neighbours[i]] = s[i] * s[idx]
        
    return s,links,neighbours_array,num_neighbours,G

def ground_truth_network_WS(N,k,p_rewiring):
    """Generates the ground truth network: Watts–Strogatz.
    Inputs:
        - N: number of nodes.
        - k: expected number of edges per node (must be even).
        - p_rewiring: probability of rewiring
    Outputs: 
        - s: 1D int32 array with values +1,-1.
        - links: shape (N,k_max) array containing the sign of the links given
            by si*sj, in the same order as neighbours_array
        - neighbours_array: shape (N,k_max) array containing the neihbours
            of each node, if k_i<k_max the rest of the values are -1.
        - num_neighbours: 1D array containing the number of neighbours 
            of each node.
        - G: network."""
        
    #nodes' values
    s=1-2*np.random.randint(0,2,N)
    
    if k % 2 != 0:
        raise ValueError("k must be even for the Watts–Strogatz\
                         model implementation.")
        
    G=nx.connected_watts_strogatz_graph(N, k, p_rewiring)
    
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
        
    #links array
    links=np.zeros_like(neighbours_array,dtype=np.int32)
    for i in range(N):
        idx = neighbours_array[i, :num_neighbours[i]]
        links[i, :num_neighbours[i]] = s[i] * s[idx]
        
    return s,links,neighbours_array,num_neighbours,G

def ground_truth_network_ER(N,k,p_rewiring):
    """Generates the ground truth network: Erdos-Renyi.
    Inputs:
        - N: number of nodes.
        - k: expected number of edges per node.
        - p_rewiring: value for WS, not used here
    Outputs: 
        - s: 1D int32 array with values +1,-1.
        - links: shape (N,k_max) array containing the sign of the links given
            by si*sj, in the same order as neighbours_array
        - neighbours_array: shape (N,k_max) array containing the neihbours
            of each node, if k_i<k_max the rest of the values are -1.
        - num_neighbours: 1D array containing the number of neighbours 
            of each node.
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
        
    #links array
    links=np.zeros_like(neighbours_array,dtype=np.int32)
    for i in range(N):
        idx = neighbours_array[i, :num_neighbours[i]]
        links[i, :num_neighbours[i]] = s[i] * s[idx]
        
    return s,links,neighbours_array,num_neighbours,G

def observer(s,links,r,n_a,n_n,network,criteria_BA):
    """Generates the obersver variables.
    Inputs:
        - s: 1D int32 array with values +1,-1.
        - links: shape (N,k_max) array containing the sign of the links given
            by si*sj, in the same order as n_a.
        - r: noise [0,0.5]
        - n_a: shape (N,k_max) array comtaineing the neihbours
            of each node, if k_i<k_max the rest of the values are -1.
        - n_n: 1D array containin the number of neighbours 
            of each node.
        - criteria_BA: "Max", "Min", "Random", only for Ba, indicates the criteria to 
        select the initial node.
    Outputs: 
        - s_o: 1D (N) array containing the observer's opinion of each node.
             Since here it acts as the initial condition, only one non-zero
             component. 
         - links_o: (N,k_max) array containing the links affected by noise, in 
             the same order as the n_a matrix.
        - defined: 1D boolean array where True corresponds to defined.
        - eligible: 1D boolean array where True corresponds to being eligible.
        -i_n: initial node"""
        
    N=len(s)
    
    i_n=0
    #initial node
    if network is ground_truth_network_BA:
        if criteria_BA=="Max":
            i_n=np.argmax(n_n)
        elif criteria_BA=="Min":
            i_n=np.argmin(n_n)
        else:
            i_n=np.random.randint(N)
            
        
    #o matrix
    s_o=np.zeros((N),dtype=int)
    s_o[i_n]=s[i_n]
    
    
    #defined nodes
    defined=np.full((N),False,dtype=np.bool_)
    defined[i_n]=True
    
    #eligible nodes
    eligible=np.full((N),False,dtype=np.bool_)
    eligible[n_a[i_n][:n_n[i_n]]]=True
    
    #links
    links_o=np.zeros_like(n_a)
    
    #noise
    for i in range(N):
        for idx in range(n_n[i]):
            j=n_a[i, idx]
            
            if i<j:
                val=links[i,idx]
                
                if np.random.rand()<r:
                    val*=-1
    
                links_o[i,idx]=val
                
                for idx2 in range(n_n[j]):
                    if n_a[j, idx2]==i:
                        links_o[j, idx2]=val
                        break
    
    return s_o,links_o,defined,eligible,i_n

def exploration(N,k,r,update_rule,strategy,GTN,observer_info,M,weight,rule):
    """Excutes the exploration of the network for the 3
    different exploration rules.
    Inputs:
        - N: number of nodes.
        - k: expected number of connections per node.
        - r: noise [0,0.5]
        - update_rule: function.
        - strategy: string with the strategy name.
        - GTN: tupple (s,J,n_a,n_n,G)
        - observer_info: tupple (o,J_o,def_nodes,eli_nodes,i_n)
        - M: float, indicates the percentage of agreeing neighbours
        needed to trust a node. (only for ambiguity bias if not ignored)
        - weight: function of one variable, used to calculate the wieght of 
        each node (only used for ordering effects)
        - rule: string with the rule name 
    Outputs: 
        - observables: time dependant observables arrays (N,5) - 
        d(t),q_def(t),q(t),d_max(t),<d>(t).
        """
    
    #only for weighted biases
    list_w_b=["mr_primacy_linear","mr_primacy_exp","mr_primacy_power",
              "rn_primacy_linear","rn_primacy_exp","rn_primacy_power",
              "mr_recency_linear","mr_recency_exp","mr_recency_power",
              "rn_recency_linear","rn_recency_exp","rn_recency_power"]
    
    w_b=False
    if rule in list_w_b:
        w_b=True

    #ground truth network
    s,links,n_a,n_n,G=GTN
    #observer
    o,links_o,def_nodes,eli_nodes,i_n=observer_info
    
    #distances
    d_a = np.full(N, -1, dtype=np.int32)
    for node, d in nx.single_source_shortest_path_length(G, i_n).items():
        d_a[node] = d
    
    #results
    observables=np.zeros((N,5))

    if strategy=="rs":
        #random selection
        #exploration
        observables[:,:]=explore_nw_r(o,links_o,update_rule,n_a,n_n,eli_nodes,\
                                 def_nodes,d_a,s,M,weight,w_b)
        
    elif strategy=="obd": 
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
        observables[:,:]=explore_nw_obd(o,links_o,update_rule,n_a,n_n,def_nodes,\
                                     order,d_a,s,M,weight,w_b)
    
    return observables

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

def main_program(N,k,r,update_rule,N_i,rule,strategy,weight,GTN_network,c_BA="Random",
                 M=0,p_r=-1):
    """Executes N_i realization and saves the data at the folder data:
    saves the time evolution in time_evo_simulations_biases folder.
    Inputs:
        - N: number of nodes.
        - k: expected number of connections per node.
        - r: noise [0,0.5]
        - update_rule: function.
        - N_i: number of executions to average.
        - rule: str of the rule to save the data file.
        - strategy: str of the strategy used.
        - weight: function of one variable, used to calculate the wieght of 
        each node (only used for ordering effects)
        - GTN_network: function indicating which topology is used.
        - c_BA: "Max", "Min", "Random", only for Ba, indicates the criteria to 
        select the initial node.
        - p_r: rewiring probability, only for Watts-Strogatz
        - M: float, indicates the percentage of agreeing neighbours
        needed to trust a node. (only for ambiguity bias if not ignored)"""
    
    
    results=np.zeros((N,5,N_i))
    
    for i in range(N_i):
        np.random.seed(i+10)
        GTN=GTN_network(N, k, p_r)
        s,links,n_a,n_n,G=GTN
        observer_info=observer(s, links, r, n_a, n_n,GTN_network,c_BA)
        obs=exploration(N,k,r,update_rule,strategy,GTN,observer_info,M,weight
                        ,rule)
        results[:,:,i]=obs[:,:]
        
    from os.path import exists
    from os import makedirs
    
    #folder
    directory="../data/time_evo_simulations_biases/"
    if not exists(directory):
        makedirs(directory)
        
    #save the data
        
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
            
    np.savez_compressed(directory+name,r=results)

#%%
r_values=np.arange(0.,0.5001,0.01)
k=20
N=1000
N_i=1000
weight=prim_lin
#%%

for r in r_values:
    #Barabasi-Albert
    """
    main_program(N,k,r,update_majority,N_i,"mr_BA","rs",weight,
                 ground_truth_network_BA, c_BA="Min")
    main_program(N,k,r,update_majority,N_i,"mr_BA","obd",weight,
                 ground_truth_network_BA, c_BA="Min")
    
    main_program(N,k,r,update_rn,N_i,"rn_BA","rs",weight,
                 ground_truth_network_BA, c_BA="Min")
    main_program(N,k,r,update_rn,N_i,"rn_BA","obd",weight,
                 ground_truth_network_BA, c_BA="Min")
    
    main_program(N,k,r,update_majority,N_i,"mr_BA","rs",weight,
                 ground_truth_network_BA, c_BA="Max")
    main_program(N,k,r,update_majority,N_i,"mr_BA","obd",weight,
                 ground_truth_network_BA, c_BA="Max")
    
    main_program(N,k,r,update_rn,N_i,"rn_BA","rs",weight,
                 ground_truth_network_BA, c_BA="Max")
    main_program(N,k,r,update_rn,N_i,"rn_BA","obd",weight,
                 ground_truth_network_BA, c_BA="Max")"""
    
#%%
p_r=0.001
r=0.05
main_program(N,k,r,update_majority,N_i,"mr_WS","rs",weight,
             ground_truth_network_WS,p_r=p_r)
main_program(N,k,r,update_majority,N_i,"mr_WS","obd",weight,
             ground_truth_network_WS,p_r=p_r)

main_program(N,k,r,update_rn,N_i,"rn_WS","rs",weight,
             ground_truth_network_WS,p_r=p_r)
main_program(N,k,r,update_rn,N_i,"rn_WS","obd",weight,
             ground_truth_network_WS,p_r=p_r)
#%%
"""r_values=np.arange(0.,0.5001,0.05)
p_r_values=np.concatenate(([0], np.logspace(-4, 0, 10)))
k=20
N=1000
N_i=1000
for r in r_values:
    for p_r in p_r_values:
        #Watts-Strogatz
        main_program(N,k,r,update_majority,N_i,"mr_WS","rs",weight,
                     ground_truth_network_WS,p_r=p_r)
        main_program(N,k,r,update_majority,N_i,"mr_WS","obd",weight,
                     ground_truth_network_WS,p_r=p_r)
        
        main_program(N,k,r,update_rn,N_i,"rn_WS","rs",weight,
                     ground_truth_network_WS,p_r=p_r)
        main_program(N,k,r,update_rn,N_i,"rn_WS","obd",weight,
                     ground_truth_network_WS,p_r=p_r)"""
        
#%%
#k_dependency
"""r_values=np.arange(0.47,0.5001,0.01)
k_values=[10,30,40,50]
N=1000
N_i=1000
weight=prim_lin
for r in r_values:
    for k in k_values:
        main_program(N,k,r,update_majority,N_i,"mr","rs",weight,
                     ground_truth_network_ER)
        main_program(N,k,r,update_majority,N_i,"mr","obd",weight,
                     ground_truth_network_ER)
        
        main_program(N,k,r,update_rn,N_i,"rn","rs",weight,
                     ground_truth_network_ER)
        main_program(N,k,r,update_rn,N_i,"rn","obd",weight,
                     ground_truth_network_ER)"""

#%%
#N_dependency
"""r_values=np.arange(0.23,0.5001,0.01)
k=20
N_values=[100,500]
N_i=1000
weight=prim_lin
for r in r_values:
    for N in N_values:
        main_program(N,k,r,update_majority,N_i,"mr","rs",weight,
                     ground_truth_network_ER)
        main_program(N,k,r,update_majority,N_i,"mr","obd",weight,
                     ground_truth_network_ER)
        
        main_program(N,k,r,update_rn,N_i,"rn","rs",weight,
                     ground_truth_network_ER)
        main_program(N,k,r,update_rn,N_i,"rn","obd",weight,
                     ground_truth_network_ER)"""
        
#%%
"""
#######################################
IMPORTANT INFORMATION BEFORE EXECUTING
#######################################
    
This program provides the code to simulate the system with different 
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
    
- Network topotlogy (without biases):
    add _WS (Watts-Strogatz) or _BA (Barabasi-Albert) to mr o rn
"""

