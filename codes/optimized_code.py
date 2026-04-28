# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 11:42:06 2026

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
@njit    
def accuracy(s_t,s_o,N):
    """Returns the accuracy. Inputs:
        - s_t: 1D array with the true values of the nodes.
        - o: shape (N,2) array with each component being (0,0),(1,0) or (0,1), 
            where (p+,p-), defined by the observer.
        - N: number of nodes."""
        
    q=np.sum(s_o*s_t)/N 
    
    return q


@njit
def update_majority(links,s_o,n,n_d,k,k_d,i_s):
    """Returns the updated o matrix using the majority rule.
    o must be a shape (N,2) array with each component from 1 to N being (0,0),
    (1,0) or (0,1). Inputs:
        - i_s: integer from 0 to N-1. 
        - J_o: int32 NxN array which contains the signed connections.
        - n: 1D array with the neighbours of i_s node. 
        - k: number of neighbours of i_s node.
        - n_d: 1D array with the defined neighbours of i_s node. 
        - k_d: number of defined neighbours of i_s node."""
    
        
    #we define the opinion considering all defined neighbours
    new_value=np.sign(np.sum(s_o[n]*links[i_s,:k]))
    
    #if there is a draw we select one randomly
    if new_value==0:
        new_value=1-2*np.random.randint(0,2) 
    #we update the results
    # update
    s_o[i_s]=new_value

    return s_o

def update_majority_anchor(links,s_o,n,n_d,k,k_d,i_s):
    """Returns the updated o matrix using the majority rule.
    o must be a shape (N,2) array with each component from 1 to N being (0,0),
    (1,0) or (0,1). Inputs:
        - i_s: integer from 0 to N-1. 
        - J_o: int32 NxN array which contains the signed connections.
        - n: 1D array with the neighbours of i_s node. 
        - k: number of neighbours of i_s node.
        - n_d: 1D array with the defined neighbours of i_s node. 
        - k_d: number of defined neighbours of i_s node."""
    
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
def update_majority_ambiguity(links,s_o,n,n_d,k,k_d,i_s):
    """Returns the updated o matrix using the majority rule.
    o must be a shape (N,2) array with each component from 1 to N being (0,0),
    (1,0) or (0,1). Inputs:
        - i_s: integer from 0 to N-1. 
        - J_o: int32 NxN array which contains the signed connections.
        - n: 1D array with the neighbours of i_s node. 
        - k: number of neighbours of i_s node.
        - n_d: 1D array with the defined neighbours of i_s node. 
        - k_d: number of defined neighbours of i_s node."""
    
    positive=0
    negative=0
    for idx in range(k):
        node=n[idx]
        val=s_o[node]*links[i_s,idx]
        if val==1:
            positive+=1
        elif val==-1:
            negative+=1
    if positive/(positive+negative)>0.75:
        s_o[i_s]=True
    else:
        s_o[i_s]=False

    return s_o

@njit
def update_rn(links,s_o,n,n_d,k,k_d,i_s):
    """Returns the updated o matrix using the random neighbour rule.
    o must be a shape (N,2) array with each component from 1 to N being (0,0),
    (1,0) or (0,1). Inputs:
        - i_s: integer from 0 to N-1. 
        - J_o: int32 NxN array which contains the signed connections.
        - n: 1D array with the neighbours of i_s node. 
        - k: number of neighbours of i_s node.
        - n_d: 1D array with the defined neighbours of i_s node. 
        - k_d: number of defined neighbours of i_s node."""
            
    #we choose a neighbour
    c_n=n_d[np.random.randint(k_d)]
    
    for idx in range(k):
        if n[idx] == c_n:
            break
    #update
    s_o[i_s]=links[i_s,idx]*s_o[c_n]

    return s_o

@njit
def update_rn_anchor(links,s_o,n,n_d,k,k_d,i_s):
    """Returns the updated o matrix using the random neighbour rule.
    o must be a shape (N,2) array with each component from 1 to N being (0,0),
    (1,0) or (0,1). Inputs:
        - i_s: integer from 0 to N-1. 
        - J_o: int32 NxN array which contains the signed connections.
        - n: 1D array with the neighbours of i_s node. 
        - k: number of neighbours of i_s node.
        - n_d: 1D array with the defined neighbours of i_s node. 
        - k_d: number of defined neighbours of i_s node."""
        
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
def explore_nw_rw(s_o,links_o,update_rule,n_a,n_n,def_nodes,modify,d,s):
    """Explores the network and returns the final o matrix. Inputs:
        - o: shape (N,2) array with each component being (0,0),(1,0) or (0,1),
            since here it acts as the initial condition, only one non-zero
            component. 
        - J_o: int32 NxN array which contains the signed connections.
        - update_rule: function.
        - n_a: shape (N,k_max) array containing the neihbours of each
            node, if k_i<k_max the rest of the values are -1. 
        - n_n: 1D array containin the number of neighbours (k_i) of each node.
        - def_nodes: 1D boolean array where True corresponts to defined.
        - modify: boolean indicating if a defined node can be modified.
        - d: array containing the distance to node 0.
        - s: 1D int32 array with values +1,-1 of the truth network.
        Outputs: 
            -observables: array (N,5) d(t),q_def(t),q(t),d_max(t),<d>(t)"""

    #number of subjects
    N=s_o.shape[0]
    #opinion definition
    s_o_new=s_o.copy()
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
            s_o_new=update_rule(links_o,s_o_new,n_s,k_s,d_n,k,i_s)
            i+=1
            
            #observables: d(t),q_def(t),q(t),d_max(t),<d>(t)
            observables[i+1,0]=d[i_s]*1.
            s_o_i=s_o_new(i_s)
            observables[i+1,1]=observables[i,1]*(i+1)/(i+2)+s_o_i*s[i_s]/(i+2)
            observables[i+1,2]=accuracy(s,s_o_new,N)
            observables[i+1,3] = max(observables[i,3], d[i_s])
            observables[i+1,4]=observables[i,4]*(i)/(i+1)+d[i_s]/(i+1)
        
        #update of tracking variables
        def_n[i_s]=True
        i_p=i_s
        

    return observables

@njit
def explore_nw_bfs(s_o,links_o,update_rule,n_a,n_n,def_nodes,order,d,s):
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
    N=s_o.shape[0]
    #opinion definition
    s_o_new=s_o.copy()
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
        s_o_new=update_rule(links_o,s_o_new,n_s,k_s,d_n,k,i_s)
        
        #update of tracking variables
        def_n[i_s]=True
        
        #observables: d(t),q_def(t),q(t),d_max(t),<d>(t)
        observables[i+1,0]=d[i_s]*1.
        s_o_i=s_o_new(i_s)
        observables[i+1,1]=observables[i,1]*(i+1)/(i+2)+s_o_i*s[i_s]/(i+2)
        observables[i+1,2]=accuracy(s,s_o_new,N)
        observables[i+1,3] = max(observables[i,3], d[i_s])
        observables[i+1,4]=observables[i,4]*(i)/(i+1)+d[i_s]/(i+1)
        
    return observables

@njit
def explore_nw_r(s_o,links_o,update_rule,n_a,n_n,e_n,def_nodes,d,s):
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
    N=s_o.shape[0]

    #time evolutions
    observables=np.zeros((N,5),dtype="float64")
    #d(t),q_def(t),q(t),d_max(t),<d>(t)
    observables[0,:]=np.array([0.,1.,1./N,0.,0.])
    
    #opinion definition
    s_o_new=s_o.copy()
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
        s_o_new=update_rule(links_o,s_o_new,n_s,k_s,d_n,k,i_s)
        
        #update of tracking variables
        e_nodes[i_s]=False
        def_n[i_s]=True
        for j in n_s:
            if def_n[j]==False:
                e_nodes[j]=True
        
        #observables: d(t),q_def(t),q(t),d_max(t),<d>(t)
        observables[i+1,0]=d[i_s]*1.
        s_o_i=s_o_new(i_s)
        observables[i+1,1]=observables[i,1]*(i+1)/(i+2)+s_o_i*s[i_s]/(i+2)
        observables[i+1,2]=accuracy(s,s_o_new,N)
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
    links=np.zeros_like(neighbours_array)
    for i in range(N):
        idx = neighbours_array[i, :num_neighbours[i]]
        links[i, :num_neighbours[i]] = s[i] * s[idx]
        
    return s,links,neighbours_array,num_neighbours,G

def observer(s,links,r,n_a,n_n):
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
    
    
    #o matrix
    s_o=np.zeros((N),dtype=int)
    s_o[0]=s[0]
    
    #defined nodes
    defined=np.full((N),False,dtype=np.bool_)
    defined[0]=True
    
    #eligible nodes
    eligible=np.full((N),False,dtype=np.bool_)
    eligible[n_a[0][:n_n[0]]]=True
    
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
    
    return s_o,links_o,defined,eligible

def exploration(N,k,r,update_rule,strategy,GTN,observer_info):
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
    s,links,n_a,n_n,G=GTN
    #observer
    o,links_o,def_nodes,eli_nodes=observer_info
    
    #distances
    d_a = np.full(N, -1, dtype=np.int32)
    for node, d in nx.single_source_shortest_path_length(G, 0).items():
        d_a[node] = d
    
    #results
    observables=np.zeros((N,5))

    if strategy=="rs":
        #random selection
        #exploration
        observables[:,:]=explore_nw_r(o,links_o,update_rule,n_a,n_n,eli_nodes,\
                                 def_nodes,d_a,s)
        
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
        observables[:,:]=explore_nw_bfs(o,links_o,update_rule,n_a,n_n,def_nodes,\
                                     order,d_a,s)
        
    elif strategy=="rw":
        #random walk no modify
        #exploration
        observables[:,:]=explore_nw_rw(o,links_o,update_rule,n_a,n_n,def_nodes,\
                                    False,d_a,s)
    
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

def main_program(N,k,r,update_rule,N_i,rule,strategy):
    """Executes realization N_i times and save the data at folder data,
    saves the time evolution.
    Inputs:
        - N: number of nodes.
        - k: expected number of connections per node.
        - r: noise [0,0.5]
        - update_rule: function.
        - N_i: number of executions to average.
        - rule: str of the rule to save the data file."""
    
    
    results=np.zeros((N,5,N_i))
    
    for i in range(N_i):
        np.random.seed(i+10)
        GTN=ground_truth_network(N, k)
        s,links,n_a,n_n,G=GTN
        observer_info=observer(s, links, r, n_a, n_n)
        obs=exploration(N, k, r, update_rule, GTN, observer_info)
        results[:,:,i]=obs[:,:]
        
    from os.path import exists
    from os import makedirs
    
    #folder
    directory="../data/time_evo_simulations/"
    if not exists(directory):
        makedirs(directory)
        
    #save the data

    name=rule+"_"+strategy+"_"+str(N)+"_"+str(k)+"_"+str(round(r,2))\
        +"_"+str(N_i)+".npz"
    
    np.savez_compressed(directory+name,r=results)

#%%
r_values=np.arange(0.01,0.5005,0.01)
k=20
N=1000
N_i=5000
#%%
for r in r_values:
    main_program(N,k,r,update_majority,N_i,"mr","rs")