import torch
import torch.nn.functional as F

import numpy as np
import random
from scipy.special import binom
from itertools import combinations

def regSHAP(data,M,model,seed,err,true_val):
    """
    This function computes the shapley value for the regression setting
    
    Args : 
        - data : Data object containing at least 
            * a tensor of nodes (x)
            * a tensor of edge indices (edge_index). 
                Please note that we suppose an undirected graph,
                meaning that an edge between 0 and 1 will be [0,1] and [1,0] in the edge indices
            * a tensor of edge attributes (edge_attr)
            * a value for the target (y)
        - M : number of submodels which will be created (Monte-Carlo number of simulations)
        - model : GNN model which takes as parameters (x,edge_index,edge_attr,batch)
        - seed : random seed
        - err : Boolean for the scoring function
            * True : means that we will compare the error wrt to true_val (see below)
            * False : means that we will only compute the difference between the prediction with or wihtout edges
        - true_val : Boolean for the scoring function
            * True : means that we will compare the error of the prediciton with the true HL gap
            * False : means that we will compare with respect to the prediction of the model on the full graph
            * None : used for err = False
    Returns :
        - A numpy array of all the shapley value for each edge of the graph.
    """
    random.seed(seed)
    model.eval()
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    x = data.x

    # Computing the approximation y
    
    if true_val:
        # real value of HOMO-LUMO gap for comparison
        HLG = data.y.item()
    elif true_val == False:
        # predicted value of HOMO-LUMO gap for comparison
        batch_HLG = torch.zeros(x.shape[0], dtype=int, device=x.device)
        HLG = model(x,edge_index,edge_attr,batch_HLG)
    else:
        pass

    n_edges = edge_index.shape[1]//2 # number of undirected edges
    print("n_edges :",n_edges)
    arange_edges = np.arange(n_edges) # 0,1,...,n_edges-1 for undirected edge indexing
    shapley_array = np.zeros(n_edges) # 1 shapley value per double-edge
    
    num_sub = binom(n_edges-1,range(0,n_edges)) # computes C(n_edges-1,k) for k in 0,...,n_edges-1
    print("num submodels : \n",num_sub)
    T = np.sum(num_sub) # sum of all subsets possible
    print("total number of possiblities :", T)
    P = num_sub/T # proportion of class k wrt to total of subsets of |E|-1
    N_class = np.round(P*M).astype(int)
    print("Number of subsets per size {0,...,|E|/2-1} : ",N_class)
    for j in range(n_edges):
        #print(f"Starting edge {j}")
        shapley_val = 0
        idx_wo_j = np.delete(arange_edges,j) #indexes without j
        #print(f"index of edges without edge {j}\n",idx_wo_j)
        
        for k in range(len(N_class)): # Loop over C_k
            #print(f"class n°{k}")
            cum = 0
            if N_class[k] == 0:
                #print(f"no subsets of size {k}")
                continue
                
            # Randomly creating N_class[k] subsets of size k from {0,...,|E|-1}\{j}
            if M == T: # we do every combination
                S_k = [np.array(x) for x in combinations(idx_wo_j, k)]
            else:
                S_k = [np.random.choice(idx_wo_j, k, replace=False) for _ in range(N_class[k])]
            
            for S_idx in S_k:
                assert not model.training # sanity check
                
                # S without j
                batch_min = torch.ones(x.shape[0], dtype=int, device=x.device)
                if S_idx.size == 0:
                    v_min = 0
                    S_min_idx = torch.tensor([])
                    S_min = torch.tensor([[],[]])
                    v_min = 0
                else:
                    S_min_idx = np.append(2*S_idx,2*S_idx+1) # Re-creating the double-index edge-set without j
                    S_min = edge_index[:,S_min_idx] # selecting the edges indexed by S_min_idx (tensor of nodes indices)
                    S_min_attr = edge_attr[S_min_idx,:]
                    
                    indices_nodes_connected = torch.unique(S_min[0])
                    batch_min[indices_nodes_connected] = 0

                    pred_min = model(x,S_min,S_min_attr,batch_min)[0]
                    #print("pred_min : ",pred_min)
                    if err:
                        v_min = (pred_min-HLG)**2
                    else:
                        v_min = pred_min
                
                # S with j
                batch_plus = torch.ones(x.shape[0], dtype=int, device=x.device)
                
                S_plus_idx = np.append(S_min_idx,[2*j,2*j+1]) # Adding j to S
                S_plus = edge_index[:,S_plus_idx]
                S_plus_attr = edge_attr[S_plus_idx,:]
                
                indices_nodes_connected = torch.unique(S_plus[0])
                batch_plus[indices_nodes_connected] = 0
                  
                pred_plus = model(x,S_plus,S_plus_attr,batch_plus)[0]
                #print("pred_max : ",pred_plus)
                #print("\n")
                if err:
                    v_plus = (pred_plus-HLG)**2
                else:
                    v_plus = pred_plus
                
                
                # Adding to cumulative sharpe val
                cum += (v_plus-v_min)
            shapley_val += cum/N_class[k]
            #print(f"class {k} done")
        shapley_array[j] = shapley_val/n_edges
        #print(shapley_array[j])
        #print(f"edge {j} done")
    return (-1)**err*shapley_array