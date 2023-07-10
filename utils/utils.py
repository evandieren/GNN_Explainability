import sys
sys.path.insert(0,'../models/pcqm4m-v2_ogb/')
sys.path.insert(0, '../../edgeSHAPer/src/')

# torch related imports
import torch
import torch.nn.functional as F
from torch_geometric.data.data import Data
from torch_geometric.utils import degree
from torch_geometric.explain import Explainer,GNNExplainer,PGExplainer
from torch_geometric.explain.metric import fidelity
from gnn import GNN
from torch_geometric.utils import to_networkx
import networkx as nx

# other
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils_edge import visualize_explanations
import matplotlib.pyplot as plt

def refactor(x: torch.Tensor,double=False,divide=False) -> torch.Tensor:
    """
    Refactors a positive array to an array of 0 to 1.
    If double is True, then we create an array of size 2*len(array)
    and copy each value twice. Divide = True divides the result by 2

    Args:
        -x : torch tensor of dimension (1xn)
        -double : boolean for the size of the array to return:
            - If True : creation of an array of size 2*n and duplication of each element next to itself
            - If False : returns a n-size array
        -divide : boolean for the division of the array to return:
            - If True : the entire returned array is divided by 2
            - If False : only returns the refactored array.
    Returns:
        The refactored tensor
    """
    min_val = min(x)
    max_val = max(x)
    range_val = max_val - min_val
    ret = (x-min_val)/range_val
    if double:
        d_ret = torch.zeros(2*len(ret))
        for i in range(len(x)):
            d_ret[2*i] = d_ret[2*i+1] = ret[i]
        return (1-0.5*divide)*d_ret
    return (1-0.5*divide)*ret

def rank_arr(x: torch.Tensor, descending: bool) -> torch.Tensor:
    """
    Returns a tensor of ranks from elements in x. descending means descending order, 
    if descending=False, we look at increasing (default) order
    
    Note:
        If two elements have the same value, they will have the same rank

    Args:
        -x : Pytorch tensor of dimension (1,n)
        -descending : boolean for the sorting:
            - If True : descending order
            - If False : increasing order

    Returns:
        - returns a tensor containing ranks of each element of x
    """
    rank = {int(val): i for i, val in enumerate(torch.sort(torch.unique(x),descending=descending).values)}
    return torch.tensor([rank[int(val)] for val in x])

""" old version
def rank_arr(x: torch.Tensor) -> torch.Tensor:
    # array : PyTorch tensor
    rank = {int(val): i for i, val in enumerate(torch.sort(torch.unique(x)).values)}
    return torch.Tensor([rank[int(val)] for val in x])
"""

""" tests
a = torch.tensor([5,9,-1,10,2,4,5,2])
b = torch.tensor([9,5,10,-1,4,2,2,5])
print(rank_arr(a,False))
print(rank_arr(b,False))
c = torch.vstack((rank_arr(a,False),rank_arr(b,False)))
print(type(c))
d = torch.tensor([0,1])
print(torch.index_select(c,1,d))
x = torch.randn(3, 4)
print(type(x))
indices = torch.tensor([0, 2])
print(torch.index_select(x, 0, indices))
print(torch.index_select(x, 1, indices))
"""

def get_top_k(x: torch.Tensor, k: int, positive: bool) -> torch.Tensor:
    """
    Returns the top-k indices from a given array. positive means that we are
    looking for the highest, if positive=False, we look at the lowest
    
    Args:
        -x : Pytorch tensor of dimension (1,n)
        -k : number of elements in the top-k set
        -positive : boolean for the sorting:
            - If True : we are looking for the k biggest elements
            - If False : we are looking for the k smallest elements

    Returns:
        Tensor containing the indices of the top-k set (biggest or smallest depending on 'positive')
    """
    sorted_indices = torch.argsort((-1)**positive*x)
    return sorted_indices[:k]

#=========#
# Metrics #
#=========#

# FID +/-
def fid(positive,model,data,n_sub,tot_expl):
    """
    Computing fid +/- for a given graph using a GNN model
    in a regression setting
    
    Args:
        - positive : True if fid +, else False
        - model : GNN model to compute the prediction
        - data : Data object representing the graph instance
        - n_sub : number of elements considered unimportant (for fid+) or important (fid-)
        - tot_expl : array of explanations between 0 and 1

    Returns:
        The value of the fid +/- for the graph instance
    """
    # taking the mean of pairs of edges
    model.eval()
    S_idx = get_top_k(tot_expl,n_sub,not positive)

    S_imp_idx = torch.cat((2*S_idx,2*S_idx+1)) # Re-creating the double-index edge-set without j
    # selecting the edges indexed by S_min_idx (tensor of nodes indices)
    S_imp = torch.index_select(data.edge_index, 1, S_imp_idx) 
    S_imp_attr = torch.index_select(data.edge_attr, 0, S_imp_idx)

    # Only selecting connected nodes
    batch = torch.ones(data.x.shape[0], dtype=int, device=data.x.device)
    indices_nodes_connected = torch.unique(S_imp[0])
    batch[indices_nodes_connected] = 0
    pred_imp = model(data.x,S_imp,S_imp_attr,batch)[0]

    batch = torch.zeros(data.x.shape[0], dtype=int, device=data.x.device)
    tot_pred = model(data.x,data.edge_index,data.edge_attr,batch)
    return torch.abs(tot_pred-pred_imp)/torch.abs(tot_pred)

def compute_fid(model,list_graphs,positive,explanations,n_sub,threshold):
    """
    Computing fid +/- for a list of graphs and returns the mean FID
    
    Args:
        - model : GNN model to compute the prediction
        - list_graphs : list of PyG Data objects representing graph instances
        - positive : True if fid +, else False
        - explanations : array of explanations between 0 and 1
        - n_sub : number of elements considered unimportant (for fid+) or important (fid-)
        - threshold : threshold for (un)important explanations

        If n_sub = None, we will use the threshold, else we will use the n_sub
    Returns:
        Value of the mean fid +/-
    """
    print("top-k",n_sub)
    cumul = 0
    fids = np.zeros(len(list_graphs))
    for i,graph in enumerate(list_graphs):
        if len(explanations[i]) == graph.edge_index.shape[1]/2:
            ref_expl = refactor(explanations[i],double=True,divide=False)
        else:
            ref_expl = refactor(explanations[i],double=False,divide=False)
        tot_expl = 0.5*(ref_expl[::2] + ref_expl[1::2]) # taking the mean for each pair of edges
        if n_sub == None:
            if positive: # taking the least important
                n_sub = sum(tot_expl<threshold)
            else:
                n_sub = sum(tot_expl>threshold)
        fids[i] = fid(positive,model,graph,n_sub,tot_expl)
    return np.mean(fids),np.std(fids)

def GEA(expl, gt_expl,gt_indices, top_k):
    """
    Computing the graph explanation accuracy
    Args:
        - expl : array of explanations (between 0 and 1)
        - gt_expl : array of ground truth explanation (between 0 and 1)
        - gt_indices : list of indices for which the feature is explained
        - k : number of elements in the top-k set of explanations
    Returns:
        Value of the graph explanation accuracy

    Source : mims-harvard/GraphXAI Github repository
    """
    if type(expl) == np.ndarray:
        expl = torch.tensor(expl)
    if type(gt_expl) == np.ndarray:
        gt_expl = torch.tensor(gt_expl)

    imp_mask = get_top_k(expl, k, True)
    gt_imp_mask = get_top_k(gt_expl, k, True)
    
    TP = 0
    FP = 0
    FN = 0
    for gt_i, feat_idx in enumerate(gt_indices):
        if gt_imp_mask[gt_i]: # GT = True
            if imp_mask[feat_idx]: # imp = True
                TPf += 1
            else: # imp = False
                FP += 1
        else: # GT = False
            if imp_mask[feat_idx]: # imp = True
                FN += 1
            # no need for TN in this metric.
    JAC = TP / (TP + FP + FN + EPS)
    return JAC

def GEF_helper(expl, data, model, k):
    """
    Inside-loop function helper for GEF

    Returns the prediction of the model on the graph, and subgraph respectively
    """

    # Computing the mean of a given edge
    if len(expl) == data.edge_index.shape[1]:
        expl = 0.5*(expl[::2] + expl[1::2])

    if k > len(expl):
        print("k is bigger than the number of bonds !")
    k = min(k,len(expl))

    #   Creating top-k subgraph
    S_idx = get_top_k(expl,k,True)
    S_imp_idx = torch.cat((2*S_idx,2*S_idx+1)) # Re-creating the double-index edge-set without j
    S_imp = torch.index_select(data.edge_index, 1, S_imp_idx) 
    S_imp_attr = torch.index_select(data.edge_attr, 0, S_imp_idx)
    # removing unconnected nodes
    indices_nodes_connected = torch.unique(S_imp[0])
    new_x = torch.index_select(data.x,0,indices_nodes_connected)
    new_edge_index = torch.vstack((rank_arr(S_imp[0],False),rank_arr(S_imp[1],False)))
    sub_graph = Data(x=new_x,edge_index=new_edge_index,edge_attr=S_imp_attr,y=data.y)
    sub_batch = torch.zeros(sub_graph.x.shape[0], dtype=int, device=sub_graph.x.device)
    batch = torch.zeros(data.x.shape[0], dtype=int, device=data.x.device)
   
    model.eval()
    sub_pred = model(sub_graph.x,sub_graph.edge_index,sub_graph.edge_attr,sub_batch).item()
    pred = model(data.x,data.edge_index,data.edge_attr,batch).item()

    return pred,sub_pred

def GEF(expl:list, list_graphs:list,model: GNN, k:int):
    """
    Computing the empirical graph explanation faithfulness for a list of graphs G(N,E) and explanations
    Args:
        - expl : list of explanations (each between 0 and 1 and of size 2*|E|, tensor)
        - list_graphs : lsit of PyG Data objects containing graph information
        - model : model used for predicting
        - k : number of features contained in the subgraph
    Returns :
    A tuple made of all predictions from GEF_Helper (len(expl) x 2) matrix and the gef value.
    """
    D_KL = lambda p,q: np.sum(p*np.where(p*q != 0, np.log(p/q), 0))
    preds = np.zeros((len(expl),2))
    for i,ex in enumerate(tqdm(expl)):
        preds[i,:] = GEF_helper(ex, list_graphs[i], model, k)

    max_val = np.amax(preds)
    min_val = np.amin(preds)
    bins = np.arange(min_val-0.01,max_val+0.01,0.01)
    pdf_tot, _ = np.histogram(preds[:,0], bins=bins, density=True)
    pdf_sub, _ = np.histogram(preds[:,1], bins=bins, density=True)
    pdf_tot /= 100
    pdf_sub /= 100
    dkl = D_KL(pdf_tot,pdf_sub) # pas besoin de divisier par le # d'elements car on prend des pourcentages
    gef = 1-np.exp(-dkl)
    return (preds,gef)


def compute_gef(idx_mols,list_graphs,algo_dicts,model,k,algorithms):
    """
    Computing the empirical graph explanation faithfulness for a list of graphs G(N,E) and explanations
    for all algorithms in the algorithms list (must be a key of algo_dicts)
    Args:
        - idx_mols : list of molecule indices (from PCQM4Mv2 dataset)
        - list_graphs : lsit of PyG Data objects containing graph information
        - algo_dicts : dictionary with keys in 'algorithms' which contain all explanations
                       for molecules in idx_mols
        - model : model used for predicting
        - k : number of features contained in the subgraph
        - algorithms : list of strings for explanation algorithms we want to compute GEF from
    Returns :
    A dictionnary containing the GEF values and predictions, keys are in 'algorithms'
    """
    dict_gef = {}
    D_KL = lambda p,q: np.sum(p*np.where(p*q != 0, np.log(p/q), 0))
    for alg in algorithms:
        if alg in ["gnnexplainer","edgeSHAPer"]:
            expl = [torch.tensor((np.abs(algo_dicts[alg][i])[::2]+
                                  np.abs(algo_dicts[alg][i])[1::2])*.5) for i in idx_mols]
        else:
            expl = [torch.tensor(np.abs(algo_dicts[alg][i])) for i in idx_mols]
        dict_gef[alg] = GEF(expl,list_graphs,model,k)
        print(f"GEF {alg} : ",dict_gef[alg][1])
    return dict_gef

#=====#
# Viz #
#=====#

def graph_subgraph(dict_gef,k,col_dict,xlim,ylim,filename):
    """
    Plots a N-subplot with N being the number of algorithms to be studied (keys in dict_gef)

    Args:
        - dict_gef : dictionnary with keys : names of algorithm, and values : tuple of GEF value and all predictions 
                     for graph and subgraph
        - col_dict : dictionnary with keys : names of algorithm, and values : string of color to be plotted
        - xlim : tuple of lower and upper bound for all N graphs for x-axis
        - ylim : tuple of lower and upper bound for all N graphs for y-axis
        - filename : name of the file to save the png plot
    Returns :
    Saves the plot to 'filename'

    """

    low = min(xlim[0],ylim[0])
    high = max(xlim[1],ylim[1])

    N = len(dict_gef.keys())
    fig,ax = plt.subplots(N,figsize=(10,10))
    for i,alg in enumerate(dict_gef.keys()):
        ax[i].scatter(dict_gef[alg][0][:,0], dict_gef[alg][0][:,1], s=100, lw = 0, color=col_dict[alg],label=alg)
        mse = np.mean((dict_gef[alg][0][:,0]-dict_gef[alg][0][:,1])**2)
        ax[i].plot(np.arange(low,high+1),np.arange(low,high+1),color="dimgrey",ls="--",lw=2)
        ax[i].text(xlim[1]-1, ylim[0]+1, "MSE : " + str(mse)[:4],
                 horizontalalignment='center',
                 verticalalignment='center',fontsize=15)
        ax[i].legend()
        ax[i].grid()
        ax[i].set_xlim(xlim[0],xlim[1])
        ax[i].set_ylim(ylim[0],ylim[1])
    fig.supxlabel('Graph predictions',fontsize=15)
    fig.supylabel('Subgraph predictions',fontsize=15)
    fig.suptitle("Comparison graph vs subgraph predictions, k="+str(k),fontsize=18)
    fig.tight_layout()
    plt.savefig(filename)


def compute_tresh(thresh,n_mols,algo_dicts,col_dict,algorithms,filename):
    """
    Computes the %edges explanation vs threhsold plot
    
    Args:
        - thresh : array of threshold values between 0 and 1
        - algo_dicts : dictionary with keys in 'algorithms' which contain all explanations
                       for molecules in idx_mols
        - col_dict : dictionnary with keys : names of algorithm, and values : string of color to be plotted
        - algorithms : list of strings for explanation algorithms we want to compute the plot
        - filename : name of the file to save the png plot
    Returns :
    Saves the plot to 'filename'

    """
    fig, ax = plt.subplots(1,figsize=(10,5))
    plot_dict = {}
    for alg in algorithms:
        print(alg)
        if alg in ["gnnexplainer","edgeSHAPer"]:
            expl = [refactor(torch.tensor((np.abs(algo_dicts[alg][i])[::2]+np.abs(algo_dicts[alg][i])[1::2])*.5),
                     False,False) for i in algo_dicts[alg]]
        else:
            expl = [refactor(torch.tensor(np.abs(algo_dicts[alg][i])),
                             False,False) for i in algo_dicts[alg]]
        cumsum = np.zeros(len(thresh))
        l_bound = 0
        list_idx_mols = list(algo_dicts[alg].keys())
        expl = expl[:n_mols] # to be improved
        for i,ex in enumerate(tqdm(expl)):
            l_bound += 1/len(ex)
            for idx,t in enumerate(thresh):
                cumsum[idx] += torch.sum(ex>=t)/len(ex)
        print("# expl : ",len(expl))
        ax.plot(thresh,cumsum/len(expl),label=alg, color=col_dict[alg])
        ax.plot(thresh,l_bound/len(expl)*np.ones(len(thresh)),ls="--",color=col_dict[alg])
    ax.plot(thresh,1-thresh,label="random")
    ax.set_xticks(np.arange(0,1.1,0.1));
    ax.set_xlabel("Threshold",fontsize=15)
    ax.set_ylabel("% edges",fontsize=15)
    ax.set_title("Percentage of edges with importance above the threshold",fontsize=18)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(filename)

def sort_edge_indices(G,expl):
    array = G.edge_index.clone()
    array = array[:,::2]
    not_sorted_idx = array[0,:] > array[1,:]
    temp = array[0,not_sorted_idx].clone()
    array[0,not_sorted_idx] = array[1,not_sorted_idx].clone()
    array[1,not_sorted_idx] = temp
    sort_idx = torch.argsort(array[0,:])
    array = array[:,sort_idx]
    sorted_expl = expl[sort_idx]
    return array,sorted_expl

def plot_nx_mol(
    G: nx.Graph,
    edge_mask=None,
    edge_type=None,
    threshold=None,
    drop_isolates=False,
    ax=None,
    fig=None,
    filename=None,
    alg=None,
    flag=None,
    flag2=None):
    """Draw molecule. Taken from https://github.com/VisiumCH/AMLD-2021-Graphs/tree/master
    Args:
        G : nx.Graph
        edge_mask : dict, optional
            Dictionary of edge/float items, by default None.
            If given the edges will be color coded. If `treshold` is given,
            `edge_mask` is used to filter edges with mask lower than value.
        edge_type : array of float, optional
            Type of bond encoded as a number, by default None.
            If given, bond width will represent the type of bond.
        threshold : float, optional
            Minumum value of `edge_mask` to include, by default None.
            Only used if `edge_mask` is given.
        drop_isolates : bool, optional
            Wether to remove isolated nodes, by default True if `treshold` is given else False.
        ax : matplotlib.axes.Axes, optional
            Axis on which to draw the molecule, by default None
        fig : matplotlib.figure
        filename : name of file for saving
        alg : name of explanation algorithm
        flag : flag to know whether we are on the left or right of plot (practical for implem)
        flag2 : same
    """
    if drop_isolates is None:
        drop_isolates = True if threshold else False
    if ax is None:
        fig, ax = plt.subplots(dpi=120)

    pos = nx.planar_layout(G)
    pos = nx.kamada_kawai_layout(G, pos=pos)

    if edge_type is None:
        widths = None
    else:
        widths = edge_type + 1

    edgelist = G.edges()

    if edge_mask is None:
        edge_color = 'black'
    else:
        if threshold is not None:
            edgelist = [
                (u, v) for u, v in G.edges() if edge_mask[(u, v)] > threshold
            ]

        edge_color = [edge_mask[(u, v)] for u, v in edgelist]

    nodelist = G.nodes()
    if drop_isolates:
        if not edgelist:  # Prevent errors
            print("No nodes left to show !")
            return

        nodelist = list(set.union(*map(set, edgelist)))

    nx.draw_networkx(
        G, pos=pos,
        nodelist=nodelist,
        node_size=200,
        width=widths,
        edgelist=edgelist,
        edge_color=edge_color, edge_cmap=plt.cm.Blues,
        edge_vmin=0., edge_vmax=1.,
        node_color='darkgrey',
        with_labels = False,    
        ax=ax
    )

    if flag:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin = min(edge_color), vmax=max(edge_color)))
        sm._A = []
        cbar = plt.colorbar(sm,ax=ax)
        if flag2:
            cbar.ax.set_title("Edge\n importance",fontsize=12)
    else:
        ax.set_ylabel(alg,fontsize=16)
    print("done")

def mol_viz(algorithms,idx_mols,figsize,dataset_PyG,algo_dicts,filename):
    """
    Plots a 2 x len(algorithms) molecule graphs with edge importances

    """

    fig, ax = plt.subplots(len(algorithms),2,figsize=figsize)
    for col,idx_mol in enumerate(idx_mols):
        G = dataset_PyG[idx_mol]
        G_nx = to_networkx(G, to_undirected=True)
        edges = G_nx.edges()
        for i,alg in enumerate(algorithms):
            if alg in ["gnnexplainer","edgeSHAPer"]:
                expl = refactor(torch.tensor((np.abs(algo_dicts[alg][idx_mol])[::2]+np.abs(algo_dicts[alg][idx_mol])[1::2])*.5),
                                     False,False)
            else:
                expl = refactor(torch.tensor(np.abs(algo_dicts[alg][idx_mol])),False,False)
            sorted_edge_idx, sorted_expl = sort_edge_indices(G,expl)
            print(len(sorted_expl))
            edge_mask = {}
            for j,pair in enumerate(edges):
                edge_mask[pair] = sorted_expl[j].item()
            flag = col == 1
            flag2 = i == 0
            plot_nx_mol(G_nx,edge_mask=edge_mask,ax=ax[i,col],fig=fig,alg=alg,flag=flag,flag2=flag2)
        ax[0,col].set_title("Molecule n°"+str(idx_mol),fontsize=15)
    fig.savefig(filename)



def get_concat_v(image_list):
    n_images = len(image_list)
    tot_height = 0
    max_width = -1
    for img in image_list:
        tot_height += img.height
        max_width = max(max_width,img.width)
    dst = Image.new('RGB', (max_width, tot_height))
    curr_height = 0
    for img in image_list:
        dst.paste(img, (0, curr_height))
        curr_height += img.height
    return dst

def show_explanation(algorithms,dataset,smile_dataset,mol_idx,algo_dicts,cmap):
    print(mol_idx)
    data = dataset[mol_idx]
    print(data)
    mol_smile = smile_dataset[mol_idx][0]
    print(mol_smile)
    
    image_list = []
    for i,alg in enumerate(algorithms):
        print(alg)
        expl = algo_dicts[alg][mol_idx]
        if len(expl) == data.edge_index.shape[1]/2:
            expl = np.array(refactor(algo_dicts[alg][mol_idx],True,True))
        else:
            expl = np.array(refactor(algo_dicts[alg][mol_idx],False,True))
        print(expl)
        image_list.append(visualize_explanations(data.edge_index, mol_smile, expl,cmap))
    return get_concat_v(image_list)

def show_barplot_explanation(algorithms,dataset,mol_idx,algo_dicts,f_name):
    print(mol_idx)
    data = dataset[mol_idx]
    print(data)
    
    x = np.arange(data.edge_index.shape[1]/2) # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    
    fig, ax = plt.subplots(1,figsize=(10,5))
    
    val_dict = {}
    for i,alg in enumerate(algorithms):
        expl = algo_dicts[alg][mol_idx]
        if len(expl) == data.edge_index.shape[1]:
            expl = 0.5*(expl[::2]+expl[1::2]) # computing mean
        expl = np.array(refactor(expl,False,False))
        val_dict[alg] = expl
    
    for i, expl in val_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, expl, width, label=i)
        #ax.bar_label(x, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Length (mm)')
    ax.set_title('Explanations wrt edges')
    ax.set_xticks(x + width, x)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 1.1)

    plt.savefig(f_name)

# Charles' functions

def get_ranks(x: torch.Tensor) -> torch.Tensor:
    tmp = x.argsort()
    ranks = torch.zeros_like(tmp)
    ranks[tmp] = torch.arange(len(x))
    return ranks

def spearman_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the Spearman Rank Correlation between two tensors.

    from https://discuss.pytorch.org/t/spearmans-correlation/91931/5

    Parameters
    ----------
    x : torch.Tensor
        shape (N,)
    y : torch.Tensor
        shape (N,)

    Returns
    -------
    torch.Tensor
        spearman correlation
    """
    x_rank = self._get_ranks(x)
    y_rank = self._get_ranks(y)

    n = x.size(0)
    upper = 6 * torch.sum((x_rank - y_rank).pow(2))
    down = n * (n**2 - 1.0)
    return 1.0 - (upper / down)

def _loss(self, explanation: torch.Tensor, ground_truth: torch.Tensor) -> float:
    return self.spearman_correlation(explanation, ground_truth).item()