import torch
import numpy as np
import random
import matplotlib.pyplot as plt

import os
from datetime import datetime

import pandas as pd
import numpy as np
import networkx as nx

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

from rdkit_heatmaps import mapvalues2mol
from rdkit_heatmaps.utils import transform2png
from rdkit import Chem
from rdkit.Chem import Draw

from torchdrug import data
from torchdrug.core import Registry as R

# functions
def set_all_seeds(SEED):
    '''
    Set all seeds for reproducibility.
    '''
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)
    random.seed(SEED)

def load_data(DATA_PATH, SMILES_FIELD_NAME, LABEL_FIELD_NAME):
    '''
    Load the data from a .csv file.
    '''
    df_data = pd.read_csv(DATA_PATH, sep = ",")
    df_data = df_data[[SMILES_FIELD_NAME, LABEL_FIELD_NAME]]
    
    return df_data


def save_model(model, MODEL_SAVE_PATH):
    '''
    Save the trained model.
    '''
    now = datetime.now()
    current_time = now.strftime("%Y_%m_%d-%H_%M_%S")

    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)

    MODEL_PATH = MODEL_SAVE_PATH + "/model_" + current_time + ".ckpt"
    torch.save(model.state_dict(), MODEL_PATH)

def create_edge_index(mol):
    """
    Create edge index for a molecule.
    """
    adj = nx.to_scipy_sparse_matrix(mol).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index

def visualize_explanations(edge_index,smiles, phi_edges,cmap, SAVE_PATH=None):
    
    test_mol = Chem.MolFromSmiles(smiles)
    test_mol = Draw.PrepareMolForDrawing(test_mol)

    num_bonds = len(test_mol.GetBonds())

    rdkit_bonds = {}

    for i in range(num_bonds):
        init_atom = test_mol.GetBondWithIdx(i).GetBeginAtomIdx()
        end_atom = test_mol.GetBondWithIdx(i).GetEndAtomIdx()
        
        rdkit_bonds[(init_atom, end_atom)] = i

    rdkit_bonds_phi = [0]*num_bonds
    for i in range(len(phi_edges)):
        phi_value = phi_edges[i]
        init_atom = edge_index[0][i].item()
        end_atom = edge_index[1][i].item()
        
        if (init_atom, end_atom) in rdkit_bonds:
            bond_index = rdkit_bonds[(init_atom, end_atom)]
            rdkit_bonds_phi[bond_index] += phi_value
        if (end_atom, init_atom) in rdkit_bonds:
            bond_index = rdkit_bonds[(end_atom, init_atom)]
            rdkit_bonds_phi[bond_index] += phi_value

    plt.clf()
    canvas = mapvalues2mol(test_mol, None, rdkit_bonds_phi, atom_width=0.2, bond_length=0.5, bond_width=0.5,color=cmap)
    img = transform2png(canvas.GetDrawingText())

    
    if SAVE_PATH is not None:

        img.save(SAVE_PATH + "/" + "EdgeSHAPer_explanations_heatmap.png", dpi = (300,300))
    
    return img  
#classes

@R.register("datasets.ChEMBL")
class ChEMBL(data.MoleculeDataset):
    '''
    Class for the molecule dataset.
    '''
    def __init__(self, path, smiles_field, target_fields, verbose=1, **kwargs):
    
        self.path = path
        self.smiles_field = smiles_field
        self.target_fields= target_fields

        self.load_csv(self.path, smiles_field=self.smiles_field, target_fields=self.target_fields,
                    verbose=verbose, **kwargs)

class ChEMBLDatasetPyG(InMemoryDataset):
    '''
    Class for the PyG version of the dataset.
    '''
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, data_list = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data_list = data_list

        data_list = self.data_list

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)

class GCN(torch.nn.Module):
    '''
    4-layer GCN model class.
    '''
    def __init__(self, node_features_dim, hidden_channels, num_classes):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(node_features_dim, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x.float(), edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        # 2. Readout layer
        x = global_mean_pool(x, batch)  

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x