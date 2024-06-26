import sys
sys.path.insert(0, '../../../../models/pcqm4m-v2_ogb')
# PyTorch related
import torch
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.explain import Explainer,GNNExplainer
from torch_geometric.utils import to_undirected
import torch.optim as optim
from gnn import GNN

# Dataset-related
import ogb
from ogb.lsc import PCQM4Mv2Dataset, PygPCQM4Mv2Dataset

# Misc.
import os
import json
from tqdm import tqdm
import time
import random
import numpy as np
import pandas as pd
from numpy.random import default_rng
import csv


path_model = "../../../../models/pcqm4m-v2_ogb"
model_pt = torch.load(path_model+"model_trained.pt",map_location=torch.device('cpu'))

shared_params = {
        'num_layers': 5,
        'emb_dim': 600,
        'drop_ratio': 0,
        'graph_pooling': 'sum'
    }

device = torch.device("cpu")
print(device)
model = GNN(gnn_type = 'gcn', virtual_node = False, **shared_params).to('cpu')
model.load_state_dict(model_pt['model_state_dict'])

optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer.load_state_dict(model_pt['optimizer_state_dict'])
epoch = model_pt['epoch']
print(epoch)

path_data = "../../../../data/"

dataset = PygPCQM4Mv2Dataset(root = path_data)
print("loading dataset ok")

mol_indices = np.loadtxt(path_data+"gr5_molecules.csv",
                 delimiter=",", dtype=int)
print("loading index OK")

explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=500),
        explanation_type="model",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        )
    )

print("creation of edge-wise explainer OK")

model.eval()
dict_out = {}
count = 0
failed = []
print(len(mol_indices))
for idx in mol_indices:
    data = dataset[idx]
    print(count)
    batch = torch.zeros(data.x.shape[0],dtype=torch.long, device=data.x.device)
    target = data.y
    try:
        explanation = explainer(x=data.x,edge_index=data.edge_index,edge_attr=data.edge_attr,batch=batch)
    except Exception as e:
        print(e)
        print(f"error with molecule {idx}")
        print("saving to failed")
        failed.append(idx)
    dict_out[int(idx)] = explanation.edge_mask
    count += 1
    if count%50 == 0:
        print("saving " + str(count))
        torch.save(dict_out,"./model_5gr_gnnexpl_dict.pt")

torch.save(dict_out,"./model_5gr_gnnexpl_dict.pt")
with open("failed.csv","w") as f:
    wr = csv.writer(f,delimiter="\n")
    wr.writerow(failed)
