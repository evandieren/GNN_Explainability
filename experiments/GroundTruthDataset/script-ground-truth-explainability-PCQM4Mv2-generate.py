from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from ogb.lsc import PCQM4Mv2Dataset
import ogb
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

print(rdkit.__version__) #2021.03.5
print(ogb.__version__) #1.3.3

# download sdf for pcqm4m-v2 dataset
# !wget http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz
# !tar -xf pcqm4m-v2-train.sdf.tar.gz

suppl = Chem.SDMolSupplier('pcqm4m-v2-train.sdf')

# get property values (HOMO-LUMO gap) for dataset
dataset = PCQM4Mv2Dataset(root = '.', only_smiles = True)
prop_values=[]
for dat in dataset:
    prop_values.append(dat[1])
prop_values_arr=np.array(prop_values)


#generate canonical SMILEs list
atomlist=[]
smilesall=[]
moliall=[]
fail=[]
isomericSmiles=False # chirality not considered
kekuleSmiles=True
for moli, mol in enumerate(tqdm(suppl)):   
    mol=suppl[moli]
    mol=Chem.RemoveHs(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    try:
        smile = Chem.MolToSmiles(mol,isomericSmiles=isomericSmiles, kekuleSmiles=kekuleSmiles, canonical=True)
        smilesall.append(smile)
        moliall.append(moli)
    except:
        fail.append(moli)
    for atom in mol.GetAtoms():
        atomidx=atom.GetAtomicNum()
        if atomidx not in atomlist:
          atomlist.append(atomidx)

smilesall2=np.array(smilesall)

# convert SMILE list to dictionary for faster lookup
smilesdict={}
for sidx, s in enumerate(smilesall):
    smilesdict[s]=sidx
    
    

results_all=[]
verbose=False
generatedsmilelist=[]
singlebond = list(Chem.MolFromSmiles("CC").GetBonds())[0]
for molidx in range(len(suppl)):
    if molidx%10000==0:
        print(molidx, end=', ')
    mol=suppl[molidx]
    Chem.Kekulize(mol, clearAromaticFlags=True)
    if mol:
        results_arr=[]
        canrm=[]
        hetero=[]
        for atomi, atom in enumerate(mol.GetAtoms()):
          numnb=len(atom.GetNeighbors())
          if numnb==1 and atom.GetAtomicNum()==6:
                canrm.append(atomi)
          if atom.GetAtomicNum()!=6:
            hetero.append(atomi)
          #print(atomi, numnb)
        nonsingle=[]
        inring=[]
        for bondi, bond in enumerate(mol.GetBonds()):
            bondtyp=bond.GetBondType()
            if bondtyp!=Chem.BondType.SINGLE:
                nonsingle.append(bondi)
            if bond.IsInRing() and bondtyp==Chem.BondType.SINGLE:
                inring.append([bondi, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        # remove atom if carbon end atom
        for rmidx in canrm:
            molcopy= Chem.RWMol(mol)
            molcopy.RemoveAtom(rmidx)
            generatedsmile=Chem.MolToSmiles(molcopy,isomericSmiles=isomericSmiles, kekuleSmiles=kekuleSmiles)
            if generatedsmile in smilesdict:
                match=smilesdict[generatedsmile]
                if verbose:
                    print(molidx,'r', rmidx, ':', match)
                results_arr.append([molidx, 'r', rmidx, match])
        # change atom to C if heteroatom
        for cidx in hetero:
            molcopy= Chem.RWMol(mol)
            (molcopy.GetAtoms()[cidx]).SetAtomicNum(6)
            try:
                generatedsmile=Chem.MolToSmiles(molcopy,isomericSmiles=isomericSmiles, kekuleSmiles=kekuleSmiles)
                if generatedsmile in smilesdict:
                    match=smilesdict[generatedsmile]
                    if verbose:
                        print(molidx,'c', cidx, ':', match)
                    results_arr.append([molidx,'c', cidx, match])
            except:
                match=0
        # saturate bond
        for bidx in nonsingle:
            molcopy= Chem.RWMol(mol)
            molcopy.ReplaceBond(bidx, singlebond, preserveProps=False)
            try:
                generatedsmile=Chem.MolToSmiles(molcopy,isomericSmiles=isomericSmiles, kekuleSmiles=kekuleSmiles)
                if generatedsmile in smilesdict:
                    match=smilesdict[generatedsmile]
                    if verbose:
                        print(molidx,'b', bidx, ':', match)
                    results_arr.append([molidx,'b', bidx, match])
            except:
                match=0
        # break ring bond if saturated
        for didx in inring:
            molcopy= Chem.RWMol(mol)
            molcopy.RemoveBond(didx[1],didx[2])
            try:
                generatedsmile=Chem.MolToSmiles(molcopy,isomericSmiles=isomericSmiles, kekuleSmiles=kekuleSmiles)
                if generatedsmile in smilesdict:
                    match=smilesdict[generatedsmile]
                    if verbose:
                        print(molidx,'d', didx[0], ':', match)
                    results_arr.append([molidx,'d', didx[0], match])
            except:
                match=0
        if results_arr!=[]:
            results_all.append(np.array(results_arr))
            
            
results_all2=np.vstack(results_all)

# split atomwise, bondwise
idx_first=results_all2[:,0].astype('int')
operatoridx=results_all2[:,1]
operatoridx[operatoridx=='c']=0
operatoridx[operatoridx=='r']=1
operatoridx[operatoridx=='b']=2
operatoridx[operatoridx=='d']=3
operatoridx=operatoridx.astype('int')
atombondidx=results_all2[:,2].astype('int')
idx_second=results_all2[:,3].astype('int')
atomwise_idx=np.argwhere(operatoridx<2)[:,0]
bondwise_idx=np.argwhere(operatoridx>=2)[:,0]

# get explanation values for pairs
explain_val=prop_values_arr[idx_first]-prop_values_arr[idx_second]
results_modif=np.vstack((idx_first, operatoridx,atombondidx,idx_second,explain_val)).T

# save to csv for atomwise
df = pd.DataFrame(results_modif[atomwise_idx,:])
list_columns=['molecule index', 'operation index', 'atom index', 'paired molecule index', 'explanation value']

df.columns =list_columns
for key in list_columns[:-1]:
    print(key)
    tmp=df[key].values.astype(int)
    df[key] = tmp

key=list_columns[-1]
tmp=df[key].values
df[key] = np.round(tmp, 10)
df.to_csv("ground-truth-explainability-PCQM4Mv2-atomwise.csv",index=False)

# save to csv for bondwise
df2 = pd.DataFrame(results_modif[bondwise_idx,:])
list_columns=['molecule index', 'operation index', 'bond index', 'paired molecule index', 'explanation value']

df2.columns =list_columns
for key in list_columns[:-1]:
    print(key)
    tmp=df2[key].values.astype(int)
    df2[key] = tmp

key=list_columns[-1]
tmp=df2[key].values
df2[key] = np.round(tmp, 10)
df2.to_csv("ground-truth-explainability-PCQM4Mv2-bondwise.csv",index=False)
