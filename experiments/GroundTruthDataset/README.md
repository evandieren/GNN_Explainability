# Ground truth explanation dataset for chemical property prediction on molecular graphs

The two csv files `ground-truth-explainability-PCQM4Mv2-*.csv` store separately atomwise and bondwise explanation values of the HOMO-LUMO gap.
Format of csv files:

```
molecule index, operation index, atom/bond index, paired molecule index, explanation value
```

Load csv files as following:

```
    import pandas as pd
    df = pd.read_csv("ground-truth-explainability-PCQM4Mv2-atomwise.csv")
    df.head()
```

These explanation values are calculated for the molecules in the training set of PCQM4Mv2 quantum chemistry dataset (PubChemQC project). Download pcqm4m-v2-train.sdf from https://ogb.stanford.edu/docs/lsc/pcqm4mv2/ or on commandline:

```
    wget http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz
    tar -xf pcqm4m-v2-train.sdf.tar.gz # extracted pcqm4m-v2-train.sdf 
```

The molecule index in csv indicates the index of molecule in the sdf file starting with 0. To preserve the correct order of atoms/bonds, it's important to load the molecules as following and not to add hydrogens:

```
    from rdkit import Chem
    suppl = Chem.SDMolSupplier('pcqm4m-v2-train.sdf')
```

## Cite paper
Hruska, E., Zhao, L. & Liu, F. Ground truth explanation dataset for chemical property prediction on molecular graphs. ChemRxiv (2022).