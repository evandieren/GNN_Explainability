# Edge-centric Explainability Methods for Graph Neural Networks in a Regression setting

The principal aim of this project is to explore edge-centric explainability methods for regression, as
previous works regarding the explanation of GNNs have primarily been targeted to classification only.
To tackle this task, we selected three explainability methods: GNNExplainer, a regression-setting
modified version of EdgeSHAPer and a variant of the latter also based on Shapley value, which we
named regSHAPer. We compare those three explainability methods using various metrics, which have been modified 
for the regression setting.

## Architecture of the repository
```bash
GNN_Explainability/
├── EdgeSHAPer/
├── experiments/
│   ├── Comparison_Explainers/
│   └── GroundTruthDataset/
├── models/
│   └── pcqm4m-v2_ogb/
├── utils/
│   ├── regSHAP.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

### Prerequisites and Installation

The first step to install the necessary packages is to create a new Python virtual environment (e.g. using conda or venv) with Python 3.9.7 (the version used for the project). The second step is to install PyTorch. This project was based on version 1.13.1. To install PyTorch 1.13.1 with pip, please run the following command:

```
pip install torch==1.13.1
```

Furthermore, please also install the library [pyg-lib](https://github.com/pyg-team/pyg-lib) from Pytorch-geomertric with the following command: 

```
pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```
where `${TORCH}` will be 1.13.0, and `${CUDA}` will be your cuda version or GPU (see link for pyg-lib for more information)

*BUILDING REQUIREMENTS IN PROGRESS*

Finally, the requirements to run the experiments are in the `requirement.txt` file. Please install the requirements using the following command from the root of the repository:

```
pip install -r requirements.txt
```
This will automatically install the required packages to run the experiments. 

Estimated time for installation: `< 1h`


## Running the experiments, training, etc.

All experiments are in the `experiments/` folder. The most important one from which the entire report is based on 
is in the sub-folder `experiments/Comparison_Explainers/`. In this folder, one will find a Jupyter notebook where we compare the three explainability methods.

If one wants to train the model, instead of working with our `models/pcqm4m-v2_ogb/model_trained.pt` checkpoint, then the following command has to be run : 
```
python3 main_gnn.py --gnn gcn --log_dir log_dir_ogb/ --checkpoint_dir check_dir_ogb/
```
It will train the model and save checkpoints in the `models/pcqm4m-v2_ogb/check_dir_ogb/` folder.

To compute explanations for the three methods, one has to run the scripts from the `experiments/Comparison_Explainers/scripts` folder.

## Author and supervisors

* **Eliott Van Dieren** (Author of the code and the final report)
* **Charles Dufour** (Supervision)

## Acknowledgments

* Charles Dufour for his guidance, continuous help, and for making me discover GNNs
* Andrea Mastropietro - EdgeSHAPer code for classification (original: https://github.com/AndMastro/EdgeSHAPer)
* Open Graph Benchmark - Graph Neural Network used for the project: https://github.com/snap-stanford/ogb/tree/master/examples/lsc/pcqm4m-v2


