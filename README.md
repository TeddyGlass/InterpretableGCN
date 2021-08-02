# InterpretableGCN
The codes is to develop [Graph Convolutional Network (GCN)](https://arxiv.org/abs/1609.02907) model for drug discovery and material informatics, and to interpret its basis of prediction. [Integrated Gradients](https://arxiv.org/abs/1703.01365) was implemented to interptet deep learning model, and this method requires no modification to the original network and is extremely simple to implement; it just needs a few calls to the standard gradient operator. As shown below, a red to blue color map is displayed on the chemical structure. Details and commentary of this implementation is available in [Qiita](https://qiita.com/kuro3210/items/811f95d070cf6b98396a).

![toppage](/images/image_.png) 


# Directory
```
.
├── data
│   └── smiles_cas_N6512.smi
├── src
│   ├── mol2graph.py
│   ├── Early_stopping.py
│   ├── MolecularGCN.py
│   ├── Integrated_Gradient.py
│   └── train.py
└── model
    └── checkpoint_model.pth
```

# Enviornment
```bash
conda create -n InterpretableGCN pyhton=3.8.1 -y
source activage InterpretableGCN
conda install --yes --file requirements.txt
```

# Training

# Visualization
