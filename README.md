# InterpretableGCN
The codes is to develop Graph Convolutional Network (GCN) model for drug discovery and material informatics and to interpret its basis of prediction. Integrated Gradients was implemented to interptet deep learning model, and this method requires no modification to the original network and is extremely simple to implement; it just needs a few calls to the standard gradient operator.
![Uploading image_.png…]()


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
│   └── smiles_cas_N6512.smi
├── model
│   └── checkpoint_model.pth
├── images
```
