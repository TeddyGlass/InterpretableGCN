# InterpretableGCN
The code provides us with the interpretation of its predictions in addition to developing a [Graph Convolutional Network (GCN)](https://arxiv.org/abs/1609.02907) model for drug discovery and materials informatics. [Integrated Gradients](https://arxiv.org/abs/1703.01365) was implemented to interpret the deep learning model. This method requires no modification to the original network and is extremely simple to implement; it just needs a few calls to the standard gradient operator. As shown below, a red to blue color map will be displayed on the chemical structure.

![toppage](/images/image_.png) 


# Directory
```
.
├── data
│     └── smiles_cas_N6512.smi
├── figure
├── images
├── model
│     ├── checkpoint_model.pth
│     └── config.ini
├── src
│     ├── mol2graph.py
│     ├── callbacks.py
│     ├── network.py
│     ├── integrated_gradients.py
│     └── train.py
├── install_packages.sh
├── train.sh
├── visualize.sh
└── README.md

```

# Installation of packages
*These packages are dependent on PyTorch version 1.9.** *.*

```bash
conda create -n InterpretableGCN python=3.8.1 -y
source activate InterpretableGCN
bash install_packages.sh
```
Confirm that the installation has completed successfully. The version of pytorch and pytorch geometric are showed your terminal.
```bash
python -c "import torch; print(torch.__version__)"
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

# Training
```bash
bash start_train.sh
```
![demo](/images/demo.png)
![learning_curve](/images/learning_curve.png)
When the training is completed, configure file with training parameters is saved in model folder.  

If you want to train model with other patameters, you can edit training parameters in start_train.sh.
```bash:start_train.sh
data=../data/smiles_cas_N6512.smi
batch_size=128
dim=64
n_conv_hidden=4
n_mlp_hidden=1
dropout=0.1
lr=1e-4
n_epochs=1000
patience=1
model_path=../model/checkpoint_model.pth

python ./src/train.py $data $batch_size $dim $n_conv_hidden $n_mlp_hidden $dropout $lr $n_epochs $patience $model_path
```

# Visualization
To work visualization of the prediction basis of GCN, you have to specify SMILES you want to predict and the configure file which was generated after training. When you have completed the ptrediction and visualization of the prediction basis, output file with name of predicted classs is saved in firuge folder.  
Please edit **visualize.sh** as following.
```bash
config=../model/config.ini
smiles='OC(=O)c1ccccc1' # You can edit it yourself !

python ./src/visualize.py $config $smiles
```
To visualize, run the following command
```bash
bash visualize.sh
```
