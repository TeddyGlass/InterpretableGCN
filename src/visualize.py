import os
os.chdir("./src")
print("Current directory: ", os.getcwd())

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import DrawingOptions

from network import MolecularGCN
import mol2graph
import integrated_gradients as ig

import argparse
import configparser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('smiles', type=str)
    args = parser.parse_args()

    # read config file
    config = configparser.ConfigParser()
    config.read(f'{args.config}')
    model_path = config.get('parameters', 'model_path')
    # batch_size = int(config.get('parameters', 'batch_size'))
    dim = int(config.get('parameters', 'dim'))
    n_conv_hidden = int(config.get('parameters', 'n_conv_hidden'))
    n_mlp_hidden = int(config.get('parameters', 'n_mlp_hidden'))
    dropout = float(config.get('parameters', 'dropout'))

    # device setting
    print('-'*100)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses CPU...')
    print('-'*100)

    # load best model
    print('-'*100)
    print('load best model')
    model = MolecularGCN(
        dim = dim,
        n_conv_hidden = n_conv_hidden,
        n_mlp_hidden = n_mlp_hidden,
        dropout = dropout
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    print(model)
    print('-'*100)
    
    # mol2graph
    print('-'*100)
    print('mol to graph')
    mol = Chem.MolFromSmiles(args.smiles)
    mol_graph = mol2graph.mol2vec(mol)
    print(mol_graph)
    print('-'*100)

    # prediction
    batch = torch.zeros(mol_graph.x.shape[0], dtype=int).to(device)
    x, edge_index = mol_graph.x, mol_graph.edge_index
    model.eval()
    with torch.no_grad():
        y_pred = model.forward(x, edge_index, batch, edge_weight=None).detach().numpy()
    predicted_class = int(np.argmax(y_pred))
    
    # integrated gradients
    edge_mask_dict = ig.integrated_grads_mask(model, mol_graph, predicted_class)
    edge_pairs_dict = ig.bond_pairs_to_id(mol_graph.edge_index.detach().numpy())
    edge_ig_dict = {key: edge_mask_dict[value] for key, value in edge_pairs_dict.items()}
    edge_igcolor_dict = {key: ig.calor_cmap(list(edge_ig_dict.values()))[i] for i, key in enumerate(edge_ig_dict.keys())}
    atom_mask_dict = ig.edge_mask_to_atom_mask(edge_mask_dict)
    atom_igcolor_dict = {key: ig.calor_cmap(list(atom_mask_dict.values()))[i] for i, key in enumerate(atom_mask_dict.keys())}

    # drawing
    drawer = rdMolDraw2D.MolDraw2DSVG(600, 300)
    drawer.drawOptions().padding = .0
    drawer.SetLineWidth(2)
    drawer.SetFontSize(.6)
    drawer.drawOptions().updateAtomPalette({k: (0, 0, 0) for k in DrawingOptions.elemDict.keys()})
    drawer.DrawMolecule(
        rdMolDraw2D.PrepareMolForDrawing(mol),
        highlightBonds = list(edge_igcolor_dict.keys()),
        highlightBondColors = edge_igcolor_dict,
        highlightAtoms = [k for k in atom_igcolor_dict .keys()], 
        highlightAtomColors=atom_igcolor_dict,
        highlightAtomRadii={i: 0.3 for i in range(len(atom_igcolor_dict))}
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:', '')
    with open('../figure/integrated_grads.svg', 'w') as f:
        f.write(svg)
