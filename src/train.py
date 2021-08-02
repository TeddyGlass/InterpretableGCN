import os
os.chdir("./src")
print("Current directory: ", os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from rdkit import Chem

import mol2graph
from network import MolecularGCN
from callbacks import EarlyStopping

import argparse
import configparser


def train(model, optimizer, loader):
    model.train()
    loss_all = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        output = model.forward(data.x, data.edge_index, data.batch).squeeze(1)
        loss =  F.cross_entropy(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(loader)


def eval(model, loader):
    model.eval()
    with torch.no_grad():
        P, S = [], []
        loss_all = 0
        for data in loader:
            data = data.to(device)
            y_true = data.y.to(device).detach().numpy()
            output = model.forward(data.x, data.edge_index, data.batch)
            y_pred = output.detach().numpy()[:,1]
            P.append(y_pred)
            S.append(roc_auc_score(y_true, y_pred))
            loss = F.cross_entropy(output, data.y)
            loss_all += loss.item() * data.num_graphs
    return  np.concatenate(P), loss_all/len(loader), np.mean(S)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('batch_size', type=int)
    parser.add_argument('dim', type=int)
    parser.add_argument('n_conv_hidden', type=int)
    parser.add_argument('n_mlp_hidden', type=int)
    parser.add_argument('dropout', type=float)
    parser.add_argument('lr', type=float)
    parser.add_argument('n_epochs', type=int)
    parser.add_argument('patience', type=int)
    parser.add_argument('model_path', type=str)
    args = parser.parse_args()

    # Load datasets
    print('-'*100)
    print('Load datasets')
    f = open(f'{args.data}', 'r', encoding='UTF-8')
    rows = f.read().split('\n')
    f.close()
    mols, properties = [], []
    failed_mol_numbers = []
    for i, r in enumerate(rows):
        mol = Chem.MolFromSmiles(r.split('\t')[0])
        property = r.split('\t')[2]
        if mol is None:
            failed_mol_numbers.append(i)
        else:
            mols.append(mol)
            properties.append(property)
    mols, properties = np.array(mols), np.array(properties, dtype=int)
    for i in failed_mol_numbers:
        print(f'{i}: SMILES can not be converted to Mol object')
    print('-'*100)
    
    # Split dataset to training, validation, and test sets
    print('-'*100)
    print('Split dataset to training, validation, and test sets')
    skf = StratifiedKFold(n_splits=5, random_state=1640, shuffle=True)
    train_idx, valid_idx = list(skf.split(mols, properties))[0]
    mols_train, y_train = mols[train_idx], properties[train_idx]
    mols_valid, y_valid = mols[valid_idx], properties[valid_idx]
    print('Training: ', mols_train.shape)
    print('Validation: ', mols_valid.shape)
    print('-'*100)

    # Mol to Graph
    print('-'*100)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses CPU...')
    print('-'*100)
   
    print('-'*100)
    print('Mol to Graph...')
    X_train = [mol2graph.mol2vec(m) for m in mols_train.tolist()]
    for i, data in enumerate(X_train):
        data.y = torch.LongTensor([y_train[i]]).to(device)
    X_valid = [mol2graph.mol2vec(m) for m in mols_valid.tolist()]
    for i, data in enumerate(X_valid):
        data.y = torch.LongTensor([y_valid[i]]).to(device)
    train_loader = DataLoader(X_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(X_valid, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print('Conveting mol to graph has been completed.')
    print('-'*100)

    # Model instance construction
    print('-'*100)
    print('Model instance construction')
    model = MolecularGCN(
        dim = args.dim,
        n_conv_hidden = args.n_conv_hidden,
        n_mlp_hidden = args.n_mlp_hidden,
        dropout = args.dropout
        ).to(device)
    print(model)
    print('-'*100)

    # Training
    lr = args.lr
    n_epochs = args.n_epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    earlystopping = EarlyStopping(patience=args.patience, path=args.model_path, verbose=True)
    history = {
        'loss_train':[],
        'loss_valid':[],
        'score_train':[],
        'score_valid':[],
    }
    for epoch in range(1, n_epochs+1):
        # training
        train(model, optimizer, train_loader)

        # performance evaluation
        y_pred_train, loss_train, score_train = eval(model, train_loader)
        y_pred_valid, loss_valid, score_valid = eval(model, valid_loader)

        # save history
        items = [
            loss_train,
            loss_valid,
            score_train,
            score_valid,
        ]
        for i, item in enumerate(items):
            keys = list(history.keys())
            history[keys[i]].append(item)
        print(f'Epoch: {epoch}/{n_epochs}, loss_train: {loss_train:.5},\
            loss_valid: {loss_valid:.5}, AUC_train: {score_train:.5}, AUC_valid: {score_valid:.5}')
        # early stopping detection
        earlystopping(loss_valid, model)
        if earlystopping.early_stop:
            print("Early Stopping!")
            print("-"*100)
            break
    
    # learning curve
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    epochs_ = np.arange(1,len(history['loss_train'])+1)
    ax1.plot(epochs_, history['loss_train'], label="loss_train", c='blue')
    ax1.plot(epochs_, history['loss_valid'], label="loss_valid", c='green')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.grid(True)
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc='lower right')
    plt.savefig('../figure/learning_curve.png', dpi=200)
    plt.show()

    # create configure file
    config = configparser.RawConfigParser()
    section = 'parameters'
    config.add_section(section)
    config.set(section, 'batch_size', args.batch_size)
    config.set(section, 'dim', args.dim)
    config.set(section, 'n_conv_hidden', args.n_conv_hidden)
    config.set(section, 'n_mlp_hidden', args.n_mlp_hidden)
    config.set(section, 'dropout', args.dropout)
    with open('../model/config.ini', 'w') as f:
        config.write(f)
