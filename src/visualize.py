import os
os.chdir("./src")
print("Current directory: ", os.getcwd())

import torch
from network import MolecularGCN

import argparse
import configparser


if __naeme__ == '__main__':



    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU!')
    else:
        device = torch.device('cpu')
        print('The code uses CPU...')