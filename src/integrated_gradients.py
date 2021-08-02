import numpy as np
import torch
from captum.attr import  IntegratedGradients
from collections import defaultdict


def integrated_grads_mask(model, data, target):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    def model_forward(edge_mask, data):
        batch = torch.zeros(data.x.shape[0], dtype=int).to(device)
        out = model(data.x, data.edge_index, batch, edge_mask)
        return out

    # integrated gradients
    input_mask = torch.ones(data.edge_index.shape[1]).requires_grad_(True).to(device)
    ig = IntegratedGradients(model_forward)
    mask = ig.attribute(
        input_mask,
        target=target,
        additional_forward_args=(data,),
        internal_batch_size=data.edge_index.shape[1]
    )
    edge_mask = mask.cpu().detach().numpy()

    # aggregate edge directions
    edge_mask_dict = defaultdict(float)
    for val, u, v in list(zip(edge_mask, *data.edge_index)):
        u, v = u.item(), v.item()
        if u > v:
            u, v = v, u
        edge_mask_dict[(u, v)] += val
    edge_mask_dict_max = max(np.abs([_ for _ in edge_mask_dict.values()]))
    if edge_mask_dict_max > 0:
        edge_mask_dict = {key: value/edge_mask_dict_max for key, value in edge_mask_dict.items()}
    return edge_mask_dict


def edge_mask_to_atom_mask(edge_mask_dict):
    # get atom indexes
    atom_indexes = []
    for key in edge_mask_dict.keys():
        i, j = key[0], key[1]
        atom_indexes.append(i)
        atom_indexes.append(j)
    # integrate edge masks into atom
    atom_mask_dict = {}
    for i in range(max(atom_indexes)+1):
            i_values = []
            for key, value in  edge_mask_dict.items():
                if i in key:
                    i_values.append(value)
            atom_mask_dict[i] = i_values[np.argmax(np.abs(i_values))]
    return atom_mask_dict


def bond_pairs_to_id (edge_index):
    bond_pairs = {}
    for i in range(edge_index.shape[1]):
        if i%2==0:
            bond_pair = sorted([edge_index[0][i], edge_index[1][i]])
            bond_pairs[int(i/2)] = (bond_pair[0], bond_pair[1])
    return bond_pairs


def calor_cmap(x):
    """Red to Blue color map
    x: list
    """
    cmaps = []
    for v in x:
        if v > 0:
            # Red cmap for positive value
            cmap = (1.0, 1.0 - v, 1.0 - v)
        else:
            # Blue cmap for negative value
            v *= -1
            cmap = (1.0 - v, 1.0 - v, 1.0)
        cmaps.append(cmap)
    return cmaps