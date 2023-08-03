print("importing")
import h5py

import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

def normalizing_flows(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device=}")

    num_layers = 4
    base_dist = distributions.StandardNormal(shape=[6])

    transform_layers = []
    for _ in range(num_layers):
        transform_layers.append(transforms.permutations.ReversePermutation(features=3))
        transform_layers.append(
            transforms.autoregressive.MaskedAffineAutoregressiveTransform(
                features=3,
                hidden_features=5
            )
        )

    transform_obj = transforms.CompositeTransform(transform_layers)

    flow_obj = flows.Flow(transform_obj, base_dist).to(device)

    optimizer = torch.optim.Adam(flow_obj.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
if __name__ == '__main__':
    normalizing_flows() 