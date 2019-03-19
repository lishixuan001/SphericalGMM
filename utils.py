import numpy as np
import torch
import h5py
from pdb import set_trace as st
import torch.nn as nn
import torch.nn.functional as func
import argparse
import math

def load_args():
    parser = argparse.ArgumentParser(description='HighDimSphere Train')
    parser.add_argument('--data_path',     default='../mnistPC', type=str,   metavar='XXX', help='Path to the model')
    parser.add_argument('--batch_size',    default=5,           type=int,   metavar='N',   help='Batch size of test set')
    parser.add_argument('--num_epochs',    default=200,          type=int,   metavar='N',   help='Epoch to run')
    parser.add_argument('--num_points',    default=512,          type=int,   metavar='N',   help='Number of points in a image')
    parser.add_argument('--log_interval',  default=10,           type=int,   metavar='N',   help='log_interval')
    parser.add_argument('--sigma',         default=0.05,         type=float, metavar='N',   help='sigma of sdt')
    parser.add_argument('--baselr',        default=0.05 ,        type=float, metavar='N',   help='sigma of sdt')
    parser.add_argument('--gpu',           default='0,1',        type=str,   metavar='XXX', help='GPU number')
    parser.add_argument('--radius',        default=0.5,          type=float, metavar='XXX', help='Radius for density')

    args = parser.parse_args()
    return args


def load_data(data_dir, batch_size, shuffle=True, num_workers=4):
    train_data = h5py.File(data_dir + ".hdf5" , 'r')
    xs = np.array(train_data['data'])
    ys = np.array(train_data['labels'])
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(xs).float(), torch.from_numpy(ys).long())
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle = shuffle, num_workers=num_workers)
    train_data.close()
    return train_loader_dataset


def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud: tensor (batch_size, num_points, num_dims, num_channels)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    B, N, D = point_cloud.size()
    
    x_norm = (point_cloud ** 2).sum(-1).view(B, -1, 1) # [B, N, 1]
    y_norm = x_norm.view(B, 1, -1) # [B, 1, N]

    dist = x_norm + y_norm - 2.0 * torch.matmul(point_cloud, point_cloud.transpose(1, 2)) # [B, C, N, N]
    
    return dist


def down_sampling(X, v, out_pts):
    B, N, _ = X.shape
    
    ind_all = []
    for b in range(B):
        indices = torch.multinomial(v[b], out_pts, replacement = False)
        ind_all.append(indices)
        
    ind_all = torch.stack(ind_all)
    idx = (torch.arange(B)*N).cuda()
    idx = idx.view((B, 1))
    k2 = ind_all + idx
    X = X.view(-1, X.shape[-1])
    return X[k2]


def density_mapping(inputs, weights, s2_grid):
    """
    inputs : [B, N, 3]
    weights : [B, N]
    s2_grid : [2b, 2b, 3]
    """
    B, N, D = inputs.size()
    b = s2_grid.size()[0] / 2
    
    s2_grid = s2_grid.view(-1, D) # [4b^2, 3]
    inputs = inputs.unsqueeze(2).repeat(1, 1, s2_grid.size()[0], D) # [B, N, 4b^2, 3]
    
    # Calculate Density
    numerator = inputs - s2_grid # [B, N, 4b^2, 3]
    numerator = numerator * numerator
    numerator = -0.5 * numerator
    numerator = torch.exp(numerator) # [B, N, 4b^2, 3]
    
    denominator = math.sqrt(math.pow(2 * math.pi, D))
    
    density = numerator / denominator # [B, N, 4b^2, 3]
    
    # Normalization Over Dimension
    density = torch.norm(density, p=2, dim=-1) # [B, N, 4b^2]
    
    # Multiply Weights
    weights = weights.unsqueeze(-1) # [B, N, 1]
    density = density * weights # [B, N, 4b^2]
    
    # Sum Over Number of Points
    density = density.sum(dim=1) # [B, 4b^2]
    
    # Adjust Dimension
    density = density.view(B, 2*b, 2*b)  # [B, 2b, 2b]
    
    return density
    
    
    
    
    



















# END