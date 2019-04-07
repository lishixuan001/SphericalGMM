import numpy as np
import torch
import h5py
from pdb import set_trace as st
import torch.nn as nn
import torch.nn.functional as func
import lie_learn.spaces.S2 as S2
import argparse
import math

def rotate_random():
    Q, _ = np.linalg.qr(np.random.rand(3,3)*0.2)
    if np.linalg.det(Q)==-1:
        Q[:,0] = -Q[:,0]
    return Q


def load_args():
    parser = argparse.ArgumentParser(description='Spherical GMM')
    parser.add_argument('--data_path',      default='../mnist',   type=str,   metavar='XXX', help='Path to the model')
    parser.add_argument('--batch_size',     default=500,          type=int,   metavar='N',   help='Batch size of test set')
    parser.add_argument('--num_epochs',     default=2000,         type=int,   metavar='N',   help='Epoch to run')
    parser.add_argument('--num_points',     default=512,          type=int,   metavar='N',   help='Number of points in a image')
    parser.add_argument('--log_interval',   default=1000,         type=int,   metavar='N',   help='log_interval')
    parser.add_argument('--sigma',          default=0.05,         type=float, metavar='N',   help='sigma of sdt')
    parser.add_argument('--baselr',         default=5e-5 ,        type=float, metavar='N',   help='learning rate')
    parser.add_argument('--gpu',            default='0,1',        type=str,   metavar='XXX', help='GPU number')
    parser.add_argument('--density_radius', default=0.2,          type=float, metavar='XXX', help='Radius for density')
    parser.add_argument('--save_interval',  default=50,           type=int,   metavar='N',   help='save_interval')
    parser.add_argument('--resume_testing', default=0,            type=int,   metavar='N',   help='load used model at iteration')
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

   
def load_data_h5(data_dir, batch_size, shuffle=True, num_workers=4, rotate=False, batch=False):
    train_data = h5py.File(data_dir + "_data.h5" , 'r')
    train_labels = h5py.File(data_dir + "_label.h5" , 'r')
    xs = np.array(train_data['data'])
    xs = np.delete(xs, 1, 2)
    if rotate:
        if batch:
            b = xs.shape[0]
            xs = np.array([np.dot(xs[i], rotate_random()) for i in range(b)]) 
        else:
            rotation_matrix = rotate_random()
            xs = np.dot(xs, rotation_matrix)
    
#     x_min = np.amin(xs, axis=1, keepdims=True)
#     x_max = np.amax(xs, axis=1, keepdims=True)
#     out_max = np.array([0.8, 0.8, 0.5])
#     out_min = np.array([-0.8, -0.8, -0.5])
#     xs = (xs - x_min) / (x_max-x_min) * (out_max - out_min) + out_min
    ys = np.array(train_labels['label'])
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(xs).float(), torch.from_numpy(ys).long())
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle = shuffle, num_workers=num_workers)
    train_data.close()
    return train_loader_dataset

def direct_load_h5(data_dir, batch_size, shuffle=False, num_workers=4):
    #This dataset is 4-dimensional, delete second one b/c y is random
    train = h5py.File(data_dir, 'r')
    xs = np.array(train['data'])
    train_labels = h5py.File("../mnist/test_label.h5" , 'r')
    
    ys = np.array(train_labels['label'])
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(xs).float(), torch.from_numpy(ys).long())
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle = shuffle, num_workers=num_workers)
    train.close()
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

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

def get_grid(b, radius=1, grid_type="Driscoll-Healy"):
    """
    :param b: the number of grids on the sphere
    :param radius: the radius of each sphere
    :param grid_type: "Driscoll-Healy"
    :return: tensor [2b, 2b, 3]
    """
    # theta in shape (2b, 2b), range [0, pi]; phi range [0, 2 * pi]
    theta, phi = S2.meshgrid(b=b, grid_type=grid_type)
    theta = torch.from_numpy(theta)  # .cuda()
    phi = torch.from_numpy(phi)  # .cuda()

    # x will be reshaped to have one dimension of 1, then can broadcast
    # look this link for more information: https://pytorch.org/docs/stable/notes/broadcasting.html
    x_ = radius * torch.sin(theta) * torch.cos(phi)
    x = x_.reshape((1, 4 * b * b))  # tensor -> [1, 4 * b * b]

    y_ = radius * torch.sin(theta) * torch.sin(phi)
    y = y_.reshape((1, 4 * b * b))

    z_ = radius * torch.cos(theta)
    z = z_.reshape((1, 4 * b * b))

    grid = torch.cat((x, y, z), dim=0)  # -> [3, 4b^2]
    grid = grid.transpose(0, 1) # -> [4b^2, 3]
    
#     fig = pyplot.figure()
#     ax = Axes3D(fig)
#     ax.scatter(grid[:, 0], grid[:, 1], grid[:, 2])
#     pyplot.savefig('books_read.png')
    
    grid = grid.view(2*b, 2*b, 3 ) # -> [2b, 2b, 3]
    
    return grid 


def density_mapping(inputs, radius, s2_grid, sigma):
    """
    inputs : [B, N, 3]
    radius : radius to count neighbor for weights
    s2_grid : [2b, 2b, 3]
    """
    
    
    
    B, N, D = inputs.size()
    b = int(s2_grid.size()[0] / 2)

    # Get Weights
    dists = pairwise_distance(inputs)
    
    # Resize inputs and grid
    s2_grid = s2_grid.view(-1, D) # -> [4b^2, 3]
    inputs = inputs.unsqueeze(2).repeat(1, 1, s2_grid.size()[0], 1) # -> [B, N, 4b^2, 3]
    
    # Calculate Density
    numerator = inputs - s2_grid # -> [B, N, 4b^2, 3]
    numerator = torch.matmul(numerator, sigma)
    numerator = numerator * numerator
    numerator = torch.sum(numerator, dim=-1) # -> [B, N, 4b^2]
    numerator = -0.5 * numerator 
    numerator = torch.exp(numerator) # -> [B, N, 4b^2]
    
    denominator = np.sqrt((2*3.14)**D * sigma**D)
    
    #density = numerator / denominator# -> [B, N, 4b^2] 
    
    
    # Sum Over Number of Points
    density = numerator.sum(dim=1) # -> [B, 4b^2]
    
    # Adjust Dimension
    density = density.view(B, 2*b, 2*b)  # -> [B, 2b, 2b]
    
    
    
    return density
    
    
    
def data_translation(inputs, bandwidth, radius, sigma):  
    """
    :param inputs: [B, N, 3]
    :param radius: radius of area to calculate weights by number of neighbors
    :return: [B, 2b, 2b]
    """
    
    inputs = inputs.cuda()
    
    
    
    s2_grid = get_grid(
        b=bandwidth
    ).float().cuda()  # -> [2b, 2b, 3]

    inputs = density_mapping(
        inputs=inputs,
        radius=radius,
        s2_grid=s2_grid,
        sigma=sigma
    ).float()  # -> (B, 2b, 2b)
    

    return inputs


     

    















# END
