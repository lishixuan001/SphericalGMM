import numpy as np
import torch
import h5py
from pdb import set_trace as st
import torch.nn as nn
import torch.nn.functional as func
import lie_learn.spaces.S2 as S2
import argparse
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rotate_random():
    Q, _ = np.linalg.qr(np.random.rand(3, 3) * 0.2)
    if np.linalg.det(Q) == -1:
        Q[:, 0] = -Q[:, 0]
    return Q


def load_args():
    parser = argparse.ArgumentParser(description='Spherical GMM')

    # Model
    parser.add_argument('--data_path', default='../dataset/mnist', type=str, metavar='XXX', help='Path to the model')
    parser.add_argument('--batch_size', default=512, type=int, metavar='N', help='Batch size of test set')
    parser.add_argument('--num_epochs', default=300, type=int, metavar='N', help='Epoch to run')
    parser.add_argument('--num_points', default=512, type=int, metavar='N', help='Number of points in a image')
    parser.add_argument('--log_interval', default=1000, type=int, metavar='N', help='log_interval')
    parser.add_argument('--baselr', default=5e-5, type=float, metavar='N', help='learning rate')
    parser.add_argument('--gpu', default='3', type=str, metavar='XXX', help='GPU number')
    parser.add_argument('--visualize', default=0, type=int, metavar='XXX', help='if do visualization')
    
    # Modal Structure
    parser.add_argument('--num_classes', default='10', type=int, metavar='XXX', help='number of classes for classification') 
    parser.add_argument('--num_so3_layers', default='3', type=int, metavar='XXX', help='number of SO3 layers')

    # Save Model
    parser.add_argument('--save_interval', default=20, type=int, metavar='N', help='save_interval')
    parser.add_argument('--resume_training', default=0, type=int, metavar='N', help='load used model at iteration')
    parser.add_argument('--resume_testing', default=None, type=str, metavar='N', help='load used model at iteration')

    # Multi-Grid
    parser.add_argument('--num_grids', default=3, type=int, metavar='N', help='number of shells')
    parser.add_argument('--base_radius', default=1, type=int, metavar='N', help='radius of the out-est shell')

    # GMM
    parser.add_argument('--density_radius', default=0.2, type=float, metavar='XXX', help='Radius for density')
    parser.add_argument('--static_sigma', default=0.05, type=float, metavar='N', help='static sigma to use')
    parser.add_argument('--use_static_sigma', default=0, type=int, metavar='N', help='if use static sigma')
    parser.add_argument('--use_weights', default=0, type=int, metavar='N', help='if use weights for each point for GMM')
    parser.add_argument('--sigma_layer_diff', default=1, type=int, metavar='N', help='if expected sigma different over shells')
    
    args = parser.parse_args()
    return args


def load_data(data_dir, batch_size, shuffle=True, num_workers=4):
    train_data = h5py.File(data_dir + ".hdf5", 'r')
    xs = np.array(train_data['data'])
    ys = np.array(train_data['labels'])
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(xs).float(), torch.from_numpy(ys).long())
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=shuffle,
                                                       num_workers=num_workers)
    train_data.close()
    return train_loader_dataset


def load_data_h5(data_dir, batch_size, shuffle=True, num_workers=4, rotate=False, batch=False):
    train_data = h5py.File(data_dir + "_data.h5", 'r')
    train_labels = h5py.File(data_dir + "_label.h5", 'r')
    xs = np.array(train_data['data'])
    xs = np.delete(xs, 1, 2)
    if rotate:
        if batch:
            b = xs.shape[0]
            xs = np.array([np.dot(xs[i], rotate_random()) for i in range(b)])
        else:
            rotation_matrix = rotate_random()
            xs = np.dot(xs, rotation_matrix)

    ys = np.array(train_labels['label'])
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(xs).float(), torch.from_numpy(ys).long())
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=shuffle,
                                                       num_workers=num_workers)
    train_data.close()
    return train_loader_dataset


def data_init_norm(inputs):
    """
    inputs : [B, N, 3]
    """
    maxs, _ = torch.max(inputs, dim=2, keepdim=True)
    mins, _ = torch.min(inputs, dim=2, keepdim=True)
    inputs = (inputs - mins) / (maxs - mins) * 2 - 1
    return inputs


def direct_load_h5(data_dir, batch_size, shuffle=False, num_workers=4):
    # This dataset is 4-dimensional, delete second one b/c y is random
    train = h5py.File(data_dir, 'r')
    xs = np.array(train['data'])
    train_labels = h5py.File("../mnist/test_label.h5", 'r')

    ys = np.array(train_labels['label'])
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(xs).float(), torch.from_numpy(ys).long())
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=shuffle,
                                                       num_workers=num_workers)
    train.close()
    return train_loader_dataset


def down_sampling(X, v, out_pts):
    B, N, _ = X.shape

    ind_all = []
    for b in range(B):
        indices = torch.multinomial(v[b], out_pts, replacement=False)
        ind_all.append(indices)

    ind_all = torch.stack(ind_all)
    idx = (torch.arange(B) * N).cuda()
    idx = idx.view((B, 1))
    k2 = ind_all + idx
    X = X.view(-1, X.shape[-1])
    return X[k2]


def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    og_batch_size = point_cloud.shape[0]  # point_cloud.get_shape().as_list()[0]
    point_cloud = torch.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = point_cloud.unsqueeze(0)  # torch.expand_dims(point_cloud, 0)

    point_cloud_transpose = point_cloud.permute(0, 2, 1)
    # torch.transpose(point_cloud, perm=[0, 2, 1])
    point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)
    point_cloud_inner = -2 * point_cloud_inner
    point_cloud_square = torch.sum(point_cloud ** 2, dim=-1, keepdim=True)
    point_cloud_square_tranpose = point_cloud_square.permute(0, 2,
                                                             1)  # torch.transpose(point_cloud_square, perm=[0, 2, 1])
    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def get_grids(b, num_grids, base_radius=1, grid_type="Driscoll-Healy"):
    """
    :param b: the number of grids on the sphere
    :param base_radius: the radius of each sphere
    :param grid_type: "Driscoll-Healy"
    :param num_grids: number of grids
    :return: [(radius, tensor([2b, 2b, 3])) * 3]
    """

    grids = list()
    radiuses = [round(i, 2) for i in list(np.linspace(0, base_radius, num_grids + 1))[1:]]

    # Each grid has differet radius, the radiuses are distributed uniformly based on number
    for radius in radiuses:

        # theta in shape (2b, 2b), range [0, pi]; phi range [0, 2 * pi]
        theta, phi = S2.meshgrid(b=b, grid_type=grid_type)
        theta = torch.from_numpy(theta)
        phi = torch.from_numpy(phi)

        # x will be reshaped to have one dimension of 1, then can broadcast
        # look this link for more information: https://pytorch.org/docs/stable/notes/broadcasting.html
        x_ = radius * torch.sin(theta) * torch.cos(phi)
        x = x_.reshape((1, 4 * b * b))  # tensor -> [1, 4 * b * b]

        y_ = radius * torch.sin(theta) * torch.sin(phi)
        y = y_.reshape((1, 4 * b * b))

        z_ = radius * torch.cos(theta)
        z = z_.reshape((1, 4 * b * b))

        grid = torch.cat((x, y, z), dim=0)  # -> [3, 4b^2]
        grid = grid.transpose(0, 1)  # -> [4b^2, 3]

        grid = grid.view(2 * b, 2 * b, 3)  # -> [2b, 2b, 3]
        grid = grid.float().cuda()

        grids.append( (radius, grid) )

    assert len(grids) == num_grids
    return grids


def visualize_grids(s2_grids, folder='grid', colors=['red', 'blue', 'orange']):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i, (_, s2_grid) in enumerate(s2_grids):
        grid = s2_grid.view(-1, 3)
        x = grid[:, 0].squeeze().cpu().numpy()
        y = grid[:, 1].squeeze().cpu().numpy()
        z = grid[:, 2].squeeze().cpu().numpy()
        ax.scatter(x, y, z, marker='o', c=colors[int(i % len(colors))])
    plt.savefig("./imgs/{}/s2_grids.png".format(folder))
    plt.close()


def visualize_raw(inputs, labels, folder='raw'):
    """
    inputs : [B, N, 3]
    labels : [B]
    """
    for i in range(10):
        label = str(labels[i].item())
        image = inputs[i].detach().cpu().numpy()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(image[:, 0], image[:, 1], image[:, 2])

        # ax = fig.add_subplot(1, 2, 2, projection='3d')
        # ax.scatter3D(image[:, 0], image[:, 1], image[:, 2])
        # ax.view_init(elev=70)  # azim=-60

        plt.savefig("./imgs/{}/{}.png".format(folder, label))
        plt.close()


def visualize_sphere(origins, data, labels, s2_grids, params, folder='sphere'):
    """
    data :  list( Tensor([B, 2b, 2b]) * num_grids )
    """
    if params['use_static_sigma']:
        subfolder = "static_sigma_{}".format(params['static_sigma'])
    else:
        subfolder = "conv_sigma"
    
    for i in range(10):
        label = str(labels[i].item())
        fig, axs = plt.subplots(1, 3)
        for j, inputs in enumerate(data):
            inputs = inputs[i][0].detach().cpu().numpy()
            # ax, fig = plt.subplots(figsize=(10, 10))
            axs[j].set_title('Layer {}'.format(j))
            axs[j].imshow(inputs)
        plt.savefig('./imgs/{}/{}/{}-map.png'.format(folder, subfolder, label))
        plt.close()
            
    for i in range(10):
        label = str(labels[i].item())
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        image = origins[i].detach().cpu().numpy()
        ax.scatter3D(image[:, 0], image[:, 1], image[:, 2], marker="^", c='red')
        for j, inputs in enumerate(data):
            _, grid = s2_grids[j]
            grid = grid.view(-1, 3)
            x = grid[:, 0].squeeze().cpu().numpy()
            y = grid[:, 1].squeeze().cpu().numpy()
            z = grid[:, 2].squeeze().cpu().numpy()
            inputs = inputs[i]
            _, B, B = inputs.shape
            inputs = inputs.view(B*B, 1)
            inputs = inputs[:, 0].detach().cpu().numpy()
            ax.scatter(x, y, z, c=inputs)
        plt.savefig('./imgs/{}/{}/{}-grid.png'.format(folder, subfolder, label))
        plt.close()

        
def data_mapping(inputs, base_radius=1):
    """
    Change the data cloud locations (scaling) to make all data points fall inside the ourier sphere (shell)
    inputs : [B, N, 3]
    return : [B, N, 3]
    """
    B, N, D = inputs.size()

    # Radiactively Mapping -> let k = sqrt(x^2 + y^2 + z^2); ratio = radius / k; update x,y,z = (x,y,z) * ratio
    k, _ = torch.max(torch.sqrt(torch.sum(torch.pow(inputs, 2), dim=2, keepdim=True)), dim=1,
                     keepdim=True)  # [B, 1, 1])
    k = k.float()
    ratio = base_radius / k
    inputs = torch.mul(inputs, ratio)
    return inputs


def density_mapping(b, inputs, data_index, density_radius, sphere_radius, s2_grid, sigma_diag, sigma_layer_diff=False, static_sigma=0.05, use_static_sigma=True, use_weights=False):
    """
    inputs : [B, N, 3]
    index : index of valid corresponding inputs
    radius : radius to count neighbor for weights
    s2_grid : [2b, 2b, 3]
    """

    B, N, D = inputs.size()

    # Get Weights
    if use_weights:
        dists = pairwise_distance(inputs)  # -> [B, N, N]
        weights = (dists <= density_radius).sum(dim=1).float()  # -> [B, N]
        weights = weights / weights.mean(dim=1, keepdim=True)

    # Resize inputs and grid
    s2_grid = s2_grid.view(-1, D)  # -> [4b^2, 3]
    inputs = inputs.unsqueeze(2).repeat(1, 1, s2_grid.size()[0], 1)  # -> [B, N, 4b^2, 3]

    # Calculate Density & Apply Cropping
    numerator = inputs - s2_grid  # -> [B, N, 4b^2, 3]
        
    sigma_inverse = 1 / sigma_diag
    
    middle = torch.mul(numerator, sigma_inverse)
    numerator = middle * numerator

    numerator = torch.sum(numerator, dim=-1)  # -> [B, N, 4b^2]
    numerator = -0.5 * numerator
    numerator = torch.exp(numerator)  # -> [B, N, 4b^2]

    sigma_det = torch.prod(sigma_diag, dim=3)  # -> [B, N, 1]

    denominator = torch.sqrt(torch.pow(sigma_det, D))  # -> [B, N, 1]
    denominator = denominator.cuda()

    density = numerator / denominator  # -> [B, N, 4b^2]
    
    # Filter out only valid data points
    density = torch.mul(density, data_index)

    # If use weightes
    if use_weights:
        # [Use Weights] Multiply Weights
        weights = weights.unsqueeze(-1) # -> [B, N, 1]
        density = density * weights # -> [B, N, 4b^2]

    # Sum Over Number of Points
    density = density.sum(dim=1)  # -> [B, 4b^2]

    # Adjust Dimension
    density = density.view(B, 2 * b, 2 * b)  # -> [B, 2b, 2b]

    return density


def data_cropping(data, inner_radius, radius):
    """
    Crop the valid data points needed for the given radius
    :param data: [B, N, 3]
    :param inner_radius: (float) bottom line for the croption
    :param radius: upper line for the croption
    :return: [B, N, 3] where only valid points have value and others all zeros
    """
    distances = torch.sqrt(torch.sum(torch.pow(data, 2), dim=2, keepdim=True)) # [B, N, 1]

    index_lower = distances >= inner_radius
    index_upper = distances <= radius
    index = index_lower * index_upper
    index = index.float() # [B, N, 1]

    return index


def data_translation(inputs, s2_grids, params, sigma_diag):
    """
    :param inputs: [B, N, 3]
    :param s2_grids: []
    :param params: parameters
    :return: list( Tensor([B, 2b, 2b]) * num_grids )
    """
    B, N, D = inputs.size()
    inputs = inputs.cuda()

    mappings = list()
    inner_radius = 0.0

    for radius, s2_grid in s2_grids:
        index = data_cropping(inputs, inner_radius, radius) # [B, N, 1] with invalid points left zeros
        mapping = density_mapping(
            b=params['bandwidth_0'],
            inputs=inputs,
            data_index=index,
            density_radius=params['density_radius'],
            sphere_radius=radius,
            s2_grid=s2_grid,
            sigma_diag=sigma_diag,
            sigma_layer_diff=params['sigma_layer_diff'],
            static_sigma=params['static_sigma'],
            use_static_sigma=params['use_static_sigma'],
            use_weights=params['use_weights']
        ).float()  # -> (B, 2b, 2b)
        mapping = mapping.view(B, 1, 2 * params['bandwidth_0'],
                                 2 * params['bandwidth_0'])  # [B, 2b0, 2b0] -> [B, 1, 2b0, 2b0]
        mappings.append(mapping)
        inner_radius = radius

    return mappings

# END
