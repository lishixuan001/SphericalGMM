import numpy as np
import torch
import h5py
from pdb import set_trace as st
import torch.nn as nn
import torch.nn.functional as func
import lie_learn.spaces.S2 as S2
import argparse
import math
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipyvolume as ipv


def load_args():
    parser = argparse.ArgumentParser(description='Spherical GMM')

    # Model
    parser.add_argument('--data_path', default='/home/lishixuan001/ICSI/datasets/mnist', type=str, metavar='XXX', help='Path to the model')
    parser.add_argument('--batch_size', default=40, type=int, metavar='N', help='Batch size of test set')
    parser.add_argument('--num_epochs', default=300, type=int, metavar='N', help='Epoch to run')
    parser.add_argument('--num_points', default=512, type=int, metavar='N', help='Number of points in a image')
    parser.add_argument('--log_interval', default=1000, type=int, metavar='N', help='log_interval')
    parser.add_argument('--baselr', default=5e-5, type=float, metavar='N', help='learning rate')
    parser.add_argument('--gpu', default='4', type=str, metavar='XXX', help='GPU number')
    parser.add_argument('--visualize', default=0, type=int, metavar='XXX', help='if do visualization')
    parser.add_argument('--save_model', default=1, type=int, metavar='XXX', help='if save model checkpoint')
    
    # Modal Structure
    parser.add_argument('--num_classes', default='10', type=int, metavar='XXX', help='number of classes for classification') 
    parser.add_argument('--num_so3_layers', default='3', type=int, metavar='XXX', help='number of SO3 layers')

    # Save Model
    parser.add_argument('--save_interval', default=20, type=int, metavar='N', help='save_interval')
    parser.add_argument('--resume_training', default=0, type=int, metavar='N', help='load used model at iteration')
    parser.add_argument('--resume_testing', default=None, type=str, metavar='N', help='load used model at iteration')

    # Multi-Grid
    parser.add_argument('--rotate_deflection', default=0.1, type=float, metavar='N', help='rotation deflection for testing')
    parser.add_argument('--num_grids', default=3, type=int, metavar='N', help='number of shells')
    parser.add_argument('--base_radius', default=1, type=int, metavar='N', help='radius of the out-est shell')

    # GMM
    parser.add_argument('--density_radius', default=0.2, type=float, metavar='XXX', help='Radius for density')
    parser.add_argument('--static_sigma', default=0.05, type=float, metavar='N', help='static sigma to use')
    parser.add_argument('--use_static_sigma', default=0, type=int, metavar='N', help='if use static sigma')
    parser.add_argument('--use_weights', default=0, type=int, metavar='N', help='if use weights for each point for GMM')
    parser.add_argument('--sigma_layer_diff', default=1, type=int, metavar='N', help='if expected sigma different over shells')

    # Parallel Computing
    parser.add_argument("--local_rank", default=0, type=int)
    
    args = parser.parse_args()
    return args


def rotate_random(deflection, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

def load_data_h5(params, data_type, shuffle=True, num_workers=4, rotate=False, batch=False):
    data_dir = params['{}_dir'.format(data_type)]
    batch_size = params['batch_size']
    
    train_data = h5py.File(data_dir + "_data.h5", 'r')
    train_labels = h5py.File(data_dir + "_label.h5", 'r')
    xs = np.array(train_data['data'])
    xs = np.delete(xs, 1, 2)
    if rotate:
        if batch:
            b = xs.shape[0]
            xs = np.array([np.dot(xs[i], rotate_random(params['rotate_deflection'])) for i in range(b)])
        else:
            rotation_matrix = rotate_random(params['rotate_deflection'])
            xs = np.dot(xs, rotation_matrix)

    ys = np.array(train_labels['label'])
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(xs).float(), torch.from_numpy(ys).long())
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=shuffle,
                                                       num_workers=num_workers)
    train_data.close()
    return train_loader_dataset


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


def get_spheres(scale=1, subdiv=1, radius=0.675):
    middle_point_cache = {}
    
    def vertex(x, y, z):
        """ Return vertex coordinates fixed to the unit sphere """
        length = np.sqrt(x**2 + y**2 + z**2)
        return [(i * scale) / length for i in (x,y,z)]


    def middle_point(point_1, point_2):
        """ Find a middle point and project to the unit sphere """
        smaller_index = min(point_1, point_2)
        greater_index = max(point_1, point_2)

        key = "{0}-{1}".format(smaller_index, greater_index)

        if key in middle_point_cache:
            return middle_point_cache[key]

        # If it's not in cache, then we can cut it
        vert_1 = verts[point_1]
        vert_2 = verts[point_2]
        middle = [sum(i)/2 for i in zip(vert_1, vert_2)]

        verts.append(vertex(*middle))

        index = len(verts) - 1
        middle_point_cache[key] = index

        return index

    PHI = (1 + np.sqrt(5)) / 2

    verts = [ vertex(-1, PHI, 0), 
         vertex( 1, PHI, 0), 
         vertex(-1, -PHI, 0), 
         vertex( 1, -PHI, 0), 
         vertex(0, -1, PHI), 
         vertex(0, 1, PHI), 
         vertex(0, -1, -PHI), 
         vertex(0, 1, -PHI), 
         vertex( PHI, 0, -1), 
         vertex( PHI, 0, 1), 
         vertex(-PHI, 0, -1), 
         vertex(-PHI, 0, 1), ]

    faces = [ 
        # 5 faces around point 0 
        [0, 11, 5], 
        [0, 5, 1], 
        [0, 1, 7], 
        [0, 7, 10], 
        [0, 10, 11], 
        # Adjacent faces 
        [1, 5, 9], 
        [5, 11, 4], 
        [11, 10, 2], 
        [10, 7, 6], 
        [7, 1, 8], 
        # 5 faces around 3 
        [3, 9, 4], 
        [3, 4, 2], 
        [3, 2, 6], 
        [3, 6, 8], 
        [3, 8, 9], 
        # Adjacent faces 
        [4, 9, 5], 
        [2, 4, 11], 
        [6, 2, 10], 
        [8, 6, 7], 
        [9, 8, 1], 
    ]
    
#     # Generate based dots on Sphere surface [12 spheres -> 42 spheres]
#     for i in range(subdiv): 
#         faces_subdiv = [] 
#         for tri in faces: 
#             v1 = middle_point(tri[0], tri[1]) 
#             v2 = middle_point(tri[1], tri[2]) 
#             v3 = middle_point(tri[2], tri[0]) 
#             faces_subdiv.append([tri[0], v1, v3]) 
#             faces_subdiv.append([tri[1], v2, v1]) 
#             faces_subdiv.append([tri[2], v3, v2]) 
#             faces_subdiv.append([v1, v2, v3]) 
#         faces = faces_subdiv

    def get_radius(scale, verts):
        min_dist = find_min_dist(verts)
        d = 0.5 * min_dist / np.cos(np.pi / 6) 

        x = verts[5]

        v = np.random.rand(3)
        v = v - np.matmul(v, x) * x
        v = v / norm(v) * d

        p = np.cos(norm(v)) * x + np.sin(norm(v)) * v / norm(v)

        origin = np.array([0, 0, 0])
        center = 0.5 * (origin + x)

        radius = euclidean_distance(p, center)

        return radius
    
    # Get embedding sphere centers
    centers = []
    origin = np.array([0, 0, 0])
    for vert in verts:
        center = 0.5 * (origin + vert)
        centers.append(center)
    
    return centers, radius

def get_grids(b, num_grids, base_radius=1, center=[0, 0, 0], grid_type="Driscoll-Healy"):
    """
    :param b: the number of grids on the sphere
    :param base_radius: the radius of each sphere
    :param grid_type: "Driscoll-Healy"
    :param num_grids: number of grids
    :return: [(radius, tensor([2b, 2b, 3])) * num_grids]
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
        x = x + center[0]

        y_ = radius * torch.sin(theta) * torch.sin(phi)
        y = y_.reshape((1, 4 * b * b))
        y = y + center[1]

        z_ = radius * torch.cos(theta)
        z = z_.reshape((1, 4 * b * b))
        z = z + center[2]

        grid = torch.cat((x, y, z), dim=0)  # -> [3, 4b^2]
        grid = grid.transpose(0, 1)  # -> [4b^2, 3]

        grid = grid.view(2 * b, 2 * b, 3)  # -> [2b, 2b, 3]
        grid = grid.float().cuda()

        grids.append( (radius, grid) )

    assert len(grids) == num_grids
    return grids


def get_sphere_grids(b, num_grids, base_radius):
    """
    return: [center, [(radius, tensor([2b, 2b, 3])) * num_grids]] * num_centers
    """
    
    centers, R = get_spheres()
    
    # Create grids based on each center
    all_grids = []
    for center in centers:
        grids = get_grids(b, num_grids, base_radius=R, center=center)
        all_grids.append([center, grids]) # tensor -> [center, [(radius, tensor([2b, 2b, 3])) * num_grids]]
    return all_grids

def visualize_sphere_grids(s2_grids, params, folder='grid', colors=["#7b0001", "#ff0001", "#ff8db4"]):
    fig = ipv.figure()
    for _, layer_grids in s2_grids:
        for i, (_, shell) in enumerate(layer_grids):
            shell = shell.reshape(-1, 3).transpose(0, 1) # tensor([3, 4b^2]
            x_axis = shell[0, :].cpu().numpy()
            y_axis = shell[1, :].cpu().numpy()
            z_axis = shell[2, :].cpu().numpy()
            ipv.scatter(x_axis, y_axis, z_axis, marker="sphere", color=colors[i])
    ipv.save("./imgs/{}/s2_sphere_sphere.html".format(folder))
    
        
def visualize_sphere_sphere(origins, data, labels, s2_grids, params, folder='sphere', num_selections=5):
    """
    data :  list( list( Tensor([B, 1, 2b0, 2b0]) * num_grids ) * num_centers)
    s2_grids: [center, [(radius, tensor([2b, 2b, 3])) * num_grids]]
    """
    if params['use_static_sigma']:
        subfolder = "static_sigma_{}".format(params['static_sigma'])
    else:
        if params['sigma_layer_diff']:
            subfolder = "conv_sigma_diff_over_layer"
        else:
            subfolder = "conv_sigma_share_over_layer"
    
    for i in range(num_selections):
        label = str(labels[i].item())
        for j, grids in enumerate(data):
            fig, axs = plt.subplots(1, 3)
            for k, grid in enumerate(grids):
                grid = grid[i][0] 
                grid = grid.detach().cpu().numpy() # [2b, 2b]
                axs[k].set_title('Label {}, Center {}, Layer {}'.format(label, j, k))
                axs[k].imshow(grid)
            plt.savefig('./imgs/{}/{}/map/sphere-label[{}]-center[{}]-map.png'.format(folder, subfolder, label, j))
            plt.close()
            
    for i in range(num_selections):
        label = str(labels[i].item())
        fig = ipv.figure()
        image = origins[i].detach().cpu().numpy()
        ipv.scatter(image[:, 0], image[:, 1], image[:, 2], marker='diamond', color='red')
        
        for j, grids in enumerate(data):
            _, center_grid = s2_grids[j]
            center_data = data[j]
            for k, shell in enumerate(grids):
                _, shell_grid = center_grid[k] # Tensor([2b, 2b, 3])
                shell_data = center_data[k][i] # Tensor([1, 2b0, 2b0])
                shell_grid = shell_grid.view(-1, 3)
                x = shell_grid[:, 0].squeeze().cpu().numpy()
                y = shell_grid[:, 1].squeeze().cpu().numpy()
                z = shell_grid[:, 2].squeeze().cpu().numpy()
                
                # For Color Showing 
                shell_data = shell_data.permute(1, 2, 0) # Tensor([2b0, 2b0, 1])
                shell_data = shell_data.view(-1, 1)[:, 0].detach().cpu().numpy() # Tensor([4b0^2])
                color_data = (shell_data - np.min(shell_data)) / (np.max(shell_data) - np.min(shell_data))
                color_data = np.expand_dims(color_data, axis=-1).repeat(2, axis=-1)
                color = np.zeros((color_data.shape[0], 4))
                color[:, 0:2] = color_data
                color[:, 2] = 0.4 # Adjust Color [NEED NOT CHANGE]
                color[:, 3] = 1 # Transparancy [DO NOT CHANGE]
                
                ipv.scatter(x, y, z, marker='sphere', color=color)
                
        ipv.save('./imgs/{}/{}/grid/sphere-label[{}]-grid.html'.format(folder, subfolder, label))

        
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


def density_mapping(b, inputs, data_index, density_radius, sphere_radius, s2_grid, sigma_layer_diff=False, static_sigma=0.05, use_static_sigma=True, use_weights=False):
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

    # If Use Static Sigma
    if use_static_sigma:
        # [Use Static Sigma] For Testing With Static Sigma
        sigma_diag = torch.tensor([static_sigma, static_sigma, static_sigma]).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1).cuda()
    else:
        # Calculate Sigma [Covariance]
        sigma = torch.matmul(numerator.transpose(2, 3), numerator)  # -> [B, N, 3, 3]
        sigma = sigma / (4 * (b ** 2))

        index = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]]).cuda()
        index = index.unsqueeze(0).unsqueeze(0)
        index = index.repeat(B, N, 1, 1)
        sigma_diag = torch.gather(sigma, 2, index)  # -> [B, N, 3, 3] -> [[diag1, diag2, diag3] * 3]
        sigma_diag = sigma_diag[:, :, 0, :]  # -> [B, N, 3]
        sigma_diag = sigma_diag.unsqueeze(2)  # -> [B, N, 1, 3]

        # Adjust Sigma Values [0.2~0.7] -> [0.02~0.07]
        # Mean Sigma for each point
        
        sigma_diag = sigma_diag / 10  
        
        # If request E[sigma] vary over diff layers of shell
        if sigma_layer_diff:
            sigma_diag = sigma_diag * sphere_radius
            
        sigma_diag = torch.mean(sigma_diag, dim=3, keepdim=True).repeat(1, 1, 1, 3)  # -> [B, N, 1, 3]
    
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


def data_sphere_cropping(data, center, inner_radius, radius):
    """
    Crop the valid data points needed for the given radius
    :param data: [B, N, 3]
    :param inner_radius: (float) bottom line for the croption
    :param radius: upper line for the croption
    :return: [B, N, 3] where only valid points have value and others all zeros
    """
    center = torch.Tensor(center).view(1, 1, 3).cuda()
    distances = torch.sqrt(torch.sum(torch.pow(data - center, 2), dim=2, keepdim=True)) # [B, N, 1]

    index_lower = distances >= inner_radius
    index_upper = distances <= radius
    index = index_lower * index_upper
    index = index.float() # [B, N, 1]

    return index


def data_sphere_translation(inputs, s2_grids, params):
    """
    s2_grids: [[center, [(radius, tensor([2b, 2b, 3])) * num_grids]] * num_centers]
    :return: list( list( Tensor([B, 2b, 2b]) * num_grids ) * num_centers)
    """
    B, N, D = inputs.size()
    inputs = inputs.cuda()
    
    all_mappings = []
    for center, grids in s2_grids:
        inner_radius = 0.0
        mappings = list()
        for i, (radius, shell) in enumerate(grids):
            index = data_sphere_cropping(inputs, center, inner_radius, radius) # [B, N, 1] with invalid points left zeros
            mapping = density_mapping(
                b=params['bandwidth_0'],
                inputs=inputs,
                data_index=index,
                density_radius=params['density_radius'],
                sphere_radius=radius,
                s2_grid=shell,
                sigma_layer_diff=params['sigma_layer_diff'],
                static_sigma=params['static_sigma'],
                use_static_sigma=params['use_static_sigma'],
                use_weights=params['use_weights']
            ).float()  # -> (B, 2b, 2b)
            mapping = mapping.view(B, 1, 2 * params['bandwidth_0'],
                                     2 * params['bandwidth_0'])  # [B, 2b0, 2b0] -> [B, 1, 2b0, 2b0]
            mappings.append(mapping)
            inner_radius = radius
        all_mappings.append(mappings)

    return all_mappings


def euclidean_distance(p1, p2):
    return np.sqrt(sum(np.square(np.array(list(map(lambda x: x[0] - x[1], zip(p1, p2)))))))






















# END