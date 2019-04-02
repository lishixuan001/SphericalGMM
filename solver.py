import os
import time
import datetime
import h5py
import numpy as np

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import utils
from os.path import join
from logger import setup_logger
import lie_learn.spaces.S2 as S2
from model import SphericalGMMNet
from pdb import set_trace as st
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

def map_points_onto_sphere(inputs, radius = 1):
    #Assume input is B*N*2 ------> want result to be B*N*3 on sphere of radius R
    
    B, N, D = inputs.shape
    inputs = inputs.cpu()
    pyplot.scatter(inputs[0, :, 0], inputs[0, :, 1])
    pyplot.savefig('books_read3.png')
    
    # Given a "mapping sphere" of radius R,
    # the Mercator projection (x,y) of a given latitude and longitude is:
    #    x = R * longitude
    #    y = R * log( tan( (latitude + pi/2)/2 ) )

    # and the inverse mapping of a given map location (x,y) is:
    #   longitude = x / R
    #   latitude = 2 * atan(exp(y/R)) - pi/2
    # To get the 3D coordinates from the result of the inverse mapping:

    # Given longitude and latitude on a sphere of radius S,
    # the 3D coordinates P = (P.x, P.y, P.z) are:
    #   P.x = S * cos(latitude) * cos(longitude)
    #   P.y = S * cos(latitude) * sin(longitude)
    #   P.z = S * sin(latitude)
    
    ##First step: scale inputs to be in [-pi/2, pi/2]^2
    
    # Result := ((Input - InputLow) / (InputHigh - InputLow))
    #           * (OutputHigh - OutputLow) + OutputLow;
    
    max_in = torch.max(inputs, dim=1, keepdim=True)[0]     #B*2
    min_in = torch.min(inputs, dim=1, keepdim=True)[0]     #B*2
    inputs = ((inputs - min_in) / (max_in - min_in)) * (3.14) + (-1.57)
    longs = inputs[:, :, 0]
    lats = inputs[:, :, 1]
    new_inputs = torch.zeros((B, N, 3))
    new_inputs[:, :, 0] = radius * torch.cos(lats) * torch.cos(longs)
    new_inputs[:, :, 1] = radius * torch.cos(lats) * torch.sin(longs)
    new_inputs[:, :, 2] = radius * torch.sin(lats)
    
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.scatter(new_inputs[0, :, 0], new_inputs[0, :, 1], new_inputs[0, :, 2])
    pyplot.savefig('books_read2.png')
    return new_inputs


        
def eval(test_iterator, model):
    acc_all = []
    for i, (inputs, labels) in enumerate(test_iterator):
        if i <=10:
            inputs = Variable(inputs).cuda()
            B, N, D = inputs.size()
            
            if inputs.shape[-1] == 2:
                zero_padding = torch.zeros((B, N, 1), dtype=inputs.dtype).cuda()
                inputs = torch.cat((inputs, zero_padding), -1) # [B, N, 3]
            inputs = utils.data_translation(inputs, params['bandwidth_0'], params['density_radius'], params['sigma'])
            inputs = inputs.view(params['batch_size'], 1, 2 * params['bandwidth_0'], 2 * params['bandwidth_0'])  # -> [B, 1, 2b0, 2b0]
            outputs = model(inputs)
            outputs = torch.argmax(outputs, dim=-1)
            acc_all.append(np.mean(outputs.detach().cpu().numpy() == labels.numpy()))
        else:
            return np.mean(np.array(acc_all))


def train(params):
    
    # Logger Setup and OS Configuration
    logger = setup_logger("SphericalGMMNet")

    logger.info("Loading Data")

    # Load Data
    test_iterator = utils.load_data_h5(params['test_dir'], batch_size=params['batch_size'])
    train_iterator = utils.load_data_h5(params['train_dir'], batch_size=params['batch_size'])

    logger.info("Model Setting Up")
    
    # Model Setup
    model = SphericalGMMNet(params).cuda()
    model = model.cuda()

    # Model Configuration Setup
    optim = torch.optim.Adam(model.parameters(), lr=params['baselr'])
    cls_criterion = torch.nn.CrossEntropyLoss().cuda()
    
    logger.info("Start Training")
    
    # Iterate by Epoch
    for epoch in range(params['num_epochs']):  # loop over the dataset multiple times
        running_loss = []
        for batch_idx, (inputs, labels) in enumerate(train_iterator):

            """ Variable Setup """
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            #print(labels[0])
            B, N, D = inputs.size()
            if inputs.shape[-1] == 2:
                #inputs = map_points_onto_sphere(inputs, params['density_radius'])
                zero_padding = torch.zeros((B, N, 1), dtype=inputs.dtype).cuda()
                inputs = torch.cat((inputs, zero_padding), -1) # [B, N, 3]
                
            #Preprocessing
            inputs = utils.data_translation(inputs, params['bandwidth_0'], params['density_radius'], params['sigma'])  # [B, N, 3] -> [B, 2b0, 2b0]
            #Print images
#             image_root = './imgs'
#             for i in range(10):
#                 img = inputs[i].cpu()
#                 pyplot.imsave(join(image_root, str(labels[i])+'.png'), img)
                
                
    
            inputs = inputs.view(params['batch_size'], 1, 2 * params['bandwidth_0'], 2 * params['bandwidth_0'])  # -> [B, 1, 2b0, 2b0]
    
            """ Run Model """
            outputs = model(inputs)
            
            """ Back Propagation """
            loss = cls_criterion(outputs, labels.squeeze())
            loss.backward(retain_graph=True)
            optim.step()
            running_loss.append(loss.item())

            # Update Loss Per Batch
            logger.info("Batch: [{batch}/{total_batch}] Epoch: [{epoch}] Loss: [{loss}]".format(batch=batch_idx,
                                                                                         total_batch=len(train_iterator),
                                                                                         epoch=epoch,
                                                                                         loss=np.mean(running_loss)))
            # Periodically Show Accuracy
            if batch_idx % params['log_interval'] == 0:
                acc = eval(test_iterator, model)
                logger.info("Accuracy: [{}]\n".format(acc))

        acc = eval(test_iterator, model)
        logger.info("Epoch: [{epoch}/{total_epoch}] Loss: [{loss}] Accuracy: [{acc}]".format(epoch=epoch,
                                                                                      total_epoch=params['num_epochs'],
                                                                                      loss=np.mean(running_loss),
                                                                                      acc=acc))

    logger.info('Finished Training')

if __name__ == '__main__':
    
    args = utils.load_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    params = {
        'train_dir' : os.path.join(args.data_path, "train"),
        'test_dir'  : os.path.join(args.data_path, "test"),
        
        'num_epochs'    : args.num_epochs,
        'batch_size'    : args.batch_size,
        'num_points'    : args.num_points,
        
        'sigma'         : args.sigma,
        'log_interval'  : args.log_interval,
        'baselr'        : args.baselr,
        'density_radius'        : args.density_radius,
        
        'feature_out1': 8,
        'feature_out2': 16, 
        'feature_out3': 32,
        'feature_out4': 64,

        'num_classes': 10, 

        'bandwidth_0':    8,
        'bandwidth_out1': 8, 
        'bandwidth_out2': 6,  
        'bandwidth_out3': 4, 
        'bandwidth_out4': 2,
    }

    train(params)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# END