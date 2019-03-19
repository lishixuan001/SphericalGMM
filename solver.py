from model import SphericalGMMNet
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import utils
import h5py
from pdb import set_trace as st
import argparse
from logger import setup_logger
import torch.utils.data
import matplotlib.pyplot as plt
import lie_learn.spaces.S2 as S2

        
def eval(test_iterator, model, sigma, grid):
    acc_all = []
    for i, (inputs, labels) in enumerate(test_iterator):
        if i <=10:
            inputs = Variable(inputs).cuda()
            inputs = utils.data_generation(inputs, grid, sigma)
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
    test_iterator = utils.load_data(params['test_dir'], batch_size=params['batch_size'])
    train_iterator = utils.load_data(params['train_dir'], batch_size=params['batch_size'])

    logger.info("Model Setting Up")
    # Model Setup
    # model = SphericalGMMNet(10, params['num_neighbors'], params['num_points'], params['grid']).cuda()
    # model = model.cuda()

    # Model Configuration Setup
    # optim = torch.optim.Adam(model.parameters(), lr=params['baselr'])
    # cls_criterion = torch.nn.CrossEntropyLoss().cuda()
    
    # Construct Grid
    #     theta : shape (2b, 2b), range [0, pi]; 
    #     phi   : range [0, 2 * pi]
    theta, phi = S2.meshgrid(b=b, grid_type="Driscoll-Healy")
    
    logger.info("Start Training")

    # Iterate by Epoch
    for epoch in range(params['num_epochs']):  # loop over the dataset multiple times
        running_loss = []
        for batch_idx, (inputs, labels) in enumerate(train_iterator):

            """ Variable Setup """
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            B, N, D = inputs.size()
            
            if inputs.shape[-1] == 2:
                logger.info("--> Data Dimension Adjustment Operated")
                zero_padding = torch.zeros((B, N, 1), dtype=inputs.dtype)
                inputs = torch.cat((inputs, zero_padding), -1) # [B, N, 3]
            
            dists = utils.pairwise_distance(inputs)
            weights = (dists <= params['radius']).sum(dim=1).float()
            weights = weights / weights.sum(dim=1, keepdim=True) # [B, D]
            
            s2_grid = None # TODO
         
            
            density = utils.density_mapping(inputs, weights, s2_grid)
            
            
            
            return
            
            
            
            
            
            
            
            
            # optim.zero_grad()

            """ Model Input/Output """
            # inputs = utils.data_generation(inputs, params['grid'], params['sigma'])        
            # outputs = model(inputs)

            """ Update Loss and Do Backprop """ 
            # loss = cls_criterion(outputs, labels.squeeze())
            # loss.backward(retain_graph=True)
            # optim.step()
            # running_loss.append(loss.item())

            # Update Loss Per Batch
#             logger.info("Batch: [{batch}/{total_batch}] Epoch: [{epoch}] Loss: [{loss}]".format(batch=batch_idx,
#                                                                                          total_batch=len(train_iterator),
#                                                                                          epoch=epoch,
#                                                                                          loss=np.mean(running_loss)))

            # Periodically Show Accuracy
#             if batch_idx % params['log_interval'] == 0:
#                 acc = eval(test_iterator, model, params['sigma'], params['grid'])
#                 logger.info("Accuracy: [{}]\n".format(acc))

#         acc = eval(test_iterator, model, params['sigma'], params['grid'])
#         logger.info("Epoch: [{epoch}/{total_epoch}] Loss: [{loss}] Accuracy: [{acc}]".format(epoch=epoch,
#                                                                                       total_epoch=params['num_epochs'],
#                                                                                       loss=np.mean(running_loss),
#                                                                                       acc=acc))

        torch.save(model.state_dict(), os.path.join(params['log_dir'], '_'.join(["manifold", str(epoch + 1)])))

    logger.info('Finished Training')

if __name__ == '__main__':
    
    args = utils.load_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    params = dict(
        train_dir = os.path.join(args.data_path, "train"),
        test_dir  = os.path.join(args.data_path, "test"),
        num_points     = args.num_points,
        num_epochs     = args.num_epochs,
        log_interval   = args.log_interval,
        batch_size     = args.batch_size,
        baselr         = args.baselr,
        radius         = args.radius
    )
    

    train(params)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# END