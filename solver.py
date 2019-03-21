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
from logger import setup_logger
import lie_learn.spaces.S2 as S2
from model import SphericalGMMNet
from pdb import set_trace as st


        
def eval(test_iterator, model):
    acc_all = []
    for i, (inputs, labels) in enumerate(test_iterator):
        if i <=10:
            inputs = Variable(inputs).cuda()
            B, N, D = inputs.size()
            
            if inputs.shape[-1] == 2:
                zero_padding = torch.zeros((B, N, 1), dtype=inputs.dtype).cuda()
                inputs = torch.cat((inputs, zero_padding), -1) # [B, N, 3]
                
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
            B, N, D = inputs.size()
            
            if inputs.shape[-1] == 2:
                zero_padding = torch.zeros((B, N, 1), dtype=inputs.dtype).cuda()
                inputs = torch.cat((inputs, zero_padding), -1) # [B, N, 3]
            
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
        
        'log_interval'  : args.log_interval,
        'baselr'        : args.baselr,
        'density_radius'        : args.density_radius,
        
        'feature_out1': 8,
        'feature_out2': 16, 
        'feature_out3': 32,
        'feature_out4': 64,

        'num_classes': 10, 

        'bandwidth_0': 10,
        'bandwidth_out1': 10, 
        'bandwidth_out2': 8,  
        'bandwidth_out3': 6, 
        'bandwidth_out4': 4,
    }

    train(params)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# END