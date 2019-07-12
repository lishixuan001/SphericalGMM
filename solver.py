import os
import time
import datetime
import h5py
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import utils
from os.path import join
from logger import setup_logger
import lie_learn.spaces.S2 as S2
from model import SphericalGMMNet
from pdb import set_trace as st
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def eval(model, params, logger, num_epochs=3, rotate=True):
    
    logger.info("================================ Eval ================================\n")
    
    s2_grids = utils.get_sphere_grids(b=params['bandwidth_0'], num_grids=params['num_grids'], base_radius=params['base_radius'])

    acc_overall = list()
    test_iterator = utils.load_data_h5(params, data_type="test", rotate=rotate, batch=False)
    for epoch in range(num_epochs):
        acc_all = []
        with torch.no_grad():
            for _, (inputs, labels) in enumerate(test_iterator):
                
                inputs = Variable(inputs).cuda()
                B, N, D = inputs.size()

                if inputs.shape[-1] == 2:
                    zero_padding = torch.zeros((B, N, 1), dtype=inputs.dtype).cuda()
                    inputs = torch.cat((inputs, zero_padding), -1)  # [B, N, 3]

                # Data Mapping
                inputs = utils.data_mapping(inputs, base_radius=params['base_radius'])  # [B, N, 3]
                inputs = utils.data_sphere_translation(inputs, s2_grids, params)

                outputs = model(inputs)
                outputs = torch.argmax(outputs, dim=-1)
                acc_all.append(np.mean(outputs.detach().cpu().numpy() == labels.numpy()))
            acc_overall.append(np.mean(np.array(acc_all)))
            logger.info('[epoch {}] Accuracy: [{}]'.format(epoch, str(np.mean(np.array(acc_all)))))
            
    logger.info("======================================================================\n")
    return np.max(acc_overall)


def test(params, model_name, num_epochs=1000):
    logger = setup_logger("SphericalGMMNet")
    logger.info("Loading Data")

    # Load Data
    logger.info("Model Setting Up")

    # Model Configuration Setup
    model = SphericalGMMNet(params).cuda()
    model = model.cuda()
    if len(params['gpu'].split(",")) >= 2:
        model = nn.DataParallel(model)

    logger.info('Loading the trained models from {model_name} ...'.format(model_name=model_name))
    model_path = os.path.join(params['save_dir'], '{model_name}'.format(model_name=model_name))
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    # Generate the grids
    s2_grids = utils.get_sphere_grids(b=params['bandwidth_0'], num_grids=params['num_grids'], base_radius=params['base_radius'])

    test_iterator = utils.load_data_h5(params, data_type="test", rotate=True, batch=False)
    for epoch in range(num_epochs):
        acc_all = []
        with torch.no_grad():
            for _, (inputs, labels) in enumerate(test_iterator):
                inputs = Variable(inputs).cuda()
                B, N, D = inputs.size()

                if inputs.shape[-1] == 2:
                    zero_padding = torch.zeros((B, N, 1), dtype=inputs.dtype).cuda()
                    inputs = torch.cat((inputs, zero_padding), -1)  # [B, N, 3]

                # Data Mapping
                inputs = utils.data_mapping(inputs, base_radius=params['base_radius'])  # [B, N, 3]

                # Data Translation
                inputs = utils.data_sphere_translation(inputs, s2_grids, params)

                outputs = model(inputs)
                outputs = torch.argmax(outputs, dim=-1)
                acc_all.append(np.mean(outputs.detach().cpu().numpy() == labels.numpy()))

            logger.info('[epoch {}] Accuracy: [{}]'.format(epoch, str(np.mean(np.array(acc_all)))))


def train(params):
    # Logger Setup and OS Configuration
    logger = setup_logger("SphericalGMMNet")
    logger.info("Loading Data")

    # Load Data
    train_iterator = utils.load_data_h5(params, data_type="train")

    # Model Setup
    logger.info("Model Setting Up")


    model = SphericalGMMNet(params).cuda()
    model = model.cuda()
    if len(params['gpu'].split(",")) >= 2:
        model = nn.DataParallel(model)

    # Model Configuration Setup
    optim = torch.optim.Adam(model.parameters(), lr=params['baselr'])
    cls_criterion = torch.nn.CrossEntropyLoss().cuda()

    # Resume If Asked
    date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if params['resume_training']:
        date_time = params['resume_training']
        model_path = os.path.join(params['save_dir'], '{date_time}-model.ckpt'.format(date_time=date_time))
        model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    # Display Parameters
    for name, value in params.items():
        logger.info("{name} : [{value}]".format(name=name, value=value))

    # Generate the grids
    s2_grids = utils.get_sphere_grids(b=params['bandwidth_0'], num_grids=params['num_grids'], base_radius=params['base_radius'])

    # TODO [Visualize Grids]
    if params['visualize']:
        utils.visualize_sphere_grids(s2_grids, params)
    
    # Keep track of max Accuracy during training
    non_rotate_acc, rotate_acc = 0, 0
    max_non_rotate_acc, max_rotate_acc = 0, 0
    
    # Iterate by Epoch
    logger.info("Start Training")
    for epoch in range(params['num_epochs']):

        # Save the model for each step
        if non_rotate_acc > max_non_rotate_acc:
            max_non_rotate_acc = non_rotate_acc
            save_path = os.path.join(params['save_dir'], '{date_time}-NR-[{acc}]-model.ckpt'.format(date_time=date_time, acc=non_rotate_acc))
            torch.save(model.state_dict(), save_path)
            logger.info('Saved model checkpoints into {}...'.format(save_path))
        if rotate_acc > max_rotate_acc:
            max_rotate_acc = rotate_acc
            save_path = os.path.join(params['save_dir'], '{date_time}-R-[{acc}]-model.ckpt'.format(date_time=date_time, acc=rotate_acc))
            torch.save(model.state_dict(), save_path)
            logger.info('Saved model checkpoints into {}...'.format(save_path))

        # Running Model
        running_loss = []
        for batch_idx, (inputs, labels) in enumerate(train_iterator):

            """ Variable Setup """
            inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
            B, N, D = inputs.size()

            if inputs.shape[-1] == 2:
                zero_padding = torch.zeros((B, N, 1), dtype=inputs.dtype).cuda()
                inputs = torch.cat((inputs, zero_padding), -1)  # [B, N, 3]

            # Data Mapping
            inputs = utils.data_mapping(inputs, base_radius=params['base_radius'])  # [B, N, 3]
            
            if params['visualize']:
                
                # TODO [Visualization [Raw]]
                origins = inputs.clone()
                # utils.visualize_raw(inputs, labels)
                
                # TODO [Visualization [Sphere]]
                print("---------- Static ------------")
                params['use_static_sigma'] = True
                inputs1 = utils.data_sphere_translation(inputs, s2_grids, params)  
                utils.visualize_sphere_sphere(origins, inputs1, labels, s2_grids, params, folder='sphere')
                
                print("\n---------- Covariance ------------")
                params['use_static_sigma'] = False
                params['sigma_layer_diff'] = False
                inputs2 = utils.data_sphere_translation(inputs, s2_grids, params)  
                utils.visualize_sphere_sphere(origins, inputs2, labels, s2_grids, params, folder='sphere')
                
                print("\n---------- Layer Diff ------------")
                params['use_static_sigma'] = False
                params['sigma_layer_diff'] = True
                inputs3 = utils.data_sphere_translation(inputs, s2_grids, params)  
                utils.visualize_sphere_sphere(origins, inputs3, labels, s2_grids, params, folder='sphere')
                return
            else:
                # Data Translation
                inputs = utils.data_sphere_translation(inputs, s2_grids, params) # list( list( Tensor([B, 2b, 2b]) * num_grids ) * num_centers)
            
            """ Run Model """
            outputs = model(inputs)

            """ Back Propagation """
            loss = cls_criterion(outputs, labels.squeeze())
            loss.backward(retain_graph=True)
            optim.step()
            running_loss.append(loss.item())

            # Update Loss Per Batch
            logger.info("Batch: [{batch}/{total_batch}] Epoch: [{epoch}] Loss: [{loss}]".format(batch=batch_idx,
                                                                                                total_batch=len(
                                                                                                    train_iterator),
                                                                                                epoch=epoch,
                                                                                                loss=np.mean(
                                                                                                    running_loss)))

        non_rotate_acc = eval(model, params, logger, rotate=False)
        logger.info(
            "**************** Epoch: [{epoch}/{total_epoch}] Accuracy: [{acc}] ****************\n".format(epoch=epoch,
                                                                                                          total_epoch=
                                                                                                          params[
                                                                                                              'num_epochs'],
                                                                                                          loss=np.mean(
                                                                                                              running_loss),
                                                                                                          acc=non_rotate_acc))
        
        rotate_acc = eval(model, params, logger, rotate=True)
        logger.info(
            "**************** Epoch: [{epoch}/{total_epoch}] Accuracy: [{acc}] ****************\n".format(epoch=epoch,
                                                                                                          total_epoch=
                                                                                                          params[
                                                                                                              'num_epochs'],
                                                                                                          loss=np.mean(
                                                                                                              running_loss),
                                                                                                          acc=rotate_acc))        

    logger.info('Finished Training')


if __name__ == '__main__':

    args = utils.load_args()

    params = {
        'train_dir': os.path.join(args.data_path, "train"),
        'test_dir' : os.path.join(args.data_path, "test"),
        'save_dir' : os.path.join('./', "save"),
        
        'gpu'       : args.gpu,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'num_points': args.num_points,
        'visualize' : bool(args.visualize),

        'log_interval': args.log_interval,
        'save_interval': args.save_interval,
        'baselr': args.baselr,
        'density_radius': args.density_radius,
 
        'rotate_deflection':  args.rotate_deflection,
        'num_grids':          args.num_grids,
        'base_radius':        args.base_radius,
        'static_sigma':       args.static_sigma,
        'use_static_sigma':   bool(args.use_static_sigma),
        'use_weights':        bool(args.use_weights),
        'sigma_layer_diff':   bool(args.sigma_layer_diff),

        'feature_out1': 8,
        'feature_out2': 16,
        'feature_out3': 32,
        'feature_out4': 64,
        'feature_out5': 128,

        'num_classes': args.num_classes,
        'num_so3_layers': args.num_so3_layers,

        'bandwidth_0': 10,
        'bandwidth_out1': 10,
        'bandwidth_out2': 8,
        'bandwidth_out3': 6,
        'bandwidth_out4': 4,
        'bandwidth_out5': 2,

        'resume_training': args.resume_training,
    }

    os.environ['CUDA_VISIBLE_DEVICES'] = params['gpu']
    if args.resume_testing:
        test(params, args.resume_testing)
    else:
        train(params)

# END
