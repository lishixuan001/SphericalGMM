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
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def eval(test_iterator, model, params, logger, num_epochs=10):
    
    logger.info("================================ Eval ================================\n")
    
    s2_grids = utils.get_grids(b=params['bandwidth_0'], num_grids=params['num_grids'], base_radius=params['base_radius'])

    acc_overall = list()
    test_iterator = utils.load_data_h5(params['test_dir'], batch_size=params['batch_size'], rotate=True, batch=False)
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
                inputs = utils.data_translation(inputs, s2_grids,
                                                params)  # [B, N, 3] -> list( Tensor([B, 2b, 2b]) * num_grids )

                outputs = model(inputs)
                outputs = torch.argmax(outputs, dim=-1)
                acc_all.append(np.mean(outputs.detach().cpu().numpy() == labels.numpy()))
            acc_overall.append(np.mean(np.array(acc_all)))
            logger.info('[epoch {}] Accuracy: [{}]'.format(epoch, str(np.mean(np.array(acc_all)))))
            
    logger.info("======================================================================\n")
    return np.max(acc_overall)


def test(params, date_time, num_epochs=1000):
    logger = setup_logger("SphericalGMMNet")
    logger.info("Loading Data")

    # Load Data
    logger.info("Model Setting Up")

    # Model Configuration Setup
    model = SphericalGMMNet(params).cuda()
    model = model.cuda()

    logger.info('Loading the trained models from {date_time} ...'.format(date_time=date_time))
    model_path = os.path.join(params['save_dir'], '{date_time}-model.ckpt'.format(date_time=date_time))
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    # Generate the grids
    # [(radius, tensor([2b, 2b, 3])) * 3]
    s2_grids = utils.get_grids(b=params['bandwidth_0'], num_grids=params['num_grids'], base_radius=params['base_radius'])

    test_iterator = utils.load_data_h5(params['test_dir'], batch_size=params['batch_size'], rotate=True, batch=False)
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
                inputs = utils.data_translation(inputs, s2_grids,
                                                params)  # [B, N, 3] -> list( Tensor([B, 2b, 2b]) * num_grids )

                outputs = model(inputs)
                outputs = torch.argmax(outputs, dim=-1)
                acc_all.append(np.mean(outputs.detach().cpu().numpy() == labels.numpy()))

            logger.info('[epoch {}] Accuracy: [{}]'.format(epoch, str(np.mean(np.array(acc_all)))))


def train(params):
    # Logger Setup and OS Configuration
    logger = setup_logger("SphericalGMMNet")
    logger.info("Loading Data")

    # Load Data
    train_iterator = utils.load_data_h5(params['train_dir'], batch_size=params['batch_size'])
    test_iterator = utils.load_data_h5(params['test_dir'], batch_size=params['batch_size'], rotate=True, batch=False)

    # Model Setup
    logger.info("Model Setting Up")
    model = SphericalGMMNet(params).cuda()
    model = model.cuda()

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
    # [(radius, tensor([2b, 2b, 3])) * 3]
    s2_grids = utils.get_grids(b=params['bandwidth_0'], num_grids=params['num_grids'], base_radius=params['base_radius'])

    # TODO [Visualize Grids]
    if params['visualize']:
        utils.visualize_grids(s2_grids)
    
    # Keep track of max Accuracy during training
    acc, max_acc = 0, 0
    
    # Iterate by Epoch
    logger.info("Start Training")
    for epoch in range(params['num_epochs']):

        # Save the model for each step
        if acc > max_acc:
            max_acc = acc
            save_path = os.path.join(params['save_dir'], '{date_time}-[{acc}]-model.ckpt'.format(date_time=date_time, acc=acc))
            torch.save(model.state_dict(), save_path)
            logger.info('Saved model checkpoints into {}...'.format(save_path))

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

        acc = eval(test_iterator, model, params, logger)
        logger.info(
            "**************** Epoch: [{epoch}/{total_epoch}] Accuracy: [{acc}] ****************\n".format(epoch=epoch,
                                                                                                          total_epoch=
                                                                                                          params[
                                                                                                              'num_epochs'],
                                                                                                          loss=np.mean(
                                                                                                              running_loss),
                                                                                                          acc=acc))

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
