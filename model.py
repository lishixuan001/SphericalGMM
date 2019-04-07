from __future__ import division
import torch
import time
import utils
import torch.nn as nn
import torch.nn.functional as F
from s2cnn import S2Convolution
from s2cnn import SO3Convolution
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid
from s2cnn import so3_integrate
from pdb import set_trace as st


class SphericalGMMNet(nn.Module):

    def __init__(self, params):
        super(SphericalGMMNet, self).__init__()

        self.params = params
        self.batch_size = self.params['batch_size']
        self.num_points = self.params['num_points']
        self.density_radius = self.params['density_radius']

        self.feature_out1 = self.params['feature_out1']
        self.feature_out2 = self.params['feature_out2']
        self.feature_out3 = self.params['feature_out3']
        self.feature_out4 = self.params['feature_out4']
        self.feature_out5 = self.params['feature_out5']
        
        self.num_classes = self.params['num_classes']
        
        self.bandwidth_0 = self.params['bandwidth_0']
        self.bandwidth_out1 = self.params['bandwidth_out1']
        self.bandwidth_out2 = self.params['bandwidth_out2']
        self.bandwidth_out3 = self.params['bandwidth_out3']
        self.bandwidth_out4 = self.params['bandwidth_out4']
        self.bandwidth_out5 = self.params['bandwidth_out5']

        self.sigma = nn.Parameter(torch.rand(3, 3))

        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        # s2 conv [Learn Pattern]
        self.conv1 = S2Convolution(
            nfeature_in=1, # [hard-code in_feature=1]
            nfeature_out=self.feature_out1,
            b_in=self.bandwidth_0,
            b_out=self.bandwidth_out1,
            grid=grid_s2
        )
        
        self.bn1 = nn.BatchNorm3d(
            num_features=self.feature_out1
        )
        

        # so3 conv (1) [Rotation Invariant]
        self.conv2 = SO3Convolution(
            nfeature_in=self.feature_out1,
            nfeature_out=self.feature_out2,
            b_in=self.bandwidth_out1,
            b_out=self.bandwidth_out2,
            grid=grid_so3
        )
        
        self.bn2 = nn.BatchNorm3d(
            num_features=self.feature_out2
        )
        
        # so3 conv (2) [Rotation Invariant]
        self.conv3 = SO3Convolution(
            nfeature_in=self.feature_out2,
            nfeature_out=self.feature_out3,
            b_in=self.bandwidth_out2,
            b_out=self.bandwidth_out3,
            grid=grid_so3
        )
        
        self.bn3 = nn.BatchNorm3d(
            num_features=self.feature_out3
        )
        
        # so3 conv (3) [Rotation Invariant]
        self.conv4 = SO3Convolution(
            nfeature_in=self.feature_out3,
            nfeature_out=self.feature_out4,
            b_in=self.bandwidth_out3,
            b_out=self.bandwidth_out4,
            grid=grid_so3
        )
        
        self.bn4 = nn.BatchNorm3d(
            num_features=self.feature_out4
        )
        
        self.conv5 = SO3Convolution(
            nfeature_in=self.feature_out4, # [hard-code in_feature=1]
            nfeature_out=self.feature_out5,
            b_in=self.bandwidth_out4,
            b_out=self.bandwidth_out5,
            grid=grid_so3
        )
        
        self.bn5 = nn.BatchNorm3d(
            num_features=self.feature_out5
        )
        

        self.out_layer = nn.Sequential(
            nn.Linear(self.feature_out5, int(self.feature_out5 / 2)),
            nn.ReLU(),
            nn.Linear(int(self.feature_out5 / 2), 10)
        )
    
        

        
    def forward(self, x):
        """
        :param x: tensor (B, 1, 2b0, 2b0)
        """
        
	inputs = utils.data_translation(inputs, self.bandwidth_0, self.density_radius, self.sigma) # -> [B, N, 3] -> [B, 2b0, 2b0]
        inputs = inputs.view(self.batch_size, 1, 2 * self.bandwidth_0, 2 * self.bandwidth_0)  # -> [B, 1, 2b0, 2b0]
        
        # S2 Conv 
        x = self.conv1(x)  # -> [B, f1, 2b1, 2b1, 2b1]
        x = F.relu(x)
        x = self.bn1(x)
        
        
        # SO3 Conv
        x = self.conv2(x)  # -> [B, f2, 2b2, 2b2, 2b2]
        x = F.relu(x)
        x = self.bn2(x)
        
        x = self.conv3(x)  # -> [B, f3, 2b3, 2b3, 2b3]
        x = F.relu(x)
        x = self.bn3(x)
        
        x = self.conv4(x)  # -> [B, f4, 2b4, 2b4, 2b4]
        x = F.relu(x)
        x = self.bn4(x)
        
        x = self.conv5(x)  # -> [B, f1, 2b1, 2b1, 2b1]
        x = F.relu(x)
        x = self.bn5(x)
        
        x = so3_integrate(x)  # -> (B, f4)
        x = self.out_layer(x)
        
        return x
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# END    
