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
        self.num_grids = self.params['num_grids']
        self.batch_size = self.params['batch_size']
        self.num_points = self.params['num_points']
        self.density_radius = self.params['density_radius']

        self.feature_out1 = self.params['feature_out1']
        self.feature_out2 = self.params['feature_out2']
        self.feature_out3 = self.params['feature_out3']
        self.feature_out4 = self.params['feature_out4']
        self.feature_out5 = self.params['feature_out5']
        
        self.num_classes = self.params['num_classes']
        self.num_so3_layers = self.params['num_so3_layers']
        
        self.bandwidth_0 = self.params['bandwidth_0']
        self.bandwidth_out1 = self.params['bandwidth_out1']
        self.bandwidth_out2 = self.params['bandwidth_out2']
        self.bandwidth_out3 = self.params['bandwidth_out3']
        self.bandwidth_out4 = self.params['bandwidth_out4']
        self.bandwidth_out5 = self.params['bandwidth_out5']

        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        # s2 conv [Learn Pattern] -----------------------------------------------
        self.conv0_0 = S2Convolution(
            nfeature_in=1,
            nfeature_out=self.feature_out1,
            b_in=self.bandwidth_0,
            b_out=self.bandwidth_out1,
            grid=grid_s2
        )
        
        self.conv0_1 = S2Convolution(
            nfeature_in=1,
            nfeature_out=self.feature_out1,
            b_in=self.bandwidth_0,
            b_out=self.bandwidth_out1,
            grid=grid_s2
        )
        
        self.conv0_2 = S2Convolution(
            nfeature_in=1, 
            nfeature_out=self.feature_out1,
            b_in=self.bandwidth_0,
            b_out=self.bandwidth_out1,
            grid=grid_s2
        )
        
        self.bn0_0 = nn.BatchNorm3d(
            num_features=self.feature_out1
        )
        
        self.bn0_1 = nn.BatchNorm3d(
            num_features=self.feature_out1
        )
        
        self.bn0_2 = nn.BatchNorm3d(
            num_features=self.feature_out1
        )

        # so3 conv (1) [Rotation Invariant] -----------------------------------------------
        self.conv1_0 = SO3Convolution(
            nfeature_in=self.feature_out1,
            nfeature_out=self.feature_out2,
            b_in=self.bandwidth_out1,
            b_out=self.bandwidth_out2,
            grid=grid_so3
        )
        
        self.conv1_1 = SO3Convolution(
            nfeature_in=self.feature_out1,
            nfeature_out=self.feature_out2,
            b_in=self.bandwidth_out1,
            b_out=self.bandwidth_out2,
            grid=grid_so3
        )
        
        self.conv1_2 = SO3Convolution(
            nfeature_in=self.feature_out1,
            nfeature_out=self.feature_out2,
            b_in=self.bandwidth_out1,
            b_out=self.bandwidth_out2,
            grid=grid_so3
        )
        
        self.bn1_0 = nn.BatchNorm3d(
            num_features=self.feature_out2
        )
        
        self.bn1_1 = nn.BatchNorm3d(
            num_features=self.feature_out2
        )
        
        self.bn1_2 = nn.BatchNorm3d(
            num_features=self.feature_out2
        )
        
        # so3 conv (2) [Rotation Invariant] -----------------------------------------------
        self.conv2_0 = SO3Convolution(
            nfeature_in=self.feature_out2,
            nfeature_out=self.feature_out3,
            b_in=self.bandwidth_out2,
            b_out=self.bandwidth_out3,
            grid=grid_so3
        )
        
        self.conv2_1 = SO3Convolution(
            nfeature_in=self.feature_out2,
            nfeature_out=self.feature_out3,
            b_in=self.bandwidth_out2,
            b_out=self.bandwidth_out3,
            grid=grid_so3
        )
        
        self.conv2_2 = SO3Convolution(
            nfeature_in=self.feature_out2,
            nfeature_out=self.feature_out3,
            b_in=self.bandwidth_out2,
            b_out=self.bandwidth_out3,
            grid=grid_so3
        )
        
        self.bn2_0 = nn.BatchNorm3d(
            num_features=self.feature_out3
        ) 
        
        self.bn2_1 = nn.BatchNorm3d(
            num_features=self.feature_out3
        ) 
        
        self.bn2_2 = nn.BatchNorm3d(
            num_features=self.feature_out3
        ) 
        
        # so3 conv (3) [Rotation Invariant] -----------------------------------------------
        self.conv3_0 = SO3Convolution(
            nfeature_in=self.feature_out3,
            nfeature_out=self.feature_out4,
            b_in=self.bandwidth_out3,
            b_out=self.bandwidth_out4,
            grid=grid_so3
        ) 
        
        self.conv3_1 = SO3Convolution(
            nfeature_in=self.feature_out3,
            nfeature_out=self.feature_out4,
            b_in=self.bandwidth_out3,
            b_out=self.bandwidth_out4,
            grid=grid_so3
        ) 
            
        self.conv3_2 = SO3Convolution(
            nfeature_in=self.feature_out3,
            nfeature_out=self.feature_out4,
            b_in=self.bandwidth_out3,
            b_out=self.bandwidth_out4,
            grid=grid_so3
        ) 
        
        self.bn3_0 = nn.BatchNorm3d(
            num_features=self.feature_out4
        ) 
        
        self.bn3_1 = nn.BatchNorm3d(
            num_features=self.feature_out4
        ) 
        
        self.bn3_2 = nn.BatchNorm3d(
            num_features=self.feature_out4
        ) 
        
        self.weights = nn.Parameter(nn.init.uniform_(torch.Tensor(self.feature_out4, self.num_grids)))

        self.out_layer = nn.Sequential(
            nn.Linear(self.feature_out4, int(self.feature_out4 / 2)),
            nn.ReLU(),
            nn.Linear(int(self.feature_out4 / 2), 10)
        )
    
    
    def S2(self, inputs):
        ys = list()
        for i, x in enumerate(inputs):
            y = self.conv1[i](x)
            ys.append(y)
        return ys
    
    def forward(self, x):
        """
        :param x: list( Tensor([B, 2b0, 2b0]) * num_grids )
        """

        # S2 Conv 
        x = [self.conv0_0(x[0]), # -> [B, f1, 2b1, 2b1, 2b1] * num_grids
             self.conv0_1(x[1]), 
             self.conv0_2(x[2])]
        x = [F.relu(x[0]), 
             F.relu(x[1]), 
             F.relu(x[2])]
        x = [self.bn0_0(x[0]), 
             self.bn0_1(x[1]), 
             self.bn0_2(x[2])]
        
        # SO3 Conv
        x = [self.conv1_0(x[0]), # -> [B, f2, 2b2, 2b2, 2b2] * num_grids
             self.conv1_1(x[1]), 
             self.conv1_2(x[2])]
        x = [F.relu(x[i]) for i in range(len(x))]
        x = [self.bn1_0(x[0]), 
             self.bn1_1(x[1]), 
             self.bn1_2(x[2])]
        
        x = [self.conv2_0(x[0]), # -> [B, f3, 2b3, 2b3, 2b3] * num_grids
             self.conv2_1(x[1]), 
             self.conv2_2(x[2])]
        x = [F.relu(x[i]) for i in range(len(x))]
        x = [self.bn2_0(x[0]), 
             self.bn2_1(x[1]), 
             self.bn2_2(x[2])]
        
        x = [self.conv3_0(x[0]), # -> [B, f4, 2b4, 2b4, 2b4] * num_grids
             self.conv3_1(x[1]), 
             self.conv3_2(x[2])]
        x = [F.relu(x[i]) for i in range(len(x))]
        x = [self.bn3_0(x[0]), 
             self.bn3_1(x[1]), 
             self.bn3_2(x[2])]

        x = [so3_integrate(x[i]) for i in range(len(x))]  # -> (B, f4) * num_grids
        
        x = [x[i].unsqueeze(0) for i in range(len(x))]
        x = torch.cat(tuple(x), dim=0)  # -> (num_grids, B, f4)

        N, B, C = x.shape

        x = x.permute(1, 2, 0)  # -> (B, f4, num_grids)
        x = torch.mul(x, torch.sigmoid(self.weights))  # -> (B, f4, num_grids)
        x = torch.sum(x, dim=-1, keepdim=False)  # -> (B, f4)

        x = self.out_layer(x)
        
        return x
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# END    
