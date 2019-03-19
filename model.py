import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import utils
from pdb import set_trace as st

class SphericalGMMNet(nn.Module):
    def __init__(self, params=None):
        super(ManifoldNet, self).__init__()

        self.k = num_neighbors
        self.points = num_points
        
        self.layer1 = None
        
        self.Last = wFM.Last(40, num_classes, 512)
       
    def forward(self, inputs):
       
        out = None

#         print("===========================")
#         print("[Output]")
#         print("Size: {}".format(out.size()))
#         print("Tensor: {}".format(out))
#         print("===========================")
        
        return out

