import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SGC(nn.Module):
    '''
        Implementation of GDN using Simple Graph Convolution. (out_feature = 1)
        - Network Encoder: Input이 (S^K)X (precomputed) 이기 때문에, adjacency matrix 고려 안 해줘도 되고 그냥 linear transformation한 번 하면 끝남.
        - Abnormality Valuator: 하나의 hidden layer -> scalar output 구조인데, hidden layer의 경우 Network Encoder가 SGC이기 때문에 Encoder의 output과 implicit 결합 가능.
    '''
    def __init__(self, in_feature, out_feature):
        super(SGC, self).__init__()
        self.fc1 = nn.Linear(in_feature, 512) # Implicit하게 Network Encoder + hidden layer of Abnormality Valuator가 표현됨
        self.bn = nn.BatchNorm1d(512) # ! 추가된 것. Batch Normalization
        self.out = nn.Linear(512, out_feature) # Output of Abnormality Valuator

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x) # ! 추가된 것
        x = F.relu(x, inplace=False) # ! 추가된 것
        x = self.out(x)
        return x
