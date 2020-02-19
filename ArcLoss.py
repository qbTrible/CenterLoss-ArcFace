# -*- coding: utf-8 -*- 
# @Time : 2019-11-19 16:00 
# @Author : Trible 

import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcLoss(nn.Module):

    def __init__(self, feature_dim, cls_dim, s=10):
        super().__init__()
        self.W = nn.Parameter(torch.randn(feature_dim, cls_dim))
        self.s = s

    def forward(self, feature):
        _W = F.normalize(self.W, dim=0)
        _X = F.normalize(feature, dim=1)
        # _X = feature
        cosa = torch.matmul(_X, _W)/self.s
        a = torch.acos(cosa)
        top = torch.exp(torch.cos(a+0.1)*self.s)
        _top = torch.exp(torch.cos(a)*self.s)
        bottom = torch.sum(torch.exp(cosa*self.s), dim=1).view(-1, 1)

        return top/(bottom-_top+top) + 1e-10