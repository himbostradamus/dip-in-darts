from collections import OrderedDict
import nni.retiarii.nn.pytorch as nn
from nni import trace
import torch

@trace
def seBlock(channel, reduction=16):
    return nn.Sequential(
        nn.Linear(channel, channel // reduction, bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(channel // reduction, channel, bias=False),
        nn.Sigmoid()
    )

    # corresponding forward function 
   
def seForward(x, fcs):
    b, c, _, _ = x.size()
    y = nn.AdaptiveAvgPool2d(1)(x).view(b, c)
    y = fcs(y).view(b, c, 1, 1)
    return x * y.expand_as(x)

