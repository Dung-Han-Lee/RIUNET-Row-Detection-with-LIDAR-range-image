#
# Author : Dung-Han Lee
# contact: dunghanlee@gmail.com
# 
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import  WeightedRandomSampler
from sys import argv

#User define modules
import data
import models
import routine
import config

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cross_entropy2d(model, inputs, target, weight_map, weight=None, size_average=True):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht and w != wt:  # upsample labels
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # CHW -> HWC -> (HW) x c
    inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(inputs, target, weight=None, reduction='none')
    
    # Apply weight map
    weight_map = weight_map.view(-1)
    loss = (loss * weight_map)
    loss = torch.mean(loss)
       
    l2_reg = None
    for W in model.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    
    return loss + config.reg_factor * l2_reg
    
def run():
    train_dataset = data.RangeViewDataset(mode = 'train')
    val_dataset   = data.RangeViewDataset(mode = 'val')
        
    train_loader = DataLoader( train_dataset, 
                               batch_size=config.batch_size,
                               shuffle=True, 
                               drop_last = False)

    val_loader = DataLoader( val_dataset, 
                             batch_size=config.batch_size, 
                             shuffle=False, 
                             drop_last = False)
    
    # Set up models and (optionally) load weights
    network = models.unet()
    if config.load:
        state_dict = torch.load('./models/'+config.model)
        network.load_state_dict(state_dict)
        print("loading weights from {}".format(config.model) )

    if not config.load:
        print("train from scracth...")

    network.train()
    network.to(device)
            
    # Define loss function then start to train
    criterion = cross_entropy2d     
    optimizer = torch.optim.Adam(network.parameters(), lr = config.lr)
    routine.train(network, train_loader, val_loader, criterion, optimizer)

if __name__ == "__main__":
    run()

