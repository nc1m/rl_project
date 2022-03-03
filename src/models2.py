from torch import nn
import torch
import numpy as np
import src.utils as utils
import torch.nn.functional as F


class OnlineEncoder(nn.Module):
    def __init__(self, inChannels, dimOut, noAugmentation):
        "dimOut = env.action_space"
        super(OnlineEncoder, self).__init__()
        inChannels_l = [inChannels, 32, 64]
        outChannels_l = [32, 64, 64]
        kernelSizes = [8, 4, 3]
        strides = [4, 2, 1]

        layers = []
        for ic, oc, kernelSize, stride in zip(inChannels_l, outChannels_l, kernelSizes, strides):
            layers.append(nn.Conv2d(ic, oc, kernelSize, stride))
            layers.append(nn.ReLU())
            if noAugmentation:
                layers.append(nn.Dropout(0.5))



        # conv1_in = nn.Conv2d(inChannels, 32, 8, 4)
        # conv2 = nn.Conv2d(32, 64, 4, 2)
        # conv3 = nn.Conv2d(64, 64, 3, 1)
        # layers = nn.Sequential([conv1_in, nn.ReLU(), conv2, nn.ReLU(), conv3, nn.ReLU()])



        # # TODO compute output size of convs
        # linear = nn.Linear(512, dimOut)
        # layers.append(linear, nn.ReLU())
        # if noAugmentation:
        #         layers.append(nn.Dropout(0.5))

        self.encoder = nn.Sequential(*layers)


    def forward(self, x):
        x = self.encoder(x)
        print(x.shape)
        return x


class DQN(nn.Module):
    def __init__(self, inChannels, dimOut, noAugmentation, Tau):
        "docstring"
        super(DQN, self).__init__()

        self.onlineEncoder = OnlineEncoder(inChannels, dimOut, noAugmentation)
        self.targetEncoder = utils.EMA(self.onlineEncoder, Tau)

        layers = []
        self.sharedLinear = nn.Linear(dimHid, 256)
