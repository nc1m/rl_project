from torch import nn
import torch
import numpy as np
import src.utils as utils
import torch.nn.functional as F


class OnlineEncoder(nn.Module):
    def __init__(self, inChannels, dimHid, noAugmentation):
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
        layers.append(nn.Flatten(1))
        print(f'dimHid: {dimHid}')
        layers.append(nn.Linear(64 * 7 * 7, dimHid))
        layers.append(nn.ReLU())
        if noAugmentation:
                layers.append(nn.Dropout(0.5))

        self.encoder = nn.Sequential(*layers)


    def forward(self, x):
        x = self.encoder(x)
        return x


class DuelingDDQN(nn.Module):
    def __init__(self, inChannels, dimHid, dimOut, noAugmentation, Tau):
        "docstring"
        super(DuelingDDQN, self).__init__()
        print(f'dimHid: {dimHid}')
        self.onlineEncoder = OnlineEncoder(inChannels, dimHid, noAugmentation)
        if noAugmentation:
            self.targetEncoder = utils.EMA(self.onlineEncoder, 0.99)
        else:
            self.targetEncoder = utils.EMA(self.onlineEncoder, 0.0)
        # TODO
        sharedLinearLayers = []
        sharedLinearLayers.append(nn.Linear(dimHid, dimHid))
        sharedLinearLayers.append(nn.ReLU())
        if noAugmentation:
            sharedLinearLayers.append(nn.Dropout(0.5))

        self.sharedLin = nn.Sequential(*sharedLinearLayers)


        stateLayers = []
        stateLayers.append(nn.Linear(dimHid, dimHid))
        stateLayers.append(nn.ReLU())
        if noAugmentation:
            stateLayers.append(nn.Dropout(0.5))
        stateLayers.append(nn.Linear(dimHid, 1))
        stateLayers.append(nn.ReLU())
        if noAugmentation:
            stateLayers.append(nn.Dropout(0.5))

        self.stateLin = nn.Sequential(*stateLayers)

        actionLayers = []
        actionLayers.append(nn.Linear(dimHid, dimHid))
        actionLayers.append(nn.ReLU())
        if noAugmentation:
            actionLayers.append(nn.Dropout(0.5))
        actionLayers.append(nn.Linear(dimHid, dimOut))
        actionLayers.append(nn.ReLU())
        if noAugmentation:
            actionLayers.append(nn.Dropout(0.5))

        self.actionLin = nn.Sequential(*actionLayers)



    def forward(self, x):
        encoding = self.onlineEncoder(x)
        print(f'encoding.shape: {encoding.shape}')
        h = self.sharedLin(encoding)
        print(f'h.shape: {h.shape}')
        actionVal = self.actionLin(h)
        print(f'actionVal.shape: {actionVal.shape}')
        actionCentered = actionVal - actionVal.mean(dim=-1, keepdim=True)
        print(f'actionCentered.shape: {actionCentered.shape}')
        stateVal = self.stateLin(h)
        print(f'stateVal.shape: {stateVal.shape}')
        q = stateVal + actionCentered
        return q
