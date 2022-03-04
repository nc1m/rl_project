from re import X
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


        self.conv = nn.Sequential(*layers)
        # conv1_in = nn.Conv2d(inChannels, 32, 8, 4)
        # conv2 = nn.Conv2d(32, 64, 4, 2)
        # conv3 = nn.Conv2d(64, 64, 3, 1)
        # layers = nn.Sequential([conv1_in, nn.ReLU(), conv2, nn.ReLU(), conv3, nn.ReLU()])



        # # TODO compute output size of convs
        linLayers = []
        linLayers.append(nn.Flatten(1))
        print(f'dimHid: {dimHid}')
        linLayers.append(nn.Linear(64 * 7 * 7, dimHid))
        linLayers.append(nn.ReLU())
        if noAugmentation:
                linLayers.append(nn.Dropout(0.5))

        self.mlp = nn.Sequential(*linLayers)



    def forward(self, x):
        convFeatureMaps = self.conv(x)
        hiddenRep = self.mlp(convFeatureMaps)
        return hiddenRep, convFeatureMaps


class ConvTransitionModel(nn.Module):
    def __init__(self, inChannels, outChannels, dimHid, dimOut, noAugmentation):
        "docstring"
        super(ConvTransitionModel, self).__init__()
        self.dimOut = dimOut

        print(f'inChannels: {inChannels}')
        print(f'outChannels: {outChannels}')
        print(f'dimOut: {dimOut}')
        print(64 + dimOut)
        convLayers = []
        convLayers.append(nn.Conv2d(64 + dimOut, dimHid, 3))
        convLayers.append(nn.ReLU())
        convLayers.append(nn.BatchNorm2d(dimHid))
        convLayers.append(nn.Conv2d(dimHid, dimHid, 3))
        convLayers.append(nn.ReLU())

        self.conv = nn.Sequential(*convLayers)

        linLayers = []
        linLayers.append(nn.Flatten(1))
        linLayers.append(nn.Linear(dimHid*3*3, dimHid))
        linLayers.append(nn.ReLU())
        if noAugmentation:
            linLayers.append(nn.Dropout(0.5))

        linLayers.append(nn.Linear(dimHid, dimHid))
        linLayers.append(nn.ReLU())
        if noAugmentation:
            linLayers.append(nn.Dropout(0.5))


        self.MLP = nn.Sequential(*linLayers)



    def forward(self, x, action):
        print(f'x.shape: {x.shape}')
        print(f'action.shape: {action.shape}')
        # TODO In Atari, an action is encoded as a one hot vector which is tiled appropriately into planes.
        batch_range = torch.arange(action.shape[0], device=action.device)
        print(f'batch_range.shape: {batch_range.shape}')
        action_onehot = torch.zeros(action.shape[0],
                                    self.dimOut, # self.num_actions,
                                    x.shape[-2],
                                    x.shape[-1],
                                    device=action.device)
        print(f'action_onehot.shape: {action_onehot.shape}')
        action_onehot[batch_range, action, :, :] = 1
        stacked_image = torch.cat([x, action_onehot], 1)
        print(f'stacked_image.shape: {stacked_image.shape}')
        stacked_image = self.conv(stacked_image)
        convTransition_hiddenRep_tk = self.MLP(stacked_image)
        print(f'convTransiton_hiddenRep_tk.shape: {convTransition_hiddenRep_tk.shape}')
        return convTransition_hiddenRep_tk






class DuelingDDQN(nn.Module):
    def __init__(self, inChannels, dimHid, dimOut, noAugmentation):
        "docstring"
        super(DuelingDDQN, self).__init__()
        print(f'dimHid: {dimHid}')
        self.dimHid = dimHid
        self.onlineEncoder = OnlineEncoder(inChannels, dimHid, noAugmentation)
        if noAugmentation:
            self.targetEncoder = utils.EMA(self.onlineEncoder, 0.99)
        else:
            self.targetEncoder = utils.EMA(self.onlineEncoder, 0.0)

        sharedLinearLayers = []
        sharedLinearLayers.append(nn.Linear(dimHid, dimHid))
        sharedLinearLayers.append(nn.ReLU())
        if noAugmentation:
            sharedLinearLayers.append(nn.Dropout(0.5))

        self.sharedLin = nn.Sequential(*sharedLinearLayers)


        valueLayersHid = []
        valueLayersHid.append(nn.Linear(dimHid, dimHid))
        valueLayersHid.append(nn.ReLU())
        if noAugmentation:
            valueLayersHid.append(nn.Dropout(0.5))

        self.valueHidMLP = nn.Sequential(*valueLayersHid)

        valueLayersOut = []
        valueLayersOut.append(nn.Linear(dimHid, 1))
        valueLayersOut.append(nn.ReLU())
        if noAugmentation:
            valueLayersOut.append(nn.Dropout(0.5))

        self.valueOutMLP = nn.Sequential(*valueLayersOut)



        advantageLayersHid = []
        advantageLayersHid.append(nn.Linear(dimHid, dimHid))
        advantageLayersHid.append(nn.ReLU())
        if noAugmentation:
            advantageLayersHid.append(nn.Dropout(0.5))

        self.advantageHidMLP = nn.Sequential(*advantageLayersHid)

        advantageLayersOut = []
        advantageLayersOut.append(nn.Linear(dimHid, dimOut))
        advantageLayersOut.append(nn.ReLU())
        if noAugmentation:
            advantageLayersOut.append(nn.Dropout(0.5))

        self.advantageOutMLP = nn.Sequential(*advantageLayersOut)

        predictorLayers = []
        predictorLayers.append(nn.Linear(dimHid, dimOut))
        predictorLayers.append(nn.ReLU())
        if noAugmentation:
            predictorLayers.append(nn.Dropout(0.5))

        self.predictor_q = nn.Sequential(*predictorLayers)

        self.convTransitionModel = ConvTransitionModel(260, dimHid, dimHid, dimOut, noAugmentation)

        # self.target_projection = nn.Sequential(nn.Linear(self.dimHid, 2*self.dimHid),
        #                                         nn.BatchNorm1d(2*self.dimHid),
        #                                         nn.ReLU(),
        #                                         nn.Linear(2*self.dimHid, self.dimHid))

    def forward(self, encoding):
        # encoding = self.onlineEncoder(x)
        # print(f'encoding.shape: {encoding.shape}')
        print(f'encoding.shape: {encoding.shape}')
        h = self.sharedLin(encoding)
        # print(f'h.shape: {h.shape}')
        valueHid = self.valueHidMLP(h)
        actionVal = self.valueOutMLP(valueHid)
        # print(f'actionVal.shape: {actionVal.shape}')
        actionCentered = actionVal - actionVal.mean(dim=-1, keepdim=True)
        # print(f'actionCentered.shape: {actionCentered.shape}')
        advantageHid = self.advantageHidMLP(h)
        stateVal = self.advantageOutMLP(advantageHid)
        # print(f'stateVal.shape: {stateVal.shape}')
        q = stateVal + actionCentered


        # print(valueHid.shape)
        # print(advantageHid.shape)
        # onlineProjectionOut = torch.cat((valueHid, advantageHid))
        # predictor_q_out = self.predictor_q(onlineProjectionOut)

        return q# , predictor_q_out

    def forward_online_encoder(self, x):
        hiddenRep, convFeatureMaps = self.onlineEncoder(x)
        print(f'hiddenRep.shape: {hiddenRep.shape}')
        print(f'convFeatureMaps.shape: {convFeatureMaps.shape}')
        return hiddenRep, convFeatureMaps

    def forward_conv_transition_model(self, x, action):
        return self.convTransitionModel(x, action)

    def forward_online_projection_and_predictor_q(self, convTransitionModelRep):
        # TODO
        h = self.sharedLin(convTransitionModelRep)
        valueHid = self.valueHidMLP(h)
        advantageHid = self.advantageHidMLP(h)
        onlineProjectionOut = torch.cat((valueHid, advantageHid))
        print(f'onlineProjectionOut.shape: {onlineProjectionOut.shape}')
        predictor_q_out = self.predictor_q(onlineProjectionOut)
        return predictor_q_out

    # def forward_online_projection(self, x):
    #     return self.onlineProjection(x)

# class OnlineProjectionHead():
#     def __init__(self, dimHid):
#         "docstring"
#         super(OnlineProjectionHead, self).__init__()

#         self.net = nn.Sequential(nn.Linear(dimHid,2*dimHid),
#                                     nn.BatchNorm1d(2*dimHid),
#                                     nn.ReLU(),
#                                     nn.Linear(2*dimHid, dimHid),
#                                     nn.ReLU(),
#                                     nn.Linear(dimHid, dimHid)) #online-q-prediction-head

#     def forward(self, x):
#         return self.net(x)

# class TargetProjectionHead():
#     def __init__(self, dimHid):
#         "docstring"
#         super(TargetProjectionHead, self).__init__()

#         self.net = nn.Sequential(nn.Linear(dimHid,2*dimHid),
#                                     nn.BatchNorm1d(2*dimHid),
#                                     nn.ReLU(),
#                                     nn.Linear(2*dimHid, dimHid))

#     def forward(self, x):
#         return self.net(x)
