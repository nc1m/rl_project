import logging
import torch
from torch import nn

from copy import deepcopy
from collections import OrderedDict
from sys import stderr

# for type hint
from torch import Tensor
import torch.nn as nn
import numpy as np
from torchvision.transforms import ColorJitter
from kornia.augmentation import RandomCrop
from collections import namedtuple
from collections import deque
import random

# Following https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class Intensity(nn.Module):     # ColorJitter from https://arxiv.org/pdf/2004.13649.pdf
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise

def process_images(images):
    images = np.array(images, dtype=np.float)
    # images = np.transpose(images, (0, 3, 1, 2)) # alternative to tensor.permute(0, 3, 1, 2)
    images = torch.tensor(images, dtype=torch.float)
    images = images.permute(0, 3, 1, 2)
    images = images / 255
    return images



class ReplayMemory(object):

    def __init__(self, capacity, use_cuda, use_augmentation, imageSize):
        self.memory = deque([], maxlen=capacity)
        self.use_augmentation = use_augmentation
        self.imageSize = imageSize
        self.use_cuda = use_cuda

    def push(self, curState, action, nextState, reward):
        """Save a transition"""
        # logging.warning(f'curState: {type(curState)}')
        # logging.warning(f'type(action): {type(action)}')
        # logging.warning(f'type(nextState): {type(nextState)}')
        # logging.warning(f'type(reward): {type(reward)}')
        self.memory.append([curState, action, nextState, reward])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states = []
        actions = []
        nextStates = []
        rewards = []
        for data in batch:
            states.append(data[0])
            actions.append(data[1])
            nextStates.append(data[2])
            rewards.append(data[3])

        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)

        states = process_images(states)
        nextStates = process_images(nextStates)

        if self.use_cuda:
            actions = actions.cuda()
            rewards = rewards.cuda()

            states = states.cuda()
            nextStates = nextStates.cuda()

        if self.use_augmentation:
            transform = nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((self.imageSize, self.imageSize)), Intensity(scale=0.5))
            # print(f'states.shape: {states.shape}')
            states = transform(states)
            nextStates = transform(nextStates)
        return states, actions, nextStates, rewards

    def __len__(self):
        return len(self.memory)



class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        # adapted from https://fyubang.com/2019/06/01/ema/
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # check if both model contains the same set of keys
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
            # shadow_variable -= (1 - decay) * (shadow_variable - variable)
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # check if both model contains the same set of keys
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            # buffers are copied
            shadow_buffers[name].copy_(buffer)

        return self.shadow
    '''
    def forward(self, inputs: Tensor, return_feature: bool = False) -> Tensor:
        if self.training:
            return self.model(inputs, return_feature)
        else:
            return self.shadow(inputs, return_feature)
    '''
