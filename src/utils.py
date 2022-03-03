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
from torchvision.transforms import Grayscale
from kornia.augmentation import ColorJitter
from kornia.augmentation import RandomCrop
from kornia.color import rgb_to_grayscale
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
    images = images.permute(0, 1, 4, 2, 3)
    images = images / 255
    images = rgb_to_grayscale(images)
    images = torch.squeeze(images)
    return images



class ReplayMemory(object):

    def __init__(self, capacity, use_cuda, use_augmentation, imageSize):
        self.memory = deque([], maxlen=capacity)
        self.use_augmentation = use_augmentation
        self.imageSize = imageSize
        self.use_cuda = use_cuda

    # def push(self, curState, action, nextState, reward):
    def push(self, framestack):
        """Save a transition"""
        # logging.warning(f'curState: {type(curState)}')
        # logging.warning(f'type(action): {type(action)}')
        # logging.warning(f'type(nextState): {type(nextState)}')
        # logging.warning(f'type(reward): {type(reward)}')
        # self.memory.append([curState, action, nextState, reward])
        self.memory.append(framestack)

    def sample(self, batch_size):
        batches = random.sample(self.memory, batch_size) # BATCH X FRAMESTACK X 4
        print(len(batches))
        print(len(batches[0]))
        print(len(batches[0][0]))
        states = []             # BATCH X FRAMESTACK X WIDTH X HEIGHT
        actions = []
        nextStates = []
        rewards = []
        for batch in batches:
            states_fs = []
            actions_fs = []
            nextStates_fs = []
            rewards_fs = []
            for framestack in batch:
                states_fs.append(framestack[0])
                actions_fs.append(framestack[1])
                nextStates_fs.append(framestack[2])
                rewards_fs.append(framestack[3])
            states.append(states_fs)
            nextStates.append(nextStates_fs)
            actions.append(actions_fs)
            rewards.append(rewards_fs)



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
            states = transform(states)
            nextStates = transform(nextStates)
        else:
            transform = nn.Sequential(RandomCrop((self.imageSize, self.imageSize)))
            states = transform(states)
            nextStates = transform(nextStates)
        print(states.shape)
        print(nextStates.shape)
        exit()
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
