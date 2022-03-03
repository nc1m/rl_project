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


#from https://github.com/astooke/rlpyt/blob/master/rlpyt/utils/tensor.py
def restore_leading_dims(tensors, lead_dim, T=1, B=1):
    """Reshapes ``tensors`` (one or `tuple`, `list`) to to have ``lead_dim``
    leading dimensions, which will become [], [B], or [T,B].  Assumes input
    tensors already have a leading Batch dimension, which might need to be
    removed. (Typically the last layer of model will compute with leading
    batch dimension.)  For use in model ``forward()`` method, so that output
    dimensions match input dimensions, and the same model can be used for any
    such case.  Use with outputs from ``infer_leading_dims()``."""
    is_seq = isinstance(tensors, (tuple, list))
    tensors = tensors if is_seq else (tensors,)
    if lead_dim == 2:  # (Put T dim.)
        tensors = tuple(t.view((T, B) + t.shape[1:]) for t in tensors)
    if lead_dim == 0:  # (Remove B=1 dim.)
        assert B == 1
        tensors = tuple(t.squeeze(0) for t in tensors)
    return tensors if is_seq else tensors[0]

#from https://github.com/astooke/rlpyt/blob/master/rlpyt/utils/tensor.py
def infer_leading_dims(tensor, dim):
    """Looks for up to two leading dimensions in ``tensor``, before
    the data dimensions, of which there are assumed to be ``dim`` number.
    For use at beginning of model's ``forward()`` method, which should
    finish with ``restore_leading_dims()`` (see that function for help.)
    Returns:
    lead_dim: int --number of leading dims found.
    T: int --size of first leading dim, if two leading dims, o/w 1.
    B: int --size of first leading dim if one, second leading dim if two, o/w 1.
    shape: tensor shape after leading dims.
    """
    lead_dim = tensor.dim() - dim
    assert lead_dim in (0, 1, 2)
    if lead_dim == 2:
        T, B = tensor.shape[:2]
    else:
        T = 1
        B = 1 if lead_dim == 0 else tensor.shape[0]
    shape = tensor.shape[lead_dim:]
    return lead_dim, T, B, shape

#from https://github.com/astooke/rlpyt/blob/master/rlpyt/models/utils.py
class ScaleGrad(torch.autograd.Function):
    """Model component to scale gradients back from layer, without affecting
    the forward pass.  Used e.g. in dueling heads DQN models."""

    @staticmethod
    def forward(ctx, tensor, scale):
        """Stores the ``scale`` input to ``ctx`` for application in
        ``backward()``; simply returns the input ``tensor``."""
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """Return the ``grad_output`` multiplied by ``ctx.scale``.  Also returns
        a ``None`` as placeholder corresponding to (non-existent) gradient of
        the input ``scale`` of ``forward()``."""
        return grad_output * ctx.scale, None


# scale_grad = ScaleGrad.apply
# Supply a dummy for documentation to render.
scale_grad = getattr(ScaleGrad, "apply", None)

#from https://github.com/astooke/rlpyt/blob/master/rlpyt/models/utils.py
def strip_ddp_state_dict(state_dict):
    """ Workaround the fact that DistributedDataParallel prepends 'module.' to
    every key, but the sampler models will not be wrapped in
    DistributedDataParallel. (Solution from PyTorch forums.)"""
    clean_state_dict = type(state_dict)()
    for k, v in state_dict.items():
        key = k[7:] if k[:7] == "module." else k
        clean_state_dict[key] = v
    return clean_state_dict

#from https://github.com/astooke/rlpyt/blob/master/rlpyt/models/utils.py
def update_state_dict(model, state_dict, tau=1, strip_ddp=True):
    """Update the state dict of ``model`` using the input ``state_dict``, which
    must match format.  ``tau==1`` applies hard update, copying the values, ``0<tau<1``
    applies soft update: ``tau * new + (1 - tau) * old``.
    """
    if strip_ddp:
        state_dict = strip_ddp_state_dict(state_dict)
    if tau == 1:
        model.load_state_dict(state_dict)
    elif tau > 0:
        update_sd = {k: tau * state_dict[k] + (1 - tau) * v
            for k, v in model.state_dict().items()}
        model.load_state_dict(update_sd)





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
        # print(states.shape)
        # print(nextStates.shape)
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
