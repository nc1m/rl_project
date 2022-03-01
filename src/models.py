from torch import nn
import torch

#TODO
class SPRModel(nn.Module):
    
    def __init__(self, input_size, output_size, time_offset):
        self.input_size = input_size
        self.output_size = output_size
        self.time_offset = time_offset
        
    
class MLPOnlineHead(nn.Module):
    def __init__(self, input_channel, output_size, hidden_size, pixels, n_atoms):
        super(MLPOnlineHead, self).__init__()
        
        self.input = nn.Linear(input_channel * pixels, hidden_size)
        self.output = nn.Linear(hidden_size, output_size * n_atoms)
        self.ReLU = nn.ReLU()
        self.flatten = nn.Flatten(-3, -1)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.input(x)
        x = self.ReLU(x)
        return self.output(x)

#TODO
class TransitionModel(nn.Module):
    def __init__(self, channels,
                 num_actions,
                 args=None,
                 blocks=16,
                 hidden_size=256,
                 pixels=36,
                 limit=300,
                 action_dim=6,
                 norm_type="bn",
                 renormalize=True,
                 residual=False):
        super(TransitionModel, self).__init__()
        '''
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.args = args
        self.renormalize = renormalize
        self.residual = residual
        layers = [Conv2dSame(channels+num_actions, hidden_size, 3),
                  nn.ReLU(),
                  init_normalization(hidden_size, norm_type)]
        for _ in range(blocks):
            layers.append(ResidualBlock(hidden_size,
                                        hidden_size,
                                        norm_type))
        layers.extend([Conv2dSame(hidden_size, channels, 3)])

        self.action_embedding = nn.Embedding(num_actions, pixels*action_dim)

        self.network = nn.Sequential(*layers)
        self.reward_predictor = RewardPredictor(channels,
                                                pixels=pixels,
                                                limit=limit,
                                                norm_type=norm_type)
        '''
        
    
class OnlineEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, paddings, nonlinearity, use_maxpool, dropout = 0.5):
        super(OnlineEncoder, self).__init__()
        
        if paddings is None:
            paddings = [0 for _ in range(len(out_channels))]
        assert len(out_channels) == len(kernel_sizes) == len(strides) == len(paddings)
        
        in_channels = [in_channels] + out_channels[:-1]
        ones = [1 for _ in range(len(strides))]
        
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=k, stride=s, padding=p) 
                       for (ic, oc, k, s, p) in zip(in_channels, out_channels, kernel_sizes, strides, paddings)]
        
        sequence = list()
        for conv_layer, maxp_stride in zip(conv_layers, maxp_strides):
            sequence.extend([conv_layer, nonlinearity()])
            if dropout > 0:
                sequence.append(nn.Dropout(dropout))
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        
        self.conv = torch.nn.Sequential(*sequence)
        
    def forward(self, x):
        return self.conv(x)