import torch
from torch import device, nn
import numpy as np

# hthttps://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class DQNConvNet(nn.Module):
    def __init__(self,shape_in,num_actions,batch_norm = False) -> None:
        """DeepQ-Learning network with three convolutions and a fully connected layer in the end

        Args:
            dim_in (_type_): Number of inputs
            num_actions (_type_): Number of outputs
        """
        super().__init__()

        if batch_norm:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=shape_in[0],out_channels=32,kernel_size=8,stride=4),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels=64,out_channels=64,kernel_size=8,stride=4),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=shape_in[0],out_channels=32,kernel_size=8,stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1),
                nn.ReLU()
            )
        
        def _get_conv_out(self, shape):
            """Size of the convolution output

            Args:
                shape (_type_): _description_

            Returns:
                _type_: _description_
            """

            o = self.conv_layers(torch.zeros(1, *shape))
            return int(np.prod(o.size())) # Product over the shape of o

        self.head = nn.Sequential(
            nn.Linear(self._get_conv_out(shape_in),num_actions),
            nn.ReLU()
        )

    def forward(self,x : torch.Tensor) -> torch.Tensor:
        """Predict action according to x

        Args:
            x (torch.Tensor): Input as image

        Returns:
            torch.Tensor: Prediction
        """
        x = x.to(device)
        x = self.conv_layers(x)
        
        return self.head(x.view(x.size(0), -1))
    

    




