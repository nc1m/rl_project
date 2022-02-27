from torch import nn
from collections import deque

class FCDQN(nn.Module):
    def __init__(self, OBS_SPACE, ACTION_SPACE, LAYERS, DEPTH, BIAS):
        super(FCDQN, self).__init__()
        
        self.layers = LAYERS
        self.input_layer = nn.Linear(OBS_SPACE, DEPTH, bias = BIAS)
        self.hidden_layer = nn.Linear(DEPTH, DEPTH, bias = BIAS)
        self.output_layer = nn.Linear(DEPTH, ACTION_SPACE, bias = BIAS)
        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        
        x = self.input_layer(x)
        x = self.ReLU(x)
        for _ in self.layers:
            x = self.hidden_layer(x)
            x = self.ReLU(x)
        
        return self.output_layer(x)
        
        


class SPRAgent(FCDQN):
    def __init__(self, buffersize) -> None:
        self.buffer = deque([], buffersize)
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
    
    def forward(self, x):
        super().forward(x)
        pass
    
    def train():
        pass
    
    def eval():
        pass
    
    def action():
        pass