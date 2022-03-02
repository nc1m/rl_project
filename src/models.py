from torch import nn
import torch
import numpy as np
import utils
import torch.nn.functional as F

class SPRModel(nn.Module):
    
    def __init__(self, input_size, output_size, time_offset, image_size, image_shape, dqn_hidden_size, jumps, model_rl):
        
        self.input_size = input_size
        self.output_size = output_size
        self.time_offset = time_offset
        self.dqn_hidden_size = dqn_hidden_size
        self.jumps = jumps
        self.model_rl = model_rl
        f, c = image_shape[:2]
        in_channels = np.prod(image_shape[:2])
        #Online Encoder
        self.o_encoder = OnlineEncoder(in_channels,
                                           out_channels = [32, 64, 64],
                                           kernel_sizes = [8, 4, 3],
                                           strides = [4, 2, 1],
                                           paddings = [0, 0, 0],
                                           nonlinearity= nn.ReLU(),
                                           use_maxpool=True,
                                           dropout = 0.5)
        
        fake_input = torch.zeros(1, f*c, image_size, image_size)
        fake_output = self.o_encoder(fake_input)
        self.hidden_size = fake_output.shape[1]
        self.pixels = fake_output.shape[-1]*fake_output.shape[-2]
        print("Spatial latent size is {}".format(fake_output.shape[1:]))
        
        #Q-Learning-Head
        self.head = MLPQHead(self.hidden_size, output_size, self.dqn_hidden_size, self.pixels)
        
        if self.jumps > 0:
            #Transition Model, falls Parameter jumps (k) > 0
            self.transition = TransitionModel(channels=self.hidden_size, 
                                                num_actions=output_size,
                                                pixels = self.pixels,
                                                hidden_size=self.hidden_size,
                                                limit = 1)
        
        #Target Encoder
        self.target_encoder = utils.EMA(model=self.o_encoder, decay=0.99)
        
        #Online Projection Head
        self.online_projection = nn.Sequential(nn.Linear(self.hidden_size, 
                                                            2*self.hidden_size),
                                                            nn.BatchNorm1d(2*self.hidden_size),
                                                            nn.ReLU(),
                                                            nn.Linear(2*self.hidden_size,
                                                            self.hidden_size))
        #Target Projection Head
        self.target_projection = self.online_projection
        
    def head_forward(self, latent):
        return self.head(latent)
    
    def o_encoder_forward(self, img):
        return self.o_encoder(img)
    
    def forward(self, observation, prev_action=None, prev_reward=None):
        #TODO Augmentation the img?
        pred_ps = []
        pred_reward = []
        pred_latents = []
        
        input_obs = observation[0].flatten(1, 2)
        #Berechne Latent Representation mit Online Encoder
        latent = self.o_encoder_forward(input_obs)
        pred_latents.append(latent)
        #Berechne Q Value von Q-Learning Head
        pred_ps.append(self.head_forward(latent))
        
        #Falls k > 0
        if self.jumps > 0:
            #Berechne zukünftige Reward von Prediction 
            pred_reward.append(self.dynamics_model.reward_predictor(pred_latents[0]))
            
            #für jeden k-Step in die Zukunft
            for j in range(1, self.jumps + 1):
                #predicte mit Transition Model die nächsten Latent State und Rewards
                latent, pred_rew = self.step(latent, prev_action[j])
                pred_rew = pred_rew[:observation.shape[1]]
                pred_latents.append(latent)
                pred_reward.append(pred_rew)
        
        #nicht ganz klar was model_rl sein soll?
        if self.model_rl > 0:
            for i in range(1, len(pred_latents)):
                pred_ps.append(self.head_forward(pred_latents[i]))
              

        pred_latents = torch.stack(pred_latents, 1)
        latents = pred_latents[:observation.shape[1]].flatten(0, 1)  # batch*jumps, *
        neg_latents = pred_latents[observation.shape[1]:].flatten(0, 1)
        #concat die jetzige latent representation und predited latent representation von Transition model
        latents = torch.cat([latents, neg_latents], 0)
        target_images = observation[self.time_offset:self.jumps + self.time_offset+1].transpose(0, 1).flatten(2, 3)
        #augmentation??
        #target_images = self.transform(target_images, True) 
        target_images = target_images[..., -1:, :, :]
        with torch.no_grad():
            #Berechne Latent representation von observation aus dem replay buffer mit target encoder
            target_latents = self.target_encoder(target_images.flatten(0, 1))
              
        
        online_latents = latents.flatten(-2, -1).permute(2, 0, 1)
        target_latents = target_latents.flatten(-2, -1).permute(2, 0, 1)
        #berechne latent representation mit online projection
        online_latents = self.online_projection(online_latents)
        
        with torch.no_grad():
            #berechne latent representation mit target projection
            target_latents = self.target_projection(target_latents)

        # online_latents = online_latents.view(-1,
        #                                    observation.shape[1],
        #                                    self.jumps+1,
        #                                    online_latents.shape[-1]).transpose(1, 2)
        # target_latents = target_latents.view(-1,
        #                                    observation.shape[1],
        #                                    self.jumps+1,
        #                                    target_latents.shape[-1]).transpose(1, 2)

        #normalisiere
        online_latents = F.normalize(online_latents.float(), p=2., dim=-1, eps=1e-3)
        target_latents = F.normalize(target_latents.float(), p=2., dim=-1, eps=1e-3)
        
        # Gradients of norrmalized L2 loss and cosine similiarity are proportional.
        # See: https://stats.stackexchange.com/a/146279 
        sprloss = F.mse_loss(online_latents, target_latents, reduction="none").sum(-1).mean(0)

        return pred_ps, pred_reward, sprloss
    
    #predicte die nächsten state und reward
    def step(self, state, action):
        next_state, reward = self.transition(state, action)
        return next_state, reward
        
class MLPQHead(nn.Module):
    def __init__(self, input_channel, output_size, hidden_size, pixels):
        super(MLPQHead, self).__init__()
        
        self.input = nn.Linear(input_channel * pixels, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.ReLU = nn.ReLU()
        self.flatten = nn.Flatten(-3, -1)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.input(x)
        x = self.ReLU(x)
        return self.output(x)

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels, affine=True),
        )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.ReLU(out)
        return out


class TransitionModel(nn.Module):
    def __init__(self, channels,
                 num_actions,
                 args=None,
                 blocks=16,
                 hidden_size=256,
                 pixels=36,
                 limit=300,
                 action_dim=6,
                 residual=False):
        super(TransitionModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.args = args
        self.residual = residual
        self.ReLU = nn.ReLU()
        layers = [nn.Conv2d(channels+num_actions, hidden_size, 3),
                  nn.ReLU(),
                  nn.BatchNorm2d(hidden_size, affine=True)]
        for _ in range(blocks):
            layers.append(ResidualBlock(hidden_size,
                                        hidden_size))
        layers.extend([nn.Conv2d(hidden_size, channels, 3)])

        self.action_embedding = nn.Embedding(num_actions, pixels*action_dim)

        self.network = nn.Sequential(*layers)
        self.reward_predictor = RewardPredictor(channels,
                                                pixels=pixels,
                                                limit=limit)
        
    def forward(self, x, action):
        batch_range = torch.arange(action.shape[0], device=action.device)
        action_onehot = torch.zeros(action.shape[0],
                                    self.num_actions,
                                    x.shape[-2],
                                    x.shape[-1],
                                    device=action.device)
        action_onehot[batch_range, action, :, :] = 1
        stacked_image = torch.cat([x, action_onehot], 1)
        next_state = self.network(stacked_image)
        if self.residual:
            next_state = next_state + x
        next_state = self.ReLU(next_state)
        next_reward = self.reward_predictor(next_state)
        return next_state, next_reward

class RewardPredictor(nn.Module):
    def __init__(self,
                 input_channels,
                 hidden_size=1,
                 pixels=36,
                 limit=300):
        super().__init__()
        self.hidden_size = hidden_size
        layers = [nn.Conv2d(input_channels, hidden_size, kernel_size=1, stride=1),
                  nn.ReLU(),
                  nn.BatchNorm2d(hidden_size, affine=True),
                  nn.Flatten(-3, -1),
                  nn.Linear(pixels*hidden_size, 256),
                  nn.ReLU(),
                  #nn.Linear(256, limit*2 + 1)]
                  nn.Linear(256, limit)]
        self.network = nn.Sequential(*layers)
        self.train()

    def forward(self, x):
        return self.network(x)
    
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