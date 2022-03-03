from torch import nn
import torch
import numpy as np
import src.utils as utils
import torch.nn.functional as F

class SPRModel(nn.Module):
    def __init__(self, output_size, time_offset, image_shape, dqn_hidden_size, jumps,
                 model_rl, noisy_nets, noisy_nets_std, n_atoms, renormalize, distributional, momentum_tau):
        super(SPRModel, self).__init__()
        self.output_size = output_size
        self.time_offset = time_offset
        self.dqn_hidden_size = dqn_hidden_size
        self.jumps = jumps
        self.model_rl = model_rl
        self.noisy = noisy_nets
        self.num_actions = output_size
        self.renormalize = renormalize
        self.distributional = distributional
        self.momentum_tau = momentum_tau
        f, c = image_shape[:2]
        in_channels = np.prod(image_shape[:2])
        n_atoms = 1 if not self.distributional else n_atoms
        #Online Encoder
        self.o_encoder = OnlineEncoder(in_channels,
                                           out_channels = [32, 64, 64],
                                           kernel_sizes = [8, 4, 3],
                                           strides = [4, 2, 1],
                                           paddings = [0, 0, 0],
                                           nonlinearity= nn.ReLU,
                                           use_maxpool=True,
                                           dropout = 0.5)

        #print(image_shape)
        fake_input = torch.zeros(1, f*c, image_shape[0], image_shape[1])
        fake_output = self.o_encoder(fake_input)
        self.hidden_size = fake_output.shape[1]
        self.pixels = fake_output.shape[-1]*fake_output.shape[-2]
        print("Spatial latent size is {}".format(fake_output.shape[1:]))

        #Q-Learning-Head
        #self.head = MLPQHead(self.hidden_size, output_size, self.dqn_hidden_size, self.pixels)
        self.head = DQNDistributionalDuelingHeadModel(self.hidden_size,
                                                output_size,
                                                hidden_size=self.dqn_hidden_size,
                                                pixels=self.pixels,
                                                noisy=self.noisy,
                                                n_atoms=n_atoms,
                                                std_init=noisy_nets_std)

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
        
        for param in (list(self.target_encoder.parameters())
                            + list(self.target_projection.parameters())):
                    param.requires_grad = False

    def set_sampling(self, sampling):
        if self.noisy:
            self.head.set_sampling(sampling)
            
    def head_forward(self, latent, logits=False):
        lead_dim, T, B, img_shape = utils.infer_leading_dims(latent, 3)
        p = self.head(latent)

        if self.distributional:
            if logits:
                p = F.log_softmax(p, dim=-1)
            else:
                p = F.softmax(p, dim=-1)
        else:
            p = p.squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        p = utils.restore_leading_dims(p, lead_dim, T, B)
        return p


    def o_encoder_forward(self, img):
        """Returns the normalized output of convolutional layers."""
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = utils.infer_leading_dims(img, 3)
        conv_out = self.o_encoder(img.view(T * B, *img_shape))  # Fold if T dimension.
        if self.renormalize:
            conv_out = renormalize(conv_out, -3)
        return conv_out
        #return self.o_encoder(img)

    def forward(self, observation, prev_action=None, prev_reward=None):
        #TODO Augmentation the img?
        pred_ps = []
        pred_reward = []
        pred_latents = []

        observation = torch.tensor(observation)
        input_obs = observation[0].flatten(1,2)
        #Berechne Latent Representation mit Online Encoder
        latent = self.o_encoder_forward(input_obs)
        
        #Berechne Q Value von Q-Learning Head
        pred_ps.append(self.head_forward(latent, logits = True))
        pred_latents.append(latent)
        #Falls k > 0
        if self.jumps > 0:
            #Berechne zukünftige Reward von Prediction
            pred_rew = self.transition.reward_predictor(pred_latents[0])
            pred_reward.append(F.log_softmax(pred_rew, -1))

            #für jeden k-Step in die Zukunft
            for j in range(1, self.jumps + 1):
                #predicte mit Transition Model die nächsten Latent State und Rewards
                latent, pred_rew = self.step(latent, prev_action[j])
                pred_rew = pred_rew[:observation.shape[1]]
                pred_latents.append(latent)
                pred_reward.append(F.log_softmax(pred_rew, -1))

        #nicht ganz klar was model_rl sein soll?
        if self.model_rl > 0:
            for i in range(1, len(pred_latents)):
                pred_ps.append(self.head_forward(pred_latents[i], logits=True))
                                               
        #do_spr
        pred_latents = torch.stack(pred_latents, 1)
        latents = pred_latents[:observation.shape[1]].flatten(0, 1)  # batch*jumps, *
        neg_latents = pred_latents[observation.shape[1]:].flatten(0, 1)
        #concat die jetzige latent representation und predited latent representation von Transition model
        latents = torch.cat([latents, neg_latents], 0)
        target_images = observation[self.time_offset:self.jumps + self.time_offset+1].transpose(0, 1).flatten(2, 3)
        #augmentation??
        #target_images = self.transform(target_images, True)
        #target_images = target_images[..., -1:, :, :]
        with torch.no_grad():
            #Berechne Latent representation von observation aus dem replay buffer mit target encoder
            target_latents = self.target_encoder(target_images.flatten(0, 1))
            if self.renormalize:
                target_latents = renormalize(target_latents, -3)

        #local_spr_loss
        #berechne latent representation mit online projection
        online_latents = latents.flatten(-2, -1).permute(2, 0, 1)
        online_latents = self.online_projection(online_latents)
        
                    #berechne latent representation mit target projection

        with torch.no_grad():
            target_latents = target_latents.flatten(-2, -1).permute(2, 0, 1)
            target_latents = self.target_projection(target_latents)

        online_latents = online_latents.view(-1,
                                           observation.shape[1],
                                           self.jumps+1,
                                           online_latents.shape[-1]).transpose(1, 2)
        target_latents = target_latents.view(-1,
                                           observation.shape[1],
                                           self.jumps+1,
                                           target_latents.shape[-1]).transpose(1, 2)

        #spr_loss
        #normalisiere
        online_latents = F.normalize(online_latents.float(), p=2., dim=-1, eps=1e-3)
        target_latents = F.normalize(target_latents.float(), p=2., dim=-1, eps=1e-3)

        # Gradients of norrmalized L2 loss and cosine similiarity are proportional.
        # See: https://stats.stackexchange.com/a/146279
        sprloss = F.mse_loss(online_latents, target_latents, reduction="none").sum(-1).mean(0)

        spr_loss = spr_loss.view(-1, observation.shape[1])# split to batch, jumps
        
        
        #TARGET ENCODER AND TARGET PROJECTION UPDATE WITH EMA
        utils.update_state_dict(self.target_encoder,
                              self.o_encoder.state_dict(),
                              self.momentum_tau)

        utils.update_state_dict(self.target_projection,
                                self.online_projection.state_dict(),
                                self.momentum_tau)
                
        return pred_ps, pred_reward, sprloss

    #predicte die nächsten state und reward
    def step(self, state, action):
        next_state, reward = self.transition(state, action)
        return next_state, reward
    
    def select_action(self, obs):
        value = self.forward(obs, None, None, train=False, eval=True)

        if self.distributional:
            value = from_categorical(value, logits=False, limit=10)
        return value

def from_categorical(distribution, limit=300, logits=True):
    distribution = distribution.float()  # Avoid any fp16 shenanigans
    if logits:
        distribution = torch.softmax(distribution, -1)
    num_atoms = distribution.shape[-1]
    weights = torch.linspace(-limit, limit, num_atoms, device=distribution.device).float()
    return distribution @ weights

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.1, bias=True):
        super(NoisyLinear, self).__init__()
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.sampling = True
        self.noise_override = None
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.bias_sigma = nn.Parameter(torch.empty(out_features), requires_grad=bias)
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        if not self.bias:
            self.bias_mu.fill_(0)
            self.bias_sigma.fill_(0)
        else:
            self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        # Self.training alone isn't a good-enough check, since we may need to
        # activate .eval() during sampling even when we want to use noise
        # (due to batchnorm, dropout, or similar).
        # The extra "sampling" flag serves to override this behavior and causes
        # noise to be used even when .eval() has been called.
        if self.noise_override is None:
            use_noise = self.training or self.sampling
        else:
            use_noise = self.noise_override
        if use_noise:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)
        
class DQNDistributionalDuelingHeadModel(torch.nn.Module):
    """An MLP head with optional noisy layers which reshapes output to [B, output_size, n_atoms]."""

    def __init__(self,
                 input_channels,
                 output_size,
                 pixels=30,
                 n_atoms=51,
                 hidden_size=256,
                 grad_scale=2 ** (-1 / 2),
                 noisy=0,
                 std_init=0.1):
        super().__init__()
        if noisy:
            self.linears = [NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                            NoisyLinear(hidden_size, output_size * n_atoms, std_init=std_init),
                            NoisyLinear(pixels * input_channels, hidden_size, std_init=std_init),
                            NoisyLinear(hidden_size, n_atoms, std_init=std_init)
                            ]
        else:
            self.linears = [nn.Linear(pixels * input_channels, hidden_size),
                            nn.Linear(hidden_size, output_size * n_atoms),
                            nn.Linear(pixels * input_channels, hidden_size),
                            nn.Linear(hidden_size, n_atoms)
                            ]
        self.advantage_layers = [nn.Flatten(-3, -1),
                                 self.linears[0],
                                 nn.ReLU(),
                                 self.linears[1]]
        self.value_layers = [nn.Flatten(-3, -1),
                             self.linears[2],
                             nn.ReLU(),
                             self.linears[3]]
        self.advantage_hidden = nn.Sequential(*self.advantage_layers[:3])
        self.advantage_out = self.advantage_layers[3]
        self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms), requires_grad=True)
        self.value = nn.Sequential(*self.value_layers)
        self.network = self.advantage_hidden
        self._grad_scale = grad_scale
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        x = utils.scale_grad(input, self._grad_scale)
        advantage = self.advantage(x)
        value = self.value(x).view(-1, 1, self._n_atoms)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def advantage(self, input):
        x = self.advantage_hidden(input)
        x = self.advantage_out(x)
        x = x.view(-1, self._output_size, self._n_atoms)
        return x + self.advantage_bias

    def reset_noise(self):
        for module in self.linears:
            module.reset_noise()

    def set_sampling(self, sampling):
        for module in self.linears:
            module.sampling = sampling


class MLPQHead(nn.Module):
    def __init__(self, input_channel, output_size, hidden_size, pixels):
        super(MLPQHead, self).__init__()

        self.input = nn.Linear(input_channel * pixels, hidden_size)
        print(hidden_size)
        print(output_size)
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
            Conv2dSame(in_channels, out_channels, 3),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels, affine=True),
            Conv2dSame(out_channels, out_channels, 3),
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
        layers = [Conv2dSame(channels+num_actions, hidden_size, 3),
                  nn.ReLU(),
                  nn.BatchNorm2d(hidden_size, affine=True)]
        for _ in range(blocks):
            layers.append(ResidualBlock(hidden_size,
                                        hidden_size))
        layers.extend([Conv2dSame(hidden_size, channels, 3)])

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
        if self.renormalize:
            next_state = renormalize(next_state, 1)
        next_reward = self.reward_predictor(next_state)
        return next_state, next_reward

class Conv2dSame(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True,
                 stride=1,
                 padding_layer=nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias,
                            stride=stride, padding=ka)
        )

    def forward(self, x):
        return self.net(x)

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
    
def renormalize(tensor, first_dim=1):
    if first_dim < 0:
        first_dim = len(tensor.shape) + first_dim
    flat_tensor = tensor.view(*tensor.shape[:first_dim], -1)
    max = torch.max(flat_tensor, first_dim, keepdim=True).values
    min = torch.min(flat_tensor, first_dim, keepdim=True).values
    flat_tensor = (flat_tensor - min)/(max - min)

    return flat_tensor.view(*tensor.shape)
