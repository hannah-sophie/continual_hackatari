import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from src.CPB.cbp_conv import CBPConv
from src.CPB.cbp_linear import CBPLinear

from .common import Predictor, layer_init


class PPODefault(Predictor):
    def __init__(self, envs, device):
        super().__init__()
        self.device = device
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class PPObj(Predictor):
    def __init__(self, envs, device, encoder_dims=(128, 64), decoder_dims=(32,)):
        super().__init__()
        self.device = device

        dims = envs.observation_space.shape
        layers = nn.ModuleList()

        in_dim = dims[-1]

        for l in encoder_dims:
            layers.append(layer_init(nn.Linear(in_dim, l)))
            layers.append(nn.ReLU())
            in_dim = l
        layers.append(nn.Flatten())
        in_dim *= np.prod(dims[:-1], dtype=int)
        l = in_dim
        for l in decoder_dims:
            layers.append(layer_init(nn.Linear(in_dim, l)))
            layers.append(nn.ReLU())
            in_dim = l

        self.network = nn.Sequential(*layers)
        self.actor = layer_init(nn.Linear(l, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(l, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class PPO_Obj(Predictor):
    def __init__(self, envs, device):
        super().__init__()
        self.device = device

        self.network = nn.Sequential(
            layer_init(nn.Linear(4, 32)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


class PPO_CBP(Predictor):
    def __init__(
        self,
        envs,
        device,
        replacement_rate=0,
        init="default",
        maturity_threshold=100,
        decay_rate=0,
    ):
        super().__init__()
        self.device = device
        self.conv1 = layer_init(nn.Conv2d(4, 32, 8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, 3, stride=1))
        self.last_filter_output = 7 * 7
        self.num_conv_outputs = 64 * self.last_filter_output
        self.fct1 = layer_init(nn.Linear(self.num_conv_outputs, 512))
        self.cbp1 = CBPConv(
            in_layer=self.conv1,
            out_layer=self.conv2,
            device=device,
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            init=init,
            decay_rate=decay_rate,
        )
        self.cbp2 = CBPConv(
            in_layer=self.conv2,
            out_layer=self.conv3,
            device=device,
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            init=init,
            decay_rate=decay_rate,
        )
        self.cbp3 = CBPConv(
            in_layer=self.conv3,
            out_layer=self.fct1,
            device=device,
            num_last_filter_outputs=self.last_filter_output,
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            init=init,
            decay_rate=decay_rate,
        )

        self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.cbp_actor = CBPLinear(
            in_layer=self.fct1,
            out_layer=self.actor,
            device=device,
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            init=init,
            decay_rate=decay_rate,
        )
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.cbp_critic = CBPLinear(
            in_layer=self.fct1,
            out_layer=self.critic,
            device=device,
            replacement_rate=replacement_rate,
            maturity_threshold=maturity_threshold,
            init=init,
            decay_rate=decay_rate,
        )

        self.network = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.cbp1,
            self.conv2,
            nn.ReLU(),
            self.cbp2,
            self.conv3,
            nn.ReLU(),
            self.cbp3,
            nn.Flatten(),
            self.fct1,
            nn.ReLU(),

        )
        self.act = nn.ReLU()

    def get_value(self, x):
        x_output = self.network(x / 255.0)
        return self.critic(self.cbp_critic(x_output))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(self.cbp_actor(hidden))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(self.cbp_critic(hidden))


def shrink_perturb_agent_weights(agent, shrink_factor=0.7, noise_scale=0.01):
    """
    Shrink the weights of the agent's network by a specified factor and add noise.

    :param agent: The PPO agent whose weights are to be shrunk.
    :param shrink_factor: The factor by which to shrink the weights.
    :param noise_scale: The scale of the noise to be added to the weights.
    """
    with torch.no_grad():
        for param in agent.network.parameters():
            if param.requires_grad:
                param.mul_(shrink_factor)
                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)
        for param in agent.actor.parameters():
            if param.requires_grad:
                param.mul_(shrink_factor)
                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)
        for param in agent.critic.parameters():
            if param.requires_grad:
                param.mul_(shrink_factor)
                noise = torch.randn_like(param) * noise_scale
                param.add_(noise)
