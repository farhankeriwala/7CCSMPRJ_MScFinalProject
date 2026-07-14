import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int = 4, action_dim: int = 2, hidden_dim:int = 64):
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # shared features used by both actor and critic
        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.actor_mean = nn.Linear(hidden_dim, action_dim)  # outputs mean in unconstrained space
        self.critic_head = nn.Linear(hidden_dim, 1)
        self.actor_log_std = nn.Linear(hidden_dim, action_dim)  # outputs log std

        self._init_weights()

    def _init_weights(self):
        for l in self.trunk:
            if isinstance(l, nn.Linear):
                nn.init.orthogonal_(l.weight, gain=2**0.5)
                nn.init.zeros_(l.bias)

        # a small actor gain to make the initial actions more exploratory
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)

        nn.init.orthogonal_(self.actor_log_std.weight, gain=0.01)
        nn.init.zeros_(self.actor_log_std.bias)

        # standard critic gain
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def get_value(self, observations: torch.Tensor):
        """

        :param observations: the observations to get the value for
        :return: the value of the observations bar the last dimension
        """


        return self.forward(observations).squeeze(-1)

    def forward(self, observations: torch.Tensor):
        """

        :param observations: the observations
        :return: the value estimate for rollout collection for critic updates
        """


        trunk_out = self.trunk(observations)
        return self.critic_head(trunk_out)

    def get_action_and_value(self, observations: torch.Tensor, action:torch.Tensor = None):
        """

        :param observations: the observations to get the action and value for
        :param action: the action to get the log probability for
        :return: the action, log probability, entropy and value estimate
        """

        trunk_out = self.trunk(observations)

        mean = torch.sigmoid(self.actor_mean(trunk_out))

        log_std = self.actor_log_std(trunk_out).clamp(-3.0, 3.0)
        std = log_std.exp()

        dist = Normal(mean, std)

        if action == None:
            action = dist.sample().clamp(0.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic_head(trunk_out).squeeze(-1)

        return action, log_prob, entropy, value