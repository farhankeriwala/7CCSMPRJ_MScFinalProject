import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

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

        # actor head outputs 4 values per dimension
        self.actor_head = nn.Linear(hidden_dim, action_dim*2)
        # scalar state value
        self.critic_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for l in self.trunk:
            if isinstance(l, nn.Linear):
                nn.init.orthogonal_(l.weight, gain=2**0.5)
                nn.init.zeros_(l.bias)

        # a small actor gain to make the initial actions more exploratory
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

        # standard critic gain
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def _get_beta_params(self, trunk_out: torch.Tensor):
        """
        A function to get the beta parameters from the actor head
        :param trunk_out: the output of the trunk
        :return: the alpha and beta parameters
        """

        # split into alpha and beta parameters for each action dimension
        raw = self.actor_head(trunk_out)
        raw_alpha, raw_beta = raw.chunk(2, dim=-1)

        alpha = F.softplus(raw_alpha) + 1.0
        beta = F.softplus(raw_beta) + 1.0

        return alpha, beta

    def get_val(self, observations: torch.Tensor):
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

        # get beta and alpha
        alpha, beta = self._get_beta_params(trunk_out)

        # contruct the beta distribution
        dist = Beta(alpha, beta)

        if action is None:
            # if there is no action, sample from the distribution
            action = dist.sample()

        # get the sum of the log probabilities across the action dimensions
        log_prob = dist.log_prob(action).sum(dim=-1)

        # sum entropy across the action dimensions
        entropy = dist.entropy().sum(dim=-1)

        # compute the critic
        val = self.critic_head(trunk_out).squeeze(-1)

        return action, log_prob, entropy, val