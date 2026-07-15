import torch
import torch.nn as nn
from torch.distributions import Normal


class DistributionalActorCritic(nn.Module):
    def __init__(self, observation_dim: int = 4, action_dim: int = 2, hidden_dim: int = 64, num_quantiles: int = 32,
                 alpha_cvar: float = 0.05):
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_quantiles = num_quantiles
        self.alpha_cvar = alpha_cvar

        # number of quantiles in CVAR tail
        self.num_tails = max(1, int(alpha_cvar * num_quantiles))

        tau = torch.FloatTensor([
            (2 * i - 1) / (2 * num_quantiles) for i in range(1, num_quantiles + 1)
        ])
        self.register_buffer('tau', tau)

        # shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # actor head (guassian policy)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Linear(hidden_dim, action_dim)

        # distributional critic head
        self.critic_head = nn.Linear(hidden_dim, num_quantiles)

        self._init_weights()

    def _init_weights(self):
        """

        """
        for l in self.trunk:
            if isinstance(l, nn.Linear):
                nn.init.orthogonal_(l.weight, gain=2 ** 0.5)
                nn.init.zeros_(l.bias)

        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)

        nn.init.orthogonal_(self.actor_log_std.weight, gain=0.01)
        nn.init.constant_(self.actor_log_std.bias, -1.0)

        # critic head with standard gain
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def get_quantiles(self, observations: torch.Tensor) -> torch.Tensor:
        """
        compute the N quantiles for the given observations
        :param observations:
        :return: the distribution
        """


        trunk_out = self.trunk(observations)
        return self.critic_head(trunk_out)

    def get_cvar(self, observations: torch.Tensor) -> torch.Tensor:
        """
        computes the CVAR for the given observations
        :param observations:
        :return: cvar
        """


        quantiles = self.get_quantiles(observations)

        # sort quantiles into ascending order so that the lowest ones are the worst outcomes i.e. the left tail.
        sorted_quantiles, _ = torch.sort(quantiles, dim=-1)

        # cvar is the mean of the num_tails lowest quantiles
        cvar = sorted_quantiles[:, :self.num_tails].mean(dim=-1)

        return cvar

    def get_action_and_value(self, observations: torch.Tensor, action: torch.Tensor = None):
        """
        returns the action, log probability, entropy and quantiles for the given observations
        :param observations:
        :param action:
        :return: action, log probability, entropy and quantiles
        """
        trunk_out = self.trunk(observations)

        mean = torch.sigmoid(self.actor_mean(trunk_out))
        log_std = self.actor_log_std(trunk_out).clamp(-2.0, -0.7)
        std = log_std.exp()

        dist = Normal(mean, std)

        if action is None:
            action = dist.sample().clamp(0.0, 1.0)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        quantiles = self.critic_head(trunk_out)

        return action, log_prob, entropy, quantiles


    def get_value(self, observations: torch.Tensor):
        quantiles = self.get_quantiles(observations)
        return quantiles.mean(dim=-1)
