import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class DistributionalActorCritic(nn.Module):
    def __init__(self, observation_dim: int = 4, action_dim: int = 2, hidden_dim:int = 64, num_quantiles: int  = 32, alpha_cvar:float = 0.05):
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
        # TODO: implement
        pass

    def get_quantiles(self, observations: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        pass

    def get_cvar(self, observations: torch.Tensor) -> torch.Tensor:
        # TODO: implement
        pass

    def get_action_and_value(self, observations: torch.Tensor, action:torch.Tensor = None):
        # TODO: implement
        pass

    def get_value(self, observations: torch.Tensor):
        # TODO: implement
        pass
