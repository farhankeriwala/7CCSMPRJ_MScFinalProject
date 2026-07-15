import torch
import numpy as np
from torch.optim import Adam
from agent.distributional_actor_critic import DistributionalActorCritic
from env.vec_hedging_env import VecHedgingEnv
from training.losses import quantile_huber_loss, cvar_advantage

class CvarPPOTrainer:
    def __init__(self):
        pass

    def _compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor, last_val: torch.Tensor) -> torch\
            .Tensor:
        pass

    def _collect_rollout(self, observations: torch.Tensor):
        pass

    def _ppo_update(self, returns: torch.Tensor):
        pass

    def train(self, total_ep: int = 150000, stage_name: str = "lambda=1.5"):
        pass

    def set_lam_override(self, lam:float):
        pass