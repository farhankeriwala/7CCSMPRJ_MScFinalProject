import torch
import numpy as np
from torch.optim import Adam
from agent.distributional_actor_critic import DistributionalActorCritic
from env.vec_hedging_env import VecHedgingEnv
from training.losses import quantile_huber_loss, cvar_advantage

class CvarPPOTrainer:
    def __init__(self, env: VecHedgingEnv, net:DistributionalActorCritic, lr: float = 1e-4, gamma: float = 1.0, gae_lambda : float = 0.95, epsilon_clip: float = 0.2, c_val:float = 1.0, c_entropy:float = 0.001, num_epochs: int = 10, batch_size:int = 64,
                 rollout_steps:int = 2048, kappa:float  = 1.0, device:str = None):

        self.env = env
        self.net = net
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon_clip = epsilon_clip
        self.c_val = c_val
        self.c_entropy = c_entropy
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.kappa = kappa

        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f'---------- Using device: {self.device} ----------')

        self.net.to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=self.lr)
        N = env.N
        observation_dim = env.observations_dim
        action_dim = env.actions_dim
        N_num_quantiles = net.num_quantiles

        # rollout buffers
        self.buffer_observations = torch.zeros(rollout_steps, N, observation_dim).to(self.device)
        self.buffer_actions = torch.zeros(rollout_steps, N, action_dim).to(self.device)
        self.buffer_log_probs = torch.zeros(rollout_steps, N).to(self.device)
        self.buffer_rewards = torch.zeros(rollout_steps, N).to(self.device)
        self.buffer_dones = torch.zeros(rollout_steps, N).to(self.device)

        self.buffer_quantiles = torch.zeros(rollout_steps, N, N_num_quantiles).to(self.device)


    def _compute_returns(self, rewards: torch.Tensor, dones: torch.Tensor, last_val: torch.Tensor) -> torch\
            .Tensor:
        """

        :param rewards:
        :param dones:
        :param last_val:
        :return:
        """


        returns = torch.zeros_like(rewards)
        bootstrap_val = last_val.clone()

        for t in reversed(range(self.rollout_steps)):
            bootstrap_val = rewards[t] + self.gamma * bootstrap_val * (1.0 - dones[t])
            returns[t] = bootstrap_val

        return returns

    def _collect_rollout(self, observations: torch.Tensor):
        """

        :param observations:
        :return:
        """


        ep_returns = []
        ep_buffer = torch.zeros(self.env.N, device=self.device)

        with torch.no_grad():
            for step in range(self.rollout_steps):
                self.buffer_observations[step] = observations

                action, log_prob,_, quantiles = self.net.get_action_and_value(observations)
                self.buffer_actions[step] = action
                self.buffer_log_probs[step] = log_prob
                self.buffer_quantiles[step] = quantiles

                observations_np, rewards_np, dones_np, _ = self.env.step(action.cpu().numpy())

                rewards = torch.tensor(rewards_np, dtype=torch.float32).to(self.device)
                dones = torch.tensor(dones_np, dtype=torch.float32).to(self.device)
                observations = torch.tensor(observations_np, dtype=torch.float32).to(self.device)

                self.buffer_rewards[step] = rewards
                self.buffer_dones[step] = dones

                ep_buffer += rewards

                if dones_np.all():
                    ep_returns.extend(ep_buffer.cpu().numpy().tolist())
                    ep_buffer = torch.zeros(self.env.N, device=self.device)
                    observations_np = self.env.reset()
                    observations = torch.tensor(observations_np, dtype=torch.float32).to(self.device)
            last_val = self.net.get_value(observations)
        return observations, last_val, ep_returns

    def _ppo_update(self, returns: torch.Tensor):
        """

        :param returns:
        :return:
        """


        observations_flat = self.buffer_observations.view(-1, self.env.observations_dim)
        actions_flat = self.buffer_actions.view(-1, self.env.actions_dim)
        log_probs_flat = self.buffer_log_probs.view(-1)
        returns_flat = returns.view(-1)
        quantiles_flat = self.buffer_quantiles.view(-1, self.net.num_quantiles)

        total_samples = observations_flat.shape[0]
        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(self.num_epochs):
            indices = torch.randperm(total_samples, device=self.device)

            for start in range(0, total_samples, self.batch_size):
                mb_i = indices[start:start + self.batch_size]
                mb_observations = observations_flat[mb_i]
                mb_actions = actions_flat[mb_i]
                mb_log_probs = log_probs_flat[mb_i]
                mb_quantiles = quantiles_flat[mb_i]
                mb_returns = returns_flat[mb_i]
                # normalise returns
                mb_returns_norm = (mb_returns - mb_returns.mean()) / (mb_returns.std() + 1e-8)

                # forward pass
                _, new_log_probs, entropy, new_quantiles = self.net.get_action_and_value(mb_observations, action=mb_actions)
                log_ratio = new_log_probs - mb_log_probs
                ratio  = log_ratio.exp()

                advantage = cvar_advantage(mb_returns, mb_quantiles, num_tail=self.net.num_tails)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * advantage

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss =  quantile_huber_loss(new_quantiles, mb_returns_norm, self.net.tau, kappa=self.kappa)

                entropy_loss = entropy.mean()

                loss = policy_loss + self.c_val * value_loss - self.c_entropy * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)

                self.optimizer.step()
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
        }

    def train(self, total_ep: int = 150000, stage_name: str = "lambda=1.5"):
        """

        :param total_ep:
        :param stage_name:
        :return:
        """


        observations_np = self.env.reset()
        observations = torch.tensor(observations_np, dtype=torch.float32).to(self.device)

        ep_completed = 0
        update_count = 0
        all_returns = []

        print(f"---------- PHASE 3: Starting training for {total_ep} episodes ----------")
        print(f"Target: {total_ep:,} episodes")
        print(f"Rollout: {self.rollout_steps} steps × "
              f"{self.env.N} envs = "
              f"{self.rollout_steps * self.env.N:,} transitions\n")

        while ep_completed < total_ep:
            observations, last_val, ep_returns = self._collect_rollout(observations)
            ep_completed += len(ep_returns)
            all_returns.extend(ep_returns)

            returns = self._compute_returns(self.buffer_rewards, self.buffer_dones, last_val)

            losses = self._ppo_update(returns)
            update_count += 1

            if update_count % 2 == 0:
                recent = all_returns[-1000:]
                mean_return = np.mean(recent)
                std_return = np.std(recent)

                arr = np.array(recent)
                threshold = np.quantile(arr, 0.05)
                rolling_cvar = float(np.mean(arr[arr <= threshold]))

                print(f"[{stage_name}] Update {update_count:3d} | "
                      f"Episodes {ep_completed:7,} | "
                      f"Mean: {mean_return:7.2f} | "
                      f"CVaR: {rolling_cvar:7.2f} | "
                      f"Entropy: {losses['entropy']:.4f} | "
                      f"VLoss: {losses['value_loss']:.4f}")
        print(f"---------- Training complete ----------")
        return all_returns

    def set_lam_override(self, lam:float):
        """

        :param lam:
        :return:
        """


        self.env.set_lam_override(lam)
        print(f"\nCurriculum: advancing to lambda={lam}")
