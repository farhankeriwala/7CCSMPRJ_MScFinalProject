import torch
import numpy as np
from torch.optim import Adam
from agent.actor_critic import ActorCritic
from env.vec_hedging_env import VecHedgingEnv

class PPOTrainer:
    """
    A custom PPO trainer for deep hedging under Merton dynamics.
    """


    def __init__(self,
                 env: VecHedgingEnv,
                 net: ActorCritic,
                 lr:float = 1e-4,
                 gamma: float = 1.0,
                 gae_lambda: float = 0.95,
                 epsilon_clip:float = 0.2,
                 c_val: float = 1.0,
                 c_entropy: float = 0.001,
                 num_epochs:int = 10,
                 batch_size: int = 64,
                 rollout_steps:int = 2048,
                 device:str = None):
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

        if device is not None:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"---------- Using device: {self.device} ----------")

        self.net.to(self.device)
        self.optimizer = Adam(self.net.parameters(), lr=self.lr)
        N = self.env.N
        observation_dim = env.observations_dim
        action_dim = env.actions_dim

        # pre allocating the rollout buffers
        self.buffer_observations = torch.zeros(rollout_steps, N, observation_dim).to(self.device)
        self.buffer_actions = torch.zeros(rollout_steps, N, action_dim).to(self.device)
        self.buffer_log_probs = torch.zeros(rollout_steps, N).to(self.device)
        self.buffer_rewards = torch.zeros(rollout_steps, N).to(self.device)
        self.buffer_dones = torch.zeros(rollout_steps, N).to(self.device)
        self.buffer_vals = torch.zeros(rollout_steps, N).to(self.device)

    def _compute_gae(self, rewards: torch.Tensor, vals: torch.Tensor, dones: torch.Tensor, last_val: torch.Tensor):
        """
        based on the Generalised Advantage Estimation (Schulman et al. 2015)
        :param rewards:
        :param vals:
        :param dones:
        :param last_val:
        :return:
        """

        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(self.env.N, device=self.device)

        # backwards iteration for standard GAR recurrence
        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = last_val
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_val = vals[t + 1]

            # TD error
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - vals[t]

            # GAE recurrence
            last_gae = delta + self.gamma *  self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns  = advantages + vals
        return advantages, returns

    def _collect_rollout(self, observations: torch.Tensor):
        ep_returns = []
        ep_buffer = torch.zeros(self.env.N, device=self.device)

        with torch.no_grad():
            for step in range(self.rollout_steps):

                self.buffer_observations[step] = observations

                action, log_prob, _, val = self.net.get_action_and_value(observations)
                self.buffer_actions[step] = action
                self.buffer_log_probs[step] = log_prob
                self.buffer_vals[step] = val

                observations_np, rewards_np, dones_np, _ = self.env.step(action.cpu().numpy())

                rewards = torch.tensor(rewards_np, dtype=torch.float32).to(self.device)
                dones = torch.tensor(dones_np, dtype=torch.float32).to(self.device)
                observations = torch.tensor(observations_np, dtype=torch.float32).to(self.device)

                self.buffer_rewards[step] = rewards
                self.buffer_dones[step] = dones

                # # Debug — print diagnostics on first step of first rollout only
                # if step == 0:
                #     print(f"\n--- Rollout Debug (step 0) ---")
                #     print(f"Action mean:  {action.mean().item():.4f}")
                #     print(f"Action std:   {action.std().item():.4f}")
                #     print(f"Action min:   {action.min().item():.4f}")
                #     print(f"Action max:   {action.max().item():.4f}")
                #     print(f"Reward mean:  {rewards.mean().item():.4f}")
                #     print(f"Reward std:   {rewards.std().item():.4f}")
                #     print(f"Value mean:   {val.mean().item():.4f}")
                #     print(f"Log prob mean:{log_prob.mean().item():.4f}")
                #     print(f"------------------------------\n")

                ep_buffer += rewards

                if dones_np.all():
                    ep_returns.extend(ep_buffer.cpu().numpy().tolist())
                    ep_buffer = torch.zeros(self.env.N, device=self.device)
                    observations_np = self.env.reset()
                    observations = torch.tensor(observations_np, dtype=torch.float32).to(self.device)

            last_val = self.net.get_value(observations)
        return observations, last_val, ep_returns

    def _ppo_update(self, advantages: torch.Tensor, returns: torch.Tensor):
        """
        this function will update the actor-critic network using PPO
        :param advantages:
        :param returns:
        :return: metrics of the update
        """
        observations_flat = self.buffer_observations.view(-1, self.env.observations_dim)
        actions_flat = self.buffer_actions.view(-1, self.env.actions_dim)
        log_probs_flat = self.buffer_log_probs.view(-1)
        advantages_flat = advantages.view(-1)
        returns_flat = returns.view(-1)

        # normalize the advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        total_samples = observations_flat.shape[0]

        policy_losses = []
        entropy_losses = []
        value_losses = []

        for _ in range(self.num_epochs):
            # random permutations for batch sampling
            indices = torch.randperm(total_samples, device=self.device)

            for start in range(0, total_samples, self.batch_size):
                mb_i = indices[start:start + self.batch_size]
                mb_observations = observations_flat[mb_i]
                mb_actions = actions_flat[mb_i]
                mb_log_probs = log_probs_flat[mb_i]
                mb_advantages = advantages_flat[mb_i]
                mb_returns = returns_flat[mb_i]
                mb_returns = (mb_returns - mb_returns.mean()) / (mb_returns.std() + 1e-8)

                # evaluate actions under current pi
                _, new_log_probs, entropy, new_vals = self.net.get_action_and_value(mb_observations, action=mb_actions)
                log_ratio = new_log_probs - mb_log_probs
                ratio  = log_ratio.exp()

                ## ppo clipped surrogate
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * ((new_vals - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + self.c_val * value_loss - self.c_entropy * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

                # append the losses and return the dict of info
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
        }


    def train(self, total_ep: int = 500000):
        observations_np = self.env.reset()
        observations = torch.tensor(observations_np, dtype=torch.float32).to(self.device)

        completed_ep = 0
        update_count = 0
        all_returns = []

        print(f"---------- Starting training for {total_ep} episodes ----------")
        print(f"---------- Using device: {self.device} ----------")
        print(f"Rollout size: {self.rollout_steps} steps × {self.env.N} envs "
              f"= {self.rollout_steps * self.env.N:,} transitions per update\n")

        while completed_ep < total_ep:

            # rollout collection
            observations, last_val, ep_returns = self._collect_rollout(observations)
            completed_ep += len(ep_returns)
            all_returns.extend(ep_returns)

            advantages, returns = self._compute_gae(self.buffer_rewards, self.buffer_vals, self.buffer_dones, last_val)

            losses = self._ppo_update(advantages, returns)
            update_count += 1

            # log updates
            if update_count % 2 == 0:
                recent_returns = all_returns[-1000:]  # last 1000 episodes
                mean_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)

                print(f"Update {update_count:4d} | "
                      f"Episodes {completed_ep:7,} | "
                      f"Mean Return: {mean_return:7.2f} | "
                      f"Std: {std_return:6.2f} | "
                      f"Policy Loss: {losses['policy_loss']:7.4f} | "
                      f"Value Loss: {losses['value_loss']:7.4f} | "
                      f"Entropy: {losses['entropy']:6.4f}")
        print(f"---------- Training complete ----------")
        return all_returns