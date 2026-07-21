import numpy as np
import torch
from env.hedging_env import HedgingEnv
from agent.distributional_actor_critic import DistributionalActorCritic

np.random.seed(42)
torch.manual_seed(42)

env = HedgingEnv(
    S0=100.0, K=100.0, B=75.0, r=0.05, sigma=0.20,
    lam=1.5, mu_J=-0.04, sigma_J=0.08, T=1.0,
    num_steps=50, transaction_cost=0.001, rho=0.5,
)

device = torch.device("cpu")
net = DistributionalActorCritic(
    observation_dim = 4,
    action_dim      = 2,
    hidden_dim      = 64,
    num_quantiles   = 32,
    alpha_cvar      = 0.05,
)
net.load_state_dict(torch.load("results/phase3_rho00.pt", map_location=device))
net.eval()

obs, _ = env.reset()
total_reward = 0.0
terminated = False
step = 0

while not terminated:
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        trunk_out = net.trunk(obs_tensor)
        action = torch.sigmoid(net.actor_mean(trunk_out))
        action = action.clamp(0.0, 1.0)

    action_np = action.squeeze(0).cpu().numpy()
    obs, reward, terminated, truncated, info = env.step(action_np)
    total_reward += reward

    # Print every step
    print(f"Step {step:2d} | action={action_np} | reward={reward:.4f} | "
          f"S={env.prices[env.t]:.2f} | cumulative={total_reward:.4f}")
    step += 1

print(f"\nFinal P&L: {total_reward:.4f}")
print(f"Knocked out: {info['knocked_out']}")