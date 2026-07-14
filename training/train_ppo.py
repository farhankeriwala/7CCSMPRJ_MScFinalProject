import os
import numpy as np
import torch
import matplotlib.pyplot as plt


from env.vec_hedging_env import VecHedgingEnv
from agent.actor_critic import ActorCritic
from training.ppo_trainer import PPOTrainer


def plot_training_curves(returns: list, save_path: str):
    """
    Plot smoothed episode returns over training.
    Uses a rolling mean with window=500 to show the learning trend
    without the noise of individual episode returns.
    """
    returns_arr = np.array(returns)
    window      = 500

    # Rolling mean — smooths out per-episode variance
    smoothed = np.convolve(
        returns_arr,
        np.ones(window) / window,
        mode="valid"   # only plot where full window is available
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # --- Top panel: raw returns + smoothed ---
    axes[0].plot(returns_arr, alpha=0.2, color="steelblue", linewidth=0.5,
                 label="Episode return")
    axes[0].plot(range(window - 1, len(returns_arr)), smoothed,
                 color="steelblue", linewidth=2.0,
                 label=f"Rolling mean (window={window})")
    axes[0].axhline(y=-9.00, color="red", linestyle="--", linewidth=1.5,
                    label="BS Baseline mean (-9.00)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total P&L")
    axes[0].set_title("PPO Training — Episode Returns")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Bottom panel: rolling CVaR at 5% ---
    # Computed over rolling window of 1000 episodes
    cvar_window = 1000
    cvar_values = []
    for i in range(cvar_window, len(returns_arr)):
        window_returns = returns_arr[i - cvar_window:i]
        threshold      = np.quantile(window_returns, 0.05)
        tail           = window_returns[window_returns <= threshold]
        cvar_values.append(np.mean(tail))

    axes[1].plot(range(cvar_window, len(returns_arr)), cvar_values,
                 color="red", linewidth=1.5, label="Rolling CVaR 5%")
    axes[1].axhline(y=-15.43, color="darkred", linestyle="--", linewidth=1.5,
                    label="BS Baseline CVaR (-15.43)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("CVaR 5%")
    axes[1].set_title("PPO Training — Rolling CVaR (window=1000 episodes)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves: {save_path}")


def main():
    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots",   exist_ok=True)

    # ------------------------------------------------------------------
    # Environment — agreed hyperparameters from dissertation spec
    # ------------------------------------------------------------------
    env = VecHedgingEnv(
        N          = 256,       # parallel environments
        S0         = 100.0,
        K          = 100.0,
        B          = 75.0,      # barrier at 0.75 * S0
        r          = 0.05,
        sigma      = 0.20,
        lam        = 1.5,       # Eraker et al. (2003) calibration
        mu_J       = -0.04,     # negative jumps
        sigma_J    = 0.08,
        T          = 1.0,
        num_steps  = 50,        # weekly rebalancing
        transaction_cost         = 0.001,     # transaction cost rate
        rho        = 0.5,       # sentiment signal quality
    )

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------
    net = ActorCritic(
        observation_dim    = env.observations_dim,   # 4
        action_dim    = env.actions_dim,   # 2
        hidden_dim = 64,
    )

    # ------------------------------------------------------------------
    # Trainer — agreed hyperparameters from dissertation spec
    # ------------------------------------------------------------------
    trainer = PPOTrainer(
        env          = env,
        net          = net,
        lr           = 3e-4,        # Adam learning rate
        gamma        = 1.0,         # finite horizon — no discounting
        gae_lambda   = 0.95,        # GAE lambda
        epsilon_clip = 0.2,         # PPO clip parameter
        c_val      = 0.5,         # value loss coefficient
        c_entropy    = 0.001,       # entropy bonus — reduced to prevent explosion
        num_epochs     = 10,          # PPO update epochs per rollout
        batch_size   = 64,          # minibatch size
        rollout_steps = 2048,       # steps per rollout
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  Phase 2 — Standard PPO Deep Hedging Agent")
    print("="*60)
    print(f"  Environments:    {env.N}")
    print(f"  Rollout steps:   {trainer.rollout_steps}")
    print(f"  Transitions/upd: {trainer.rollout_steps * env.N:,}")
    print(f"  Target episodes: 500,000")
    print(f"  Device:          {trainer.device}")
    print("="*60 + "\n")

    returns = trainer.train(total_ep=5000)

    # ------------------------------------------------------------------
    # Save model weights and training returns
    # ------------------------------------------------------------------
    torch.save(net.state_dict(), "results/ppo_phase2.pt")
    np.save("results/returns_phase2.npy", np.array(returns))
    print("\nSaved: results/ppo_phase2.pt")
    print("Saved: results/returns_phase2.npy")

    # ------------------------------------------------------------------
    # Training curves
    # ------------------------------------------------------------------
    plot_training_curves(returns, "plots/ppo_phase2_training.png")

    # ------------------------------------------------------------------
    # Quick summary statistics at end of training
    # ------------------------------------------------------------------
    returns_arr    = np.array(returns)
    final_returns  = returns_arr[-10_000:]   # last 10k episodes
    threshold      = np.quantile(final_returns, 0.05)
    tail           = final_returns[final_returns <= threshold]
    final_cvar     = float(np.mean(tail))

    print("\n" + "="*60)
    print("  Phase 2 Training Complete — Final 10,000 Episodes")
    print("="*60)
    print(f"  Mean P&L:   {np.mean(final_returns):>8.4f}  (BS: -9.00)")
    print(f"  Std P&L:    {np.std(final_returns):>8.4f}  (BS:  2.49)")
    print(f"  CVaR 5%:    {final_cvar:>8.4f}  (BS: -15.43)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()