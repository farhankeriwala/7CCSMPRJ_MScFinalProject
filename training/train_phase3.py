import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from env.vec_hedging_env import VecHedgingEnv
from agent.distributional_actor_critic import DistributionalActorCritic
from training.cvar_ppo_trainer import CvarPPOTrainer


def plot_training_curves(
    all_returns: list,
    stage_boundaries: list,
    rho: float,
    save_path: str,
):
    """
    Plot training curves with curriculum stage boundaries marked.
    stage_boundaries: list of episode counts where each stage ends
    """
    returns_arr = np.array(all_returns)
    window      = 500

    smoothed = np.convolve(
        returns_arr,
        np.ones(window) / window,
        mode="valid"
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # --- Top panel: returns ---
    axes[0].plot(returns_arr, alpha=0.15, color="steelblue", linewidth=0.5)
    axes[0].plot(
        range(window - 1, len(returns_arr)), smoothed,
        color="steelblue", linewidth=2.0,
        label=f"Rolling mean (window={window})"
    )
    axes[0].axhline(
        y=-9.00, color="red", linestyle="--",
        linewidth=1.5, label="BS Baseline mean (-9.00)"
    )

    # Mark curriculum stage boundaries
    colours = ["orange", "green"]
    labels  = ["λ=0.5→1.0", "λ=1.0→1.5"]
    for i, boundary in enumerate(stage_boundaries):
        axes[0].axvline(
            x=boundary, color=colours[i], linestyle=":",
            linewidth=1.5, label=labels[i]
        )

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total P&L")
    axes[0].set_title(f"Phase 3 CVaR-PPO Training — ρ={rho}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # --- Bottom panel: rolling CVaR ---
    cvar_window = 1000
    cvar_values = []
    for i in range(cvar_window, len(returns_arr)):
        w         = returns_arr[i - cvar_window:i]
        threshold = np.quantile(w, 0.05)
        cvar_values.append(float(np.mean(w[w <= threshold])))

    axes[1].plot(
        range(cvar_window, len(returns_arr)), cvar_values,
        color="red", linewidth=1.5, label="Rolling CVaR 5%"
    )
    axes[1].axhline(
        y=-15.43, color="darkred", linestyle="--",
        linewidth=1.5, label="BS Baseline CVaR (-15.43)"
    )
    axes[1].axhline(
        y=-73.02, color="darkorange", linestyle="--",
        linewidth=1.5, label="PPO Phase 2 CVaR (-73.02)"
    )

    for i, boundary in enumerate(stage_boundaries):
        axes[1].axvline(
            x=boundary, color=colours[i], linestyle=":",
            linewidth=1.5
        )

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("CVaR 5%")
    axes[1].set_title("Phase 3 — Rolling CVaR (window=1,000 episodes)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def run_phase3(rho: float = 0.5):
    """
    Full Phase 3 training run with curriculum learning.

    Curriculum stages:
        Stage 1: λ=0.5  — easy, few jumps    — 150,000 episodes
        Stage 2: λ=1.0  — medium             — 150,000 episodes
        Stage 3: λ=1.5  — target calibration — 150,000 episodes

    Actor weights are retained across stages.
    Critic is reinitialised at each stage transition to avoid
    quantile estimates calibrated to wrong jump intensity.
    """
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs("results", exist_ok=True)
    os.makedirs("plots",   exist_ok=True)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    env = VecHedgingEnv(
        N                = 256,
        S0               = 100.0,
        K                = 100.0,
        B                = 75.0,
        r                = 0.05,
        sigma            = 0.20,
        lam              = 1.5,
        mu_J             = -0.04,
        sigma_J          = 0.08,
        T                = 1.0,
        num_steps        = 50,
        transaction_cost = 0.001,
        rho              = rho,
        lam_override     = 0.5,    # start at easy stage
    )

    # ------------------------------------------------------------------
    # Network — use your actual parameter names
    # ------------------------------------------------------------------
    net = DistributionalActorCritic(
        observation_dim = env.observations_dim,
        action_dim      = env.actions_dim,
        hidden_dim      = 64,
        num_quantiles   = 32,
        alpha_cvar      = 0.05,
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    def make_trainer(env, net):
        return CvarPPOTrainer(
            env           = env,
            net           = net,
            lr            = 1e-4,
            gamma         = 1.0,
            gae_lambda    = 0.95,
            epsilon_clip  = 0.2,
            c_val         = 1.0,
            c_entropy     = 0.01,
            num_epochs    = 10,
            batch_size    = 64,
            rollout_steps = 2048,
            kappa         = 1.0,
        )

    trainer = make_trainer(env, net)

    # ------------------------------------------------------------------
    # Curriculum training
    # ------------------------------------------------------------------
    all_returns      = []
    stage_boundaries = []

    print("\n" + "="*60)
    print(f"  Phase 3 — CVaR-PPO with Curriculum  (ρ={rho})")
    print("="*60)

    # Stage 1 — λ=0.5
    print("\nStage 1: λ=0.5 (easy)")
    env.set_lam_override(0.5)
    returns_1 = trainer.train(total_ep=150_000, stage_name="λ=0.5")
    all_returns.extend(returns_1)
    stage_boundaries.append(len(all_returns))

    # Stage transition — reinitialise critic only, keep actor
    print("\nReinitialising critic for stage transition...")
    _reinit_critic(net)

    # Stage 2 — λ=1.0
    print("\nStage 2: λ=1.0 (medium)")
    env.set_lam_override(1.0)
    returns_2 = trainer.train(total_ep=150_000, stage_name="λ=1.0")
    all_returns.extend(returns_2)
    stage_boundaries.append(len(all_returns))

    # Stage transition — reinitialise critic only
    print("\nReinitialising critic for stage transition...")
    _reinit_critic(net)

    # Stage 3 — λ=1.5 (target)
    print("\nStage 3: λ=1.5 (target)")
    env.set_lam_override(1.5)
    returns_3 = trainer.train(total_ep=150_000, stage_name="λ=1.5")
    all_returns.extend(returns_3)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    rho_str      = str(rho).replace(".", "")
    model_path   = f"results/phase3_rho{rho_str}.pt"
    returns_path = f"results/returns_phase3_rho{rho_str}.npy"

    torch.save(net.state_dict(), model_path)
    np.save(returns_path, np.array(all_returns))
    print(f"\nSaved: {model_path}")
    print(f"Saved: {returns_path}")

    # ------------------------------------------------------------------
    # Training curves
    # ------------------------------------------------------------------
    plot_path = f"plots/phase3_training_rho{rho_str}.png"
    plot_training_curves(all_returns, stage_boundaries, rho, plot_path)

    # ------------------------------------------------------------------
    # Final summary — safe slice in case fewer than 10k episodes
    # ------------------------------------------------------------------
    n_final       = min(10_000, len(all_returns))
    returns_arr   = np.array(all_returns[-n_final:])
    threshold     = np.quantile(returns_arr, 0.05)
    final_cvar    = float(np.mean(returns_arr[returns_arr <= threshold]))

    print("\n" + "="*60)
    print(f"  Phase 3 Complete (ρ={rho}) — Final {n_final:,} Episodes")
    print("="*60)
    print(f"  Mean P&L:  {np.mean(returns_arr):>8.4f}  (BS: -9.00)")
    print(f"  Std P&L:   {np.std(returns_arr):>8.4f}  (BS:  2.49)")
    print(f"  CVaR 5%:   {final_cvar:>8.4f}  (BS: -15.43)")
    print("="*60 + "\n")

    # Colab download helper
    try:
        from google.colab import files
        files.download(model_path)
        files.download(returns_path)
        print("Files downloaded via Colab.")
    except ImportError:
        pass

    return all_returns, net


def _reinit_critic(net: DistributionalActorCritic):
    """
    Reinitialise critic head weights only — actor weights are retained.

    Bengio et al. (2009): when moving to a harder task, the value
    function must be recalibrated but the policy can transfer.
    """
    torch.nn.init.orthogonal_(net.critic_head.weight, gain=1.0)
    torch.nn.init.zeros_(net.critic_head.bias)
    print("Critic head reinitialised — actor weights retained.")


if __name__ == "__main__":
    import sys
    rho = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    print(f"Running Phase 3 with ρ={rho}")
    run_phase3(rho=rho)