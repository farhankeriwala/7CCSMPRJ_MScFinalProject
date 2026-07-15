import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from env.hedging_env import HedgingEnv
from agent.distributional_actor_critic import DistributionalActorCritic
from evaluation.evaluate_baseline import eval_baseline_agent


def compute_cvar(pnl: np.ndarray, alpha: float = 0.05) -> float:
    """CVaR at level alpha — mean of worst alpha fraction of outcomes."""
    threshold = np.quantile(pnl, alpha)
    tail      = pnl[pnl <= threshold]
    return float(np.mean(tail))


def evaluate_phase3_agent(
    model_path: str   = "results/phase3_rho05.pt",
    num_episodes: int = 10_000,
    seed: int         = 42,
    rho: float        = 0.5,
) -> tuple:
    """
    Evaluate a trained Phase 3 agent deterministically.
    Uses sigmoid mean action — no sampling.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = HedgingEnv(
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
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = DistributionalActorCritic(
        observation_dim = 4,
        action_dim      = 2,
        hidden_dim      = 64,
        num_quantiles   = 32,
        alpha_cvar      = 0.05,
    )
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    pnl_list          = []
    knocked_out_count = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        terminated   = False

        while not terminated:
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                # Deterministic mean action — no sampling
                trunk_out = net.trunk(obs_tensor)
                action    = torch.sigmoid(net.actor_mean(trunk_out))
                action    = action.clamp(0.0, 1.0)

            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward

        pnl_list.append(total_reward)
        if info["knocked_out"]:
            knocked_out_count += 1

        if (ep + 1) % 1000 == 0:
            print(f"  Episode {ep + 1}/{num_episodes} complete")

    pnl = np.array(pnl_list)

    metrics = {
        "mean_pnl":      float(np.mean(pnl)),
        "std_pnl":       float(np.std(pnl)),
        "var_05":        float(np.quantile(pnl, 0.05)),
        "cvar_05":       compute_cvar(pnl, alpha=0.05),
        "min_pnl":       float(np.min(pnl)),
        "max_pnl":       float(np.max(pnl)),
        "knockout_rate": knocked_out_count / num_episodes,
    }

    return pnl, metrics


def plot_three_way_comparison(
    bs_pnl: np.ndarray,
    ppo_pnl: np.ndarray,
    phase3_pnl: np.ndarray,
    bs_metrics: dict,
    ppo_metrics: dict,
    phase3_metrics: dict,
    rho: float,
    save_path: str,
):
    """Three-way P&L distribution comparison."""
    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.hist(bs_pnl,     bins=100, alpha=0.5, color="steelblue",
            label="BS Baseline", density=True)
    ax.hist(ppo_pnl,    bins=100, alpha=0.4, color="darkorange",
            label="Standard PPO (Phase 2)", density=True)
    ax.hist(phase3_pnl, bins=100, alpha=0.5, color="green",
            label=f"CVaR-PPO Phase 3 (ρ={rho})", density=True)

    # CVaR lines
    ax.axvline(bs_metrics["cvar_05"],     color="steelblue",  linestyle="--",
               linewidth=1.5,
               label=f"BS CVaR: {bs_metrics['cvar_05']:.2f}")
    ax.axvline(ppo_metrics["cvar_05"],    color="darkorange", linestyle="--",
               linewidth=1.5,
               label=f"PPO CVaR: {ppo_metrics['cvar_05']:.2f}")
    ax.axvline(phase3_metrics["cvar_05"], color="green",      linestyle="--",
               linewidth=1.5,
               label=f"Phase 3 CVaR: {phase3_metrics['cvar_05']:.2f}")

    ax.set_xlabel("Total Episode P&L")
    ax.set_ylabel("Density")
    ax.set_title(
        f"P&L Distribution: BS Baseline vs Standard PPO vs CVaR-PPO (ρ={rho})"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def print_three_way_comparison(
    bs_metrics: dict,
    ppo_metrics: dict,
    phase3_metrics: dict,
    rho: float,
):
    print("\n" + "="*70)
    print(f"  Evaluation Results — Three-Way Comparison (ρ={rho})")
    print("="*70)
    print(f"{'Metric':<20} {'BS Baseline':>12} {'PPO Phase2':>12} "
          f"{'CVaR-PPO':>12} {'Δ vs PPO':>10}")
    print("-"*70)

    metrics_to_show = [
        ("Mean P&L",  "mean_pnl"),
        ("Std P&L",   "std_pnl"),
        ("VaR 5%",    "var_05"),
        ("CVaR 5%",   "cvar_05"),
        ("Min P&L",   "min_pnl"),
        ("Max P&L",   "max_pnl"),
    ]

    for label, key in metrics_to_show:
        bs_val     = bs_metrics[key]
        ppo_val    = ppo_metrics[key]
        p3_val     = phase3_metrics[key]
        delta_ppo  = p3_val - ppo_val
        print(f"{label:<20} {bs_val:>12.4f} {ppo_val:>12.4f} "
              f"{p3_val:>12.4f} {delta_ppo:>+10.4f}")

    print("-"*70)
    print(f"{'Knock-out rate':<20} "
          f"{bs_metrics['knockout_rate']:>11.2%} "
          f"{ppo_metrics['knockout_rate']:>12.2%} "
          f"{phase3_metrics['knockout_rate']:>12.2%}")
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys

    # Accept rho as command line argument
    rho     = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    rho_str = str(rho).replace(".", "")

    model_path  = f"results/phase3_rho{rho_str}.pt"
    ppo_pnl_path = "results/ppo_phase2_pnl.npy"

    # ------------------------------------------------------------------
    # BS Baseline
    # ------------------------------------------------------------------
    print("Running BS baseline evaluation...")
    bs_pnl, bs_raw_metrics = eval_baseline_agent(num_ep=10_000, seed=42)
    bs_metrics = {
        "mean_pnl":      bs_raw_metrics["avg_pnl"],
        "std_pnl":       bs_raw_metrics["std_pnl"],
        "var_05":        bs_raw_metrics["var_5"],
        "cvar_05":       bs_raw_metrics["cvar_5"],
        "min_pnl":       bs_raw_metrics["min_pnl"],
        "max_pnl":       bs_raw_metrics["max_pnl"],
        "knockout_rate": bs_raw_metrics["knock_out_rate"] / 100,
    }

    # ---------------------------