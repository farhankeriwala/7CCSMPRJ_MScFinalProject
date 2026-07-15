import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from env.hedging_env import HedgingEnv
from agent.actor_critic import ActorCritic
from env.vec_hedging_env import VecHedgingEnv


def compute_cvar(pnl: np.ndarray, alpha: float = 0.05) -> float:
    """CVaR at level alpha — mean of worst alpha fraction of outcomes."""
    threshold = np.quantile(pnl, alpha)
    tail      = pnl[pnl <= threshold]
    return float(np.mean(tail))


def evaluate_ppo(
    model_path: str    = "results/ppo_phase2.pt",
    num_episodes: int  = 10_000,
    seed: int          = 42,
) -> tuple:

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load environment — single env for evaluation, same params as training
    env = HedgingEnv(
        S0       = 100.0,
        K        = 100.0,
        B        = 75.0,
        r        = 0.05,
        sigma    = 0.20,
        lam      = 1.5,
        mu_J     = -0.04,
        sigma_J  = 0.08,
        T        = 1.0,
        num_steps = 50,
        transaction_cost       = 0.001,
        rho      = 0.5,
    )

    # Load trained network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net    = ActorCritic(observation_dim=4, action_dim=2, hidden_dim=64)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()   # disable dropout etc. — deterministic evaluation

    pnl_list          = []
    knocked_out_count = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        terminated   = False

        while not terminated:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                trunk_out = net.trunk(obs_tensor)
                action = torch.sigmoid(net.actor_mean(trunk_out))
                action = action.clamp(0.0, 1.0)

            # Debug first episode first step only
            if ep == 0:
                print(f"obs_tensor: {obs_tensor}")
                print(f"action: {action}")
                print(f"S0: {env.S0}, K: {env.K}, B: {env.B}")

            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, info = env.step(action_np)
            total_reward += reward

        pnl_list.append(total_reward)
        if info["knocked_out"]:
            knocked_out_count += 1

        if (ep + 1) % 1000 == 0:
            print(f"Episode {ep + 1}/{num_episodes} complete")

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


def plot_pnl_comparison(
    bs_pnl: np.ndarray,
    ppo_pnl: np.ndarray,
    bs_metrics: dict,
    ppo_metrics: dict,
):
    """Overlay histogram of BS baseline vs PPO P&L distributions."""
    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(bs_pnl,  bins=100, alpha=0.5, color="steelblue",
            label="BS Baseline", density=True)
    ax.hist(ppo_pnl, bins=100, alpha=0.5, color="darkorange",
            label="Standard PPO", density=True)

    # CVaR lines
    ax.axvline(bs_metrics["cvar_05"],  color="steelblue",  linestyle="--",
               linewidth=1.5, label=f"BS CVaR 5%: {bs_metrics['cvar_05']:.2f}")
    ax.axvline(ppo_metrics["cvar_05"], color="darkorange", linestyle="--",
               linewidth=1.5, label=f"PPO CVaR 5%: {ppo_metrics['cvar_05']:.2f}")

    # Mean lines
    ax.axvline(bs_metrics["mean_pnl"],  color="steelblue",  linestyle=":",
               linewidth=1.5, label=f"BS Mean: {bs_metrics['mean_pnl']:.2f}")
    ax.axvline(ppo_metrics["mean_pnl"], color="darkorange", linestyle=":",
               linewidth=1.5, label=f"PPO Mean: {ppo_metrics['mean_pnl']:.2f}")

    ax.set_xlabel("Total Episode P&L")
    ax.set_ylabel("Density")
    ax.set_title("P&L Distribution: Black-Scholes Baseline vs Standard PPO")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("plots/ppo_phase2_pnl.png", dpi=150)
    plt.close()
    print("Saved: plots/ppo_phase2_pnl.png")


def print_comparison(bs_metrics: dict, ppo_metrics: dict):
    print("\n" + "="*55)
    print("  Evaluation Results — BS Baseline vs Standard PPO")
    print("="*55)
    print(f"{'Metric':<20} {'BS Baseline':>12} {'PPO':>12} {'Delta':>10}")
    print("-"*55)

    metrics_to_show = [
        ("Mean P&L",      "mean_pnl"),
        ("Std P&L",       "std_pnl"),
        ("VaR 5%",        "var_05"),
        ("CVaR 5%",       "cvar_05"),
        ("Min P&L",       "min_pnl"),
        ("Max P&L",       "max_pnl"),
    ]

    for label, key in metrics_to_show:
        bs_val  = bs_metrics[key]
        ppo_val = ppo_metrics[key]
        delta   = ppo_val - bs_val
        print(f"{label:<20} {bs_val:>12.4f} {ppo_val:>12.4f} {delta:>+10.4f}")

    print("-"*55)
    print(f"{'Knock-out rate':<20} {bs_metrics['knockout_rate']:>11.2%} "
          f"{ppo_metrics['knockout_rate']:>11.2%}")
    print("="*55 + "\n")


if __name__ == "__main__":
    # Load BS baseline P&L from Phase 1
    # Re-run baseline evaluation to get matched P&L array
    from evaluation.evaluate_baseline import eval_baseline_agent as evaluate_baseline
    print("Running BS baseline evaluation for comparison...")
    bs_pnl, bs_metrics = evaluate_baseline(num_ep=10_000, seed=42)

    # Map baseline keys to standard names used by print_comparison
    bs_metrics = {
        "mean_pnl":      bs_metrics["avg_pnl"],
        "std_pnl":       bs_metrics["std_pnl"],
        "var_05":        bs_metrics["var_5"],
        "cvar_05":       bs_metrics["cvar_5"],
        "min_pnl":       bs_metrics["min_pnl"],
        "max_pnl":       bs_metrics["max_pnl"],
        "knockout_rate": bs_metrics["knock_out_rate"] / 100,  # convert % back to fraction
    }
    # Evaluate PPO agent
    print("\n Running PPO agent evaluation...")
    ppo_pnl, ppo_metrics = evaluate_ppo(
        model_path   = "results/ppo_phase2.pt",
        num_episodes = 10_000,
        seed         = 42,
    )

    # Save PPO P&L array for significance testing
    np.save("results/ppo_phase2_pnl.npy",      ppo_pnl)
    np.save("results/bs_baseline_pnl.npy",      bs_pnl)

    print_comparison(bs_metrics, ppo_metrics)
    plot_pnl_comparison(bs_pnl, ppo_pnl, bs_metrics, ppo_metrics)
