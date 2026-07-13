import numpy as np
import matplotlib.pyplot as plt
import os
from env.hedging_env import HedgingEnv
from agent.bs_baseline import BlackScholesBaselineAgent

def compuate_cvar(pnl:np.ndarray ,alpha:float = 0.05):
    threshold = np.quantile(pnl, alpha)
    tail = pnl[pnl <= threshold]
    return float(np.mean(tail))

def eval_baseline_agent(num_ep: int = 10_000, seed: int = 42):
    np.random.seed(seed)
    env = HedgingEnv()
    agent = BlackScholesBaselineAgent(env)

    pnls = []
    knock_out_count = 0

    for e in range(num_ep):
        res = agent.run_ep()
        pnls.append(res["total_pnl"])

        if res["knocked_out"]:
            knock_out_count += 1

        if (e + 1) % 1000 == 0:
            print(f"----- Episode {e + 1}/{num_ep} Completed -----")

    pnl = np.array(pnls)

    metrics = {
        "avg_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl)),
        "var_5": float(np.quantile(pnl, 0.05)),
        "cvar_5": float(compuate_cvar(pnl, alpha=0.05)),
        "min_pnl": float(np.min(pnl)),
        "max_pnl": float(np.max(pnl)),
        "knock_out_rate": float(knock_out_count / num_ep) * 100,
        "knock_out_count": knock_out_count,
    }

    return pnl, metrics

def plot_pnl_dist(pnl: np.ndarray, metrics: dict):
    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(pnl, bins=100, color="steelblue", alpha=0.7, edgecolor="none", label="Black-Scholes Baseline Agent P&L", density=True)

    ax.axvline(metrics["cvar_5"], color="red", linestyle="--", linewidth=2, label=f"CVaR 5%: {metrics['cvar_5']:.2f}")
    ax.axvline(metrics["var_5"], color="orange", linestyle="--", linewidth=2, label=f"5%: {metrics['var_5']:.2f}")
    ax.axvline(metrics["min_pnl"], color="black", linestyle=":", linewidth=1, label=f"Min P&L: {metrics['min_pnl']:.2f}")
    ax.axvline(metrics["avg_pnl"], color="black", linestyle=":", linewidth=1, label=f"Avg P&L: {metrics['avg_pnl']:.2f}")
    ax.axvline(metrics["max_pnl"], color="black", linestyle=":", linewidth=1, label=f"Max P&L: {metrics['max_pnl']:.2f}")

    ax.set_xlabel("P&L")
    ax.set_ylabel("Frequency")

    ax.set_title("Distribution of P&L for Black-Scholes Baseline Agent (10,000 episodes)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("plots/baseline_agent_pnl_dist.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved: plots/bs_baseline_pnl.png")

def print_metrics(metrics: dict):
    print(f"Average P&L: {metrics['avg_pnl']:.2f}")
    print(f"Standard Deviation: {metrics['std_pnl']:.2f}")
    print(f"CVaR 5%: {metrics['cvar_5']:.2f}")
    print(f"VaR 5%: {metrics['var_5']:.2f}")
    print(f"Min P&L: {metrics['min_pnl']:.2f}")
    print(f"Max P&L: {metrics['max_pnl']:.2f}")
    print(f"Knock-out Rate: {metrics['knock_out_rate']:.2f}%")

if __name__ == "__main__":
    pnl, metrics = eval_baseline_agent()
    plot_pnl_dist(pnl, metrics)
    print_metrics(metrics)