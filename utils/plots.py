"""
utils/plots.py
==============
Visualisation functions for the Merton simulator.

Each function produces a publication-ready figure for the
dissertation and saves it to the plots/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.merton import MertonSimulator

# Consistent style for all dissertation figures
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

PLOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "plots"
)
os.makedirs(PLOT_DIR, exist_ok=True)


# =============================================================
# FIGURE 1 — Sample Price Paths
# For: Chapter 4 (Section 4.2.1 Data / Environment)
# Purpose: Visually demonstrate jump discontinuities in Merton
#          dynamics compared to smooth GBM paths.
# =============================================================

def plot_price_paths(n_paths=10, seed=42):
    """
    Plot multiple Merton price paths showing jump discontinuities.

    This figure answers: "What does simulated Merton data look like?"
    The viewer should be able to see sudden vertical moves (jumps)
    that are absent in standard GBM simulations.
    """
    np.random.seed(seed)
    sim = MertonSimulator()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    time_axis = np.linspace(0, sim.T, sim.num_steps + 1)

    # --- Left panel: GBM (lambda=0) ---
    ax = axes[0]
    for i in range(n_paths):
        prices, _ = sim.simulate_price_path(lam_override=0.0)
        ax.plot(time_axis, prices, alpha=0.6, linewidth=0.8)

    ax.axhline(y=100, color="black", linestyle=":", alpha=0.3)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock Price")
    ax.set_title("GBM (λ=0, no jumps)")

    # --- Right panel: Merton (lambda=1.5) ---
    ax = axes[1]
    for i in range(n_paths):
        prices, jumps = sim.simulate_price_path()
        color = "steelblue"
        ax.plot(time_axis, prices, alpha=0.6, linewidth=0.8,
                color=color)

        # Mark jump locations with red dots
        jump_times = np.where(jumps)[0]
        for jt in jump_times:
            ax.plot(time_axis[jt+1], prices[jt+1], 'ro',
                    markersize=3, alpha=0.7)

    ax.axhline(y=100, color="black", linestyle=":", alpha=0.3)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock Price")
    ax.set_title(f"Merton (λ={sim.lam}, jumps marked ●)")

    fig.suptitle(
        "Simulated Price Paths: GBM vs Merton Jump-Diffusion",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "price_paths_gbm_vs_merton.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()


# =============================================================
# FIGURE 2 — Terminal Price Distribution
# For: Chapter 4 (Section 4.2.1)
# Purpose: Show the fat left tail produced by jumps,
#          motivating the need for CVaR-targeted hedging.
# =============================================================

def plot_terminal_distribution(n_paths=50_000, seed=42):
    """
    Compare the terminal price distributions under GBM vs Merton.

    This is the key motivation figure: the Merton distribution
    has a heavier left tail that VaR and CVaR must capture.
    """
    np.random.seed(seed)
    sim = MertonSimulator()

    # Simulate under both models
    prices_gbm, _    = sim.simulate_price_paths(
        num_paths=n_paths, lam_override=0.0
    )
    prices_merton, _ = sim.simulate_price_paths(
        num_paths=n_paths
    )

    final_gbm    = prices_gbm[:, -1]
    final_merton = prices_merton[:, -1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: Terminal price distributions ---
    ax = axes[0]
    bins = np.linspace(30, 250, 80)
    ax.hist(final_gbm, bins=bins, alpha=0.5, density=True,
            color="steelblue", label=f"GBM (λ=0)")
    ax.hist(final_merton, bins=bins, alpha=0.5, density=True,
            color="red", label=f"Merton (λ={sim.lam})")
    ax.set_xlabel("Terminal Stock Price S_T")
    ax.set_ylabel("Density")
    ax.set_title("Terminal Price Distribution")
    ax.legend()

    # --- Right: Log-return distributions ---
    ax = axes[1]
    log_ret_gbm    = np.log(final_gbm / sim.S0)
    log_ret_merton = np.log(final_merton / sim.S0)
    bins_lr = np.linspace(-1.0, 1.0, 80)

    ax.hist(log_ret_gbm, bins=bins_lr, alpha=0.5, density=True,
            color="steelblue", label="GBM")
    ax.hist(log_ret_merton, bins=bins_lr, alpha=0.5, density=True,
            color="red", label="Merton")
    ax.set_xlabel("Log-Return ln(S_T / S_0)")
    ax.set_ylabel("Density")
    ax.set_title("Log-Return Distribution")
    ax.legend()

    # Compute and annotate skewness and kurtosis
    from scipy.stats import skew, kurtosis
    sk_g = skew(log_ret_gbm)
    sk_m = skew(log_ret_merton)
    ku_g = kurtosis(log_ret_gbm)
    ku_m = kurtosis(log_ret_merton)

    ax.text(0.98, 0.95,
        f"GBM:    skew={sk_g:.3f}, kurt={ku_g:.3f}\n"
        f"Merton: skew={sk_m:.3f}, kurt={ku_m:.3f}",
        transform=ax.transAxes, fontsize=9,
        va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    fig.suptitle(
        "Fat Tails Under Jump-Diffusion Dynamics",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "terminal_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()

    # Print summary statistics for the report
    print(f"\nSummary Statistics ({n_paths:,} paths)")
    print(f"{'Metric':<25} {'GBM':>10} {'Merton':>10}")
    print("-" * 47)
    print(f"{'Mean S_T':<25} {final_gbm.mean():>10.4f}"
          f" {final_merton.mean():>10.4f}")
    print(f"{'Std S_T':<25} {final_gbm.std():>10.4f}"
          f" {final_merton.std():>10.4f}")
    print(f"{'Skewness (log-ret)':<25} {sk_g:>10.4f}"
          f" {sk_m:>10.4f}")
    print(f"{'Excess Kurtosis':<25} {ku_g:>10.4f}"
          f" {ku_m:>10.4f}")
    print(f"{'5th percentile S_T':<25} {np.percentile(final_gbm,5):>10.4f}"
          f" {np.percentile(final_merton,5):>10.4f}")


# =============================================================
# FIGURE 3 — Monte Carlo Convergence
# For: Chapter 4 (Section 4.3 Validation)
# Purpose: Show that MC price converges to analytical price,
#          proving the simulator is correctly implemented.
#          This directly addresses the supervisor's concern.
# =============================================================

def plot_mc_convergence(seed=42):
    """
    Plot Monte Carlo option price as a function of number of paths,
    showing convergence to the analytical Merton price.

    This is the validation figure that answers the question:
    "How do you know the simulator is correct?"
    """
    np.random.seed(seed)
    sim = MertonSimulator()
    K   = 100.0

    analytical = sim.compute_merton_call_price(K=K)

    # Simulate a large batch once, then compute running averages
    n_total   = 200_000
    prices, _ = sim.simulate_price_paths(num_paths=n_total)
    final     = prices[:, -1]
    payoffs   = np.maximum(final - K, 0.0)

    # Running MC estimate as n increases
    checkpoints = np.logspace(2, np.log10(n_total), 100).astype(int)
    checkpoints = np.unique(checkpoints)

    mc_prices = []
    for n in checkpoints:
        mc = np.exp(-sim.r * sim.T) * payoffs[:n].mean()
        mc_prices.append(mc)

    mc_prices = np.array(mc_prices)
    errors    = np.abs(mc_prices - analytical)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.semilogx(checkpoints, mc_prices, color="steelblue",
                linewidth=1.5, label="Monte Carlo estimate")
    ax.axhline(analytical, color="red", linestyle="--",
               linewidth=1.5, label=f"Analytical: {analytical:.4f}")
    ax.fill_between(checkpoints,
                    analytical - 0.5, analytical + 0.5,
                    alpha=0.1, color="red")
    ax.set_xlabel("Number of Simulated Paths")
    ax.set_ylabel("European Call Price")
    ax.set_title("Monte Carlo Price Convergence")
    ax.legend(fontsize=9)

    # --- Right: Absolute error ---
    ax = axes[1]
    ax.loglog(checkpoints, errors, color="steelblue",
              linewidth=1.5, label="Absolute error")
    # Theoretical 1/sqrt(n) convergence rate
    scale = errors[len(errors)//4] * np.sqrt(checkpoints[len(errors)//4])
    theoretical = scale / np.sqrt(checkpoints)
    ax.loglog(checkpoints, theoretical, color="red", linestyle="--",
              linewidth=1.0, label=r"$O(1/\sqrt{n})$ reference")
    ax.set_xlabel("Number of Simulated Paths")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Convergence Rate")
    ax.legend(fontsize=9)

    fig.suptitle(
        "Simulator Validation with Monte Carlo",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "simulator_validation_with_mc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()

    print(f"\nFinal MC price ({n_total:,} paths): "
          f"{mc_prices[-1]:.4f}")
    print(f"Analytical price: {analytical:.4f}")
    print(f"Final error: {errors[-1]:.4f}")


# =============================================================
# FIGURE 4 — Jump Impact on Hedging
# For: Chapter 2 / Chapter 5
# Purpose: Show WHY jumps make hedging harder — the P&L of a
#          delta-hedged position is much worse under Merton
#          than under GBM, motivating the RL approach.
# =============================================================

def plot_jump_impact_on_hedging(n_paths=10_000, seed=42):
    """
    Compare delta-hedge P&L under GBM vs Merton dynamics.

    Under GBM the BS delta hedge is near-perfect.
    Under Merton it fails because jumps cannot be hedged
    by continuous trading — this is the core gap the RL
    agent addresses.
    """
    np.random.seed(seed)
    sim = MertonSimulator()
    K   = 100.0

    def run_delta_hedge(lam_override, n):
        """Run n episodes of delta hedging and collect terminal P&L."""
        pnls = []
        for _ in range(n):
            prices, _ = sim.simulate_price_path(lam_override=lam_override)
            pnl = 0.0
            prev_delta = 0.0

            for t in range(sim.num_steps):
                S   = prices[t]
                tau = (sim.num_steps - t) * sim.dt

                # BS delta at current state
                delta = sim.black_scholes_delta(S=S, K=K, tau=tau)

                # P&L from holding delta shares over one step
                price_change = prices[t+1] - prices[t]
                pnl += delta * price_change

                # Transaction cost
                pnl -= 0.001 * abs(delta - prev_delta) * S
                prev_delta = delta

            # Terminal: subtract option payoff (agent is short the call)
            payoff = max(prices[-1] - K, 0.0)
            pnl -= payoff

            pnls.append(pnl)
        return np.array(pnls)

    print("Running delta hedge under GBM...")
    pnls_gbm    = run_delta_hedge(lam_override=0.0, n=n_paths)
    print("Running delta hedge under Merton...")
    pnls_merton = run_delta_hedge(lam_override=None, n=n_paths)

    # Compute CVaR
    def cvar(pnls, alpha=0.05):
        threshold = np.percentile(pnls, alpha * 100)
        return pnls[pnls <= threshold].mean()

    cvar_gbm    = cvar(pnls_gbm)
    cvar_merton = cvar(pnls_merton)

    fig, ax = plt.subplots(figsize=(10, 5))

    bins = np.linspace(
        min(pnls_gbm.min(), pnls_merton.min()),
        max(pnls_gbm.max(), pnls_merton.max()),
        80
    )
    ax.hist(pnls_gbm, bins=bins, alpha=0.5, density=True,
            color="steelblue", label="GBM (λ=0)")
    ax.hist(pnls_merton, bins=bins, alpha=0.5, density=True,
            color="red", label=f"Merton (λ={sim.lam})")

    ax.axvline(cvar_gbm, color="steelblue", linestyle="--",
               linewidth=2, label=f"CVaR 5% GBM: {cvar_gbm:.2f}")
    ax.axvline(cvar_merton, color="darkred", linestyle="--",
               linewidth=2, label=f"CVaR 5% Merton: {cvar_merton:.2f}")

    ax.set_xlabel("Delta-Hedging Terminal P&L")
    ax.set_ylabel("Density")
    ax.set_title(
        "Black-Scholes Delta Hedge P&L: GBM vs Merton"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "delta_hedge_pnl.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()

    print(f"\nDelta Hedge P&L Comparison ({n_paths:,} episodes)")
    print(f"{'Metric':<20} {'GBM':>10} {'Merton':>10}")
    print("-" * 42)
    print(f"{'Mean P&L':<20} {pnls_gbm.mean():>10.4f}"
          f" {pnls_merton.mean():>10.4f}")
    print(f"{'Std P&L':<20} {pnls_gbm.std():>10.4f}"
          f" {pnls_merton.std():>10.4f}")
    print(f"{'CVaR 5%':<20} {cvar_gbm:>10.4f}"
          f" {cvar_merton:>10.4f}")
    print(f"{'Min P&L':<20} {pnls_gbm.min():>10.4f}"
          f" {pnls_merton.min():>10.4f}")


# =============================================================
# RUN ALL PLOTS
# =============================================================

if __name__ == "__main__":
    print("Generating dissertation figures...\n")

    print("=" * 55)
    print("Figure 1: Price Paths (GBM vs Merton)")
    print("=" * 55)
    plot_price_paths()

    print("\n" + "=" * 55)
    print("Figure 2: Terminal Distribution")
    print("=" * 55)
    plot_terminal_distribution()

    print("\n" + "=" * 55)
    print("Figure 3: Monte Carlo Convergence (Validation)")
    print("=" * 55)
    plot_mc_convergence()

    print("\n" + "=" * 55)
    print("Figure 4: Jump Impact on Delta Hedging")
    print("=" * 55)
    plot_jump_impact_on_hedging()

    print("\nAll figures saved to plots/ directory.")