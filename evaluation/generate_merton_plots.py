import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.merton import MertonSimulator
from utils.plots import (
    plot_sample_paths,
    # plot_terminal_distribution,
    # plot_mc_convergence,
    # plot_jump_distribution,
)

PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    np.random.seed(42)

    sim = MertonSimulator()
    prices, jumps = sim.simulate_price_paths(num_paths=50_000)

    plot_sample_paths(prices[:25], sim.T, os.path.join(PLOTS_DIR, "merton_sample_paths.png"))
    # plot_terminal_distribution(prices[:, -1], sim, os.path.join(PLOTS_DIR, "merton_terminal_distribution.png"))
    # plot_mc_convergence(prices[:, -1], sim, K=100.0, output_path=os.path.join(PLOTS_DIR, "merton_mc_convergence.png"))
    # plot_jump_distribution(jumps, sim, os.path.join(PLOTS_DIR, "merton_jump_distribution.png"))


if __name__ == "__main__":
    main()