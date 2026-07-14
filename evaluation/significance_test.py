import numpy as np
from scipy import stats


def run_significance_test(
    bs_pnl_path:  str = "results/bs_baseline_pnl.npy",
    ppo_pnl_path: str = "results/ppo_phase2_pnl.npy",
    alpha: float      = 0.05,
):
    """
    Wilcoxon signed-rank test on paired BS vs PPO episode returns.

    Why Wilcoxon not t-test:
        P&L distributions under Merton dynamics are non-Gaussian with
        heavy left tails. Wilcoxon makes no normality assumption —
        it tests whether one distribution is systematically shifted
        relative to the other, which is the relevant question here.

    Both arrays must have the same length (same number of episodes,
    same random seed) to ensure pairing is valid.
    """
    bs_pnl  = np.load(bs_pnl_path)
    ppo_pnl = np.load(ppo_pnl_path)

    assert len(bs_pnl) == len(ppo_pnl), \
        f"Arrays must be same length: {len(bs_pnl)} vs {len(ppo_pnl)}"

    # Wilcoxon signed-rank test — two-sided
    # Tests H0: the two distributions are identical
    # H1: PPO distribution is shifted relative to BS
    statistic, p_value = stats.wilcoxon(bs_pnl, ppo_pnl, alternative="two-sided")

    print("\n" + "="*50)
    print("  Wilcoxon Signed-Rank Test")
    print("="*50)
    print(f"  N episodes:     {len(bs_pnl):,}")
    print(f"  W statistic:    {statistic:.4f}")
    print(f"  p-value:        {p_value:.6f}")
    print(f"  Significance:   alpha = {alpha}")
    print("-"*50)

    if p_value < alpha:
        print(f"  Result: SIGNIFICANT (p < {alpha})")
        print(f"  The PPO and BS P&L distributions differ")
        print(f"  significantly at the {alpha:.0%} level.")
    else:
        print(f"  Result: NOT SIGNIFICANT (p >= {alpha})")
        print(f"  Insufficient evidence to conclude the")
        print(f"  distributions differ at the {alpha:.0%} level.")

    print("="*50 + "\n")

    # Additional: one-sided test — is PPO worse than BS?
    # H1: PPO returns are systematically lower (worse) than BS
    stat_one, p_one = stats.wilcoxon(bs_pnl, ppo_pnl, alternative="greater")
    print(f"  One-sided test (BS > PPO): p = {p_one:.6f}")
    if p_one < alpha:
        print(f"  BS baseline is significantly BETTER than PPO (p < {alpha})")
    print()

    return {
        "statistic":   statistic,
        "p_value":     p_value,
        "p_one_sided": p_one,
        "significant": p_value < alpha,
        "n":           len(bs_pnl),
    }


if __name__ == "__main__":
    results = run_significance_test()