import numpy as np
from scipy import stats

bs_pnl     = np.load("results/bs_baseline_pnl.npy")
p3_rho00   = np.load("results/phase3_pnl_rho00.npy")
p3_rho05   = np.load("results/phase3_pnl_rho05.npy")
p3_rho10   = np.load("results/phase3_pnl_rho10.npy")

for rho, p3_pnl in [("0.0", p3_rho00), ("0.5", p3_rho05), ("1.0", p3_rho10)]:

    # Phase 3 vs BS
    stat, p_two    = stats.wilcoxon(bs_pnl, p3_pnl, alternative="two-sided")
    _, p_bs_better = stats.wilcoxon(bs_pnl, p3_pnl, alternative="less")

    print(f"\nPhase 3 (ρ={rho}) vs BS Baseline:")
    print(f"  W={stat:.0f}")
    print(f"  p (two-sided):    {p_two:.6f}")
    print(f"  p (BS > Phase 3): {p_bs_better:.6f}")

    if p_bs_better < 0.05:
        print(f"  BS is significantly BETTER than Phase 3 (p < 0.05)")
    else:
        print(f"  No significant difference between BS and Phase 3")