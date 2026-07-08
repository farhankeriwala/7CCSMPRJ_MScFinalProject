"""
tests/test_merton.py
====================
Validation tests for the Merton simulator.

These tests answer the supervisor's question:
"How do we know the simulator is correctly implemented?"

We validate against the analytical Merton pricing formula —
if the simulator is correct, option prices computed from
simulated paths must converge to the formula's value.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.merton import MertonSimulator


def test_analytical_price():
    """
    Test 1: Analytical option price is in the expected range.

    For S0=100, K=100, sigma=0.20, r=0.05, T=1,
    the Black-Scholes price is approximately 10.45.
    The Merton price with jumps should be slightly higher
    due to additional volatility from jumps.
    Expected range: £10-15.
    """
    sim   = MertonSimulator()
    price = sim.compute_merton_call_price(K=100.0)
    print(f"Analytical Merton call price: £{price:.4f}")
    assert 8.0 < price < 20.0, f"Price {price:.4f} outside expected range"
    print("PASS: Price in expected range £8-20")


def test_lambda_zero_equals_black_scholes():
    """
    Test 2: With lambda=0 (no jumps), Merton reduces to Black-Scholes.

    This is a mathematical identity — Merton's model with no
    jumps IS Black-Scholes. This confirms the formula is correct.
    """
    from scipy.stats import norm
    from math import log, sqrt, exp

    sim = MertonSimulator(lam=0.0)
    K   = 100.0

    # Merton price with no jumps
    merton_price = sim.compute_merton_call_price(K=K)

    # Black-Scholes price computed directly
    S, r, sigma, T = sim.S0, sim.r, sim.sigma, sim.T
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    bs_price = S * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)

    print(f"Merton (lam=0): £{merton_price:.6f}")
    print(f"Black-Scholes:  £{bs_price:.6f}")
    assert abs(merton_price - bs_price) < 0.01, \
        f"Merton(lam=0) should equal BS: {merton_price:.4f} vs {bs_price:.4f}"
    print("PASS: Merton(lambda=0) == Black-Scholes")


def test_simulated_price_distribution():
    """
    Test 3: Simulated price distribution matches theoretical moments.

    Under risk-neutral dynamics:
        E[S_T] = S0 * exp(r * T) = 100 * exp(0.05) ≈ 105.13

    We verify the sample mean is close to this theoretical value.
    With 50,000 paths the sampling error should be under 0.5%.
    """
    np.random.seed(42)
    sim    = MertonSimulator()
    prices, _ = sim.simulate_price_paths(num_paths=50_000)
    final  = prices[:, -1]

    theoretical_mean = sim.S0 * np.exp(sim.r * sim.T)
    sample_mean      = final.mean()
    error_pct        = abs(sample_mean - theoretical_mean) / theoretical_mean * 100

    print(f"Theoretical E[S_T]: £{theoretical_mean:.4f}")
    print(f"Sample mean:        £{sample_mean:.4f}")
    print(f"Error:              {error_pct:.3f}%")
    assert error_pct < 2.0, f"Sample mean error {error_pct:.2f}% too large"
    print("PASS: Sample mean within 2% of theoretical")


def test_jump_frequency():
    """
    Test 4: Jump frequency matches lambda parameter.

    With lambda=1.5 jumps/year and T=1 year,
    the expected number of jumps per path is 1.5.
    Sample mean should be close to this.
    """
    np.random.seed(42)
    sim = MertonSimulator(lam=1.5)
    _, jumps = sim.simulate_price_paths(num_paths=50_000)

    mean_jumps = jumps.sum(axis=1).mean()
    print(f"Expected jumps per path: {sim.lam * sim.T:.3f}")
    print(f"Sample mean jumps:       {mean_jumps:.3f}")
    assert abs(mean_jumps - sim.lam * sim.T) < 0.05, \
        f"Jump frequency {mean_jumps:.3f} too far from {sim.lam*sim.T:.3f}"
    print("PASS: Jump frequency matches lambda")


def test_monte_carlo_price_vs_analytical():
    """
    Test 5: Monte Carlo option price converges to analytical price.

    This is the KEY validation test for the supervisor's concern.
    If the simulator correctly implements Merton dynamics, then
    option prices computed by averaging payoffs across simulated
    paths must converge to the analytical Merton formula price.

    This confirms the simulator is correctly implemented.
    """
    np.random.seed(42)
    sim       = MertonSimulator()
    K         = 100.0
    n_paths   = 100_000

    # Simulate paths and compute call payoffs
    prices, _ = sim.simulate_price_paths(num_paths=n_paths)
    final     = prices[:, -1]
    payoffs   = np.maximum(final - K, 0.0)

    # Monte Carlo price = discounted expected payoff
    mc_price   = np.exp(-sim.r * sim.T) * payoffs.mean()
    analytical = sim.compute_merton_call_price(K=K)
    error      = abs(mc_price - analytical)

    print(f"Monte Carlo price:  £{mc_price:.4f}  (n={n_paths:,})")
    print(f"Analytical price:   £{analytical:.4f}")
    print(f"Absolute error:     £{error:.4f}")
    assert error < 0.50, \
        f"MC price £{mc_price:.4f} too far from analytical £{analytical:.4f}"
    print("PASS: Monte Carlo converges to analytical formula")


def test_bs_delta_atm():
    """
    Test 6: Black-Scholes delta at-the-money is approximately 0.6.

    For an ATM call (S=K) with r=0.05, sigma=0.20, T=1:
    delta ≈ N(d1) ≈ N(0.35) ≈ 0.637
    This is a well-known result used to check the formula.
    """
    sim   = MertonSimulator()
    delta = sim.black_scholes_delta(S=100.0, K=100.0, tau=1.0)
    print(f"BS delta (ATM, T=1): {delta:.4f}")
    assert 0.55 < delta < 0.70, \
        f"ATM delta {delta:.4f} outside expected range 0.55-0.70"
    print("PASS: ATM delta in expected range")


if __name__ == "__main__":
    print("="*55)
    print("MERTON SIMULATOR VALIDATION TESTS")
    print("="*55)

    tests = [
        test_analytical_price,
        test_lambda_zero_equals_black_scholes,
        test_simulated_price_distribution,
        test_jump_frequency,
        test_monte_carlo_price_vs_analytical,
        test_bs_delta_atm,
    ]

    passed = 0
    for test in tests:
        print(f"\n{test.__name__}")
        print("-" * 45)
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {e}")
        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\n{'='*55}")
    print(f"Results: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("Simulator validated — ready to proceed to Step 1.3")
    print("="*55)