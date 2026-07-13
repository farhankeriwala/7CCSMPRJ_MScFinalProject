import numpy as np
import pytest
from env.vec_hedging_env import VecHedgingEnv


# Use a small N for most tests — faster, still confirms vectorisation
N = 256


@pytest.fixture
def env():
    """Fresh environment for each test."""
    return VecHedgingEnv(N=N)


# ------------------------------------------------------------------
# Test 1 — Reset returns correct shapes and bounded values
# ------------------------------------------------------------------

def test_reset_shape_and_bounds(env):
    obs = env.reset()

    # Observation must be (N, 4)
    assert obs.shape == (N, 4), f"Expected ({N}, 4), got {obs.shape}"

    # Normalised price S/S0 — should be exactly 1.0 at t=0
    assert np.allclose(obs[:, 0], 1.0), "S/S0 should be 1.0 at reset"

    # Time remaining — should be exactly 1.0 at t=0
    assert np.allclose(obs[:, 1], 1.0), "Time fraction should be 1.0 at reset"

    # Previous delta — should be 0.0 at reset
    assert np.allclose(obs[:, 2], 0.0), "delta_prev should be 0.0 at reset"

    # Sentiment signal — should be in [-1, 1]
    assert np.all(obs[:, 3] >= -1.0) and np.all(obs[:, 3] <= 1.0), \
        "Sentiment signal out of [-1, 1] range"


# ------------------------------------------------------------------
# Test 2 — Step returns correct shapes and types
# ------------------------------------------------------------------

def test_step_output_shapes(env):
    env.reset()

    # Random actions in [0, 1] for all N envs
    actions = np.random.uniform(0.0, 1.0, size=(N, 2)).astype(np.float32)
    obs, rewards, dones, info = env.step(actions)

    # Check shapes
    assert obs.shape     == (N, 4), f"obs shape wrong: {obs.shape}"
    assert rewards.shape == (N,),   f"rewards shape wrong: {rewards.shape}"
    assert dones.shape   == (N,),   f"dones shape wrong: {dones.shape}"

    # Check types
    assert obs.dtype     == np.float32, "obs should be float32"
    assert rewards.dtype == np.float32, "rewards should be float32"
    assert dones.dtype   == bool,       "dones should be bool"

    # Check info keys
    assert "pnl"         in info
    assert "transaction_cost"     in info
    assert "knocked_out" in info
    assert info["pnl"].shape == (N,)


# ------------------------------------------------------------------
# Test 3 — Episode terminates correctly at step 50
# ------------------------------------------------------------------

def test_episode_termination(env):
    env.reset()

    actions = np.random.uniform(0.0, 1.0, size=(N, 2)).astype(np.float32)

    # Steps 1 to 49 — dones should be all False
    for step in range(env.num_steps - 1):
        _, _, dones, _ = env.step(actions)
        assert not np.any(dones), \
            f"dones should be all False at step {step + 1}, got {dones.sum()} True"

    # Step 50 — dones should be all True
    _, _, dones, _ = env.step(actions)
    assert np.all(dones), \
        f"dones should be all True at step 50, got {dones.sum()} True out of {N}"


# ------------------------------------------------------------------
# Test 4 — Knock-out rate is approximately 15%
# ------------------------------------------------------------------

def test_knockout_rate(env):
    """
    Run 1000 full episodes and check that the knock-out rate
    is within 3 percentage points of the theoretical 15% target.
    B = 0.75 * S0 is calibrated to produce ~15% knock-out rate.
    """
    num_episodes = 1000
    knockout_counts = 0

    actions = np.zeros((N, 2), dtype=np.float32)  # zero actions — isolates KO logic

    for _ in range(num_episodes):
        env.reset()

        # Run full episode
        for _ in range(env.num_steps):
            _, _, dones, info = env.step(actions)

        # Count knocked-out envs at end of episode
        knockout_counts += info["knocked_out"].sum()

    total_episodes = num_episodes * N
    knockout_rate  = knockout_counts / total_episodes

    print(f"\nKnock-out rate: {knockout_rate:.2%} (target ~15%)")

    # Allow ±3 percentage points tolerance
    assert 0.12 <= knockout_rate <= 0.18, \
        f"Knock-out rate {knockout_rate:.2%} outside expected range [12%, 18%]"