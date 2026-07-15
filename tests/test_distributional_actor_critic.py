import torch
import numpy as np
import pytest
from agent.distributional_actor_critic import DistributionalActorCritic


OBS_DIM    = 4
ACT_DIM    = 2
N_QUANTILES = 32
BATCH      = 256


@pytest.fixture
def net():
    return DistributionalActorCritic(
        observation_dim     = OBS_DIM,
        action_dim     = ACT_DIM,
        hidden_dim  = 64,
        num_quantiles = N_QUANTILES,
        alpha_cvar  = 0.05,
    )


def random_obs(batch=BATCH):
    return torch.rand(batch, OBS_DIM)


# ------------------------------------------------------------------
# Test 1 — get_quantiles returns correct shape
# ------------------------------------------------------------------

def test_get_quantiles_shape(net):
    obs      = random_obs()
    quantiles = net.get_quantiles(obs)

    # Must be (batch, N_QUANTILES)
    assert quantiles.shape == (BATCH, N_QUANTILES), \
        f"Expected ({BATCH}, {N_QUANTILES}), got {quantiles.shape}"
    assert quantiles.dtype == torch.float32


# ------------------------------------------------------------------
# Test 2 — get_cvar returns correct shape and is finite
# ------------------------------------------------------------------

def test_get_cvar_shape_and_finite(net):
    obs  = random_obs()
    cvar = net.get_cvar(obs)

    assert cvar.shape == (BATCH,), f"Expected ({BATCH},), got {cvar.shape}"
    assert torch.all(torch.isfinite(cvar)), "Non-finite CVaR detected"


# ------------------------------------------------------------------
# Test 3 — CVaR <= mean of quantiles (tail mean <= full mean)
# ------------------------------------------------------------------

def test_cvar_leq_mean(net):
    """
    CVaR is the mean of the worst quantiles so must be <= mean of all quantiles.
    This is a fundamental property of CVaR as a risk measure.
    """
    obs      = random_obs()
    quantiles = net.get_quantiles(obs)
    cvar     = net.get_cvar(obs)
    mean_q   = quantiles.mean(dim=-1)

    assert torch.all(cvar <= mean_q + 1e-5), \
        "CVaR must be <= mean of quantiles — coherence property violated"


# ------------------------------------------------------------------
# Test 4 — get_action_and_value returns correct shapes
# ------------------------------------------------------------------

def test_get_action_and_value_shapes(net):
    obs = random_obs()
    action, log_prob, entropy, quantiles = net.get_action_and_value(obs)

    assert action.shape    == (BATCH, ACT_DIM),    f"action shape wrong: {action.shape}"
    assert log_prob.shape  == (BATCH,),             f"log_prob shape wrong: {log_prob.shape}"
    assert entropy.shape   == (BATCH,),             f"entropy shape wrong: {entropy.shape}"
    assert quantiles.shape == (BATCH, N_QUANTILES), f"quantiles shape wrong: {quantiles.shape}"


# ------------------------------------------------------------------
# Test 5 — Actions bounded in [0, 1]
# ------------------------------------------------------------------

def test_actions_bounded(net):
    obs    = random_obs()
    action, _, _, _ = net.get_action_and_value(obs)

    assert torch.all(action >= 0.0), "Actions below 0 detected"
    assert torch.all(action <= 1.0), "Actions above 1 detected"


# ------------------------------------------------------------------
# Test 6 — get_value returns mean of quantiles
# ------------------------------------------------------------------

def test_get_value_equals_mean_quantiles(net):
    obs      = random_obs()
    value    = net.get_value(obs)
    quantiles = net.get_quantiles(obs)

    assert value.shape == (BATCH,), f"value shape wrong: {value.shape}"
    assert torch.allclose(value, quantiles.mean(dim=-1), atol=1e-5), \
        "get_value should return mean of quantiles"


# ------------------------------------------------------------------
# Test 7 — tau buffer has correct values
# ------------------------------------------------------------------

def test_tau_values(net):
    """
    Quantile midpoints tau_i = (2i-1)/(2N) for i=1,...,N.
    First value: (2*1-1)/(2*32) = 1/64 ≈ 0.0156
    Last value:  (2*32-1)/(2*32) = 63/64 ≈ 0.9844
    """
    assert net.tau.shape == (N_QUANTILES,)
    assert torch.isclose(net.tau[0],  torch.tensor(1  / (2 * N_QUANTILES)))
    assert torch.isclose(net.tau[-1], torch.tensor((2 * N_QUANTILES - 1) / (2 * N_QUANTILES)))


# ------------------------------------------------------------------
# Test 8 — n_tail is correct
# ------------------------------------------------------------------

def test_n_tail(net):
    """
    n_tail = max(1, floor(alpha * N)) = max(1, floor(0.05 * 32)) = max(1, 1) = 1
    """
    assert net.num_tails == max(1, int(0.05 * N_QUANTILES))


# ------------------------------------------------------------------
# Test 9 — Log probs finite and negative
# ------------------------------------------------------------------

def test_log_probs_finite(net):
    obs = random_obs()
    _, log_prob, entropy, _ = net.get_action_and_value(obs)

    assert torch.all(torch.isfinite(log_prob)), "Non-finite log_probs"
    assert torch.all(torch.isfinite(entropy)),  "Non-finite entropy"


# ------------------------------------------------------------------
# Test 10 — Gradients flow through quantiles and actor
# ------------------------------------------------------------------

def test_gradients_flow(net):
    obs = random_obs()
    action, log_prob, entropy, quantiles = net.get_action_and_value(obs)

    # Loss combining all outputs
    loss = -log_prob.mean() + quantiles.mean() + entropy.mean()
    loss.backward()

    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for: {name}"
        assert torch.all(torch.isfinite(param.grad)), \
            f"Non-finite gradient for: {name}"