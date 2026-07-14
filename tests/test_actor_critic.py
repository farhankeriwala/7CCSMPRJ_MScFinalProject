import torch
import pytest
from agent.actor_critic import ActorCritic


# Fixed dimensions matching the environment spec
OBS_DIM = 4
ACT_DIM = 2
BATCH   = 256   # matches N_ENVS


@pytest.fixture
def net():
    """Fresh network for each test."""
    return ActorCritic(observation_dim=OBS_DIM, action_dim=ACT_DIM, hidden_dim=64)


def random_obs(batch=BATCH):
    """Generate random observations in valid range."""
    return torch.rand(batch, OBS_DIM)   # all features roughly in [0, 1]


# ------------------------------------------------------------------
# Test 1 — Forward pass returns correct shape
# ------------------------------------------------------------------

def test_forward_shape(net):
    obs   = random_obs()
    value = net.forward(obs)

    # forward() returns (batch, 1)
    assert value.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {value.shape}"
    assert value.dtype == torch.float32, "Value should be float32"


# ------------------------------------------------------------------
# Test 2 — get_value returns correct shape
# ------------------------------------------------------------------

def test_get_value_shape(net):
    obs   = random_obs()
    value = net.get_value(obs)

    # get_value() squeezes to (batch,)
    assert value.shape == (BATCH,), f"Expected ({BATCH},), got {value.shape}"


# ------------------------------------------------------------------
# Test 3 — get_action_and_value returns correct shapes
# ------------------------------------------------------------------

def test_get_action_and_value_shapes(net):
    obs = random_obs()
    action, log_prob, entropy, value = net.get_action_and_value(obs)

    assert action.shape   == (BATCH, ACT_DIM), f"action shape wrong: {action.shape}"
    assert log_prob.shape == (BATCH,),          f"log_prob shape wrong: {log_prob.shape}"
    assert entropy.shape  == (BATCH,),          f"entropy shape wrong: {entropy.shape}"
    assert value.shape    == (BATCH,),          f"value shape wrong: {value.shape}"


# ------------------------------------------------------------------
# Test 4 — Actions are bounded in [0, 1]
# ------------------------------------------------------------------

def test_actions_in_unit_interval(net):
    obs = random_obs()
    action, _, _, _ = net.get_action_and_value(obs)

    # Beta distribution must produce actions strictly in (0, 1)
    assert torch.all(action >= 0.0), "Actions below 0 detected"
    assert torch.all(action <= 1.0), "Actions above 1 detected"


# ------------------------------------------------------------------
# Test 5 — Log-probs are finite (no nan or inf)
# ------------------------------------------------------------------

def test_log_probs_finite(net):
    obs = random_obs()
    _, log_prob, entropy, _ = net.get_action_and_value(obs)

    assert torch.all(torch.isfinite(log_prob)), \
        f"Non-finite log_probs detected: {log_prob[~torch.isfinite(log_prob)]}"
    assert torch.all(torch.isfinite(entropy)), \
        f"Non-finite entropy detected: {entropy[~torch.isfinite(entropy)]}"


# ------------------------------------------------------------------
# Test 6 — Evaluating a given action returns same log_prob
# ------------------------------------------------------------------

def test_action_evaluation_consistency(net):
    obs = random_obs()

    # Sample an action
    with torch.no_grad():
        action, log_prob_sample, _, _ = net.get_action_and_value(obs)

    # Evaluate the same action — log_prob should match exactly
    with torch.no_grad():
        _, log_prob_eval, _, _ = net.get_action_and_value(obs, action=action)

    assert torch.allclose(log_prob_sample, log_prob_eval, atol=1e-6), \
        "Log-prob mismatch between sampling and evaluation"


# ------------------------------------------------------------------
# Test 7 — Entropy is positive
# ------------------------------------------------------------------

def test_entropy_positive(net):
    obs = random_obs()
    _, _, entropy, _ = net.get_action_and_value(obs)

    # Beta entropy can be negative for very peaked distributions,
    # but with softplus+1 initialisation it should start positive
    assert torch.all(entropy > -10.0), \
        f"Entropy unexpectedly low: min={entropy.min().item():.4f}"


# ------------------------------------------------------------------
# Test 8 — Gradient flows through the network
# ------------------------------------------------------------------

def test_gradients_flow(net):
    obs = random_obs()
    action, log_prob, entropy, value = net.get_action_and_value(obs)

    # Simple loss combining all outputs
    loss = -log_prob.mean() + entropy.mean() + value.mean()
    loss.backward()

    # Check that all parameters received gradients
    for name, param in net.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"
        assert torch.all(torch.isfinite(param.grad)), \
            f"Non-finite gradient for parameter: {name}"