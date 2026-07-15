import torch
import pytest
from training.losses import quantile_huber_loss, cvar_advantage


N_QUANTILES = 32
BATCH       = 256
N_TAIL      = max(1, int(0.05 * N_QUANTILES))  # = 1


@pytest.fixture
def tau():
    """Fixed quantile midpoints — same as in DistributionalActorCritic."""
    return torch.FloatTensor(
        [(2 * i - 1) / (2 * N_QUANTILES) for i in range(1, N_QUANTILES + 1)]
    )


# ------------------------------------------------------------------
# Test 1 — quantile_huber_loss returns a scalar
# ------------------------------------------------------------------

def test_qhl_returns_scalar(tau):
    quantiles = torch.randn(BATCH, N_QUANTILES)
    returns   = torch.randn(BATCH)

    loss = quantile_huber_loss(quantiles, returns, tau)

    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"
    assert torch.isfinite(loss), "Loss is not finite"


# ------------------------------------------------------------------
# Test 2 — loss is non-negative
# ------------------------------------------------------------------

def test_qhl_non_negative(tau):
    """
    Huber loss is always >= 0, and the asymmetric weight is always >= 0,
    so the quantile Huber loss must be non-negative.
    """
    quantiles = torch.randn(BATCH, N_QUANTILES)
    returns   = torch.randn(BATCH)

    loss = quantile_huber_loss(quantiles, returns, tau)

    assert loss.item() >= 0.0, f"Loss should be non-negative, got {loss.item()}"


# ------------------------------------------------------------------
# Test 3 — loss is zero when quantiles perfectly match returns
# ------------------------------------------------------------------

def test_qhl_zero_at_perfect_fit(tau):
    """
    When all quantile estimates equal the return exactly,
    u_i = R_t - theta_i = 0 for all i, so loss = 0.
    """
    returns   = torch.ones(BATCH) * 5.0       # all returns = 5.0
    quantiles = torch.ones(BATCH, N_QUANTILES) * 5.0  # all estimates = 5.0

    loss = quantile_huber_loss(quantiles, returns, tau)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6), \
        f"Loss should be 0 at perfect fit, got {loss.item()}"


# ------------------------------------------------------------------
# Test 4 — loss decreases as quantiles approach returns
# ------------------------------------------------------------------

def test_qhl_decreases_toward_target(tau):
    """
    Loss should be smaller when quantile estimates are closer to returns.
    """
    returns = torch.zeros(BATCH)

    # Far from target
    quantiles_far  = torch.ones(BATCH, N_QUANTILES) * 10.0
    loss_far       = quantile_huber_loss(quantiles_far, returns, tau)

    # Close to target
    quantiles_near = torch.ones(BATCH, N_QUANTILES) * 0.1
    loss_near      = quantile_huber_loss(quantiles_near, returns, tau)

    assert loss_near < loss_far, \
        f"Loss should decrease as estimates approach target: {loss_near} vs {loss_far}"


# ------------------------------------------------------------------
# Test 5 — gradients flow through quantile Huber loss
# ------------------------------------------------------------------

def test_qhl_gradients_flow(tau):
    quantiles = torch.randn(BATCH, N_QUANTILES, requires_grad=True)
    returns   = torch.randn(BATCH)

    loss = quantile_huber_loss(quantiles, returns, tau)
    loss.backward()

    assert quantiles.grad is not None, "No gradient for quantiles"
    assert torch.all(torch.isfinite(quantiles.grad)), \
        "Non-finite gradients in quantile Huber loss"


# ------------------------------------------------------------------
# Test 6 — cvar_advantage returns correct shape
# ------------------------------------------------------------------

def test_cvar_advantage_shape():
    returns   = torch.randn(BATCH)
    quantiles = torch.randn(BATCH, N_QUANTILES)

    adv = cvar_advantage(returns, quantiles, n_tail=N_TAIL)

    assert adv.shape == (BATCH,), f"Expected ({BATCH},), got {adv.shape}"
    assert torch.all(torch.isfinite(adv)), "Non-finite CVaR advantages"


# ------------------------------------------------------------------
# Test 7 — cvar_advantage uses left tail correctly
# ------------------------------------------------------------------

def test_cvar_advantage_uses_tail():
    """
    CVaR baseline = mean of n_tail lowest quantiles.
    Advantage = return - CVaR.
    If return > CVaR: advantage positive (good outcome).
    If return < CVaR: advantage negative (tail outcome, penalise).
    """
    # Set quantiles so CVaR is clearly -10 (lowest quantile)
    quantiles      = torch.zeros(BATCH, N_QUANTILES)
    quantiles[:, 0] = -10.0   # lowest quantile = -10 → CVaR = -10

    # Return of -5 is above CVaR(-10) → positive advantage
    returns_above = torch.ones(BATCH) * -5.0
    adv_above     = cvar_advantage(returns_above, quantiles, n_tail=N_TAIL)
    assert torch.all(adv_above > 0), \
        "Return above CVaR should give positive advantage"

    # Return of -15 is below CVaR(-10) → negative advantage
    returns_below = torch.ones(BATCH) * -15.0
    adv_below     = cvar_advantage(returns_below, quantiles, n_tail=N_TAIL)
    assert torch.all(adv_below < 0), \
        "Return below CVaR should give negative advantage"


# ------------------------------------------------------------------
# Test 8 — cvar_advantage with n_tail=1 equals return minus min quantile
# ------------------------------------------------------------------

def test_cvar_advantage_n_tail_1():
    """
    With n_tail=1, CVaR = single lowest quantile.
    Advantage = return - min(quantiles).
    """
    returns   = torch.zeros(BATCH)
    quantiles = torch.randn(BATCH, N_QUANTILES)

    adv          = cvar_advantage(returns, quantiles, n_tail=1)
    min_quantile = quantiles.min(dim=-1).values
    expected     = returns - min_quantile

    assert torch.allclose(adv, expected, atol=1e-5), \
        "With n_tail=1, advantage should equal return - min quantile"


# ------------------------------------------------------------------
# Test 9 — asymmetric weighting: tail quantiles get higher weight
# ------------------------------------------------------------------

def test_asymmetric_weighting(tau):
    """
    For the lowest quantile tau_1 ≈ 0.016:
    - Overestimation weight = 1 - tau_1 ≈ 0.984
    - Underestimation weight = tau_1 ≈ 0.016
    Test this on a single quantile to avoid averaging cancellation.
    """
    returns = torch.zeros(1)

    # Single quantile at tau_1 level only
    tau_single = tau[:1]   # just the first (lowest) quantile

    # Overestimate: quantile > return → u < 0 → weight = 1 - tau_1 ≈ 0.984
    quantiles_over  = torch.ones(1, 1) * 1.0
    loss_over       = quantile_huber_loss(quantiles_over, returns, tau_single)

    # Underestimate: quantile < return → u > 0 → weight = tau_1 ≈ 0.016
    quantiles_under = torch.ones(1, 1) * -1.0
    loss_under      = quantile_huber_loss(quantiles_under, returns, tau_single)

    # For lowest quantile, overestimation (weight≈0.984) >> underestimation (weight≈0.016)
    assert loss_over > loss_under, \
        f"For tau_1≈0.016: overestimation loss {loss_over:.4f} should " \
        f"exceed underestimation loss {loss_under:.4f}"