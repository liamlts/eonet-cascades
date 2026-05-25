"""CTLSTM cell math tests."""

from __future__ import annotations

import torch

from eonet_cascades.models.components.ctlstm import CTLSTMCell


def test_update_output_shapes():
    cell = CTLSTMCell(input_dim=8, hidden_dim=16)
    x = torch.randn(2, 8)
    h_prev = torch.zeros(2, 16)
    c_prev = torch.zeros(2, 16)
    c_bar_prev = torch.zeros(2, 16)
    h, c, c_bar, delta, o = cell.update(x, h_prev, c_prev, c_bar_prev)
    assert h.shape == (2, 16)
    assert c.shape == (2, 16)
    assert c_bar.shape == (2, 16)
    assert delta.shape == (2, 16)
    assert o.shape == (2, 16)


def test_evolve_at_dt_zero_equals_event_state():
    """h(t_event + 0) should equal h just after the event update."""
    cell = CTLSTMCell(input_dim=4, hidden_dim=8)
    x = torch.randn(1, 4)
    h0 = torch.zeros(1, 8)
    c0 = torch.zeros(1, 8)
    c_bar0 = torch.zeros(1, 8)
    h_event, c_post, c_bar, delta, o = cell.update(x, h0, c0, c_bar0)
    h_t, c_t = cell.evolve(c_post, c_bar, delta, o, dt=torch.zeros(1, 1))
    assert torch.allclose(h_t, h_event, atol=1e-6)
    assert torch.allclose(c_t, c_post, atol=1e-6)


def test_evolve_at_large_dt_approaches_cbar_state():
    """As dt -> infinity, c(t) -> c_bar and h(t) -> o * tanh(c_bar)."""
    cell = CTLSTMCell(input_dim=4, hidden_dim=8)
    x = torch.randn(1, 4)
    _h_event, c_post, c_bar, delta, o = cell.update(
        x, torch.zeros(1, 8), torch.zeros(1, 8), torch.zeros(1, 8)
    )
    h_far, c_far = cell.evolve(c_post, c_bar, delta, o, dt=torch.full((1, 1), 50.0))
    expected_c_far = c_bar  # exp(-delta * 50) approx 0
    expected_h_far = o * torch.tanh(c_bar)
    assert torch.allclose(c_far, expected_c_far, atol=1e-3)
    assert torch.allclose(h_far, expected_h_far, atol=1e-3)


def test_evolve_decays_monotonically_between_event_and_asymptote():
    """|c(t) - c_bar| should monotonically non-increase as dt grows."""
    cell = CTLSTMCell(input_dim=4, hidden_dim=8)
    torch.manual_seed(0)
    x = torch.randn(1, 4)
    _h_event, c_post, c_bar, delta, o = cell.update(
        x, torch.zeros(1, 8), torch.zeros(1, 8), torch.zeros(1, 8)
    )
    diff = (c_post - c_bar).abs().max()
    assert diff > 1e-3, "test setup degenerate: c_post == c_bar"
    dts = torch.tensor([[0.0], [0.5], [1.0], [2.0], [5.0]])
    cs = [cell.evolve(c_post, c_bar, delta, o, dt=dt)[1] for dt in dts]
    norms = [(c - c_bar).abs().mean().item() for c in cs]
    for i in range(len(norms) - 1):
        assert norms[i] >= norms[i + 1] - 1e-6, f"non-monotone: {norms}"
