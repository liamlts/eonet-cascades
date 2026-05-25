"""MDN spatial head tests."""

from __future__ import annotations

import torch

from eonet_cascades.models.components.mdn_head import MDNHead


def test_log_prob_shape():
    head = MDNHead(input_dim=32, n_components=8)
    x = torch.tensor([[-100.0, 35.0], [-110.0, 40.0]])
    h = torch.randn(2, 32)
    lp = head.log_prob(h, x)
    assert lp.shape == (2,)


def test_log_prob_increases_when_point_near_mean():
    """Log-prob at a high-density point should exceed log-prob at a far point."""
    torch.manual_seed(0)
    head = MDNHead(input_dim=16, n_components=4)
    h = torch.zeros(1, 16)
    x_near = torch.tensor([[0.0, 0.0]])
    x_far = torch.tensor([[100.0, 100.0]])
    lp_near = head.log_prob(h, x_near)
    lp_far = head.log_prob(h, x_far)
    assert lp_near.item() > lp_far.item()


def test_sample_shape():
    head = MDNHead(input_dim=16, n_components=4)
    h = torch.randn(5, 16)
    samples = head.sample(h)
    assert samples.shape == (5, 2)


def test_log_prob_integrates_to_one_approximately_on_grid():
    """For a single 1-component case at the origin, log-prob integrated over a
    fine grid should be near 1 (probability mass)."""
    torch.manual_seed(1)
    head = MDNHead(input_dim=4, n_components=1)
    h = torch.zeros(1, 4)
    # 41x41 grid over [-5, 5]^2 → cell area 0.0625
    lon = torch.linspace(-5.0, 5.0, 41)
    lat = torch.linspace(-5.0, 5.0, 41)
    lon_grid, lat_grid = torch.meshgrid(lon, lat, indexing="xy")
    xx = torch.stack([lon_grid.flatten(), lat_grid.flatten()], dim=-1)  # (1681, 2)
    h_rep = h.expand(xx.shape[0], -1)
    lp = head.log_prob(h_rep, xx)
    integral = float(torch.exp(lp).sum() * 0.0625)
    assert 0.3 < integral < 3.0, f"integral {integral} not within order of magnitude of 1"
