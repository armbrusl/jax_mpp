"""Minimal forward-pass tests for jax_mpp (no checkpoint required)."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_platform_name", "cpu")


@pytest.fixture(scope="module")
def tiny_model():
    """AViT-Ti with default config (smallest available variant)."""
    from jax_mpp.configs import avit_Ti

    return avit_Ti()


@pytest.fixture(scope="module")
def tiny_inputs():
    # (T, B, C, H, W) – time-first, channels-first, matching PyTorch interface
    T, B, C, H, W = 2, 1, 3, 32, 32
    x = jnp.ones((T, B, C, H, W))
    state_labels = jnp.arange(C, dtype=jnp.int32)
    bcs = jnp.zeros((B, 2), dtype=jnp.int32)  # (B, n_spatial_axes) – all open
    return x, state_labels, bcs


@pytest.fixture(scope="module")
def tiny_params(tiny_model, tiny_inputs):
    x, state_labels, bcs = tiny_inputs
    rng = jax.random.PRNGKey(0)
    return tiny_model.init(
        {"params": rng, "drop_path": rng},
        x,
        state_labels,
        bcs,
        deterministic=True,
    )


def test_import():
    from jax_mpp import AViT
    from jax_mpp.configs import avit_Ti, avit_S, avit_B, avit_L, AVIT_CONFIGS

    assert "Ti" in AVIT_CONFIGS


def test_init(tiny_params):
    leaves = jax.tree_util.tree_leaves(tiny_params)
    assert len(leaves) > 0


def test_param_count(tiny_params):
    n = sum(x.size for x in jax.tree_util.tree_leaves(tiny_params))
    assert n > 0


def test_forward_shape(tiny_model, tiny_inputs, tiny_params):
    x, state_labels, bcs = tiny_inputs
    B, C = x.shape[1], x.shape[2]
    H, W = x.shape[3], x.shape[4]
    y = tiny_model.apply(
        {"params": tiny_params["params"]},
        x,
        state_labels,
        bcs,
        deterministic=True,
    )
    # Returns last timestep: (B, C, H, W)
    assert y.shape == (B, C, H, W)


def test_forward_finite(tiny_model, tiny_inputs, tiny_params):
    x, state_labels, bcs = tiny_inputs
    y = tiny_model.apply(
        {"params": tiny_params["params"]},
        x,
        state_labels,
        bcs,
        deterministic=True,
    )
    assert jnp.all(jnp.isfinite(y))


def test_convenience_constructors():
    from jax_mpp.configs import avit_Ti, avit_S, avit_B, avit_L

    assert avit_Ti() is not None
    assert avit_S() is not None
