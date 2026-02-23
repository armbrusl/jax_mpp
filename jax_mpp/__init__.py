"""
jax_mpp: JAX/Flax translation of the Multiple Physics Pretraining (MPP) model.

Provides a 1-to-1 translation of the **AViT** architecture from
`Multiple Physics Pretraining for Physical Surrogate Models
<https://openreview.net/forum?id=DKSI3bULiZ>`_ (McCabe et al., NeurIPS 2024)
into JAX/Flax linen, maintaining exact weight compatibility with the
original PyTorch implementation.

Quick start::

    import jax, jax.numpy as jnp
    from jax_mpp import avit_B

    model = avit_B(n_states=12)
    rng = jax.random.PRNGKey(0)

    T, B, C, H, W = 4, 2, 3, 128, 128
    x = jnp.ones((T, B, C, H, W))
    labels = jnp.array([0, 1, 2])
    bcs = jnp.zeros((B, 2), dtype=jnp.int32)

    params = model.init(
        {"params": rng, "drop_path": rng},
        x, labels, bcs, deterministic=True,
    )
    y = model.apply(params, x, labels, bcs, deterministic=True)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jax_mpp")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

# Main model
from jax_mpp.avit import AViT

# Sub-modules (for advanced / component-level usage)
from jax_mpp.mixed_modules import SpaceTimeBlock
from jax_mpp.spatial_modules import (
    AxialAttentionBlock,
    RMSInstanceNorm2d,
    SubsampledLinear,
    hMLP_output,
    hMLP_stem,
)
from jax_mpp.time_modules import AttentionBlock, InstanceNorm2d
from jax_mpp.shared_modules import (
    ContinuousPositionBias1D,
    MLP,
    RelativePositionBias,
    AbsolutePositionBias,
)

# Configs & convenience constructors
from jax_mpp.configs import (
    AVIT_CONFIGS,
    avit_B,
    avit_L,
    avit_S,
    avit_Ti,
)

# Weight conversion
from jax_mpp.convert_weights import (
    convert_pytorch_to_jax_params,
    load_pytorch_state_dict,
)

__all__ = [
    "__version__",
    # Model
    "AViT",
    # Sub-modules
    "SpaceTimeBlock",
    "AxialAttentionBlock",
    "AttentionBlock",
    "RMSInstanceNorm2d",
    "InstanceNorm2d",
    "SubsampledLinear",
    "hMLP_stem",
    "hMLP_output",
    "MLP",
    "ContinuousPositionBias1D",
    "RelativePositionBias",
    "AbsolutePositionBias",
    # Configs
    "AVIT_CONFIGS",
    "avit_Ti",
    "avit_S",
    "avit_B",
    "avit_L",
    # Weight conversion
    "load_pytorch_state_dict",
    "convert_pytorch_to_jax_params",
]
