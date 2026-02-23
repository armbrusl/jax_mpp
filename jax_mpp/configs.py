"""
Model variant configurations and convenience constructors.

Defines the four standard AViT variants (Ti / S / B / L) from the
`Multiple Physics Pretraining <https://openreview.net/forum?id=DKSI3bULiZ>`_ paper.

Each function returns an uninitialised :class:`AViT` instance.
"""

from jax_mpp.avit import AViT

# ---- Variant specs --------------------------------------------------------
# embed_dim / num_heads / processor_blocks
AVIT_CONFIGS = {
    "Ti": {"embed_dim": 192, "num_heads": 3, "processor_blocks": 12, "n_states": 12},
    "S": {"embed_dim": 384, "num_heads": 6, "processor_blocks": 12, "n_states": 12},
    "B": {"embed_dim": 768, "num_heads": 12, "processor_blocks": 12, "n_states": 12},
    "L": {"embed_dim": 1024, "num_heads": 16, "processor_blocks": 24, "n_states": 12},
}


def _make_model(variant: str, **overrides) -> AViT:
    """Build an :class:`AViT` with variant-specific defaults."""
    cfg = AVIT_CONFIGS[variant].copy()
    cfg.update(overrides)
    return AViT(**cfg)


def avit_Ti(**overrides) -> AViT:
    """AViT-Tiny: embed_dim=192, heads=3, blocks=12."""
    return _make_model("Ti", **overrides)


def avit_S(**overrides) -> AViT:
    """AViT-Small: embed_dim=384, heads=6, blocks=12."""
    return _make_model("S", **overrides)


def avit_B(**overrides) -> AViT:
    """AViT-Base: embed_dim=768, heads=12, blocks=12."""
    return _make_model("B", **overrides)


def avit_L(**overrides) -> AViT:
    """AViT-Large: embed_dim=1024, heads=16, blocks=24."""
    return _make_model("L", **overrides)
