"""
Mixed (space-time) modules: factored space-time block.

JAX/Flax translation of ``models.mixed_modules`` from the MPP codebase.
Provides:

- :class:`SpaceTimeBlock` — alternates temporal attention and spatial
  axial attention within a single transformer block.
"""

from einops import rearrange

import flax.linen as nn

from jax_mpp.spatial_modules import AxialAttentionBlock
from jax_mpp.time_modules import AttentionBlock


class SpaceTimeBlock(nn.Module):
    """
    Factored space-time transformer block.

    Applies temporal attention first (over the *T* axis), then spatial
    axial attention (over *H* and *W* axes).  The MLP is part of the
    spatial :class:`AxialAttentionBlock`.

    Input/output: ``(T, B, H, W, C)`` (channels-last, time-first).

    Attributes:
        hidden_dim: Channel dimension.
        num_heads: Number of attention heads.
        drop_path: Stochastic depth rate.
        bias_type: Position-bias type (``"rel"`` | ``"continuous"`` | ``"none"``).
    """

    hidden_dim: int = 768
    num_heads: int = 12
    drop_path: float = 0.0
    bias_type: str = "rel"

    @nn.compact
    def __call__(self, x, bcs, deterministic: bool = True):
        """
        Args:
            x: ``(T, B, H, W, C)``
            bcs: Boundary condition tensor ``(B, 2)``
            deterministic: Disable stochastic depth when ``True``.

        Returns:
            ``(T, B, H, W, C)``
        """
        T = x.shape[0]

        # ---- Temporal attention ----
        x = AttentionBlock(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            drop_path=self.drop_path,
            bias_type=self.bias_type,
            name="temporal",
        )(x, deterministic=deterministic)

        # ---- Spatial axial attention ----
        # Merge T into batch for spatial processing
        x = rearrange(x, "t b h w c -> (t b) h w c")
        x = AxialAttentionBlock(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            drop_path=self.drop_path,
            bias_type=self.bias_type,
            name="spatial",
        )(x, bcs, deterministic=deterministic)
        x = rearrange(x, "(t b) h w c -> t b h w c", t=T)

        return x
