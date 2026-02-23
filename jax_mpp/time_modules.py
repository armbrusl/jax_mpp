"""
Temporal attention module.

JAX/Flax translation of ``models.time_modules`` from the MPP codebase.
Provides:

- :class:`AttentionBlock` — full attention over the time axis with
  instance normalisation, relative position bias, layer scale, and
  stochastic depth.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange

from jax_mpp.shared_modules import (
    ContinuousPositionBias1D,
    RelativePositionBias,
)
from jax_mpp.spatial_modules import _drop_path, _scaled_dot_product_attention


class InstanceNorm2d(nn.Module):
    """
    Instance normalisation over spatial dims, channels-last ``(B, H, W, C)``.

    Computes per-instance, per-channel mean/std over ``(H, W)`` and
    optionally applies a learnable affine transform.

    Attributes:
        dim: Number of channels.
        affine: Apply learnable scale and bias.
        eps: Epsilon for numerical stability.
    """

    dim: int
    affine: bool = True
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        mean = jnp.mean(x, axis=(1, 2), keepdims=True)
        var = jnp.var(x, axis=(1, 2), keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)

        if self.affine:
            scale = self.param("weight", nn.initializers.ones, (self.dim,))
            bias = self.param("bias", nn.initializers.zeros, (self.dim,))
            x = x * scale[None, None, None, :] + bias[None, None, None, :]
        return x


class AttentionBlock(nn.Module):
    """
    Temporal attention block.

    Performs full attention over the *time* axis. Spatial dimensions
    are batched together so that each spatial token attends
    independently across time steps.

    Input shape: ``(T, B, H, W, C)`` (channels-last, time-first).
    Output shape: same as input.

    Attributes:
        hidden_dim: Channel dimension.
        num_heads: Attention heads.
        drop_path: Stochastic depth rate.
        layer_scale_init_value: Initial layer-scale value (0 = disabled).
        bias_type: ``"rel"`` | ``"continuous"`` | ``"none"``.
    """

    hidden_dim: int = 768
    num_heads: int = 12
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6
    bias_type: str = "rel"

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        """
        Args:
            x: ``(T, B, H, W, C)``
            deterministic: Disable stochastic depth when ``True``.

        Returns:
            ``(T, B, H, W, C)``
        """
        T, B, H, W, C = x.shape
        head_dim = C // self.num_heads

        residual = x

        # Pre-norm — apply over flattened (T*B) batch
        x_flat = rearrange(x, "t b h w c -> (t b) h w c")
        x_flat = InstanceNorm2d(self.hidden_dim, name="norm1")(x_flat)

        # QKV projection via 1×1 conv
        x_flat = nn.Conv(3 * self.hidden_dim, kernel_size=(1, 1), name="input_head")(
            x_flat
        )

        # Rearrange for temporal attention: (B*H*W, heads, T, d)
        x_tmp = rearrange(
            x_flat, "(t b) h w (he c) -> (b h w) he t c", t=T, he=self.num_heads
        )
        q, k, v = jnp.split(x_tmp, 3, axis=-1)

        q = nn.LayerNorm(name="qnorm")(q)
        k = nn.LayerNorm(name="knorm")(k)

        # Position bias
        if self.bias_type == "continuous":
            pos_bias_mod = ContinuousPositionBias1D(
                n_heads=self.num_heads, name="rel_pos_bias"
            )
        elif self.bias_type == "rel":
            pos_bias_mod = RelativePositionBias(
                n_heads=self.num_heads, name="rel_pos_bias"
            )
        else:
            pos_bias_mod = None

        if pos_bias_mod is not None:
            rel_pos_bias = pos_bias_mod(T, T)
        else:
            rel_pos_bias = None

        out = _scaled_dot_product_attention(q, k, v, rel_pos_bias)

        # Back to spatial layout
        out = rearrange(
            out, "(b h w) he t c -> (t b) h w (he c)", h=H, w=W, he=self.num_heads
        )

        # Post-norm + output projection
        out = InstanceNorm2d(self.hidden_dim, name="norm2")(out)
        out = nn.Conv(self.hidden_dim, kernel_size=(1, 1), name="output_head")(out)
        out = rearrange(out, "(t b) h w c -> t b h w c", t=T)

        # Layer scale
        if self.layer_scale_init_value > 0:
            gamma = self.param(
                "gamma",
                lambda _rng, _shape: jnp.full(_shape, self.layer_scale_init_value),
                (self.hidden_dim,),
            )
            out = out * gamma[None, None, None, None, :]

        # Drop path + residual
        # Flatten to (T*B, ...) for drop path then reshape back
        dp_rng = self.make_rng("drop_path") if not deterministic else None
        out_flat = rearrange(out, "t b h w c -> (t b) h w c")
        out_flat = _drop_path(out_flat, self.drop_path, deterministic, dp_rng)
        out = rearrange(out_flat, "(t b) h w c -> t b h w c", t=T)

        return out + residual
