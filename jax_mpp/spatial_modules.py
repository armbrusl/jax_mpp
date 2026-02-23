"""
Spatial modules: patch embedding, axial spatial attention, and output projection.

JAX/Flax translation of ``models.spatial_modules`` from the MPP codebase.
Provides:

- :class:`RMSInstanceNorm2d` — RMS-based instance normalization (no mean subtraction)
- :class:`SubsampledLinear` — Linear layer with state-variable sub-selection
- :class:`hMLP_stem` — Hierarchical convolutional patch embedding
- :class:`hMLP_output` — Hierarchical convolutional patch de-embedding
- :class:`AxialAttentionBlock` — Dual-axis (X + Y) spatial attention with
  relative position bias, layer scale, and stochastic depth
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange

from jax_mpp.shared_modules import (
    ContinuousPositionBias1D,
    MLP,
    RelativePositionBias,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drop_path(x, rate: float, deterministic: bool, rng_key):
    """Stochastic depth: randomly drop entire samples."""
    if rate == 0.0 or deterministic:
        return x
    keep = 1.0 - rate
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = jax.random.bernoulli(rng_key, keep, shape=shape).astype(x.dtype)
    return x * mask / keep


# ---------------------------------------------------------------------------
# RMSInstanceNorm2d
# ---------------------------------------------------------------------------


class RMSInstanceNorm2d(nn.Module):
    """
    RMS-based instance normalisation (channels-last: ``(B, H, W, C)``).

    Normalises over the spatial dimensions ``(H, W)`` only by dividing
    by the RMS (standard deviation without mean subtraction).

    Attributes:
        dim: Number of channels.
        affine: Whether to apply a learnable scale.
        eps: Epsilon for numerical stability.
    """

    dim: int
    affine: bool = True
    eps: float = 1e-8

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Input of shape ``(B, H, W, C)`` (channels-last).

        Returns:
            Normalised tensor of the same shape.
        """
        # std over spatial dims (H, W) — axes 1, 2 for channels-last
        # ddof=1 matches PyTorch torch.std_mean default (Bessel correction)
        std = jnp.std(x, axis=(1, 2), keepdims=True, ddof=1)
        x = x / (std + self.eps)

        if self.affine:
            scale = self.param("weight", nn.initializers.ones, (self.dim,))
            # The original PyTorch code also allocates a bias parameter but
            # never uses it in the forward pass.  We keep an identical
            # (unused) bias so that converted weights have the right tree
            # structure.
            _bias = self.param("bias", nn.initializers.zeros, (self.dim,))  # noqa: F841
            x = x * scale[None, None, None, :]
        return x


# ---------------------------------------------------------------------------
# SubsampledLinear
# ---------------------------------------------------------------------------


class SubsampledLinear(nn.Module):
    """
    Linear layer that only uses a subset of input/output features
    determined at run-time by *state_labels*.

    During the forward pass, ``weight`` is indexed by ``labels`` along
    the input (``subsample_in=True``) or output dimension.  A scaling
    factor corrects for the reduced fan-in.

    Attributes:
        dim_in: Full input dimension (= ``n_states``).
        dim_out: Full output dimension.
        subsample_in: If ``True``, subsample input columns; else output rows.
    """

    dim_in: int
    dim_out: int
    subsample_in: bool = True

    @nn.compact
    def __call__(self, x, labels):
        """
        Args:
            x: Input tensor ``(..., C_sub)`` where ``C_sub == len(labels)``.
            labels: 1-D integer array selecting active state variables.

        Returns:
            Projected tensor ``(..., dim_out)`` (or ``(..., len(labels))``
            when ``subsample_in=False``).
        """
        # Dense-equivalent parameters stored in full size
        weight = self.param(
            "weight",
            nn.initializers.lecun_normal(),
            (self.dim_out, self.dim_in),
        )
        bias = self.param("bias", nn.initializers.zeros, (self.dim_out,))

        # labels is a 1-D array of active state-variable indices
        label_size = labels.shape[0]

        if self.subsample_in:
            scale = (self.dim_in / label_size) ** 0.5
            # weight[:, labels] for PyTorch (O, I) layout
            # Our weight is (O, I) to match PyTorch for weight conversion;
            # F.linear(x, W, b) = x @ W^T + b
            w_sub = weight[:, labels]  # (dim_out, label_size)
            out = scale * (x @ w_sub.T + bias)
        else:
            w_sub = weight[labels]  # (label_size, dim_in)
            b_sub = bias[labels]
            out = x @ w_sub.T + b_sub
        return out


# ---------------------------------------------------------------------------
# hMLP_stem  (hierarchical convolutional patch embedding)
# ---------------------------------------------------------------------------


class hMLP_stem(nn.Module):
    """
    Hierarchical convolutional patch embedding.

    Three stages each halving spatial resolution:

    1. Conv 4×4 stride 4 → RMSInstanceNorm → GELU
    2. Conv 2×2 stride 2 → RMSInstanceNorm → GELU
    3. Conv 2×2 stride 2 → RMSInstanceNorm

    Overall stride = 16 (matching ``patch_size=(16, 16)``).

    Input:  ``(B, H, W, in_chans)``  channels-last
    Output: ``(B, H', W', embed_dim)`` channels-last  (``H' = H/16``)

    Attributes:
        patch_size: Spatial patch size (unused in forward; kept for compat).
        in_chans: Number of input channels.
        embed_dim: Final embedding dimension.
    """

    patch_size: tuple = (16, 16)
    in_chans: int = 3
    embed_dim: int = 768

    @nn.compact
    def __call__(self, x):
        q = self.embed_dim // 4

        # Stage 1: conv 4×4 stride 4
        x = nn.Conv(q, kernel_size=(4, 4), strides=(4, 4), use_bias=False, name="in_proj_0")(x)
        x = RMSInstanceNorm2d(q, name="in_proj_1")(x)
        x = nn.gelu(x, approximate=False)

        # Stage 2: conv 2×2 stride 2
        x = nn.Conv(q, kernel_size=(2, 2), strides=(2, 2), use_bias=False, name="in_proj_3")(x)
        x = RMSInstanceNorm2d(q, name="in_proj_4")(x)
        x = nn.gelu(x, approximate=False)

        # Stage 3: conv 2×2 stride 2
        x = nn.Conv(self.embed_dim, kernel_size=(2, 2), strides=(2, 2), use_bias=False, name="in_proj_6")(x)
        x = RMSInstanceNorm2d(self.embed_dim, name="in_proj_7")(x)

        return x


# ---------------------------------------------------------------------------
# hMLP_output  (hierarchical convolutional patch de-embedding)
# ---------------------------------------------------------------------------


class hMLP_output(nn.Module):
    """
    Hierarchical convolutional output projection (patch de-embedding).

    Two transposed-convolution stages each doubling resolution, followed
    by a final transposed convolution (4×4, stride 4) whose kernel and
    bias are indexed by ``state_labels`` so only the relevant output
    channels are produced.

    Input:  ``(B, H', W', embed_dim)``  channels-last
    Output: ``(B, H, W, n_out)``        channels-last

    Attributes:
        patch_size: Spatial patch size (unused; kept for config compat).
        out_chans: Maximum number of output channels (``n_states``).
        embed_dim: Embedding dimension.
    """

    patch_size: tuple = (16, 16)
    out_chans: int = 3
    embed_dim: int = 768

    @nn.compact
    def __call__(self, x, state_labels):
        """
        Args:
            x: Latent of shape ``(B, H', W', embed_dim)``.
            state_labels: 1-D integer array of active output channels.

        Returns:
            Reconstructed field ``(B, H, W, n_out)`` where
            ``n_out = len(state_labels)``.
        """
        q = self.embed_dim // 4

        # Stage 1: transposed conv 2×2 stride 2
        # transpose_kernel=True matches PyTorch ConvTranspose2d convention
        x = nn.ConvTranspose(
            q, kernel_size=(2, 2), strides=(2, 2), use_bias=False,
            transpose_kernel=True, name="out_proj_0",
        )(x)
        x = RMSInstanceNorm2d(q, name="out_proj_1")(x)
        x = nn.gelu(x, approximate=False)

        # Stage 2: transposed conv 2×2 stride 2
        x = nn.ConvTranspose(
            q, kernel_size=(2, 2), strides=(2, 2), use_bias=False,
            transpose_kernel=True, name="out_proj_3",
        )(x)
        x = RMSInstanceNorm2d(q, name="out_proj_4")(x)
        x = nn.gelu(x, approximate=False)

        # Stage 3: transposed conv 4×4 stride 4, subsampled by state_labels
        # We store out_kernel as (n_states, q, 4, 4) to match PyTorch and
        # subsample on n_states, then reshape for lax.conv_transpose.
        out_kernel = self.param(
            "out_kernel",
            nn.initializers.lecun_normal(),
            (self.out_chans, q, 4, 4),
        )
        out_bias = self.param("out_bias", nn.initializers.zeros, (self.out_chans,))

        # Select only the channels corresponding to state_labels
        kernel_sub = out_kernel[state_labels]  # (n_out, q, 4, 4)
        # transpose_kernel=True: kernel is (kH, kW, C_out, C_in)
        kernel_flax = kernel_sub.transpose(2, 3, 0, 1)  # (4, 4, n_out, q)
        bias_sub = out_bias[state_labels]  # (n_out,)

        x = jax.lax.conv_transpose(
            x,  # (B, H', W', q)
            kernel_flax,  # (4, 4, n_out, q)
            strides=(4, 4),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            transpose_kernel=True,
        )
        x = x + bias_sub[None, None, None, :]
        return x


# ---------------------------------------------------------------------------
# AxialAttentionBlock
# ---------------------------------------------------------------------------


class AxialAttentionBlock(nn.Module):
    """
    Dual-axis spatial attention with MLP, layer scale, and stochastic depth.

    Attention is applied independently along the X and Y axes (averaged)
    with optional relative position bias.

    Input/output: ``(B, H, W, C)`` (channels-last).

    Attributes:
        hidden_dim: Channel dimension.
        num_heads: Number of attention heads.
        drop_path: Stochastic depth rate.
        layer_scale_init_value: Initial value for layer-scale parameters
            (0 disables layer scale).
        bias_type: ``"rel"`` | ``"continuous"`` | ``"none"``.
    """

    hidden_dim: int = 768
    num_heads: int = 12
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6
    bias_type: str = "rel"

    @nn.compact
    def __call__(self, x, bcs, deterministic: bool = True):
        """
        Args:
            x: ``(B, H, W, C)`` spatial feature map.
            bcs: Boundary condition tensor ``(B, 2)`` — one flag per axis.
            deterministic: Disable stochastic depth when ``True``.

        Returns:
            Output tensor ``(B, H, W, C)``.
        """
        B, H, W, C = x.shape
        head_dim = C // self.num_heads

        # ---- Attention branch ------------------------------------------------
        residual = x
        x = RMSInstanceNorm2d(self.hidden_dim, name="norm1")(x)

        # QKV projection: (B, H, W, C) -> (B, H, W, 3C)
        x = nn.Conv(3 * self.hidden_dim, kernel_size=(1, 1), name="input_head")(x)

        # (B, H, W, 3C) -> (B, heads, H, W, 3*head_dim)
        x = rearrange(x, "b h w (he c) -> b he h w c", he=self.num_heads)
        q, k, v = jnp.split(x, 3, axis=-1)

        # QK normalisation
        q = nn.LayerNorm(name="qnorm")(q)
        k = nn.LayerNorm(name="knorm")(k)

        # Position bias module
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

        # --- X-axis attention ---
        # (B, heads, H, W, d) -> (B*H, heads, W, d)
        qx = rearrange(q, "b he h w c -> (b h) he w c")
        kx = rearrange(k, "b he h w c -> (b h) he w c")
        vx = rearrange(v, "b he h w c -> (b h) he w c")

        if pos_bias_mod is not None:
            rel_x = pos_bias_mod(W, W, bcs[0, 0])
        else:
            rel_x = None

        xx = _scaled_dot_product_attention(qx, kx, vx, rel_x)
        xx = rearrange(xx, "(b h) he w c -> b (he c) h w", h=H)

        # --- Y-axis attention ---
        qy = rearrange(q, "b he h w c -> (b w) he h c")
        ky = rearrange(k, "b he h w c -> (b w) he h c")
        vy = rearrange(v, "b he h w c -> (b w) he h c")

        if pos_bias_mod is not None:
            rel_y = pos_bias_mod(H, H, bcs[0, 1])
        else:
            rel_y = None

        xy = _scaled_dot_product_attention(qy, ky, vy, rel_y)
        xy = rearrange(xy, "(b w) he h c -> b (he c) h w", w=W)

        # Average over both axes
        x = (xx + xy) / 2.0  # (B, C, H, W)

        # Norm + output projection (keep channels-first briefly)
        x = rearrange(x, "b c h w -> b h w c")
        x = RMSInstanceNorm2d(self.hidden_dim, name="norm2")(x)
        x = rearrange(x, "b h w c -> b c h w")
        x = rearrange(x, "b (he c) h w -> b h w (he c)", he=self.num_heads)

        x = nn.Conv(self.hidden_dim, kernel_size=(1, 1), name="output_head")(x)
        x = rearrange(x, "b h w c -> b c h w")

        # Layer scale
        if self.layer_scale_init_value > 0:
            gamma_att = self.param(
                "gamma_att",
                lambda _rng, _shape: jnp.full(
                    _shape, self.layer_scale_init_value
                ),
                (self.hidden_dim,),
            )
            x = x * gamma_att[None, :, None, None]

        # Drop path + residual
        dp_rng = self.make_rng("drop_path") if not deterministic else None
        x = _drop_path(x, self.drop_path, deterministic, dp_rng)
        x = rearrange(x, "b c h w -> b h w c")
        x = x + residual

        # ---- MLP branch -----------------------------------------------------
        residual = x
        x = rearrange(x, "b h w c -> b h w c")  # already channels-last
        x = MLP(hidden_dim=self.hidden_dim, name="mlp")(x)
        x = rearrange(x, "b h w c -> b c h w")
        x = RMSInstanceNorm2d(self.hidden_dim, name="mlp_norm")(
            rearrange(x, "b c h w -> b h w c")
        )
        x = rearrange(x, "b h w c -> b c h w")

        if self.layer_scale_init_value > 0:
            gamma_mlp = self.param(
                "gamma_mlp",
                lambda _rng, _shape: jnp.full(
                    _shape, self.layer_scale_init_value
                ),
                (self.hidden_dim,),
            )
            x = x * gamma_mlp[None, :, None, None]

        dp_rng2 = self.make_rng("drop_path") if not deterministic else None
        x = _drop_path(x, self.drop_path, deterministic, dp_rng2)
        x = rearrange(x, "b c h w -> b h w c")
        x = x + residual

        return x


# ---------------------------------------------------------------------------
# Scaled dot-product attention helper
# ---------------------------------------------------------------------------


def _scaled_dot_product_attention(q, k, v, attn_mask=None):
    """
    Scaled dot-product attention (no dropout).

    Args:
        q: ``(B, heads, N_q, d)``
        k: ``(B, heads, N_k, d)``
        v: ``(B, heads, N_k, d)``
        attn_mask: Optional bias ``(1, heads, N_q, N_k)`` added to logits.

    Returns:
        Output ``(B, heads, N_q, d)``.
    """
    d = q.shape[-1]
    scale = d ** -0.5
    attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = jax.nn.softmax(attn, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", attn, v)
