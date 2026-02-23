"""
AViT: Axial Vision Transformer for Multiple Physics Pretraining.

JAX/Flax translation of ``models.avit`` from the MPP codebase.
This is the main model class implementing the full encode → process → decode
pipeline:

1. **Instance normalisation** over time and space
2. **Sparse projection** via :class:`SubsampledLinear`
3. **Hierarchical patch embedding** (``hMLP_stem``)
4. **N × SpaceTimeBlock** (factored temporal + axial spatial attention)
5. **Hierarchical patch output** (``hMLP_output``, subsampled by state labels)
6. **De-normalisation**
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import rearrange

from jax_mpp.spatial_modules import SubsampledLinear, hMLP_output, hMLP_stem
from jax_mpp.mixed_modules import SpaceTimeBlock


class AViT(nn.Module):
    """
    Axial Vision Transformer for spatiotemporal PDE surrogate modeling.

    Given a time-series of 2-D fields ``(T, B, C, H, W)`` the model:

    * normalises per-sample across time and space,
    * projects the ``C`` state-variable channels (selected by
      ``state_labels``) into the embedding space,
    * encodes spatial patches via a hierarchical ConvNet stem,
    * applies ``processor_blocks`` factored space-time transformer
      blocks,
    * decodes back to the original resolution via a transposed-conv
      output head,
    * de-normalises using the original statistics.

    The model returns the prediction for the **last time step only**.

    Attributes:
        patch_size: Spatial patch size.
        embed_dim: Embedding/hidden dimension.
        processor_blocks: Number of SpaceTimeBlock layers.
        n_states: Number of possible input state variables.
        drop_path: Maximum stochastic-depth rate (linearly increased).
        bias_type: Position bias type (``"rel"`` | ``"continuous"`` | ``"none"``).
        num_heads: Number of attention heads.
    """

    patch_size: tuple = (16, 16)
    embed_dim: int = 768
    processor_blocks: int = 8
    n_states: int = 6
    drop_path: float = 0.2
    bias_type: str = "rel"
    num_heads: int = 12

    @nn.compact
    def __call__(self, x, state_labels, bcs, deterministic: bool = True):
        """
        Forward pass.

        Args:
            x: Input tensor ``(T, B, C, H, W)`` — time-first, channels-first
                to match the original PyTorch interface.
            state_labels: List or 1-D array of active state-variable indices.
            bcs: Boundary-condition flags ``(B, 2)`` (one per spatial axis).
            deterministic: If ``True``, disable dropout / stochastic depth.

        Returns:
            Prediction for the last time step ``(B, C, H, W)``.
        """
        T, B, C, H, W = x.shape
        state_labels = jnp.asarray(state_labels)

        # ---- 1. Normalise (per sample, across time + space) ----
        # ddof=1 matches PyTorch torch.std_mean default (Bessel correction)
        data_mean = jnp.mean(x, axis=(0, 3, 4), keepdims=True)  # (1, B, C, 1, 1)
        data_std = jnp.std(x, axis=(0, 3, 4), keepdims=True, ddof=1) + 1e-7
        x = (x - data_mean) / data_std

        # ---- 2. Sparse channel projection ----
        # (T, B, C, H, W) -> (T, B, H, W, C) channels-last
        x = rearrange(x, "t b c h w -> t b h w c")
        x = SubsampledLinear(
            dim_in=self.n_states,
            dim_out=self.embed_dim // 4,
            name="space_bag",
        )(x, state_labels)

        # ---- 3. Patch embedding ----
        # Merge T into batch for spatial conv
        x = rearrange(x, "t b h w c -> (t b) h w c")
        x = hMLP_stem(
            patch_size=self.patch_size,
            in_chans=self.embed_dim // 4,
            embed_dim=self.embed_dim,
            name="embed",
        )(x)
        x = rearrange(x, "(t b) h w c -> t b h w c", t=T)

        # ---- 4. Processor blocks ----
        dp_rates = np.linspace(0, self.drop_path, self.processor_blocks)
        for i in range(self.processor_blocks):
            x = SpaceTimeBlock(
                hidden_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=float(dp_rates[i]),
                bias_type=self.bias_type,
                name=f"blocks_{i}",
            )(x, bcs, deterministic=deterministic)

        # ---- 5. Output projection ----
        x = rearrange(x, "t b h w c -> (t b) h w c")
        x = hMLP_output(
            patch_size=self.patch_size,
            out_chans=self.n_states,
            embed_dim=self.embed_dim,
            name="debed",
        )(x, state_labels)
        # hMLP_output returns (TB, H, W, n_out) channels-last
        x = rearrange(x, "(t b) h w c -> t b c h w", t=T)

        # ---- 6. De-normalise ----
        x = x * data_std + data_mean

        # Return last time step
        return x[-1]
