"""
Shared modules: position biases and MLP.

JAX/Flax translation of ``models.shared_modules`` from the MPP codebase.
Provides:

- :class:`ContinuousPositionBias1D` — MLP-based continuous relative position bias
- :class:`RelativePositionBias` — T5-style bucketed relative position bias
- :class:`AbsolutePositionBias` — learnable absolute position bias
- :class:`MLP` — two-layer feed-forward with GELU activation
"""

import math

import jax
import jax.numpy as jnp
import flax.linen as nn


class ContinuousPositionBias1D(nn.Module):
    """
    Continuous relative position bias computed via a small MLP.

    Maps relative coordinate differences through a two-layer MLP
    (Linear → ReLU → Linear) and applies sigmoid scaling (×16).

    Attributes:
        n_heads: Number of attention heads.
    """

    n_heads: int

    @nn.compact
    def __call__(self, h: int, h2: int, bc: int = 0):
        """
        Compute position bias matrix.

        Args:
            h: Query sequence length.
            h2: Key sequence length (unused, same as h in practice).
            bc: Boundary condition flag. 0 = edges are endpoints,
                1 = periodic boundary conditions.

        Returns:
            Bias tensor of shape ``(1, n_heads, h, h)``.
        """
        # Compute both coord layouts; select via jnp.where so the code
        # remains valid when ``bc`` is a traced JAX value (e.g. inside
        # jax.lax.scan).
        #   bc==0 (or anything else): endpoints   -(h-1) .. h-1
        #   bc==1:                     periodic wrapping
        # Both branches are padded to the same length (2h-1) so that
        # jnp.where works on equal shapes.
        coords_open = jnp.arange(-(h - 1), h, dtype=jnp.float32) / (h - 1)

        periodic_parts = jnp.concatenate(
            [
                jnp.arange(1, h // 2 + 1, dtype=jnp.float32),
                jnp.arange(-(h // 2 - 1), h // 2 + 1, dtype=jnp.float32),
                jnp.arange(-(h // 2 - 1), 0, dtype=jnp.float32),
            ]
        ) / (h - 1)
        # Pad periodic_parts to length 2h-1 (it is 2h-2 naturally)
        pad_len = (2 * h - 1) - periodic_parts.shape[0]
        coords_periodic = jnp.concatenate(
            [periodic_parts, jnp.zeros(pad_len, dtype=jnp.float32)]
        )

        is_periodic = (bc == 1)
        relative_coords = jnp.where(is_periodic, coords_periodic, coords_open)

        coords = jnp.arange(h, dtype=jnp.float32)
        coords = coords[None, :] - coords[:, None]  # (h, h)
        coords = coords + (h - 1)

        # Two-layer MLP: (2h-1, 1) -> 512 -> n_heads
        x = relative_coords[:, None]  # (2h-1, 1)
        x = nn.Dense(512, use_bias=True, name="cpb_mlp_0")(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_heads, use_bias=False, name="cpb_mlp_2")(x)

        rel_pos_model = 16.0 * jax.nn.sigmoid(x.squeeze())  # (2h-1, n_heads)

        biases = rel_pos_model[coords.astype(jnp.int32)]  # (h, h, n_heads)
        return biases.transpose(2, 0, 1)[None, :, :, :]  # (1, n_heads, h, h)


class RelativePositionBias(nn.Module):
    """
    T5-style bucketed relative position bias.

    Buckets relative positions into a fixed number of bins and looks up
    a learned bias per head from an embedding table.

    Attributes:
        bidirectional: Whether attention is bidirectional.
        num_buckets: Number of relative position buckets.
        max_distance: Maximum distance for bucketing.
        n_heads: Number of attention heads.
    """

    bidirectional: bool = True
    num_buckets: int = 32
    max_distance: int = 128
    n_heads: int = 2

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        bidirectional=True,
        num_buckets=32,
        max_distance=32,
    ):
        """Translate relative position to a bucket number."""
        ret = jnp.zeros_like(relative_position, dtype=jnp.int32)
        n = -relative_position

        if bidirectional:
            num_buckets //= 2
            ret = ret + (n < 0).astype(jnp.int32) * num_buckets
            n = jnp.abs(n)
        else:
            n = jnp.maximum(n, 0)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            jnp.log(n.astype(jnp.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(jnp.int32)
        val_if_large = jnp.minimum(val_if_large, num_buckets - 1)

        ret = ret + jnp.where(is_small, n, val_if_large)
        return ret

    @nn.compact
    def __call__(self, qlen: int, klen: int, bc: int = 0):
        """
        Compute relative position bias.

        Args:
            qlen: Query sequence length.
            klen: Key sequence length.
            bc: Boundary condition flag (1 = periodic wrapping).

        Returns:
            Bias tensor of shape ``(1, n_heads, qlen, klen)``.
        """
        context_position = jnp.arange(qlen)[:, None]
        memory_position = jnp.arange(klen)[None, :]
        relative_position = memory_position - context_position  # (qlen, klen)

        # Periodic wrapping when bc == 1.  Use jnp.where instead of a
        # Python ``if`` so the code stays valid inside jax.lax.scan
        # (where ``bc`` may be a traced value).
        is_periodic = (bc == 1)
        thresh = klen // 2
        rp_wrapped = jnp.where(
            relative_position < -thresh,
            relative_position % thresh,
            relative_position,
        )
        rp_wrapped = jnp.where(
            rp_wrapped > thresh,
            rp_wrapped % (-thresh),
            rp_wrapped,
        )
        relative_position = jnp.where(is_periodic, rp_wrapped, relative_position)

        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
        )

        # Embedding: (num_buckets, n_heads)
        embedding = nn.Embed(
            num_embeddings=self.num_buckets,
            features=self.n_heads,
            name="relative_attention_bias",
        )
        values = embedding(rp_bucket)  # (qlen, klen, n_heads)
        return values.transpose(2, 0, 1)[None, :, :, :]  # (1, n_heads, qlen, klen)


class AbsolutePositionBias(nn.Module):
    """
    Learnable absolute position bias.

    Attributes:
        hidden_dim: Feature dimension.
        n_tokens: Maximum number of tokens.
    """

    hidden_dim: int
    n_tokens: int

    @nn.compact
    def __call__(self):
        """Returns bias of shape ``(1, n_tokens, hidden_dim)``."""
        bias = self.param(
            "bias",
            nn.initializers.normal(stddev=0.02),
            (1, self.n_tokens, self.hidden_dim),
        )
        return bias


class MLP(nn.Module):
    """
    Two-layer feed-forward network with GELU activation.

    Attributes:
        hidden_dim: Input/output feature dimension.
        exp_factor: Expansion factor for the hidden layer.
    """

    hidden_dim: int
    exp_factor: float = 4.0

    @nn.compact
    def __call__(self, x):
        inner_dim = int(self.hidden_dim * self.exp_factor)
        x = nn.Dense(inner_dim, name="fc1")(x)
        x = nn.gelu(x, approximate=False)
        x = nn.Dense(self.hidden_dim, name="fc2")(x)
        return x
