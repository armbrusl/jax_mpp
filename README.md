# jax_mpp

JAX/Flax translation of the **Multiple Physics Pretraining (MPP)** AViT model,
maintaining exact 1-to-1 weight compatibility with the original PyTorch
implementation.

## Overview

MPP is a pretraining strategy that jointly normalizes and embeds multiple sets
of physical dynamics into a single space for prediction. The model uses an
**Axial Vision Transformer (AViT)** architecture with factored space-time
attention.

```
Input (T, B, C, H, W)
        │
        ▼
┌─── Instance Norm ──────┐
│  normalize per sample   │
│  across time + space    │
└────────┬───────────────┘
         ▼
┌─── SubsampledLinear ───┐   Sparse channel projection
│  state vocab → embed/4  │   (selects active state vars)
└────────┬───────────────┘
         ▼
┌─── hMLP_stem ──────────┐   Hierarchical conv embedding
│  Conv4s4 → Conv2s2 →    │   (3 stages, stride 16 total)
│  Conv2s2 each w/ RMS-IN │
└────────┬───────────────┘
         ▼
┌─── N × SpaceTimeBlock ─┐
│  ┌────────────────────┐ │
│  │ Temporal Attention  │ │   Full attention over T axis
│  │ (InstanceNorm, QKV, │ │   with relative position bias
│  │  RPB, LayerScale)   │ │
│  └────────┬───────────┘ │
│           ▼              │
│  ┌────────────────────┐ │
│  │ Axial Spatial Attn  │ │   X-axis + Y-axis attention
│  │ (RMSInstanceNorm,   │ │   averaged, with RPB, MLP,
│  │  QKV, RPB, MLP,     │ │   layer scale, drop path
│  │  LayerScale)         │ │
│  └────────┬───────────┘ │
└───────────┼─────────────┘
            ▼
┌─── hMLP_output ────────┐   Hierarchical conv de-embedding
│  ConvT2s2 → ConvT2s2 → │   (3 stages, stride 16 total)
│  ConvT4s4 (subsampled)  │
└────────┬───────────────┘
         ▼
┌─── De-normalise ───────┐
│  x * std + mean         │
└────────┬───────────────┘
         ▼
Output (B, C, H, W)  [last time step]
```

## Model Variants

| Variant | embed_dim | heads | blocks | Params (approx.) |
|---------|-----------|-------|--------|-------------------|
| Ti      |   192     |   3   |   12   |   ~5.5 M          |
| S       |   384     |   6   |   12   |  ~21 M            |
| B       |   768     |  12   |   12   |  ~83 M            |
| L       |  1024     |  16   |   24   | ~300 M            |

All variants use `patch_size=(16, 16)`, `n_states=12`, `bias_type="rel"`.

## Reference

| | |
|---|---|
| **Paper** | [Multiple Physics Pretraining for Physical Surrogate Models](https://openreview.net/forum?id=DKSI3bULiZ) (NeurIPS 2024) |
| **Weights** | [Google Drive](https://drive.google.com/drive/folders/1Qaqa-RnzUDOO8-Gi4zlf4BE53SfWqDwx?usp=sharing) |
| **Original code** | [PolymathicAI/multiple_physics_pretraining](https://github.com/PolymathicAI/multiple_physics_pretraining) |

## Installation

```bash
uv venv && source .venv/bin/activate
uv pip install -e .

# With GPU support:
uv pip install -e ".[gpu]"

# For weight conversion from PyTorch:
uv pip install -e ".[convert]"
```

## Usage

### Quick Start

```python
import jax
import jax.numpy as jnp
from jax_mpp import avit_B

# Create model
model = avit_B(n_states=12)

# Dummy inputs
rng = jax.random.PRNGKey(0)
T, B, C, H, W = 4, 2, 3, 128, 128
x = jnp.ones((T, B, C, H, W))
labels = jnp.array([0, 1, 2])
bcs = jnp.zeros((B, 2), dtype=jnp.int32)

# Initialize parameters
params = model.init(
    {"params": rng, "drop_path": rng},
    x, labels, bcs, deterministic=True,
)

# Forward pass
y = model.apply(params, x, labels, bcs, deterministic=True)
print(y.shape)  # (2, 3, 128, 128)
```

### Loading Pretrained Weights

```python
from jax_mpp import avit_B, load_pytorch_state_dict, convert_pytorch_to_jax_params

# Load and convert PyTorch checkpoint
pt_state_dict = load_pytorch_state_dict("path/to/checkpoint.tar")
jax_params = convert_pytorch_to_jax_params(pt_state_dict)

# Create model and run
model = avit_B(n_states=12)
y = model.apply({"params": jax_params}, x, labels, bcs, deterministic=True)
```

### Explicit Construction

```python
from jax_mpp import AViT

model = AViT(
    patch_size=(16, 16),
    embed_dim=768,
    processor_blocks=12,
    n_states=12,
    num_heads=12,
    drop_path=0.1,
    bias_type="rel",
)
```

### Individual Components

```python
from jax_mpp import (
    AxialAttentionBlock,
    AttentionBlock,
    SpaceTimeBlock,
    SubsampledLinear,
    hMLP_stem,
    hMLP_output,
)
```

## Weight Conversion

```bash
uv run python scripts/convert.py \
    --checkpoint path/to/ckpt.tar \
    --output weights.msgpack \
    --variant B
```

### Key Mapping Rules

| PyTorch | Flax |
|---------|------|
| `blocks.{i}.*` | `blocks_{i}.*` |
| `nn.Linear.weight` | `.kernel` (transposed) |
| `nn.Conv2d.weight` | `.kernel` (OIHW → HWIO) |
| `nn.LayerNorm.weight` | `.scale` |
| `nn.Embedding.weight` | `.embedding` |
| `embed.in_proj.{i}.*` | `embed.in_proj_{i}.*` |
| `debed.out_proj.{i}.*` | `debed.out_proj_{i}.*` |

## Project Structure

```
jax_mpp/
├── jax_mpp/
│   ├── __init__.py            # Public API + version
│   ├── avit.py                # Main AViT model
│   ├── configs.py             # Variant configs (Ti/S/B/L) & constructors
│   ├── convert_weights.py     # PyTorch → Flax weight mapping
│   ├── mixed_modules.py       # SpaceTimeBlock (temporal + spatial)
│   ├── shared_modules.py      # Position biases, MLP
│   ├── spatial_modules.py     # Patch embed/de-embed, axial attention
│   └── time_modules.py        # Temporal attention
├── scripts/
│   ├── convert.py             # CLI: PyTorch → msgpack conversion
│   └── compare.py             # CLI: equivalence testing
├── pyproject.toml
├── README.md
└── LICENSE
```

## Module Details

| Module | Description |
|--------|-------------|
| `avit.py` | Top-level AViT: normalize → embed → process → de-embed → denormalize |
| `configs.py` | Variant definitions (Ti/S/B/L) and `avit_*()` constructors |
| `mixed_modules.py` | `SpaceTimeBlock` — sequential temporal + spatial attention |
| `spatial_modules.py` | `hMLP_stem`/`hMLP_output` (hierarchical conv), `AxialAttentionBlock`, `SubsampledLinear`, `RMSInstanceNorm2d` |
| `time_modules.py` | `AttentionBlock` — full attention over the time axis |
| `shared_modules.py` | `RelativePositionBias`, `ContinuousPositionBias1D`, `MLP` |
| `convert_weights.py` | PyTorch state_dict → Flax params conversion |

## Implementation Notes

- All modules use **channels-last** (`NHWC`) convention internally; the
  top-level `AViT` accepts the PyTorch-style `(T, B, C, H, W)` input for
  API compatibility and rearranges internally.
- **`flax.linen`** (functional API) is used throughout &mdash; consistent
  with the other JAX translation packages.
- **Stochastic depth** (`drop_path`) is controlled via the `deterministic`
  flag and the `"drop_path"` RNG key.
- The `SubsampledLinear` and `hMLP_output` modules perform dynamic weight
  indexing based on `state_labels`, replicating the original sparse
  projection mechanism.

## License

MIT
