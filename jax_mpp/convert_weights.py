"""
Weight conversion utility: PyTorch → JAX/Flax parameter mapping.

Maps every key in the PyTorch ``AViT`` state_dict to the corresponding
leaf in the Flax parameter tree, applying transpositions where needed.

Mapping rules
-------------
- ``space_bag.weight`` (O, I)            → kept as-is (our code does ``x @ w.T``)
- ``embed.in_proj.{i}.weight`` Conv 4-D  → ``.kernel`` (OIHW → HWIO)
- ``embed.in_proj.{i}.weight`` Norm 1-D  → ``.weight`` (kept)
- ``debed.out_proj.{i}.weight`` Conv 4-D → ``.kernel`` (IOHW → HWIO)
- ``debed.out_proj.{i}.weight`` Norm 1-D → ``.weight`` (kept)
- ``debed.out_kernel`` (q, n, 4, 4)      → (n, q, 4, 4) [transpose axes 0↔1]
- ``*.input_head.weight`` (O,I,1,1)      → ``.kernel`` (1,1,I,O)
- ``*.output_head.weight`` (O,I,1,1)     → ``.kernel`` (1,1,I,O)
- ``*.qnorm.weight`` / ``*.knorm.weight``→ ``.scale``
- ``*.mlp.fc1.weight`` (O, I)            → ``.kernel`` (I, O)
- ``*.mlp.fc2.weight`` (O, I)            → ``.kernel`` (I, O)
- ``*.relative_attention_bias.weight``    → ``.embedding``
- ``nn.InstanceNorm2d.weight``            → ``.weight`` (kept as-is, 1-D)
- ``RMSInstanceNorm2d.weight``            → ``.weight`` (kept as-is, 1-D)
- ``gamma_att``, ``gamma_mlp``, ``gamma`` → kept as-is
- ``blocks.{i}.*``                        → ``blocks_{i}.*``
"""

from __future__ import annotations

import re
from typing import Any, Dict

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def torch_to_numpy(tensor):
    """Convert a PyTorch tensor to numpy, handling GPU tensors."""
    if torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def load_pytorch_state_dict(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load a PyTorch MPP checkpoint and extract the model state dict.

    Handles common checkpoint formats::

        raw state_dict (OrderedDict)
        {'model_state': state_dict}
        {'model': state_dict}
        {'state_dict': state_dict}
    """
    assert torch is not None, "PyTorch is required to load checkpoints"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            return ckpt["model_state"]
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def convert_pytorch_to_jax_params(
    pytorch_state_dict: Dict[str, Any],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convert a PyTorch AViT state_dict to a Flax parameter dict.

    Returns:
        A nested dict suitable for ``model.apply({'params': params}, ...)``.
    """
    jax_params: Dict[str, Any] = {}
    unmapped: list[str] = []

    for pt_key, pt_val in pytorch_state_dict.items():
        val = torch_to_numpy(pt_val)
        jax_key, val = _convert_one(pt_key, val)

        if jax_key is None:
            unmapped.append(pt_key)
            continue

        _set_nested(jax_params, jax_key.split("."), val)

    if unmapped and verbose:
        print(f"[convert] {len(unmapped)} unmapped keys:")
        for k in unmapped:
            print(f"  {k}")

    return jax_params


# ---------------------------------------------------------------------------
# Single-key conversion
# ---------------------------------------------------------------------------

def _convert_one(pt_key: str, val: np.ndarray):
    """
    Convert one PyTorch (key, value) pair to Flax (key, value).

    Returns ``(jax_key, transposed_val)`` or ``(None, val)`` when unmapped.
    """
    key = pt_key

    # ---- structural renaming ----
    key = re.sub(r"blocks\.(\d+)\.", r"blocks_\1.", key)
    key = re.sub(r"embed\.in_proj\.(\d+)", r"embed.in_proj_\1", key)
    key = re.sub(r"debed\.out_proj\.(\d+)", r"debed.out_proj_\1", key)
    # continuous bias MLP indices
    key = re.sub(r"rel_pos_bias\.cpb_mlp\.(\d+)", r"rel_pos_bias.cpb_mlp_\1", key)

    # ---- special keys ----
    # debed.out_kernel: PT (q, n_states, 4, 4) -> Flax (n_states, q, 4, 4)
    if key == "debed.out_kernel":
        return key, val.transpose(1, 0, 2, 3)

    # debed.out_bias: kept as-is
    if key == "debed.out_bias":
        return key, val

    # space_bag.weight: (O, I) kept as-is (our Flax code stores (O, I))
    if key in ("space_bag.weight", "space_bag.bias"):
        return key, val

    # ---- 4-D conv weights (Conv2d and ConvTranspose2d) ----
    # Conv2d:          PT (O, I, kH, kW) → Flax (kH, kW, I, O) = transpose(2,3,1,0)
    # ConvTranspose2d: PT (I, O, kH, kW) → Flax (kH, kW, O, I) = transpose(2,3,1,0)
    #   (with transpose_kernel=True the kernel layout is (kH,kW,C_out,C_in);
    #    C_out of the transpose = C_in of the forward, matching the PT 2nd axis)
    # Both use the same transpose(2,3,1,0).
    if key.endswith(".weight") and val.ndim == 4:
        jax_key = key[:-len(".weight")] + ".kernel"
        return jax_key, val.transpose(2, 3, 1, 0)

    # ---- 2-D linear weights: (O, I) -> (I, O) ----
    if key.endswith(".weight") and val.ndim == 2 and _is_dense_weight(key):
        jax_key = key[:-len(".weight")] + ".kernel"
        return jax_key, val.T

    # ---- LayerNorm weight -> scale ----
    if key.endswith(".weight") and _is_layernorm_weight(key):
        jax_key = key[:-len(".weight")] + ".scale"
        return jax_key, val

    # ---- Embedding weight ----
    if "relative_attention_bias.weight" in key:
        jax_key = key[:-len(".weight")] + ".embedding"
        return jax_key, val

    # ---- everything else (norm .weight/.bias, gamma, bias, etc.) → keep verbatim ----
    return key, val


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _is_dense_weight(key: str) -> bool:
    """2-D linear weights that must be transposed."""
    return any(n in key for n in (".fc1.weight", ".fc2.weight"))


def _is_layernorm_weight(key: str) -> bool:
    """LayerNorm whose .weight should become .scale."""
    return any(n in key for n in (".qnorm.weight", ".knorm.weight"))


def _is_conv_transpose_weight(key: str) -> bool:
    """ConvTranspose2d weights (debed.out_proj_*) have (I, O, H, W) layout."""
    # Called before .weight → .kernel rename; val.ndim==4 guard means only
    # actual conv weights (not InstanceNorm 1-D weights) reach here.
    return "out_proj_" in key


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _set_nested(d: dict, keys: list, val):
    """Set a value in a nested dict by a list of path segments."""
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val
