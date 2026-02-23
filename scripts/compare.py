#!/usr/bin/env python
"""
Compare the original PyTorch AViT with the JAX/Flax translation.

Loads both implementations, seeds them identically, and reports
maximum / mean absolute differences.

Usage::

    PYTHONPATH=/path/to/multiple_physics_pretraining uv run python scripts/compare.py \\
        --checkpoint path/to/MPP_AViT_Ti \\
        --variant Ti
"""

import argparse
import sys

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch vs JAX MPP outputs")
    parser.add_argument("--checkpoint", type=str, required=True, help="PyTorch checkpoint path")
    parser.add_argument("--variant", type=str, default="Ti", choices=["Ti", "S", "B", "L"])
    parser.add_argument("--n_steps", type=int, default=4, help="Number of time steps")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--n_channels", type=int, default=None,
                        help="Active channels (default: all n_states)")
    args = parser.parse_args()

    assert torch is not None, "PyTorch is required for comparison"
    import jax
    import jax.numpy as jnp
    from jax_mpp import load_pytorch_state_dict, convert_pytorch_to_jax_params
    from jax_mpp.configs import AVIT_CONFIGS, _make_model

    cfg = AVIT_CONFIGS[args.variant]
    n_states = cfg["n_states"]
    C = args.n_channels if args.n_channels is not None else n_states

    # ---- Create random input ----
    np.random.seed(42)
    T, B = args.n_steps, args.batch
    H = W = args.resolution
    x_np = np.random.randn(T, B, C, H, W).astype(np.float32)
    bcs_np = np.zeros((B, 4), dtype=np.int64)

    # State labels: PyTorch expects [[0,1,...,C-1]] (nested list)
    #               JAX expects jnp.arange(C) (1-D array)
    labels_list = list(range(C))
    labels_pt = [labels_list]       # nested list for PT SubsampledLinear
    labels_jax = jnp.arange(C)

    # ---- Load checkpoint ----
    print(f"Loading checkpoint: {args.checkpoint}")
    pt_state = load_pytorch_state_dict(args.checkpoint)

    # ---- PyTorch forward ----
    print("Setting up PyTorch model...")
    try:
        from models.avit import build_avit
    except ImportError:
        print("ERROR: Cannot import original PyTorch AViT.")
        print("Try: PYTHONPATH=/path/to/multiple_physics_pretraining python scripts/compare.py ...")
        sys.exit(1)

    # Build a params namespace matching the variant config
    class Params:
        pass
    params = Params()
    params.embed_dim = cfg["embed_dim"]
    params.processor_blocks = cfg["processor_blocks"]
    params.n_states = n_states
    params.num_heads = cfg["num_heads"]
    params.patch_size = (16, 16)
    params.bias_type = "rel"
    params.block_type = "axial"
    params.space_type = "axial_attention"
    params.time_type = "attention"
    params.gradient_checkpointing = False

    pt_model = build_avit(params)
    pt_model.load_state_dict(pt_state, strict=True)
    pt_model.eval()

    x_pt = torch.from_numpy(x_np)
    bcs_pt = torch.from_numpy(bcs_np)
    print(f"Running PyTorch forward ({args.variant}, T={T}, B={B}, C={C}, H={H})...")
    with torch.no_grad():
        y_pt = pt_model(x_pt, labels_pt, bcs_pt).numpy()
    print(f"  PT output shape: {y_pt.shape}")

    # ---- JAX forward ----
    print("Setting up JAX model...")
    jax_params = convert_pytorch_to_jax_params(pt_state, verbose=False)
    jax_model = _make_model(args.variant)

    x_jax = jnp.array(x_np)
    bcs_jax = jnp.array(bcs_np)
    print(f"Running JAX forward ({args.variant}, T={T}, B={B}, C={C}, H={H})...")
    y_jax = jax_model.apply(
        {"params": jax_params}, x_jax, labels_jax, bcs_jax, deterministic=True
    )
    y_jax_np = np.array(y_jax)
    print(f"  JAX output shape: {y_jax_np.shape}")

    # ---- Compare ----
    max_diff = np.max(np.abs(y_pt - y_jax_np))
    mean_diff = np.mean(np.abs(y_pt - y_jax_np))
    rel_diff = max_diff / (np.max(np.abs(y_pt)) + 1e-8)
    print(f"\nResults for AViT-{args.variant}:")
    print(f"  Max  absolute diff: {max_diff:.6e}")
    print(f"  Mean absolute diff: {mean_diff:.6e}")
    print(f"  Max relative diff:  {rel_diff:.6e}")
    print(f"  PyTorch output range: [{y_pt.min():.4f}, {y_pt.max():.4f}]")
    print(f"  JAX output range:     [{y_jax_np.min():.4f}, {y_jax_np.max():.4f}]")

    if max_diff < 1e-4:
        print("\n  PASS: Outputs match within 1e-4")
    elif max_diff < 1e-2:
        print("\n  CLOSE: Small differences (likely float32 precision)")
    else:
        print("\n  FAIL: Significant differences detected")


if __name__ == "__main__":
    main()
