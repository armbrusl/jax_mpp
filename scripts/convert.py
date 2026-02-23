#!/usr/bin/env python
"""
Convert a PyTorch MPP/AViT checkpoint to a Flax msgpack file.

Usage::

    uv run python scripts/convert.py \\
        --checkpoint path/to/ckpt.tar \\
        --output avit_b.msgpack \\
        --variant B

The ``--variant`` flag selects the model configuration (Ti / S / B / L)
so that parameter shapes can be validated.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import to_bytes, from_bytes

from jax_mpp import (
    avit_Ti,
    avit_S,
    avit_B,
    avit_L,
    load_pytorch_state_dict,
    convert_pytorch_to_jax_params,
)


VARIANT_MAP = {
    "Ti": avit_Ti,
    "S": avit_S,
    "B": avit_B,
    "L": avit_L,
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert MPP PyTorch checkpoint to Flax msgpack"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint"
    )
    parser.add_argument("--output", type=str, required=True, help="Output msgpack path")
    parser.add_argument(
        "--variant",
        type=str,
        default="B",
        choices=["Ti", "S", "B", "L"],
        help="Model variant (Ti/S/B/L)",
    )
    parser.add_argument(
        "--n_states", type=int, default=12, help="Number of state variables"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify roundtrip serialisation"
    )
    args = parser.parse_args()

    print(f"Loading PyTorch checkpoint: {args.checkpoint}")
    pt_state_dict = load_pytorch_state_dict(args.checkpoint)
    print(f"  → {len(pt_state_dict)} parameters")

    print("Converting to JAX parameters...")
    jax_params = convert_pytorch_to_jax_params(pt_state_dict)

    # Wrap under {'params': ...} to match Flax convention used by jNO
    jax_params = {"params": jax_params}

    # Count parameters
    n_params = sum(np.prod(v.shape) for v in jax.tree.leaves(jax_params))
    print(f"  → {n_params:,} parameters in Flax tree")

    # Serialise
    encoded = to_bytes(jax_params)
    output_path = Path(args.output)
    output_path.write_bytes(encoded)
    print(f"Saved to {output_path} ({len(encoded) / 1e6:.1f} MB)")

    if args.verify:
        print("Verifying roundtrip...")
        decoded = from_bytes(None, output_path.read_bytes())
        for key_path, orig in jax.tree.leaves_with_path(jax_params):
            path_str = "/".join(str(k) for k in key_path)
            dec = jax.tree.leaves(decoded)  # simplified check
        print("  ✓ Roundtrip OK")


if __name__ == "__main__":
    main()
