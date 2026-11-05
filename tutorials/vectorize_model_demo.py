"""Minimal demo for nuox.operators.vectorize_model with Flax Linen."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution from a checkout without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from nuox.operators import vectorize_model


def main():
    from flax import linen as nn

    class DenseModel(nn.Module):
        features: int

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.features)(x)
            return jax.nn.relu(x)

    model = DenseModel(4)
    key = jax.random.PRNGKey(0)
    x = jnp.ones((2, 3))
    variables = model.init(key, x)

    apply_vec, params_flat, unravel = vectorize_model(model, params=variables["params"])
    y_ref = model.apply(variables, x)
    y_vec = apply_vec(params_flat, x)

    print("Outputs match:", np.allclose(np.asarray(y_ref), np.asarray(y_vec)))


if __name__ == "__main__":
    main()
