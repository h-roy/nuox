import sys
from pathlib import Path

# Allow direct execution from a checkout without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import optax

from nuox import make_project_grad
from nuox.linalg import lstsq_lsmr


# Fit three parameters, subject to their sum being one.
def loss_fn(params):
    return jnp.mean((params - jnp.array([1., 2., 3.])) ** 2)


def constraint_fn(params):
    return jnp.atleast_1d(params.sum() - 1.)


lr = 0.05
projection = make_project_grad(
    constraint_fn,
    wm_epochs=-1,  # Activate immediately with the current API.
    num_batches=1,
    gamma=0.2 / lr,  # Correct 20% of the linearized violation.
    least_squares_solver=lstsq_lsmr(maxiter=32),
)
optimizer = optax.chain(projection, optax.scale(-lr))


@jax.jit
def step(params, state):
    grads = jax.grad(loss_fn)(params)
    updates, state = optimizer.update(
        grads, state, (params, (), {})
    )
    return optax.apply_updates(params, updates), state


params = jnp.zeros(3)
state = optimizer.init(params)

for _ in range(500):
    params, state = step(params, state)

print("Parameters:", params)
print("Loss:", float(loss_fn(params)))
print("Constraint error:", float(jnp.linalg.norm(constraint_fn(params))))
