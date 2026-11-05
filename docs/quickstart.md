# Quick-start

Fit three parameters to `[1, 2, 3]` while requiring their sum to equal one. This is a complete CPU-compatible example; paste it into a notebook cell or run `python tutorials/minimal.py` from the repository root after installing dependencies.

```python
import jax
import jax.numpy as jnp
import optax

from nuox import make_project_grad
from nuox.linalg import lstsq_lsmr


# 1. Define the loss and constraint residual.
def loss_fn(params):
    return jnp.mean((params - jnp.array([1., 2., 3.])) ** 2)


def constraint_fn(params):
    return jnp.atleast_1d(params.sum() - 1.)


# 2. Create a constrained SGD optimizer.
lr = 0.05
projection = make_project_grad(
    constraint_fn,
    wm_epochs=-1,  # Activate from the first update.
    num_batches=1,
    gamma=0.2 / lr,  # Correct 20% of the linearized violation per step.
    least_squares_solver=lstsq_lsmr(maxiter=32),
)
optimizer = optax.chain(projection, optax.scale(-lr))


# 3. Train with the usual JAX/Optax loop.
@jax.jit
def step(params, state):
    grads = jax.grad(loss_fn)(params)
    updates, state = optimizer.update(grads, state, (params, (), {}))
    return optax.apply_updates(params, updates), state


params = jnp.zeros(3)
state = optimizer.init(params)

for _ in range(500):
    params, state = step(params, state)

print("Parameters:", params)
print("Constraint error:", float(jnp.linalg.norm(constraint_fn(params))))
```

The solution should be close to `[-0.6667, 0.3333, 1.3333]`, with constraint error below `1e-5` in this example. This checks both feasibility and the correct constrained optimum.

`constraint_fn` returns a residual, not a penalty loss. `make_project_grad` defines the constrained direction; `optax.scale(-lr)` turns it into a parameter update. The extra tuple passed to `update` supplies parameters and arguments for evaluating the constraint.

`wm_epochs=-1` activates projection on the first update. With the current implementation, `wm_epochs=0` skips the first update. See the [guide](guide.md) for the correction strength and warmup semantics.
