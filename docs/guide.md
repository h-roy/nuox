# Constraints, updates, and solvers

## Return a vector of residuals

For a condition $f_\theta(x)=y$, return `f(params, x) - y`. For a scalar condition, use `jnp.atleast_1d`. For multiple outputs, flatten and concatenate the residuals:

```python
# Fragment: predict is your model's forward function.
def constraint_fn(params, inputs, *, targets):
    return (predict(params, inputs) - targets).reshape(-1)

# Inside a training step:
updates, state = optimizer.update(
    grads, state, (params, (inputs,), {"targets": targets})
)
```

Keep array shapes fixed across compiled steps. Pass changing batches through the argument tuple rather than mutating captured Python state. Keep differentiable computations in JAX.

Do not replace a vector residual by its squared norm just to reduce the number of constraints: the derivative of that scalar vanishes at feasibility, changing the projection problem. Scale residual components deliberately when their units or magnitudes differ. Unequal scaling affects numerical conditioning and, for inconsistent systems, the least-squares compromise.

## Flatten model parameters

The bundled LSMR implementation expects a one-dimensional parameter array and a one-dimensional residual array. Use `ravel_pytree` for a model with nested parameters:

```python
from jax.flatten_util import ravel_pytree

# Fragment: initial_params, model_loss and model_constraint are your model's API.
params, unravel = ravel_pytree(initial_params)

def loss_fn(flat_params, batch):
    return model_loss(unravel(flat_params), batch)

def constraint_fn(flat_params, inputs):
    return model_constraint(unravel(flat_params), inputs).reshape(-1)
```

The optimized vector can be converted back with `unravel(params)`. For Flax Linen, `vectorize_model` provides a forward function that accepts this vector:

```python
from nuox import vectorize_model

# Fragment: model is an initialized Linen module and variables its collections.
apply_vec, params, unravel = vectorize_model(
    model, params=variables["params"], variables=variables
)
# Predictions: apply_vec(params, inputs)
```

The helper is for a functional forward pass. It does not persist changed mutable collections; manage state such as running statistics explicitly when training a stateful model.

## Compose the optimizer

The simplest composition is constrained SGD:

```python
optimizer = optax.chain(projection, optax.scale(-lr))
```

If using Adam's adaptive direction, apply it before projection:

```python
optimizer = optax.chain(
    optax.scale_by_adam(),
    projection,
    optax.scale(-lr),
)
```

This preserves the linearized constraint relation of the projected direction, up to solve error. It does not guarantee convergence to the same constrained optimum as projected SGD. Coordinate-wise scaling, momentum, or weight decay applied after projection can invalidate that relation. The `params` argument received by every chained transform is NUOX's `(params, args, kwargs)` tuple, so transforms that expect an ordinary parameter tree need a suitable adapter.

## Correction strength and warmup

Write $J$ for the constraint Jacobian and $g$ for the direction entering NUOX. With an exact, undamped, minimum-norm solve, the transform returns

$$
\widetilde g = g-J^+(Jg-\gamma c),
\qquad \Delta\theta=-\eta\widetilde g.
$$

If the linearized correction is feasible, $J\Delta\theta=-\eta\gamma c$. For an affine constraint this reduces the residual by a factor $1-\eta\gamma$. For nonlinear constraints this is a first-order relation. A rank-deficient or inconsistent system may leave an uncorrectable residual.

The quick-start sets `gamma=0.2 / lr`, correcting 20% of the linearized violation per step. `gamma=0` only projects the direction and does not repair an existing violation. With a learning-rate schedule and fixed `gamma`, the correction fraction changes with the learning rate. `scale_gamma=True` additionally multiplies `gamma` by the current residual norm; leave it off for the simple fixed-fraction interpretation.

Projection runs when `state.count > wm_epochs * num_batches`, with the count initialized to zero. Thus `wm_epochs=-1, num_batches=1` starts immediately, while `wm_epochs=0` leaves the first update unconstrained. These arguments control a step-count threshold; NUOX does not manage epochs or batching.

## Choose a least-squares solver

Start with an explicit iteration budget:

```python
from nuox.linalg import lstsq_lsmr

solver = lstsq_lsmr(atol=1e-6, btol=1e-6, maxiter=64)
```

`64` is a starting budget, not a universal convergence guarantee. Increase it if the linear solve is too inaccurate. `atol` and `btol` govern residual stopping criteria; `ctol` is a reciprocal condition-estimate threshold. `damp` regularizes the solve and changes the projection/correction. The current default `maxiter` is one million, so set a finite budget appropriate to your problem.

`lstsq_jnp()` materializes the Jacobian and is useful for small debugging problems. Normal-equation solvers are also available, but forming normal equations can worsen conditioning. Dense solvers are unsuitable for large parameter Jacobians.

The projection transform currently discards solver statistics. Direct solver calls return `(solution, stats)`, with LSMR statistics including `iteration_count`, `norm_residual`, `norm_At_residual`, and `istop`.

## Monitor what matters

Track the task loss and the residual after applying each update:

```python
residual = constraint_fn(params, inputs, targets=targets)
rms = jnp.sqrt(jnp.mean(residual**2))
maximum = jnp.max(jnp.abs(residual))
```

If residuals increase, check the optimizer order and warmup first. Then inspect solver accuracy, reduce the learning rate or correction fraction, and check whether the requested conditions are mutually compatible. Separately evaluate held-out inputs when the intended property should generalize beyond the constraint points.
