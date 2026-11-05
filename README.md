# NUOX: Constrained optimisation for JAX and Optax

**Define a loss. Define a constraint. Train with NUOX.**

NUOX provides Optax-compatible transformations for training models with differentiable equality constraints:

$$
\min_\theta \mathcal{L}(\theta)
\qquad \text{subject to} \qquad c(\theta)=0.
$$

Write your constraints as a JAX function returning residuals. NUOX projects the training direction onto the null space of the constraint Jacobian and adds a correction for existing violations. Its matrix-free least-squares solver uses Jacobian products without constructing the full parameter Jacobian.

- **A small interface:** add a constraint transform to an Optax training loop.
- **Flexible constraints:** express initial conditions, model preservation, or other differentiable equality constraints in JAX.
- **JAX-native:** work with parameter PyTrees, automatic differentiation, and JIT compilation.

## Installation

Install from a local checkout:

```bash
git clone https://github.com/h-roy/nuox.git
cd nuox
python -m pip install -e .
```

For GPU execution, install the appropriate JAX build using the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

## Quick-start

Fit three parameters to `[1, 2, 3]`, subject to their sum being one. The workflow is **define the loss and constraint → create the optimizer → train**.

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

The constrained optimum is approximately `[-0.6667, 0.3333, 1.3333]`, with a sum of one. The numerical constraint error should be close to floating-point precision.

The third argument to `optimizer.update` is `(params, constraint_args, constraint_kwargs)`. Here the constraint only needs the parameters, so its extra arguments are empty. For `constraint_fn(params, inputs)`, pass `(params, (inputs,), {})`.

### How the update works

NUOX removes the part of the proposed direction that changes the linearized constraints, then adds a correction that reduces their residual. With the final scaling `optax.scale(-lr)`, `lr * gamma` sets the fraction of the linearized violation corrected per step.

Optimizer order matters: coordinate-wise scaling after projection can change the projected direction. This quick-start uses SGD; the tutorials show the compositions used for their respective tasks.

Constraint enforcement is numerical. For nonlinear constraints, residuals depend on the step size, local linearization, feasibility, and solver accuracy. Monitor constraint error alongside the loss; constraints imposed at selected points do not guarantee satisfaction everywhere.

## Tutorials

Three notebooks demonstrate the same workflow across different learning problems:

| Tutorial | Learning objective | Constraint |
| --- | --- | --- |
| [PINNs](tutorials/pinns/pinns.ipynb) | Fit a solution to a differential equation | Enforce the initial condition at selected points. |
| [One-step flow matching](tutorials/flow_matching/smile.ipynb) | Distil a flow model into a one-step generator | Preserve selected instantaneous teacher velocities. |
| [LLM editing](tutorials/llm_editing/llm_editing.ipynb) | Teach a language model an edited fact | Restrict changes to an editable weight matrix using an orthogonal preservation constraint. |

Each notebook introduces the problem, defines the loss and constraint, constructs the NUOX optimizer, and visualizes the outcome. The flow tutorial compares ordinary flow matching with NUOX distillation at one and a few generation steps.

Keep each notebook with its companion Python and data files. Follow its setup cell when running in VS Code or Colab; GPU execution requires a GPU-backed notebook kernel.

## Documentation and development

- [Documentation](https://h-roy.github.io/nuox/)
- [Source code](https://github.com/h-roy/nuox)
- [Report a bug or suggest an example](https://github.com/h-roy/nuox/issues)

Install development dependencies and run the tests from the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

To preview the documentation locally:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

Contributions, bug reports, and new tutorials are welcome.

## License

NUOX is released under the [MIT License](LICENSE).
