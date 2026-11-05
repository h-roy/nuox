# NUOX: constrained learning with JAX and Optax

**Define a loss. Define a constraint. Train with NUOX.**

NUOX provides an Optax transformation for differentiable equality-constrained optimisation:

$$
\min_\theta \mathcal L(\theta) \quad\text{subject to}\quad c(\theta)=0.
$$

The constraint function returns residuals: zero means the specified conditions are satisfied. NUOX projects a proposed direction onto the null space of the constraint Jacobian and adds a correction for existing violations. LSMR computes the required least-squares solution using Jacobian products, without constructing the full Jacobian.

Start with the [runnable quick-start](quickstart.md), then read about [constraints and optimizer composition](guide.md). The [tutorials](tutorials.md) apply the same workflow to PINNs, one-step generation, and LLM editing. The [API reference](api.md) describes the current interfaces.

## Installation

From a checkout containing `pyproject.toml`:

```bash
git clone https://github.com/h-roy/nuox.git
cd nuox
python -m pip install -e .
```

Use the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html) to configure GPU support. The core quick-start also runs on CPU. Tutorial-specific dependencies belong to each notebook's setup.

## Scope

Constraints must be differentiable with respect to the optimized parameters. Enforcement is numerical and local: nonlinear residuals depend on the step size, solver accuracy, and feasibility. A constraint evaluated at selected inputs protects those inputs; it does not establish a global property of the model.

The bundled LSMR solver operates on one-dimensional arrays. Flatten parameter PyTrees and return a flat residual vector, as shown in the guide. Core training does not require differentiating through the optimizer or wrapping its least-squares solver in a custom derivative.

[Source code](https://github.com/h-roy/nuox) · [Issues](https://github.com/h-roy/nuox/issues) · [MIT license](https://github.com/h-roy/nuox/blob/main/LICENSE)
