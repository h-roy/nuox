# API reference

## Projection transform

`make_project_grad` returns an Optax transformation. Initialize it with the optimized parameter vector and call `update(direction, state, (params, args, kwargs))`. The constraint is evaluated as `constraint_fn(params, *args, **kwargs)`.

::: nuox.optax_nullspace.make_project_grad

## Model vectorization

`vectorize_model` returns `(apply_vec, params_vec, unravel)`. For Linen, pass `params=variables["params"]`; optionally pass the other variable collections. For NNX, the default trainable filter is `nnx.Param`, with additional state filters available through `nnx_state_filters`. The helper does not persist mutable state updates.

::: nuox.operators.vectorize_model

## Jacobian products

`build_constraint_ops(constraint_fn)(*args, **kwargs)` returns `(matvec, vecmat)`. Their signatures are `matvec(v, params)` for $Jv$ and `vecmat(w, params)` for $J^T w$.

::: nuox.operators.build_constraint_ops

## LSMR

The factory returns a callable with signature `solver(vecmat, rhs, vecmat_args=())`. Here `vecmat(w, *vecmat_args)` must compute $A^T w$; the solver returns a solution to the least-squares problem $Ax \approx \mathrm{rhs}$ and a statistics dictionary. Inputs and outputs are one-dimensional arrays. The [guide](guide.md) explains tolerances and iteration budgets.

::: nuox.linalg.lstsq_lsmr

## Small dense reference solver

`lstsq_jnp()` uses the same operator interface but materializes the matrix. Use it only for small problems.

::: nuox.linalg.lstsq_jnp

## Legacy training helper

`make_update_fn(loss_fn, optim, constr_fn)` returns a jitted function accepting `(params, opt_state, x, y, **constr_params)`. It returns `(loss_before_update, new_params, new_opt_state, constraint_after_update)`. New examples use an explicit training step to make the constraint arguments visible.

::: nuox.optax_nullspace.make_update_fn
