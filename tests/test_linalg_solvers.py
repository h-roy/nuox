"""Regression tests for NUOX linear solvers.

These mirror the structure of the original code-projected-constraints tests but
cover only the solvers we ship (LSMR, normal-equation + CG, normal-equation + LU,
 and the simple JAX least-squares fallback).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nuox.linalg import (
    dense_solve_lu,
    lstsq_custom_vjp,
    lstsq_lsmr,
    lstsq_via_normaleq,
    solve_iterative_cg,
    solve_materialize,
)


def _random_problem(shape: tuple[int, int], seed: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    key = jax.random.PRNGKey(seed)
    a = jax.random.normal(key, shape, dtype=jnp.float32)
    key, subkey = jax.random.split(key)
    x_true = jax.random.normal(subkey, (shape[1],), dtype=jnp.float32)
    b = a @ x_true
    return a, b


def solver_lsmr():
    return lstsq_lsmr(atol=1e-6, btol=1e-6, maxiter=10_000)


def solver_normaleq_cg():
    return lstsq_via_normaleq(solve_iterative_cg(maxiter=5_000, tol=1e-6))


def solver_normaleq_lu():
    return lstsq_via_normaleq(solve_materialize(dense_solve_lu()))


SOLVER_FACTORIES = [solver_lsmr, solver_normaleq_cg, solver_normaleq_lu]
SHAPES = [(8, 5), (5, 8), (6, 6)]


@pytest.mark.parametrize("solver_fn", SOLVER_FACTORIES)
@pytest.mark.parametrize("shape", SHAPES)
def test_forward_matches_numpy(solver_fn, shape):
    a, b = _random_problem(shape, seed=1)
    matrix_t = a.T
    rhs = b
    solver = solver_fn()

    def vecmat(v, *_):
        return matrix_t @ v

    sol, _ = solver(vecmat, rhs)
    expected = np.linalg.lstsq(np.asarray(a), np.asarray(b), rcond=None)[0].astype(np.float32)
    np.testing.assert_allclose(np.asarray(sol), expected, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("case", ["zero_rhs", "zero_operator", "orthogonal_rhs"])
def test_lsmr_exact_stationary_initial_point(case):
    # A nonzero b can also have x=0 as its minimum-norm least-squares solution.
    matrix = jnp.array([[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]])
    rhs = jnp.array([0.0, 3.0, 0.0])
    if case == "zero_rhs":
        rhs = jnp.zeros(3)
    elif case == "zero_operator":
        matrix = jnp.zeros_like(matrix)
    solver = lstsq_lsmr(maxiter=16)

    @jax.jit
    def solve(matrix, rhs):
        return solver(lambda value: matrix.T @ value, rhs)

    solution, stats = solve(matrix, rhs)
    expected = np.linalg.lstsq(np.asarray(matrix), np.asarray(rhs), rcond=None)[0]
    np.testing.assert_array_equal(np.asarray(solution), expected)
    assert int(stats["iteration_count"]) == 0
    assert int(stats["istop"]) == (8 if case == "zero_rhs" else 9)
    assert all(np.isfinite(np.asarray(value)).all() for value in stats.values())
    np.testing.assert_allclose(stats["norm_residual"], np.linalg.norm(np.asarray(rhs)))
    assert float(stats["norm_At_residual"]) == 0


def test_custom_vjp_matches_finite_difference():
    a0, b0 = _random_problem((6, 4), seed=2)
    a1, _ = _random_problem((6, 4), seed=3)

    def vecmat(v, theta):
        matrix = a0 + theta * a1
        return matrix.T @ v

    solver = lstsq_custom_vjp(lstsq_lsmr(atol=1e-6, btol=1e-6, maxiter=10_000))

    def loss(theta):
        sol, _ = solver(vecmat, b0, vecmat_args=(theta,))
        return jnp.sum(sol**2)

    theta = 0.1
    grad = jax.grad(loss)(theta)
    eps = 1e-3
    fd = (loss(theta + eps) - loss(theta - eps)) / (2 * eps)
    np.testing.assert_allclose(np.asarray(grad), np.asarray(fd), rtol=1e-2, atol=1e-3)
