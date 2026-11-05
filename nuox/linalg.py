"""Least-squares solvers used by the null-space projection."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

LARGE_VALUE = 1e10

VecMat = Callable[[jnp.ndarray, Any], jnp.ndarray]
Solver = Callable[[VecMat, jnp.ndarray, Tuple[Any, ...]], Tuple[jnp.ndarray, dict]]

try:
    register_dataclass = jax.tree_util.register_dataclass
except AttributeError:  # pragma: no cover

    def register_dataclass(cls):
        fields = dataclasses.fields(cls)

        def to_tree(instance):
            return [getattr(instance, f.name) for f in fields], None

        def from_tree(_, children):
            return cls(*children)

        jax.tree_util.register_pytree_node(cls, to_tree, from_tree)
        return cls


def lstsq_lsmr(
    *,
    atol: float = 1e-6,
    btol: float = 1e-6,
    ctol: float = 1e-8,
    maxiter: int = 1_000_000,
    damp=0.0,
    while_loop=jax.lax.while_loop,
) -> Solver:
    """LSMR -- like in SciPy, but using JAX."""

    @register_dataclass
    @dataclasses.dataclass
    class State:
        itn: int
        alpha: float
        u: jax.Array
        v: jax.Array
        alphabar: float
        rhobar: float
        rho: float
        zeta: float
        sbar: float
        cbar: float
        zetabar: float
        hbar: jax.Array
        h: jax.Array
        x: jax.Array
        betadd: float
        thetatilde: float
        rhodold: float
        betad: float
        tautildeold: float
        d: float
        normA2: float
        maxrbar: float
        minrbar: float
        normA: float
        condA: float
        normx: float
        normar: float
        normr: float
        istop: int

    def run(vecmat, b, vecmat_args=()):
        def vecmat_noargs(v):
            return vecmat(v, *vecmat_args)

        (ncols,) = jax.eval_shape(vecmat, b, *vecmat_args).shape
        state, normb, matvec_noargs = init(vecmat_noargs, b, ncols=ncols)
        step_fun = make_step(matvec_noargs, normb=normb)
        cond_fun = make_cond_fun()
        state = while_loop(cond_fun, step_fun, state)
        stats_ = stats(state)
        return state.x, stats_

    def init(vecmat, b, ncols: int):
        normb = jnp.linalg.norm(b)
        x = jnp.zeros(ncols, b.dtype)
        beta = normb

        u = b / jnp.where(beta > 0, beta, 1.0)

        v, matvec = jax.vjp(vecmat, u)
        alpha = jnp.linalg.norm(v)
        v = v / jnp.where(alpha > 0, alpha, 1)
        v = jnp.where(beta == 0, jnp.zeros_like(v), v)
        alpha = jnp.where(beta == 0, jnp.zeros_like(alpha), alpha)

        zetabar = alpha * beta
        alphabar = alpha
        rho = 1.0
        rhobar = 1.0
        cbar = 1.0
        sbar = 0.0

        h = v
        hbar = jnp.zeros(ncols, b.dtype)

        betadd = beta
        betad = 0.0
        rhodold = 1.0
        tautildeold = 0.0
        thetatilde = 0.0
        zeta = 0.0
        d = 0.0

        normA2 = alpha * alpha
        maxrbar = 0.0
        minrbar = 1e10
        normA = jnp.sqrt(normA2)
        condA = 1.0
        normx = 0.0

        normr = beta
        normar = alpha * beta

        state = State(
            itn=0,
            alpha=alpha,
            u=u,
            v=v,
            alphabar=alphabar,
            rho=rho,
            rhobar=rhobar,
            zeta=zeta,
            sbar=sbar,
            cbar=cbar,
            zetabar=zetabar,
            hbar=hbar,
            h=h,
            x=x,
            betadd=betadd,
            thetatilde=thetatilde,
            rhodold=rhodold,
            betad=betad,
            tautildeold=tautildeold,
            d=d,
            normA2=normA2,
            maxrbar=maxrbar,
            minrbar=minrbar,
            normar=normar,
            normr=normr,
            normA=normA,
            condA=condA,
            normx=normx,
            # x=0 is already the minimum-norm solution if b=0 or A.T @ b=0.
            # A bidiagonalization step would divide by zero in either exact case.
            istop=jnp.where(beta == 0, 8, jnp.where(alpha == 0, 9, 0)),
        )
        state = jax.tree_util.tree_map(lambda z: jnp.asarray(z), state)
        return state, normb, lambda *a: matvec(*a)[0]

    def make_step(matvec, normb: float) -> Callable:
        def step(state: State) -> State:
            Av, A_t = jax.vjp(matvec, state.v)
            u = Av - state.alpha * state.u
            beta = jnp.linalg.norm(u)

            u = u / jnp.where(beta > 0, beta, 1.0)
            v = A_t(u)[0] - beta * state.v
            alpha = jnp.linalg.norm(v)
            v = v / jnp.where(alpha > 0, alpha, 1)

            chat, shat, alphahat = _sym_ortho(state.alphabar, damp)

            rhoold = state.rho
            c, s, rho = _sym_ortho(alphahat, beta)
            thetanew = s * alpha
            alphabar = c * alpha

            rhobarold = state.rhobar
            zetaold = state.zeta
            thetabar = state.sbar * rho
            rhotemp = state.cbar * rho
            cbar, sbar, rhobar = _sym_ortho(rhotemp, thetanew)
            zeta = cbar * state.zetabar
            zetabar = -sbar * state.zetabar

            hbar = state.h - state.hbar * (thetabar * rho / (rhoold * rhobarold))
            x = state.x + (zeta / (rho * rhobar)) * hbar
            h = v - state.h * (thetanew / rho)

            betaacute = chat * state.betadd
            betacheck = -shat * state.betadd

            betahat = c * betaacute
            betadd = -s * betaacute

            thetatildeold = state.thetatilde
            ctildeold, stildeold, rhotildeold = _sym_ortho(state.rhodold, thetabar)
            thetatilde = stildeold * rhobar
            rhodold = ctildeold * rhobar
            betad = -stildeold * state.betad + ctildeold * betahat

            tautildeold = (zetaold - thetatildeold * state.tautildeold) / rhotildeold
            taud = (zeta - thetatilde * tautildeold) / rhodold
            d = state.d + betacheck * betacheck
            normr = jnp.sqrt(d + (betad - taud) ** 2 + betadd * betadd)

            normA2 = state.normA2 + beta * beta
            normA = jnp.sqrt(normA2)
            normA2 = normA2 + alpha * alpha

            maxrbar = jnp.maximum(state.maxrbar, rhobarold)
            minrbar = jnp.where(state.itn > 1, jnp.minimum(state.minrbar, rhobarold), state.minrbar)
            condA = jnp.maximum(maxrbar, rhotemp) / jnp.minimum(minrbar, rhotemp)

            normar = jnp.abs(zetabar)
            normx = jnp.linalg.norm(x)

            itn = state.itn + 1
            test1 = normr / normb
            z = normA * normr
            z_safe = jnp.where(z != 0, z, 1.0)
            test2 = jnp.where(z != 0, normar / z_safe, LARGE_VALUE)
            test3 = 1 / condA
            t1 = test1 / (1 + normA * normx / normb)
            rtol = btol + atol * normA * normx / normb

            istop = 0
            istop = jnp.where(normar == 0, 9, istop)
            istop = jnp.where(normb == 0, 8, istop)
            istop = jnp.where(itn >= maxiter, 7, istop)
            istop = jnp.where(1 + test3 <= 1, 6, istop)
            istop = jnp.where(1 + test2 <= 1, 5, istop)
            istop = jnp.where(1 + t1 <= 1, 4, istop)
            istop = jnp.where(test3 <= ctol, 3, istop)
            istop = jnp.where(test2 <= atol, 2, istop)
            istop = jnp.where(test1 <= rtol, 1, istop)

            return State(
                itn=itn,
                alpha=alpha,
                u=u,
                v=v,
                alphabar=alphabar,
                rho=rho,
                rhobar=rhobar,
                zeta=zeta,
                sbar=sbar,
                cbar=cbar,
                zetabar=zetabar,
                hbar=hbar,
                h=h,
                x=x,
                betadd=betadd,
                thetatilde=thetatilde,
                rhodold=rhodold,
                betad=betad,
                tautildeold=tautildeold,
                d=d,
                normA2=normA2,
                maxrbar=maxrbar,
                minrbar=minrbar,
                normar=normar,
                normr=normr,
                normA=normA,
                condA=condA,
                normx=normx,
                istop=istop,
            )

        return step

    def make_cond_fun() -> Callable:
        def cond(state):
            state_flat, _ = jax.flatten_util.ravel_pytree(state)
            no_nans = jnp.logical_not(jnp.any(jnp.isnan(state_flat)))
            proceed = jnp.where(state.istop == 0, True, False)
            return jnp.logical_and(proceed, no_nans)

        return cond

    def stats(state: State) -> dict:
        return {
            "iteration_count": state.itn,
            "norm_residual": state.normr,
            "norm_At_residual": state.normar,
            "norm_A": state.normA,
            "cond_A": state.condA,
            "norm_x": state.normx,
            "istop": state.istop,
        }

    return run


def _sym_ortho(a, b):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    dtype = jnp.result_type(a.dtype, b.dtype)
    a = a.astype(dtype)
    b = b.astype(dtype)

    idx = 3
    idx = jnp.where(jnp.abs(b) > jnp.abs(a), 2, idx)
    idx = jnp.where(a == 0, 1, idx)
    idx = jnp.where(b == 0, 0, idx)

    branches = [_sym_ortho_0, _sym_ortho_1, _sym_ortho_2, _sym_ortho_3]
    return jax.lax.switch(idx, branches, a, b)


def _sym_ortho_0(a, b):
    zero = jnp.zeros((), dtype=a.dtype)
    return jnp.sign(a), zero, jnp.abs(a)


def _sym_ortho_1(a, b):
    zero = jnp.zeros((), dtype=a.dtype)
    return zero, jnp.sign(b), jnp.abs(b)


def _sym_ortho_2(a, b):
    tau = a / b
    s = jnp.sign(b) / jnp.sqrt(1 + tau * tau)
    c = s * tau
    r = b / s
    return c, s, r


def _sym_ortho_3(a, b):
    tau = b / a
    c = jnp.sign(a) / jnp.sqrt(1 + tau * tau)
    s = c * tau
    r = a / c
    return c, s, r


def lstsq_via_normaleq(solve: Callable) -> Solver:
    def lstsq(vecmat: VecMat, rhs: jnp.ndarray, vecmat_args: Tuple[Any, ...] = ()):  # noqa: D401
        def vecmat_noargs(v):
            return vecmat(v, *vecmat_args)

        x_like = jax.eval_shape(vecmat_noargs, rhs)
        if rhs.size <= x_like.size:
            return lstsq_A_wide(vecmat_noargs, rhs)
        return lstsq_A_tall(vecmat_noargs, rhs)

    def lstsq_A_wide(vecmat_noargs, rhs):
        def matvec_noargs(v):
            return jax.linear_transpose(vecmat_noargs, rhs)(v)[0]

        y, stats = solve(lambda s: matvec_noargs(vecmat_noargs(s)), rhs)
        return vecmat_noargs(y), stats

    def lstsq_A_tall(vecmat_noargs, rhs):
        def matvec_noargs(v):
            return jax.linear_transpose(vecmat_noargs, rhs)(v)[0]

        return solve(lambda s: vecmat_noargs(matvec_noargs(s)), vecmat_noargs(rhs))

    return lstsq


def solve_iterative_cg(maxiter: int, tol=1e-5) -> Callable:
    def solve(matvec, rhs):
        sol, stats = jax.scipy.sparse.linalg.cg(matvec, rhs, maxiter=maxiter, tol=tol)
        return sol, stats

    return solve


def solve_materialize(dense_solve: Callable) -> Callable:
    def solve(matvec, rhs):
        return dense_solve(jax.jacfwd(matvec)(rhs), rhs)

    return solve


def dense_solve_lu() -> Callable:
    def solve(A, b):
        dtype = A.dtype
        a64 = A.astype(jnp.float64)
        b64 = b.astype(jnp.float64)
        sol64 = jnp.linalg.solve(a64, b64)
        return sol64.astype(dtype), {}

    return solve


def _materialise_matrix(
    vecmat: VecMat, rhs: jnp.ndarray, vecmat_args: Tuple[Any, ...]
) -> jnp.ndarray:
    def apply(v):
        return vecmat(v, *vecmat_args)

    eye = jnp.eye(rhs.shape[0], dtype=rhs.dtype)
    columns = jax.vmap(apply)(eye)
    return columns.T


def lstsq_jnp():
    def lstsq_(vecmat, rhs, vecmat_args=()):
        matrix_T = jax.jacrev(lambda s: vecmat(s, *vecmat_args))(rhs)
        return jnp.linalg.lstsq(matrix_T.T, rhs)[0], {}

    return lstsq_


def lstsq_custom_vjp(lstsq_fun: Callable) -> Callable:
    def lstsq_public(vecmat, rhs, vecmat_args: tuple = ()):  # noqa: D401
        vecmat_, args = jax.closure_convert(lambda s: vecmat(s, *vecmat_args), rhs)
        return lstsq_fun(vecmat_, rhs, args)

    def lstsq_fwd(vecmat, rhs, vecmat_args: tuple = ()):  # noqa: D401
        x, stats = lstsq_public(vecmat, rhs, vecmat_args=vecmat_args)
        cache = {"x": x, "rhs": rhs, "vecmat_args": vecmat_args}
        return (x, stats), cache

    def lstsq_rev(vecmat, cache, dmu_dx):
        dmu_dx, _ = dmu_dx
        x_like = jax.eval_shape(vecmat, cache["rhs"], *cache["vecmat_args"])
        if cache["rhs"].size <= x_like.size:
            return lstsq_rev_wide(vecmat, cache, dmu_dx)
        return lstsq_rev_tall(vecmat, cache, dmu_dx)

    def lstsq_rev_tall(vecmat, cache, dmu_dx):
        x = cache["x"]
        rhs = cache["rhs"]
        vecmat_args = cache["vecmat_args"]

        def vecmat_noargs(z):
            return vecmat(z, *vecmat_args)

        def matvec_noargs(z):
            return jax.vjp(vecmat_noargs, rhs)[1](z)[0]

        dmu_db = lstsq_public(matvec_noargs, dmu_dx)[0]
        p = lstsq_public(vecmat_noargs, -dmu_db)[0]

        Ax_minus_b = matvec_noargs(x) - rhs
        Ap = matvec_noargs(p)

        @jax.grad
        def grad_theta(theta):
            rA = vecmat(Ax_minus_b, *theta)
            pAA = vecmat(Ap, *theta)
            return jnp.dot(rA, p) + jnp.dot(pAA, x)

        dmu_dparams = grad_theta(vecmat_args)
        return dmu_db, dmu_dparams

    def lstsq_rev_wide(vecmat, cache, dmu_dx):
        x = cache["x"]
        rhs = cache["rhs"]
        vecmat_args = cache["vecmat_args"]

        def vecmat_noargs(z):
            return vecmat(z, *vecmat_args)

        def matvec_noargs(z):
            return jax.linear_transpose(vecmat_noargs, rhs)(z)[0]

        y = lstsq_public(matvec_noargs, x)[0]
        p = dmu_dx - lstsq_public(vecmat_noargs, matvec_noargs(dmu_dx))[0]
        q = lstsq_public(vecmat_noargs, p - dmu_dx)[0]

        @jax.grad
        def grad_theta(theta):
            yA = vecmat(y, *theta)
            qA = vecmat(q, *theta)
            return jnp.dot(yA, p) + jnp.dot(qA, x)

        grad_vecmat_args = grad_theta(vecmat_args)
        grad_rhs = -q
        return grad_rhs, grad_vecmat_args

    lstsq_fun = jax.custom_vjp(lstsq_fun, nondiff_argnums=(0,))
    lstsq_fun.defvjp(lstsq_fwd, lstsq_rev)
    return lstsq_public
