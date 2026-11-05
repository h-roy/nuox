"""Gauss–Newton geometry, fixed curricula and line search for pinns.ipynb.

The notebook supplies the residual, loss, constraint and actual NUOX step.
Dense PDE Jacobians and small Gram matrices are used for Gauss–Newton;
the NUOX constraint projection uses the package's matrix-free LSMR.
"""
from time import perf_counter
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh, cho_factor, cho_solve
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

NUOX_PHASES = [([(b, 20) for b in (1, 2, 4, 6, 8, 10, 15, 20)]+[(25, 18)], 256, 128, 1e-6),
               ([(25, 40), (30, 40)], 512, 256, 1e-7)]
PINN_PHASES = [[(b, 30) for b in (1, 2, 4, 6, 8, 10, 15, 20, 25, 30)], [(30, 30)]]


def whiten(v, B, eigenvalues, damping):
    """Apply (R.T R + damping I)^(-1/2), without a parameter-space matrix."""
    a, b = jnp.sqrt(damping), jnp.sqrt(eigenvalues+damping)
    return v/a - B.T @ ((B @ v)/(a*b*(a+b)))


def geometry(R):
    gram = R @ R.T
    eigenvalues, Q = eigh((gram+gram.T)*.5, check_finite=False)
    return jnp.asarray(Q.T @ R), jnp.asarray(np.maximum(eigenvalues, 0.))


def train(theta, residual, loss, constraint, direction=None, refinement_steps=30):
    """Run the fixed curriculum and optional refinement. None selects the penalty baseline.

    direction(theta, g, B, eigenvalues, damping, maxiter, tol) returns a descent
    update. There is no target field error or analytic solution in this loop.
    """
    is_nuox = direction is not None
    name = "NUOX" if is_nuox else "Penalty PINN"
    guard = jnp.asarray(qmc.Sobol(2, scramble=True, seed=2027).random_base2(13)*[2*np.pi, 1.])

    @jax.jit
    def pde_model(p, xy, beta):
        def single(w, z):
            return residual(w, z[None, :], beta)[0]
        scale = jnp.sqrt(len(xy))
        return (residual(p, xy, beta)/scale,
                jax.vmap(jax.grad(single), in_axes=(None, 0))(p, xy)/scale)

    @jax.jit
    def conditions(p):
        c = constraint(p)
        return jnp.concatenate((c[:128]/jnp.sqrt(128.), c[128:]/jnp.sqrt(100.)))

    condition_jacobian = jax.jit(jax.jacrev(conditions))

    @jax.jit
    def parts(p, beta):
        c = constraint(p)
        return loss(p, guard, beta), jnp.mean(c[:128]**2)+jnp.mean(c[128:]**2)

    @jax.jit
    def linear_error(p, update):
        return constraint(p)+jax.jvp(constraint, (p,), (update,))[1]

    phases = NUOX_PHASES if is_nuox else [(s, 1000, 0, 0.) for s in PINN_PHASES]
    history, start = [], perf_counter()
    # Limit BLAS oversubscription; JAX still uses the selected CPU/GPU device.
    with threadpool_limits(limits=4, user_api="blas"):
        for phase, (schedule, n, maxiter, tol) in enumerate(phases, 1):
            sampler = qmc.Sobol(2, scramble=True, seed=2026)
            trials = 3 if is_nuox or phase == 2 else 6
            backtracks = 6 if is_nuox else (7 if phase == 1 else 4)
            weight = 10. if is_nuox else 1.
            for beta, steps in schedule:
                damping = 1e-3
                for _ in range(steps):
                    # Non-power-of-two sample count is intentional in the baseline.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="The balance properties of Sobol")
                        xy = jnp.asarray(sampler.random(n)*[2*np.pi, 1.])
                    r, R = (np.asarray(v) for v in pde_model(theta, xy, float(beta)))
                    if is_nuox:
                        B, ev = geometry(R)
                        g = jnp.asarray(R.T @ r)
                    else:
                        D = np.concatenate([R, np.asarray(condition_jacobian(theta))])
                        b = np.concatenate([r, np.asarray(conditions(theta))])
                        gram = D @ D.T
                        gram = (gram+gram.T)*.5
                    old_f, old_c = (float(v) for v in parts(theta, float(beta)))
                    accepted, alpha, linear_max = False, 0., None
                    for _ in range(trials):
                        if is_nuox:
                            update = direction(theta, g, B, ev, damping, maxiter, tol)
                            linear_max = float(jnp.max(jnp.abs(linear_error(theta, update))))
                        else:
                            factor = cho_factor(gram+damping*np.eye(len(b)), lower=True, check_finite=False)
                            update = -jnp.asarray(D.T @ cho_solve(factor, b, check_finite=False))
                        if np.isfinite(np.asarray(update)).all():
                            alpha = min(1., .5/max(float(jnp.linalg.norm(update)), 1e-30))
                            for _ in range(backtracks):
                                candidate = theta+alpha*update
                                f, c = (float(v) for v in parts(candidate, float(beta)))
                                if np.isfinite(f+c) and f+weight*c < old_f+weight*old_c:
                                    theta, accepted = candidate, True
                                    break
                                alpha *= .5
                        if accepted:
                            cutoff, floor = (.5, 1e-8) if is_nuox else (.8, 1e-6)
                            damping = max(floor, damping*(.7 if alpha > cutoff else 1.))
                            break
                        damping *= 10.
                    f, c = (float(v) for v in parts(theta, float(beta)))
                    history.append(dict(step=len(history)+1, phase=phase, beta=beta,
                                        pde_mse=f, condition_mse=c, accepted=accepted,
                                        constraint_max=float(jnp.max(jnp.abs(constraint(theta)))),
                                        linear_max=linear_max, alpha=alpha, damping=damping,
                                        seconds=perf_counter()-start))
                    if not accepted and (is_nuox or phase == 2):
                        raise RuntimeError(f"{name} stopped at update {len(history)}: line search failed. "
                                           "The full recipe did not complete; do not interpret this as convergence.")
                print(f"{name:12s}  β={beta:2d}  step {len(history):3d}  "
                      f"PDE RMS {np.sqrt(f):.2e}  max |c| {history[-1]['constraint_max']:.2e}", flush=True)
    seconds = perf_counter()-start
    print(f"{name}: {sum(h['accepted'] for h in history)}/{len(history)} accepted; {seconds:.1f} s")
    if is_nuox and refinement_steps:
        return refine(theta, history, seconds, residual, loss, constraint, direction,
                      steps=refinement_steps)
    return theta, history, seconds


def refine(theta, history, seconds, residual, loss, constraint, direction, steps=30):
    """Finish with damped, correction-only NUOX updates in PDE Gauss–Newton geometry.

    The exact solution is never used. Accept lower nonlinear constraint MSE while
    keeping guard PDE MSE within four times its value before refinement.
    """
    start = perf_counter()
    history = list(history)
    guard = jnp.asarray(qmc.Sobol(2, scramble=True, seed=2027).random_base2(13)*[2*np.pi, 1.])
    points = jnp.asarray(qmc.Sobol(2, scramble=True, seed=2029).random_base2(9)*[2*np.pi, 1.])

    @jax.jit
    def pde_jacobian(p, xy):
        def single(w, z):
            return residual(w, z[None, :], 30.)[0]
        return jax.vmap(jax.grad(single), in_axes=(None, 0))(p, xy)/jnp.sqrt(len(xy))

    @jax.jit
    def check(p):
        c = constraint(p)
        return jnp.mean(c*c), jnp.max(jnp.abs(c)), jnp.sqrt(jnp.mean(c[:128]**2)), loss(p, guard, 30.)

    base = [float(v) for v in check(theta)]
    regularization = 1.
    corrections = 0
    with threadpool_limits(limits=4, user_api="blas"):
        for k in range(steps):
            old = check(theta)
            if float(old[1]) < 1e-7:
                break
            B, ev = geometry(np.asarray(pde_jacobian(theta, points)))
            accepted = False
            for _ in range(4):
                # The same notebook optimizer, now with zero task gradient and
                # a damped LSMR solve to avoid unstable feasibility corrections.
                update = direction(theta, jnp.zeros_like(theta), B, ev, 1e-3,
                                   256, 1e-10, regularization)
                alpha = 1.
                for _ in range(8):
                    candidate = theta+alpha*update
                    values = [float(v) for v in check(candidate)]
                    if np.isfinite(values).all() and values[0] < float(old[0]) and values[3] < 4*base[3]:
                        theta, accepted = candidate, True
                        break
                    alpha *= .5
                if accepted:
                    break
                regularization *= 10.
            if accepted:
                corrections += 1
                if alpha == 1.:
                    regularization = max(1e-4, regularization*.5)
                elif alpha < .25:
                    regularization *= 2.
            c = constraint(theta)
            values = [float(v) for v in check(theta)]
            history.append(dict(step=len(history)+1, phase=3, beta=30, accepted=accepted,
                                pde_mse=values[3], constraint_mse=values[0], ic_rms=values[2],
                                condition_mse=float(jnp.mean(c[:128]**2)+jnp.mean(c[128:]**2)),
                                constraint_max=values[1], linear_max=None, alpha=alpha,
                                damping=1e-3, solver_damping_next=regularization,
                                seconds=seconds+perf_counter()-start))
            if k == 0 or (k+1) % 10 == 0 or not accepted:
                print(f"NUOX refinement {k+1:2d}: IC RMS {values[2]:.2e}; "
                      f"PDE RMS {np.sqrt(values[3]):.2e}", flush=True)
            if not accepted:
                print("Refinement stopped: no acceptable correction; retaining the last accepted parameters.")
                break
    elapsed = perf_counter()-start
    print(f"NUOX: {corrections} refinement updates in {elapsed:.1f} s; "
          f"{seconds+elapsed:.1f} s total training")
    return theta, history, seconds+elapsed
