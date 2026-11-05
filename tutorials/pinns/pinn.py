"""The model and data for pinns.ipynb. No exact interior labels enter training."""
from pathlib import Path
from time import perf_counter
import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import qmc

jax.config.update("jax_enable_x64", True)
WIDTHS = (2, 50, 50, 50, 50, 1)


@jax.jit
def predict(theta, xy):
    offset = 0
    for i, (a, b) in enumerate(zip(WIDTHS[:-1], WIDTHS[1:])):
        w = theta[offset:offset + a*b].reshape(a, b)
        offset += a*b
        bias = theta[offset:offset + b]
        offset += b
        xy = xy @ w + bias
        if i < 4:
            xy = jnp.tanh(xy)
    return xy.ravel()


def initial_parameters():
    # Seed-0 PyTorch Linear draws, saved before any training. No torch dependency.
    theta = np.load(Path(__file__).with_name("initial_weights.npy"))
    digest = hashlib.sha256(theta.astype("<f8").tobytes()).hexdigest()
    assert digest == "b4967199ddc0c484a8f017c78ca1927ccc27e7674f6516ab0e18b9a880af0243"
    return jnp.asarray(theta)


def boundary_points():
    x = np.linspace(0., 2*np.pi, 128, endpoint=False)
    t = np.linspace(0., 1., 100)
    initial = jnp.asarray(np.c_[x, x*0])
    left = jnp.asarray(np.c_[t*0, t])
    right = jnp.asarray(np.c_[t*0 + 2*np.pi, t])
    # Retain the original reference's FFT round trip at t=0.
    values = jnp.asarray(np.fft.ifft(np.fft.fft(np.sin(x))).real)
    return initial, left, right, values


def prefit(theta, constraint):
    """One shared boundary-only initialization: 5,000 Adam steps, learning rate .002."""
    start = perf_counter()
    optimizer = optax.adam(.002)

    @jax.jit
    def block(theta, state):
        def step(carry, _):
            theta, state = carry
            grad = jax.grad(lambda p: jnp.mean(constraint(p)**2))(theta)
            updates, state = optimizer.update(grad, state, theta)
            return (optax.apply_updates(theta, updates), state), None
        return jax.lax.scan(step, (theta, state), None, length=1000)[0]

    state = optimizer.init(theta)
    for _ in range(5):
        theta, state = block(theta, state)
    theta.block_until_ready()
    print(f"Shared boundary prefit: {perf_counter()-start:.1f} s")
    return theta


def audit(theta, residual, constraint):
    """Independent evaluation after training; never used to choose a checkpoint."""
    x, t = np.linspace(0., 2*np.pi, 513), np.linspace(0., 1., 201)
    xx, tt = np.meshgrid(x, t)
    xy = np.c_[xx.ravel(), tt.ravel()]
    prediction = np.concatenate([np.asarray(predict(theta, jnp.asarray(a)))
                                 for a in np.array_split(xy, 16)]).reshape(xx.shape)
    exact = np.sin(xx-30*tt)
    coords = qmc.Sobol(2, scramble=True, seed=654321).random_base2(13)
    xs, ts = 2*np.pi*coords[:, 0], coords[:, 1]
    ic = np.asarray(predict(theta, jnp.asarray(np.c_[xs, xs*0])))-np.sin(xs)
    bc = np.asarray(predict(theta, jnp.asarray(np.c_[ts*0, ts]))
                    - predict(theta, jnp.asarray(np.c_[ts*0+2*np.pi, ts])))
    r = np.asarray(jax.jit(residual)(theta, jnp.asarray(coords*[2*np.pi, 1.]), 30.))
    c = np.asarray(constraint(theta))
    metrics = dict(rmse=float(np.sqrt(np.mean((prediction-exact)**2))),
                   relative_l2=float(np.linalg.norm(prediction-exact)/np.linalg.norm(exact)),
                   pde_rms=float(np.sqrt(np.mean(r*r))),
                   ic_rms=float(np.sqrt(np.mean(ic*ic))),
                   ic_max=float(np.abs(ic).max()),
                   periodic_rms=float(np.sqrt(np.mean(bc*bc))),
                   training_constraint_max=float(np.abs(c).max()),
                   independent_constraint_max=float(max(np.abs(ic).max(), np.abs(bc).max())))
    assert np.isfinite(prediction).all() and np.isfinite(r).all()
    arrays = dict(x=x, t=t, prediction=prediction, exact=exact, ic_x=xs, ic_error=ic,
                  bc_t=ts, bc_error=bc, pde_residual=r, training_conditions=c)
    return metrics, arrays
