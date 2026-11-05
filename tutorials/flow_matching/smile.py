"""Small models, data, and display helpers for the one-step NUOX notebook.

Smile data: Alec Helbling and Polo Chau, Diffusion Explorer,
commit 43cea55f5553782488d53bb4056efb6035e14cbd. All 300 original rows retain
uniform mass. Coordinates, duplicates, and screen-coordinate orientation remain.
"""
import json
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

def data_points():
    """Return the original coordinates, without normalization or extra noise."""
    path = Path(__file__).with_name("smiley_face.json")
    points = np.asarray(json.loads(path.read_text())["points"], dtype=np.float32)
    if points.shape != (300, 2):
        raise ValueError("The pinned smile dataset must contain 300 two-dimensional rows.")
    return points


DATA = data_points()


def model(width=128, depth=3, seed=0):
    """Flat parameters and predict(params, z, r, t); 33,922 default parameters."""
    sizes = [4] + [width] * depth + [2]
    keys = jax.random.split(jax.random.PRNGKey(seed), depth + 1)
    layers = [(jax.random.normal(key, (a, b), dtype=jnp.float32) / jnp.sqrt(a),
               jnp.zeros(b, dtype=jnp.float32))
              for key, a, b in zip(keys, sizes[:-1], sizes[1:])]
    # Initially the network represents an instantaneous flow at every interval.
    layers[0] = (layers[0][0].at[-1].set(0), layers[0][1])
    params, unpack = ravel_pytree(layers)

    def predict(params, z, r, t):
        hidden = jnp.concatenate([z, 1-t, t-r], axis=-1)
        weights = unpack(params)
        for matrix, bias in weights[:-1]:
            hidden = jax.nn.silu(hidden @ matrix + bias)
        matrix, bias = weights[-1]
        return -(hidden @ matrix + bias)

    return params, predict


def fm_batch(index, n=512, seed=51001):
    """Uniform empirical data, independent Gaussian noise, and uniform time."""
    key = jax.random.fold_in(jax.random.PRNGKey(seed), index)
    data_key, noise_key, time_key = jax.random.split(key, 3)
    data = jnp.asarray(DATA)[jax.random.randint(data_key, (n,), 0, len(DATA))]
    noise = jax.random.normal(noise_key, (n, 2), dtype=jnp.float32)
    t = jax.random.uniform(time_key, (n, 1), dtype=jnp.float32)
    return (1-t)*data+t*noise, t, noise-data


def velocity_batch(index, n=128, seed=62002):
    """Fresh independent FM-interpolation collocation with uniform t in [0,1)."""
    z, t, _ = fm_batch(index, n, seed)
    return z, t


def sampler(predict):
    """Noise at t=1 to data at t=0; ordinary FM always queries r=t."""
    def sample(params, noise, nfe, ordinary=False):
        h = 1/nfe

        def step(i, z):
            t = jnp.full((len(z), 1), 1-i*h, dtype=z.dtype)
            r = t if ordinary else t-h
            return z-h*predict(params, z, r, t)

        return jax.lax.fori_loop(0, nfe, step, noise)

    return jax.jit(sample, static_argnums=(2, 3))


def sliced_w1(samples, target=None, directions=128):
    """Exact 1D empirical W1 averaged over fixed directions, including unequal n.

    Integrate the two empirical quantile step functions over their merged break
    points. No target resampling, quantile interpolation, or sample clipping.
    """
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(DATA if target is None else target, dtype=np.float64)
    if (samples.ndim != 2 or target.ndim != 2 or samples.shape[1] != 2
            or target.shape[1] != 2 or not len(samples) or not len(target)):
        raise ValueError("SW1 requires nonempty arrays of two-dimensional points.")
    if not np.isfinite(samples).all() or not np.isfinite(target).all():
        raise FloatingPointError("SW1 requires finite coordinates.")
    angles = np.linspace(0, np.pi, directions, endpoint=False)
    axes = np.stack([np.cos(angles), np.sin(angles)])
    projected, reference = np.sort(samples @ axes, axis=0), np.sort(target @ axes, axis=0)
    n, m = len(samples), len(target)
    breaks = np.unique(np.concatenate([np.arange(n+1)/n, np.arange(m+1)/m]))
    midpoint = (breaks[1:]+breaks[:-1])/2
    first = np.minimum((midpoint*n).astype(int), n-1)
    second = np.minimum((midpoint*m).astype(int), m-1)
    errors = np.abs(projected[first]-reference[second])
    return float(np.mean(np.diff(breaks) @ errors))


def cache_trajectories(predict, teacher, noise, steps=256, chunk=1024):
    """Draw noise before calling this function; chunks do not change the draw."""
    @jax.jit
    def integrate(initial):
        def step(z, index):
            t = jnp.full((len(z), 1), 1-index/steps)
            z = z-predict(teacher, z, t, t)/steps
            return z, z
        _, path = jax.lax.scan(step, initial, jnp.arange(steps))
        return jnp.concatenate([initial[None], path], axis=0).transpose(1, 0, 2)
    pieces = [np.asarray(integrate(noise[i:i+chunk])) for i in range(0, len(noise), chunk)]
    return jnp.asarray(np.concatenate(pieces))


def interpolate(cache, ids, t):
    position = (1-t[:, 0])*(cache.shape[1]-1)
    left = jnp.floor(position).astype(jnp.int32)
    right = jnp.minimum(left+1, cache.shape[1]-1)
    weight = (position-left)[:, None]
    return (1-weight)*cache[ids, left]+weight*cache[ids, right]


def trajectory_batch(cache, index, n=512):
    """Equal groups of interior, noise-start, data-end, and full intervals."""
    key = jax.random.fold_in(jax.random.PRNGKey(62001), index)
    ki, ka, kb = jax.random.split(key, 3)
    ids = jax.random.randint(ki, (n,), 0, len(cache))
    a, b = jax.random.uniform(ka, (n, 1)), jax.random.uniform(kb, (n, 1))
    group = (jnp.arange(n) % 4)[:, None]
    r, t = jnp.minimum(a, b), jnp.maximum(a, b)
    r, t = jnp.where(group == 1, a, r), jnp.where(group == 1, 1., t)
    r, t = jnp.where(group == 2, 0., r), jnp.where(group == 2, a, t)
    r, t = jnp.where(group == 3, 0., r), jnp.where(group == 3, 1., t)
    return interpolate(cache, ids, t), r, t, interpolate(cache, ids, r)


def train(step, optimizer, initial, steps, *args, label="Training"):
    """Run the supplied update; return fixed-decay EMA weights and a small log.

    Neither the objective nor the constraint is redefined here. Every stage
    starts with fresh optimizer state. Both stages use EMA decay 0.99, with no
    checkpoint selection. The returned model is the one evaluated and plotted.
    """
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    params, average, state = initial, initial, optimizer.init(initial)

    @jax.jit
    def block(params, state, average, start, count, args):
        def body(i, carry):
            p, s, ema, _ = carry
            p, s, value = step(p, s, start+i, *args)
            return p, s, .99*ema+.01*p, value
        return jax.lax.fori_loop(0, count, body,
                                (params, state, average, jnp.array(0., jnp.float32)))

    print(f"{label}: {steps:,} updates (the first block compiles)…", flush=True)
    tick, history = perf_counter(), {"step": [], "loss": []}
    for start in range(0, steps, 100):
        count = min(100, steps-start)
        params, state, average, value = block(params, state, average, start, count, args)
        value = float(value)  # Synchronize each block, keeping the notebook responsive.
        if not np.isfinite(value):
            raise FloatingPointError("Training produced a nonfinite loss.")
        history["step"].append(start+count-1)
        history["loss"].append(value)
    jax.block_until_ready(average)
    history["seconds"] = perf_counter()-tick
    print(f"{label} finished in {history['seconds']:.1f}s, including compilation.")
    return average, history


def train_flow(predict, initial, steps=8000):
    """Ordinary conditional flow matching: regress noise-data at uniform times."""
    schedule = optax.cosine_decay_schedule(1e-3, steps, alpha=.1)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(schedule))

    def step(params, state, index):
        z, t, target = fm_batch(index, n=256)
        objective = lambda p: jnp.mean((predict(p, z, t, t)-target)**2)
        value, gradient = jax.value_and_grad(objective)(params)
        updates, state = optimizer.update(gradient, state, params)
        return optax.apply_updates(params, updates), state, value

    return train(step, optimizer, initial, steps, label="Ordinary flow matching")


def make_paths(predict, teacher, count=4096, steps=128):
    """Generate the training targets from independent noise and the frozen FM model."""
    print(f"Caching {count:,} teacher trajectories with {steps} Euler steps…", flush=True)
    tick = perf_counter()
    noise = jax.random.normal(jax.random.PRNGKey(61001), (count, 2))
    paths = cache_trajectories(predict, teacher, noise, steps=steps)
    jax.block_until_ready(paths)
    print(f"Trajectory cache ready in {perf_counter()-tick:.1f}s.")
    return paths


def evaluate(predict, teacher, student, one_step, count=4096, seed=61013):
    """One independent test draw for both models and all step counts; no selection."""
    nfes = (1, 2, 4, 8, 16, 128)
    noise = jax.random.normal(jax.random.PRNGKey(seed), (count, 2))
    sample = sampler(predict)
    samples, scores = {"target": DATA, "noise": np.asarray(noise)}, {"steps": nfes}
    for method, params in (("fm", teacher), ("nuox", student)):
        scores[method] = []
        for n in nfes:
            points = np.asarray(one_step(params, noise) if method == "nuox" and n == 1
                                else sample(params, noise, n, ordinary=method == "fm"))
            samples[f"{method}_{n}"] = points
            scores[method].append(sliced_w1(points))
    z, t = velocity_batch(0, n=4096, seed=64013)
    target = predict(teacher, z, t, t)
    error = predict(student, z, t, t)-target
    scores["velocity_rms"] = float(jnp.sqrt(jnp.mean(error**2)))
    scores["velocity_relative_error"] = float(jnp.linalg.norm(error)/jnp.linalg.norm(target))
    return samples, scores


def _axes(ax, limit):
    ax.set(xlim=(-limit, limit), ylim=(limit, -limit), aspect="equal")
    ax.grid(alpha=.12)
    ax.set_axisbelow(True)


def plot_target(target=DATA):
    """Show the complete target as a standalone figure."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)
    ax.scatter(*target.T, s=8, color="#344054", alpha=.65)
    _axes(ax, max(2., 1.08*float(np.abs(target).max())))
    ax.set_title("Target smile")
    plt.show()
    return fig


def plot_comparison(samples, scores, count=2048):
    """Compare both models at 1, 2, 4, and 8 evaluations on the same noise prefix."""
    import matplotlib.pyplot as plt

    nfes = (1, 2, 4, 8)
    methods = (("fm", "Flow Matching", "#e67e36"), ("nuox", "NUOX", "#4263eb"))
    limit = max(2., 1.08*max(float(np.abs(samples[f"{method}_{n}"][:count]).max())
                            for method, _, _ in methods for n in nfes))
    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.5), sharex=True, sharey=True,
                             constrained_layout=True)
    for row, (method, label, color) in zip(axes, methods):
        row[0].set_ylabel(label)
        for ax, n in zip(row, nfes):
            points = samples[f"{method}_{n}"][:count]
            ax.scatter(*points.T, s=3, color=color, alpha=.65)
            _axes(ax, limit)
            score = scores[method][scores["steps"].index(n)]
            ax.set_xlabel(f"SW₁ = {score:.4f}")
    for ax, n in zip(axes[0], nfes):
        ax.set_title(f"{n} evaluation{'s' if n > 1 else ''}", fontsize=11)
    plt.show()
    return fig


def plot_quality(scores, history):
    """Show the inference-step trade-off and the actual student training loss."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3), constrained_layout=True)
    for method, label, color in [("fm", "Ordinary FM", "#e67e36"),
                                 ("nuox", "NUOX distillation", "#4263eb")]:
        axes[0].plot(scores["steps"], scores[method], "o-", color=color, label=label)
    axes[0].set(xscale="log", yscale="log", xlabel="Network evaluations",
                ylabel="Sliced Wasserstein-1 ↓")
    axes[0].set_xticks(scores["steps"], labels=list(map(str, scores["steps"])))
    axes[0].legend(frameon=False)
    axes[1].plot(history["step"], history["loss"], color="#4263eb")
    axes[1].set(xlabel="NUOX update", ylabel="Training endpoint MSE", yscale="log")
    for ax in axes:
        ax.grid(alpha=.2)
    plt.show()
    return fig


def show_steps(predict, teacher, student, steps=1, count=2048):
    """Interactive comparison on fixed, fresh noise; no training or post-correction."""
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    import matplotlib.pyplot as plt

    noise = jax.random.normal(jax.random.PRNGKey(61014), (count, 2))
    sample = sampler(predict)
    outputs = [np.asarray(sample(p, noise, steps, ordinary=ordinary))
               for p, ordinary in ((teacher, True), (student, False))]
    limit = max(2., 1.08*max(float(np.abs(points).max()) for points in outputs))
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), constrained_layout=True)
    for ax, points, label, color in zip(axes, outputs, ("Ordinary FM", "NUOX distillation"),
                                       ("#e67e36", "#4263eb")):
        ax.scatter(*points.T, s=3, color=color, alpha=.6)
        _axes(ax, limit)
        ax.set_title(f"{label} · {steps} step(s)")
    plt.show()
    return fig
