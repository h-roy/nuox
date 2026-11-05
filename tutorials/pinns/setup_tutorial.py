"""Small CPU/Colab-GPU setup, kept out of the teaching cells."""
from importlib import metadata, util
from pathlib import Path
import os
import subprocess
import sys

REVISION = "15ce4a69944a1b85bee69d8175557fe4ecd6867d"


def _installed(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _gpu_available():
    try:
        check = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        return check.returncode == 0 and "GPU" in check.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def setup():
    # Set before JAX initializes. Existing user settings take precedence.
    os.environ.setdefault("JAX_ENABLE_X64", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    # Prefer the actual checkout package; an outer folder called nuox is not enough.
    for root in (Path.cwd(), *Path.cwd().parents):
        if (root/"nuox/__init__.py").is_file() and (root/"nuox/optax_nullspace.py").is_file():
            sys.path.insert(0, str(root))
            break
    requirements = []
    if not _installed("jax"):
        requirements.append("jax[cuda12]==0.8.2" if _gpu_available() else "jax==0.8.2")
    elif _gpu_available() and not (_installed("jax-cuda12-plugin") or _installed("jax-cuda13-plugin")):
        requirements.append(f"jax[cuda12]=={_installed('jax')}")
    for name, requirement in [("optax", "optax==0.2.8"), ("flax", "flax==0.12.2"),
                              ("chex", "chex>=0.1.86"), ("matfree", "matfree==0.5.2"),
                              ("scipy", "scipy>=1.13"), ("matplotlib", "matplotlib>=3.8"),
                              ("threadpoolctl", "threadpoolctl>=3")]:
        if not _installed(name):
            requirements.append(requirement)
    if requirements:
        print("Installing tutorial dependencies…")
        # Constrain already installed JAX so installing a helper cannot silently upgrade it.
        if _installed("jax"):
            requirements.append(f"jax=={_installed('jax')}")
        else:
            requirements.append("jax==0.8.2")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *requirements])
    spec = util.find_spec("nuox")
    if spec is None or spec.origin is None:
        print("Installing NUOX…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
                               f"git+https://github.com/h-roy/nuox.git@{REVISION}"])
    import jax
    import nuox
    jax.config.update("jax_enable_x64", True)
    if not hasattr(nuox, "make_project_grad"):
        raise ImportError("NUOX was shadowed by another folder. Restart the kernel and run setup first.")
    devices = jax.devices()
    print(f"JAX {jax.__version__} · {devices[0].device_kind} · float64")
    if _gpu_available() and not any(d.platform == "gpu" for d in devices):
        print("GPU is visible but JAX is using CPU. Restart the runtime after setup to activate the GPU.")
