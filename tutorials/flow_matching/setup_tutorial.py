"""Portable numerical setup for the smile notebook; CPU or NVIDIA GPU."""

import importlib.util
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import sys
import tempfile

NUOX_REVISION = "15ce4a69944a1b85bee69d8175557fe4ecd6867d"
PACKAGES = [
    "optax==0.2.6", "flax==0.11.2", "matplotlib>=3.8",
]


def _pip(*requirements):
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
         "--quiet", *requirements], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        print(result.stdout)
        raise RuntimeError("Installation failed; see the package error above.")
    Path(tempfile.gettempdir(), "nuox-install.log").write_text(result.stdout)
    importlib.invalidate_caches()


def _gpu_available():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _prepare_nuox():
    """Prefer the real checkout; an outer nuox directory can be a namespace."""
    root = next((p for p in Path(__file__).resolve().parents
                 if (p / "pyproject.toml").is_file()
                 and (p / "nuox" / "__init__.py").is_file()
                 and (p / "nuox" / "optax_nullspace.py").is_file()), None)
    loaded = sys.modules.get("nuox")
    if root is not None:
        origin = getattr(loaded, "__file__", None)
        if origin and Path(origin).resolve() != (root / "nuox" / "__init__.py").resolve():
            raise RuntimeError("NUOX is loaded from another location. Restart the kernel and run Setup.")
        _pip("-e", str(root))
        # Newly installed editable .pth files do not take effect in this Python session.
        sys.path.insert(0, str(root))
    else:
        try:
            spec = importlib.util.find_spec("nuox")
        except ValueError:  # A partially imported module may have no spec.
            spec = None
        if spec is None or spec.origin is None:
            _pip(f"nuox @ git+https://github.com/h-roy/nuox.git@{NUOX_REVISION}")
    # Changing sys.path alone does not replace a namespace already in sys.modules.
    if loaded is not None and getattr(loaded, "__file__", None) is None:
        for name in list(sys.modules):
            if name == "nuox" or name.startswith("nuox."):
                del sys.modules[name]
    importlib.invalidate_caches()


def setup(device="auto"):
    """Install compatible dependencies before importing JAX; prefer an available GPU.

    NUOX comes from a local checkout when available, otherwise from a pinned
    public revision until this tutorial has a PyPI release to pin.
    """
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError("device must be 'auto', 'cpu', or 'gpu'")
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("Use Python 3.11–3.13 for this pinned JAX stack.")
    gpu = _gpu_available() if device != "cpu" else False
    if device == "gpu" and not gpu:
        raise RuntimeError("Select a GPU runtime in Colab, then rerun Setup.")

    backend = "cuda" if gpu else "cpu"
    if "jax" in sys.modules and os.environ.get("JAX_PLATFORMS") != backend:
        raise RuntimeError("Restart the kernel/session and run Setup first to select the device.")
    os.environ.update(
        JAX_PLATFORMS=backend, XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    if not gpu:
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = ""
    elif os.environ.get("JAX_CUDA_VISIBLE_DEVICES") == "":
        os.environ.pop("JAX_CUDA_VISIBLE_DEVICES")

    loaded = {}
    for name in ("numpy", "jax", "jaxlib", "flax", "optax"):
        if name in sys.modules:
            loaded[name] = getattr(sys.modules[name], "__version__", None)

    print(f"Preparing {'GPU (CUDA 12)' if gpu else 'CPU'} dependencies…")
    jax_package = "jax[cuda12]==0.8.2" if gpu else "jax==0.8.2"
    _pip(jax_package, *PACKAGES)
    _prepare_nuox()

    for name, previous in loaded.items():
        try:
            changed = previous is not None and previous != version(name)
        except PackageNotFoundError:
            changed = False
        if changed:
            raise RuntimeError("Packages updated. Restart the kernel/session once, then Run all.")
    # Check the API, not just whether a directory with this name can be found.
    from nuox import make_project_grad
    from nuox.linalg import lstsq_lsmr
    if not callable(make_project_grad) or not callable(lstsq_lsmr):
        raise ImportError("The imported nuox package does not expose the required optimizer API.")
    print("Ready. The smile data is included; both models train from scratch.")
