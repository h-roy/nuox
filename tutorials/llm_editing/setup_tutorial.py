"""One-time notebook setup. Uses the same model and precision on CPU and GPU."""

import importlib.util
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import sys
import tempfile

NUOX_REVISION = "15ce4a69944a1b85bee69d8175557fe4ecd6867d"
PACKAGES = [
    "easydel==0.2.0.2", "optax==0.2.6", "transformers==4.57.3",
    "matplotlib>=3.8", "ipywidgets>=8",
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


def setup(device="auto"):
    """Install compatible dependencies before importing JAX; prefer an available GPU.

    PyTorch is used only to load the checkpoint. It can be a CPU build even when
    JAX trains on a GPU. NUOX comes from a local checkout when available, otherwise
    from a pinned public revision until this tutorial has a PyPI release to pin.
    """
    if device not in ("auto", "cpu", "gpu"):
        raise ValueError("device must be 'auto', 'cpu', or 'gpu'")
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("Use Python 3.11–3.13 for this pinned EasyDeL version.")
    gpu = _gpu_available() if device != "cpu" else False
    if device == "gpu" and not gpu:
        raise RuntimeError("Select a GPU runtime in Colab, then rerun Setup.")

    backend = "cuda" if gpu else "cpu"
    if "jax" in sys.modules and os.environ.get("JAX_PLATFORMS") != backend:
        raise RuntimeError("Restart the kernel/session and run Setup first to select the device.")
    os.environ.update(
        JAX_PLATFORMS=backend, EASYDEL_AUTO="0", ENABLE_DISTRIBUTED_INIT="0",
        TOKENIZERS_PARALLELISM="false", XLA_PYTHON_CLIENT_PREALLOCATE="false",
    )
    if not gpu:
        os.environ["JAX_CUDA_VISIBLE_DEVICES"] = ""
    elif os.environ.get("JAX_CUDA_VISIBLE_DEVICES") == "":
        os.environ.pop("JAX_CUDA_VISIBLE_DEVICES")

    loaded = {}
    for name in ("numpy", "jax", "jaxlib", "flax", "easydel", "transformers", "optax"):
        if name in sys.modules:
            loaded[name] = getattr(sys.modules[name], "__version__", None)

    print(f"Preparing {'GPU (CUDA 12)' if gpu else 'CPU'} dependencies…")
    jax_package = "jax[cuda12]==0.8.2" if gpu else "jax==0.8.2"
    _pip(jax_package, *PACKAGES)
    if importlib.util.find_spec("torch") is None:
        _pip("torch==2.8.0", "--index-url", "https://download.pytorch.org/whl/cpu")
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.find_spec("nuox")
    if spec is None or spec.origin is None:
        sys.modules.pop("nuox", None)  # Discard a cached namespace for the outer checkout folder.
        if (root / "nuox" / "optax_nullspace.py").is_file():
            _pip("-e", str(root))
        else:
            _pip(f"nuox @ git+https://github.com/h-roy/nuox.git@{NUOX_REVISION}")
    if (root / "nuox" / "optax_nullspace.py").is_file():
        sys.path.insert(0, str(root))  # Editable .pth files load only on the next Python start.

    for name, previous in loaded.items():
        try:
            changed = previous is not None and previous != version(name)
        except PackageNotFoundError:
            changed = False
        if changed:
            raise RuntimeError("Packages updated. Restart the kernel/session once, then Run all.")
    print("Ready. The first model load downloads the public Qwen checkpoint (about 1 GB).")
