# Quick checks and troubleshooting

Run these from the repository root, which contains `pyproject.toml`:

```bash
python -m pip install -e ".[dev,docs]"
python -c "import nuox; from nuox import make_project_grad; print(nuox.__file__)"
python tutorials/minimal.py
python tutorials/vectorize_model_demo.py
python -m pytest -q
python -m mkdocs build --strict
```

The import should point to the inner `nuox/__init__.py`. The minimal example should reach approximately `[-0.6667, 0.3333, 1.3333]` with constraint error below `1e-5`. Tests must pass, and the strict documentation build must exit successfully. These checks do not require the large tutorials or model downloads.

To also execute the exact code printed in the quick-start, run:

```bash
python scripts/check_quickstart.py
```

To preview the site:

```bash
python -m mkdocs serve
```

Open the URL printed by MkDocs, normally `http://127.0.0.1:8000`. Check that equations render and navigation works. If running on a remote machine, forward port 8000 through your editor or SSH connection. A successful build checks local documentation links, but does not verify external websites or whether tutorial notebooks have been pushed to GitHub.

## ImportError: cannot import name 'make_project_grad'

If the error says `unknown location`, Python may have found the outer repository folder as a namespace rather than the installed package. Install with the same interpreter used to run the example:

```bash
python -m pip install -e .
python -c "import sys, nuox; print(sys.executable); print(nuox.__file__)"
```

Run the installation from the directory containing `pyproject.toml`, not its parent. In a notebook, use `%pip install -e /absolute/path/to/nuox` for your checkout, then restart the kernel. A cached namespace import can survive an installation until the kernel restarts. Avoid naming a local script `nuox.py`.

## GPU not visible

Check `jax.devices()` in the process that actually runs training. On a cluster, that process or notebook kernel must run inside a GPU allocation. An allocated GPU shell does not move a kernel already running on a login node. Use the JAX installation guide linked on the home page for the appropriate GPU build.

## Constraints are not being enforced

Check that projection has started (`wm_epochs=-1` starts immediately), the constraint returns residuals rather than a squared penalty, and no coordinate-wise transform follows projection. Inspect solver accuracy and the learning-rate/correction settings in the [guide](guide.md).
