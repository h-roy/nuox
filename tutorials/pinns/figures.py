"""Shared-scale figures and a small measured-results table for the notebook."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown

COLORS = {"Penalty PINN": "#d17934", "NUOX": "#2475b0"}


def style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 120,
                         "savefig.dpi": 180})


def metric_table(results):
    rows = [("Independent IC RMS ↓", "ic_rms"),
            ("Independent IC maximum ↓", "ic_max"),
            ("Field RMSE ↓", "rmse"), ("Relative L2 ↓", "relative_l2"),
            ("Independent PDE RMS ↓", "pde_rms"),
            ("Training seconds (with compilation)", "training_seconds")]
    names = list(results)
    lines = ["| Measurement | " + " | ".join(names) + " |",
             "|---|" + "---:|"*len(names)]
    for label, key in rows:
        values = [f"{results[n]['metrics'][key]:.7g}" for n in names]
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    for label, key in [("Accepted updates", "accepted_updates"), ("Attempted updates", "attempted_updates")]:
        lines.append(f"| {label} | " + " | ".join(str(results[n]["metrics"][key]) for n in names) + " |")
    return Markdown("\n".join(lines))


def solution_plot(results):
    style()
    arrays = next(iter(results.values()))["arrays"]
    extent = [0, 2*np.pi, 0, 1]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), layout="constrained", sharey=True)
    fields = [("Exact solution", arrays["exact"])]
    fields += [(f"{n}\nRMSE {r['metrics']['rmse']:.2e}", r["arrays"]["prediction"]) for n, r in results.items()]
    for ax, (title, field) in zip(axes, fields):
        im = ax.imshow(field, origin="lower", aspect="auto", extent=extent, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set(title=title, xlabel="Position x", xticks=[0, np.pi, 2*np.pi], xticklabels=["0", "π", "2π"])
    axes[0].set_ylabel("Time t")
    fig.colorbar(im, ax=axes, label="u(x,t)", shrink=.9)
    fig.suptitle("Convection at β = 30", fontsize=15)
    return fig


def error_plot(results):
    style()
    errors = [np.abs(r["arrays"]["prediction"]-r["arrays"]["exact"]) for r in results.values()]
    vmax = max(a.max() for a in errors)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), layout="constrained", sharey=True)
    for ax, (name, result), error in zip(axes, results.items(), errors):
        im = ax.imshow(error, origin="lower", aspect="auto", extent=[0, 2*np.pi, 0, 1],
                       cmap="magma", vmin=0, vmax=vmax)
        ax.set(title=f"{name} · RMSE {result['metrics']['rmse']:.2e}", xlabel="Position x",
               xticks=[0, np.pi, 2*np.pi], xticklabels=["0", "π", "2π"])
    axes[0].set_ylabel("Time t")
    fig.colorbar(im, ax=axes, label="Absolute error", shrink=.9)
    fig.suptitle("Where does the solution differ? · common color scale", fontsize=14)
    return fig


def condition_plot(results):
    """Only the initial condition: signed errors and their absolute magnitudes."""
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for name, result in results.items():
        a = result["arrays"]
        order = np.argsort(a["ic_x"])
        x, error = a["ic_x"][order], a["ic_error"][order]
        axes[0].plot(x, error, label=name, color=COLORS[name], lw=1.8)
        axes[1].semilogy(x, np.where(np.abs(error)>0, np.abs(error), np.nan),
                         color=COLORS[name], lw=1.5)
    for ax in axes:
        ax.set(xlabel="Position x", xticks=[0, np.pi, 2*np.pi], xticklabels=["0", "π", "2π"])
        ax.grid(alpha=.15)
    axes[0].axhline(0, color="gray", ls=":", lw=.8)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].set(title="Initial-condition residual", ylabel="u(x,0) − sin(x)")
    axes[1].set(title="Absolute error · logarithmic scale", ylabel="Absolute IC residual")
    axes[0].legend(frameon=False, loc="upper left")
    ratio = results["Penalty PINN"]["metrics"]["ic_rms"]/results["NUOX"]["metrics"]["ic_rms"]
    fig.suptitle(f"Initial-condition preservation · {ratio:.1f}× lower RMS with NUOX", fontsize=14)
    return fig


def training_plot(results):
    style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), layout="constrained")
    for name, result in results.items():
        h = result["history"]
        steps = [r["step"] for r in h]
        axes[0].semilogy(steps, np.maximum([r["pde_mse"] for r in h], 1e-16), color=COLORS[name], label=name)
        refinement = [r for r in h if r.get("phase") == 3 and "ic_rms" in r]
        if refinement:
            axes[1].semilogy(range(1, len(refinement)+1), [r["ic_rms"] for r in refinement],
                             color=COLORS[name])
        axes[2].step(steps, [r["beta"] for r in h], color=COLORS[name])
    for ax, title in zip(axes, ["PDE loss at the current β", "NUOX IC refinement · training RMS", "Speed curriculum"]):
        ax.set(title=title, xlabel="Attempted update")
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Refinement update")
    axes[2].set_ylabel("β")
    return fig


def save_results(results, figures, folder="results"):
    """Save the freshly measured endpoints; training never reads this directory."""
    folder = Path(folder)
    folder.mkdir(exist_ok=True)
    metrics = {}
    for name, result in results.items():
        key = "nuox" if name == "NUOX" else "pinn"
        metrics[name] = result["metrics"]
        np.savez_compressed(folder/f"{key}.npz", params=np.asarray(result["params"]), **result["arrays"])
        (folder/f"{key}_history.json").write_text(json.dumps(result["history"], indent=2))
    (folder/"metrics.json").write_text(json.dumps(metrics, indent=2))
    for name, fig in figures.items():
        fig.savefig(folder/f"{name}.png", bbox_inches="tight")
        fig.savefig(folder/f"{name}.pdf", bbox_inches="tight")
    print(f"Saved measured results and figures to {folder}/")
