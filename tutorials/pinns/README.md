# PINNs with NUOX: initial-condition preservation

Open `pinns.ipynb` and **Run All**. The main result is the initial-condition error plot. It shows signed residuals on one common axis and their absolute magnitudes on a log axis. There is no boundary-condition plot.

## Run

- **VS Code / Jupyter:** extract all files into `tutorials/pinns/` or a standalone folder, restart the Python kernel, and run all. Starting from the repository root also works.
- **Colab:** open/upload `pinns.ipynb`, optionally select a GPU runtime, and run all. Upload `pinns_tutorial.zip` when setup prompts.

Setup installs missing dependencies into the current interpreter, prefers an actual local NUOX checkout, and avoids mistaking an outer folder named `nuox` for the package. Standalone installation pins NUOX to `15ce4a69944a1b85bee69d8175557fe4ecd6867d`. Fresh setup uses JAX 0.8.2, Optax 0.2.8 and Flax 0.12.2; existing installations are retained. Both CPU and GPU use float64. GPU hardware execution was not available for verification. If JAX was initialized before installing its GPU plugin, restart the runtime once.

## Measured result

| Independent measurement | NUOX | Unchanged penalty PINN |
|---|---:|---:|
| IC RMS error | 0.0000169469 | 0.000195411 |
| Maximum IC error | 0.0000604279 | 0.000795138 |
| Field RMSE | 0.000400617 | 0.000411375 |
| PDE RMS | 0.001554571 | 0.000337710 |

NUOX's IC RMS error is **11.5× lower** and its maximum IC error is **13.2× lower**. Field accuracy remains similar. Relative to the earlier NUOX result, refinement improves IC preservation while raising PDE residual and slightly raising field RMSE. This is a tutorial comparison of different training recipes, not an isolated optimizer benchmark or exact-feasibility guarantee.

The original 258-update NUOX curriculum and 330-update penalty baseline were executed from scratch on four CPU cores. The new 30-update refinement was then verified from that exact saved NUOX endpoint, followed by independent evaluation and reexecution of the changed plotting code. It took **53 seconds** including compilation; cumulative measured training was 335.4 seconds for NUOX and 164.1 seconds for the baseline, plus shared prefit/evaluation/plotting. The revised notebook was not rerun as one uninterrupted fresh execution. Its default still trains everything from untrained weights; saved result checkpoints are never loaded by the notebook's training cells.

## Formulation and preserved core recipe

Solve `u_t + beta*u_x = 0` on `[0,2*pi] × [0,1]`, with `u(x,0)=sin(x)` and periodicity. Periodicity remains part of the training problem, while the presentation focuses on IC preservation. The raw-input `[2,50,50,50,50,1]` tanh network has 7,851 float64 parameters. Constraints concatenate 128 initial-condition and 100 periodic residuals. The exact interior solution is used only in the final audit.

Both methods start from the same boundary-only prefit: 5,000 persistent Adam steps, learning rate .002, on `mean(c**2)`. The bundled `initial_weights.npy` contains only the original **untrained** seed-0 PyTorch Linear draws, promoted to float64; PyTorch is not required. Parameter SHA256: `b4967199ddc0c484a8f017c78ca1927ccc27e7674f6516ab0e18b9a880af0243`.

NUOX transforms both gradient and constraint by `A=(R.T R + mu I)^(-1/2)` for the normalized PDE residual Jacobian R. Its notebook-visible Optax chain combines `make_project_grad` with `optax.scale(-1)`. The PDE Jacobian and small Gram eigendecomposition are dense; the NUOX constraint projection uses native matrix-free LSMR. No dense constraint solve or full parameter-space inverse is used.

The first 258 NUOX updates retain the supplied recipe:

- Phase 1: beta `1,2,4,6,8,10,15,20` × 20 updates, beta 25 × 18; 256 fresh Sobol points, LSMR maxiter 128, atol/btol 1e-6.
- Phase 2: beta 25 and 30 × 40 updates; 512 fresh points, maxiter 256, atol/btol 1e-7.
- Gamma=1, no projection warmup, LSMR ctol=0 and solver damping zero. Gauss–Newton damping resets to .001 per stage; three damping trials, six backtracks and .5 maximum displacement. Acceptance reduces guard PDE MSE + 10*(IC MSE + periodic MSE).

The strong penalty baseline is unchanged: 30 attempts at each beta `1,2,4,6,8,10,15,20,25,30`, followed by 30 more at beta 30; 1,000 fresh Sobol points, equal-weight PDE/IC/periodic losses, dense Cholesky for its penalty Gauss–Newton direction, and the original damping/backtracking recipe.

## Added feasibility refinement

`refinement_steps=30` adds 30 correction-only applications of the **same notebook NUOX optimizer**. Set it to zero to recover the previous recipe.

Each refinement recomputes PDE Gauss–Newton geometry on 512 fixed seed-2029 Sobol points, with geometry damping .001. The task gradient is zero: this stage reduces nonlinear constraint error. LSMR uses maxiter 256 and atol/btol 1e-10. Its regularization starts at 1, halves after full accepted steps (floor 1e-4), doubles after accepted step lengths below .25, and increases tenfold after a failed trial. There are four regularization trials and eight backtracks per update.

Accept only a decrease in nonlinear `mean(c**2)` while guard PDE MSE remains below four times its value before refinement. Stop after 30 updates, if the full training constraint maximum falls below 1e-7, or if no acceptable correction exists. The measured run accepted all 30 and stopped at the update limit; the full constraint maximum remained approximately 1.54e-4. No exact interior labels, independent audit points, or analytic field error enter training or stopping.

Each original training phase restarts Sobol seed 2026; the guard is 8,192 points at seed 2027. Final evaluation uses a 513×201 field grid and 8,192 independent points at seed 654321. The reported endpoint is the last accepted update.

## Files

- `pinns.ipynb`: 74 code lines with the actual residual, loss, constraint, NUOX optimizer, training calls and plots.
- `pinn.py`: model, random initialization, data, shared prefit and independent audit.
- `solver.py`: Gauss–Newton geometry, fixed curricula, line search and final refinement.
- `figures.py`: IC-only comparison, measured table, solution/error maps and training curves.
- `setup_tutorial.py`: setup and import-path handling.
- `initial_weights.npy`: untrained weights.
- `results/`: measured parameters, evaluation arrays, histories, metrics, provenance, validation and PNG/PDF figures. The notebook's final cell overwrites only this tutorial's exported result files when rerun.
