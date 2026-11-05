# Tutorials

Each notebook follows **task → loss → constraint → NUOX optimizer → training → visualization**. These examples demonstrate different uses of the same interface.

| Notebook | Objective | Constraint | What to inspect |
| --- | --- | --- | --- |
| [PINNs](https://github.com/h-roy/nuox/blob/main/tutorials/pinns/pinns.ipynb) | Minimize a differential-equation residual | Initial conditions at selected points | Solution heatmaps and initial-condition errors. |
| [One-step flow matching](https://github.com/h-roy/nuox/blob/main/tutorials/flow_matching/smile.ipynb) | Distil a flow model into a fast generator | Preserve selected instantaneous teacher velocities | Ordinary FM and NUOX samples at one, two, and four steps. |
| [LLM editing](https://github.com/h-roy/nuox/blob/main/tutorials/llm_editing/llm_editing.ipynb) | Fit an edited fact | Restrict changes to an editable weight matrix | Original and edited answers, including held-out prompts. |

## Run a notebook

Keep the notebook, its Python helpers, and data files together. In VS Code, select the intended Python kernel, restart it, and run from the setup cell. In Colab, follow the setup cell and upload the tutorial bundle if prompted. A GPU runtime must be selected before initializing JAX to use a GPU.

The notebooks include setup for CPU and GPU. Runtime depends on the model, hardware, compilation, and first-time downloads. Check the notebook's actual device before attributing a timing to GPU execution.

## Interpret the examples

For PINNs, satisfying sampled initial conditions does not by itself guarantee an accurate solution throughout the domain. For flow distillation, assess generation quality alongside velocity-constraint residuals; the student also receives distillation supervision. For LLM editing, preserving selected weight-space directions does not guarantee unchanged answers or correct logical consequences everywhere.

The notebooks contain the experiment settings and visual comparisons. They are demonstrations of constrained learning, not a claim of universal improvement over unconstrained methods.
