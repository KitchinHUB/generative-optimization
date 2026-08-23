# Generative Machine Learning Approaches to Optimization

This repository contains supporting code and examples for the paper:

> **Generative approaches to optimization**
> Victor Alves and John R. Kitchin
> Preprint available at: https://chemrxiv.org/doi/full/10.26434/chemrxiv-2025-hk886

## Overview

We demonstrate how generative machine learning models can solve optimization and inverse design problems by learning joint distributions over inputs and outputs, enabling bidirectional inference. The repository includes implementations and examples using:

- **Gaussian Mixture Models (GMM)** for conditional generation via Gaussian Mixture Regression
- **Conditional Flow Matching** for inverse design using neural ODE-based generative models

Both approaches learn to generate inputs conditioned on desired outputs, providing:
- Multi-modal solution discovery (finding all solutions, not just one)
- Uncertainty quantification through sample distributions
- No gradient requirements for the objective function

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for Python environment management.

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew (macOS)
brew install uv

# Create environment and install dependencies
uv sync
```

This creates a `.venv/` directory with all required dependencies including:
- NumPy, SciPy, Matplotlib
- scikit-learn, [gmr](https://github.com/AlexanderFabisch/gmr) (Gaussian Mixture Regression)
- PyTorch (Flow Matching neural networks)
- JAX
- Jupyter

### Running Notebooks and Scripts

```bash
# Run a Python script
uv run python your_script.py

# Start Jupyter
uv run jupyter notebook

# Or activate the environment
source .venv/bin/activate
```

### VS Code Integration

The repository is pre-configured for VS Code. After running `uv sync`:
1. Reload VS Code (`Cmd+Shift+P` → "Developer: Reload Window")
2. Select the interpreter: `Cmd+Shift+P` → "Python: Select Interpreter" → `.venv/bin/python`
3. For Jupyter notebooks, select the `.venv` kernel from the kernel picker

## Repository Structure

```
├── generative_optimization.py     # Core module with GMM and Flow Matching utilities
├── readme.ipynb                   # Interactive setup and navigation guide
├── SKILL.md                       # Claude Code skill for expert guidance
├── pyproject.toml                 # uv/Python project configuration
│
├── GMM Examples (00*.ipynb):
│   ├── 00a_root_finding           # PR-EOS cubic root finding
│   ├── 00b_optimization           # Unconstrained optimization
│   ├── 00c_equality_constraints   # Gasoline blending (Lagrangian)
│   ├── 00d_inequality_constraints # Barrier method + reaction equilibrium
│   ├── 00e_parameter_estimation   # Series reaction rate constants
│   └── 00h_space_mapping          # CSTR input/output mapping
│
├── Flow Matching Examples (01*.ipynb):
│   ├── 01a_fm_root_finding        # Polynomial root finding
│   ├── 01b_fm_optimization        # Unconstrained optimization
│   ├── 01c_fm_equality_constraints
│   ├── 01d_fm_inequality_constraints
│   ├── 01e_fm_parameter_estimation
│   ├── 01f_fm_space_mapping       # CSTR design
│   └── 01g_fm_eos_fitting         # Van der Waals EOS fitting
│
├── Visualizations:
│   ├── 02_flow_field_visualization
│   └── 03_space_mapping_flow_matching
│
└── Results (results-*.ipynb):     # Publication figures comparing GMM and FM
```

## Manuscript Figures

Each figure in `manuscript.tex` is produced by exactly one notebook. Re-running the
notebook below regenerates that figure in place.

| Figure file | Authoritative notebook | Example |
|---|---|---|
| `fig_gmm_example.png` | `00-example_gmm_fm.ipynb` | GMM illustration |
| `fig_fm_example.png` | `00-example_gmm_fm.ipynb` | Flow matching illustration |
| `preos_comparison.png` | `results-00-root-finding-preos.ipynb` | PR-EOS root finding |
| `optimization_comparison.png` | `results-01-optimization.ipynb` | Unconstrained optimization |
| `blending_comparison.png` | `results-02-equality-constrained-blending.ipynb` | Gasoline blending (Lagrangian) |
| `equilibrium_comparison.png` | `results-03-reaction-equilibrium.ipynb` | Reaction equilibrium (barrier method) |
| `k1-k2-Ca-Cb.png` | `00e_parameter_estimation.ipynb` | Parameter estimation setup |
| `parameter_estimation_comparison.png` | `results-04-parameter-estimation.ipynb` | Parameter estimation |
| `space_mapping_comparison.png` | `results-05-space-mapping.ipynb` | CSTR design-space mapping |
| `evaporator_training_coverage.png` | `evaporator_feasibility_data.ipynb` | Evaporator feasibility |
| `evaporator_relative_error.png` | `evaporator_feasibility_data.ipynb` | Evaporator feasibility |
| `evaporator_summary_comparison.png` | `evaporator_feasibility_data.ipynb` | Evaporator feasibility |
| `optimization_flowchart.pdf` | *(hand-drawn, not notebook-generated)* | Method schematic |
| `evaporator_schematic.pdf` | *(hand-drawn, not notebook-generated)* | Evaporator schematic |
| `toc.png` | *(hand-assembled)* | Table-of-contents graphic |

`results-and-figures.ipynb` is an earlier exploratory all-in-one notebook. It uses
different problem setups from the manuscript (its equilibrium example is plain root
finding rather than the barrier formulation), so it writes its figures with a
`draft_` prefix and is not the source of any manuscript figure.

## Quick Start

```python
import numpy as np
from generative_optimization import generate_samples, ConditionalFlowMatching

# Define a forward model: y = x^2
def forward_model(x):
    return x ** 2

# Generate training data
x_data = generate_samples(bounds=[[-2, 2]], n_samples=512)
y_data = forward_model(x_data)

# Train flow matching for inverse: given y, find x
fm = ConditionalFlowMatching(x_dim=1, c_dim=1, hidden_dim=64, n_layers=3)
fm.fit(x_data, y_data, epochs=500)

# Inverse problem: find x where y = 1
samples = fm.sample(c_values=[[1.0]], n_samples=500)
print(f"Found x values: {samples.mean():.3f} ± {samples.std():.3f}")
# Expected: x = ±1.0 (both solutions discovered)
```

## Claude Code Skill

The repository includes `SKILL.md`, a [Claude Code](https://claude.com/claude-code) skill that provides expert guidance on applying generative optimization to your own problems.

### What the Skill Provides

- **Problem assessment framework**: When to use generative optimization vs traditional methods
- **Method selection criteria**: GMM vs Flow Matching decision guide
- **Implementation patterns**: Complete code for common problem types (unconstrained, equality/inequality constraints, inverse problems, parameter estimation, multi-objective)
- **Best practices**: Training tips, hyperparameter selection, troubleshooting

### Using the Skill

**Option 1: Local skill (this repository)**

The `SKILL.md` file is already present. When you use Claude Code in this directory, it will automatically have access to the skill.

**Option 2: Global skill (use across all projects)**

```bash
mkdir -p ~/.claude/skills
cp SKILL.md ~/.claude/skills/generative-optimization.md
```

Then describe your optimization problem to Claude Code, and it will help you formulate, implement, and analyze solutions using these techniques.

## Citation

If you use this code in your research, please cite our preprint:

```bibtex
@article{alves2025generative,
  title={Generative approaches to optimization},
  author={Alves, Victor and Kitchin, John R.},
  journal={ChemRxiv},
  year={2025},
  doi={10.26434/chemrxiv-2025-hk886}
}
```

## License

See the repository for license information.
