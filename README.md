# Linear-ODE

## Overview

This repository contains a structured implementation for running and comparing Linear ODE learning methods under a shared experimental pipeline.

The project is designed to make experimentation clean, modular, and reproducible. Solver implementations, experiment orchestration, and utility functions are separated so that multiple methods can be benchmarked under the same settings in a consistent way. The main workflow is notebook-based, making it easy to run experiments, compare methods, visualize optimization behavior, and inspect formatted summary tables.

At present, the repository supports the following model names in the main experiment pipeline:

- `autodiff`
- `our_model`

The primary interface for running experiments is `example.ipynb`.

---

## Repository Structure

```text
Linear-ODE/                               (PROJECT ROOT)
├── configs/                              # Configuration files
│   └── default_config.py                 # Default experiment settings
│
├── solvers/                              # Solver implementations
│   ├── __init__.py
│   ├── autodiff.py                       # Autodiff-based Linear ODE solver
│   └── our_model.py                      # Custom proposed solver
│
├── utils/                                # Utility Python modules
│   ├── __init__.py
│   ├── data_generation.py                # Data generation and mask construction
│   ├── frechet.py                        # Frechet / matrix exponential utilities
│   ├── plotting.py                       # Plotting helpers
│   └── summary.py                        # Summary table creation and styling
│
├── example.ipynb                         # Main notebook for running experiments
├── main_code_experiment.py               # High-level experiment pipeline
├── model_runner.py                       # Model registry and execution dispatcher
├── requirements.txt                      # Python dependencies
├── README.md
└── .gitignore
```

---

## Set Up Virtual Environment

### Using venv

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### On Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Using Conda

```bash
conda create -n linearode python=3.12
conda activate linearode
pip install -r requirements.txt
```

---

## How to Run

Run the `example.ipynb` notebook to execute experiments and reproduce results interactively.

The notebook provides a simple interface to:

- choose which models to run
- define experiment settings
- set seeds and training epochs
- visualize training behavior
- inspect the final summary table

### Typical notebook inputs

```python
MODELS_TO_RUN = ["autodiff", "our_model"]
CONFIGS = [{"NumAllGene": 2000, "NumTF": 1500, "B": 1}]
DATA_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
```

By default, the notebook shows plots and the summary table inline and does not save files unless saving is explicitly enabled.

To start Jupyter, run:

```bash
jupyter notebook
```

Then open:

```text
example.ipynb
```

and run the cells step by step.

---

## Main Components

### `example.ipynb`

This is the main entry point of the project.

It is intended for:

- selecting models
- setting experiment configurations
- running experiments
- visualizing plots inline
- viewing the final formatted summary table

### `main_code_experiment.py`

This module contains the main experiment pipeline. It coordinates:

- experiment setup
- data generation
- model execution
- result collection
- plotting
- summary generation

### `model_runner.py`

This file manages model-level execution logic, including:

- model registration
- model name normalization
- shared initialization
- dispatching selected solvers

### `solvers/`

This folder contains the actual solver implementations used in the experiments.

- `autodiff.py`  
  Implements the autodiff-based Linear ODE method.

- `our_model.py`  
  Implements the custom proposed method.

### `utils/`

This folder contains helper modules used across the project.

- `data_generation.py`  
  Generates trajectory data and constructs masks.
- `frechet.py`  
  Provides Frechet derivative and matrix exponential related utilities.
- `plotting.py`  
  Contains helper functions for visualizing loss curves and trajectories.
- `summary.py`  
  Builds and styles the final experiment summary table.

---

## Typical Workflow

1. Open `example.ipynb`
2. Select one or more models
3. Set the experiment configuration
4. Choose data seeds and training epochs
5. Run the notebook cells
6. Review:
   - loss curves
   - runtime comparison
   - final loss comparison
   - formatted summary table

---

## Methods

We currently include the following methods in the experiment pipeline:

| Method      | Description                      |
|-------------|----------------------------------|
| `autodiff`  | Autodiff-based Linear ODE solver |
| `our_model` | Custom proposed Linear ODE solver |

Both methods can be selected directly by name from the notebook interface.

---

## Design Principles

### Modularity

Core logic, solver implementations, and utilities are separated.

### Reproducibility

Experiments are controlled through explicit settings and data seeds.

### Readability

The notebook stays simple while most implementation details live in Python modules.

### Extensibility

New solvers can be added without restructuring the repository.

---

## Requirements

All required Python packages are listed in `requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

---

## Notes

- The primary interface is notebook-based.
- Plots and summary tables are displayed inline by default.
- The current workflow does not require a dedicated output directory.
- If file saving is added later, generated figures and tables can be organized separately without changing the overall structure.
