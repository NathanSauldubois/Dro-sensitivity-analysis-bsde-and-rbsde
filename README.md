# Sensitivity Analysis of Distributionally Robust BSDEs and RBSDEs

This repository contains the code used to reproduce the numerical simulations presented in **"Sensitivity Analysis of Distributionally Robust BSDEs and RBSDEs"**.

The scripts implement deep BSDE/RBSDE solvers in PyTorch and run parameter sweeps used to estimate prices and sensitivity quantities.

## Repository Contents

- `american_option.py`: numerical simulations for the reflected BSDE / American option example.
- `optimal_control.py`: numerical simulations for the BSDE optimal control example.

Both scripts include:

- `TEST` and `RUN` configurations through the global `ETAT` variable.
- Neural-network solvers based on PyTorch.
- Warm-started parameter sweeps over strike values.
- Incremental CSV exports.
- Checkpointing to resume long computations.

## Requirements

The code requires Python 3 and PyTorch.

Install PyTorch following the instructions for your machine from the official PyTorch installation page. The default configuration uses CUDA:

```python
DEVICE = "cuda"
```

If CUDA is not available, change the relevant configuration class in the script to:

```python
DEVICE = "cpu"
```

## Running the Simulations

From the repository root, run:

```bash
python american_option.py
```

or:

```bash
python optimal_control.py
```

By default, both scripts use:

```python
ETAT = "RUN"
```

This mode is intended for full numerical experiments and may be computationally expensive. For a smaller smoke test, edit the script and set:

```python
ETAT = "TEST"
```

## Outputs

The scripts write results under the `OFFICIAL/` directory.

Typical outputs include:

- `OFFICIAL/sensitivities.csv`: sensitivities for the American option / RBSDE experiment.
- `OFFICIAL/control_sensitivities.csv`: sensitivities for the optimal control / BSDE experiment.
- `OFFICIAL/progress.txt`: live progress for the current sweep.
- `OFFICIAL/checkpoints/`: checkpoints used to resume long runs.

Generated output files are not required before running the scripts; they are created automatically.

## Reproducibility Notes

The numerical grids and solver hyperparameters are defined in the `main()` function and configuration dataclasses of each script. To reproduce a specific experiment, check:

- The global `ETAT` mode.
- The selected parameter grid in `main()`.
- The solver configuration class used by that mode.
- The random seed defined in the configuration.
