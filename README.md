# ML Assignment 3 — CIFAR-10: MLPs, CNNs, autoencoders, SVMs and clustering

All code, the saved split, the configuration, final logs and figure scripts are in this repository.

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt      # or: conda env create -f environment.yml
```

Hardware used: Apple M4 Max (PyTorch MPS backend). The code falls back to CUDA or CPU automatically.

## Configuration

`configs/config.json` holds every hyperparameter. **`roll_number` sets the seed:** all the numeric
characters of the roll number, kept in order, form the integer seed used for the stratified
45,000/5,000 split (`splits/split_seed<seed>.npz`). The same seed drives every model initialisation, the
label-noise draw, the fixed permutations, the test-noise draw and the SVM subset.

## Reproduce everything

```bash
./run_all.sh            # about 2-3 h on an M4 Max; writes results/, logs/, figures/
```

or one step at a time (each step reuses the outputs of the steps before it):

| Step | Command (run from `src/`) | Produces |
|---|---|---|
| split | `../.venv/bin/python -c "from common import *; get_split(load_config())"` | `splits/split_seed*.npz` |
| Q1(a) | `../.venv/bin/python q1.py --part a` | `results/q1a/`, checkpoints for MLP-S/M/L and CNN |
| Q1(b) | `../.venv/bin/python q1.py --part b` | `results/q1b/` |
| Q2 | `../.venv/bin/python q2.py` | `results/q2/` (reuses the Q1(a) MLP-M and CNN) |
| Q3 | `../.venv/bin/python q3.py` | `results/q3/` (uses the Q1(a) CNN as the fixed classifier) |
| Q4 | `../.venv/bin/python q4.py` | `results/q4/` (uses Q3 codes) |
| figures | `../.venv/bin/python figures.py` | `figures/*.png` |
| report | `cd report && tectonic report.tex` | `report/report.pdf` |

## Layout

- `src/common.py` — data loading, split, fixed transforms (pixel/block permutation, grayscale), models
  (MLP, CNN, ConvAE), training loop
- `src/q1.py` … `src/q4.py` — one script per question
- `src/figures.py` — every figure in the report
- `results/**/summary.json` — the numbers quoted in the report; `logs/*.log` — the training logs

## Protocol

- The validation set is used for model, architecture, hyperparameter, early-stopping-epoch, bottleneck,
  representation, SVM and clustering selection.
- The official test set is evaluated only after every choice is fixed.
- No pretrained models, external data or architecture search are used.
