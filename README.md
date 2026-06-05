```text
cDFT LAB
●─f₁₂─●∘
```
Our logo is a Mayer *f*-bond, $f_{12}=e^{-\beta u(r_{12})}-1$, with continuation into the many-body fluid.

Classical density functional theory (cDFT) & learned functionals for fluids.

- Physics-first: variational structure, nonlocality, reproducible numerics.
- Learning augments the baseline; it does not replace the physics.
- Focus: bridging particle and continuum scales.

## Repositories
- **Paper artefacts**: reproducible scripts/configs to regenerate figures & tables.
- **Core toolbox**: solvers, reference functionals, discretisation utilities.
- **Learned components**: constrained corrections (kernels, EOS scalers, parameter maps).

## Reference
- *Learning Density Functionals to Bridge Particle and Continuum Scales* (Monti, Yatsyshin, et al.)

## Maintainers
- Peter Yatsyshin
- Edoardo Monti

---

## Project structure

```
cdft_lab2/
├── requirements.txt          # Python dependencies
├── data/
│   ├── dataset/
│   │   ├── md_planar_wc/   # MD profiles, weakly attractive wall (N4000–N8000)
│   │   ├── md_planar_wn2/  # MD profiles, wn2 wall (N8000)
│   │   ├── lda_planar_wc/  # LDA reference profiles for wc (use with --skip-lda)
│   │   ├── lda_planar_wn2/ # LDA reference profiles for wn2
│   │   ├── pkl/            # Preprocessed pickles and isotherm trajectories
│   │   └── postproc/       # Post-processed artefacts
│   └── ml_model/
│       ├── ml_dicts/       # Default trained surrogate weights (shipped)
│       └── intermediate_models/
├── output/                 # Generated plots and pickles 
└── src/
    ├── libs/               # cDFT solvers, ML surrogates, I/O utilities
    ├── preprocess_data.py
    ├── adj_train.py
    ├── compute_bulk_tmd.py
    ├── lg_continuation.py
    ├── compute_gamma_lg.py
    └── adsoprtion_continuation.py
```

Each MD/LDA dataset follows the layout `N*/T_*/z_profile.csv` (plus auxiliary CSVs and figures). Generated `*.pkl` files and everything under `output/` are gitignored but preserved locally between runs.

---

## Installation

### Requirements

- Python 3.10 or newer (tested with 3.12)
- A PyTorch build matching your hardware (CPU, CUDA, or Apple MPS)
- Packages listed in `requirements.txt`: `torch`, `numpy`, `pandas`, `matplotlib`, `tqdm`

GPU is optional but recommended for training and long continuation runs. Set `TORCH_DEVICE` (see below) to force a backend.

### Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA or Apple MPS, install PyTorch from the [official install selector](https://pytorch.org/get-started/locally/) first if the default wheel from `requirements.txt` does not match your hardware, then install the remaining packages:

```bash
pip install numpy pandas matplotlib tqdm
```

### Data and trained models

| Path | Contents |
|------|----------|
| `data/dataset/md_planar_wc/` | MD density profiles for the `wc` wall (`N4000`–`N8000`, `T_0.55`–`T_1.00`) |
| `data/dataset/md_planar_wn2/` | MD density profiles for the `wn2` wall (`N8000`) |
| `data/dataset/lda_planar_wc/` | Pre-computed LDA reference profiles for `wc` |
| `data/dataset/lda_planar_wn2/` | Pre-computed LDA reference profiles for `wn2` |
| `data/dataset/pkl/profiles_wl_<wall>/` | `z_profiles.pkl`, `z_profiles_0.pkl` from preprocessing |
| `data/dataset/pkl/profiles_lg/` | Liquid–gas isotherm trajectories |
| `data/dataset/pkl/bulk_tmd/` | Shipped phase-curve pickles (`pc_*_operative.pkl`) |
| `data/dataset/postproc/` | Post-processed data |
| `data/ml_model/ml_dicts/` | Default trained weights (`dnn_*.dict`, `wda_fn.dict`, `U_guess.pkl`) |
| `output/` | Fresh results from script runs (plots, updated pickles) |

Pre-trained weights are shipped under `data/ml_model/ml_dicts/`. To retrain them, run the preprocessing and training workflow below.

---

## Running scripts

**All scripts must be run from the `src/` directory** so that `from libs...` imports resolve correctly:

```bash
cd src
python <script>.py
```

### Device selection

Most scripts set `DEVICE_KIND` near the top of the file (`"auto"`, `"cuda"`, `"mps"`, or `"cpu"`). Override without editing code:

```bash
export TORCH_DEVICE=cuda    # or cpu | mps
python compute_bulk_tmd.py
```

On multi-GPU machines, some scripts set `CUDA_VISIBLE_DEVICES`; adjust that variable in the script if needed. Recent compatibility fixes ensure tensors are created on the active device, so the same scripts run on CPU and GPU (prefer `torch.float32` on MPS).

### Script overview

| Script | Purpose |
|--------|---------|
| `preprocess_data.py` | Build `z_profiles.pkl` / `z_profiles_0.pkl` from profile CSVs |
| `adj_train.py` | Adjoint training of DNN / WDA surrogates |
| `compute_bulk_tmd.py` | Bulk phase diagram and critical point (MF + neural vs MD) |
| `lg_continuation.py` | Liquid–gas planar isotherms via Newton + continuation |
| `compute_gamma_lg.py` | Liquid–gas surface tension \(\gamma_{lg}\) from isotherms |
| `adsoprtion_continuation.py` | Wall–liquid adsorption isotherms (`wn2` / `wc` walls) |

Most scripts (except `preprocess_data.py`) are configured by editing constants at the top of the file—`WALL`, temperatures, `DEVICE_KIND`, `USE_DBH_DIAMETER`, training flags, and output paths.

### Key configuration flags

| Flag | Meaning |
|------|---------|
| `USE_MODEL` | `1`: neural surrogates; `0`: mean-field LDA only |
| `USE_DBH_DIAMETER` | `1`: Barker–Henderson diameter scaling (training / adsorption); `0`: default inference weights |
| `RESTART_ML_MODEL` | `1`: load `data/ml_model/ml_dicts/`; `0`: train or solve from scratch |
| `WALL` | Selects external potential and data paths: `"wc"`, `"wn2"`, or `"lg"` |

Default flag values differ by script (e.g. inference scripts use `USE_DBH_DIAMETER = 0`; `adj_train.py` uses `1`). Check the header of each script before running.

---

### Recommended workflows

#### Inference only (use shipped ML weights)

```bash
cd src

# 1. Bulk coexistence curve and critical point
python compute_bulk_tmd.py
# → output/bulk_tmd/phase_curve.{svg,png}, pc_*_operative.pkl, critical_point.txt

# 2. Liquid–gas isotherms, then surface tension
python lg_continuation.py
# → data/dataset/pkl/profiles_lg/isotherm_profiles_lg_lg.pkl
# → output/plot_lg_isotherm/

python compute_gamma_lg.py
# → output/plot_lg_isotherm/gamma_lg/
# → data/dataset/pkl/bulk_tmd/pc_md_operative_with_gamma.pkl

# 3. Wall–liquid adsorption (set WALL = "wc" or "wn2" in the script)
python adsoprtion_continuation.py
# → data/dataset/pkl/profiles_wl_<wall>/isotherm_profiles_wl_<wall>.pkl
# → output/plot_adsorption_isotherm/
```

`compute_gamma_lg.py` reads the phase curve from `data/dataset/pkl/bulk_tmd/pc_nn_operative.pkl` (shipped). After re-running `compute_bulk_tmd.py`, copy or symlink the updated `pc_nn_operative.pkl` from `output/bulk_tmd/` if you want downstream scripts to use the fresh result.

#### Full retraining from MD data

```bash
cd src

# 1a. Pack MD profiles (wc wall — testing / default wall potential)
python preprocess_data.py --wall wc
# → data/dataset/pkl/profiles_wl_wc/z_profiles.pkl

# 1b. Pack MD profiles (wn2 wall — training wall potential)
python preprocess_data.py --wall wn2 \
  --source ../data/dataset/md_planar_wn2
# → data/dataset/pkl/profiles_wl_wn2/z_profiles.pkl

# Faster alternative: use pre-computed LDA references instead of an on-the-fly LDA solve
python preprocess_data.py --skip-lda --wall wc \
  --source ../data/dataset/lda_planar_wc
python preprocess_data.py --skip-lda --wall wn2 \
  --source ../data/dataset/lda_planar_wn2

# 2. Train surrogates (set WALL to match step 1; edit epochs, train_T in adj_train.py)
python adj_train.py
# → data/ml_model/ml_dicts/

# 3. Run the inference workflow above
```

---

### Per-script details

#### `preprocess_data.py`

The only script with a CLI. Gathers `z_profile.csv` files from an `N*/T_*/` tree and writes `z_profiles.pkl` plus `z_profiles_0.pkl` (reference profiles for adjoint training).

```bash
cd src
python preprocess_data.py --help

# wc wall (default source: md_planar_wc)
python preprocess_data.py --wall wc
python preprocess_data.py --source ../data/dataset/md_planar_wc \
  --output ../data/dataset/pkl/profiles_wl_wc

# wn2 wall
python preprocess_data.py --wall wn2 \
  --source ../data/dataset/md_planar_wn2

# Use shipped LDA references instead of solving LDA inline
python preprocess_data.py --skip-lda --wall wc \
  --source ../data/dataset/lda_planar_wc
python preprocess_data.py --skip-lda --wall wn2 \
  --source ../data/dataset/lda_planar_wn2
```

With `--skip-lda`, the source profiles are copied into `z_profiles_0.pkl` directly (no inline LDA Newton solve).

#### `adj_train.py`

Trains the neural corrections (DNN, repulsive DNN, WDA) against MD profiles. Key settings at the top of the file:

- `WALL`: `"wc"` or `"wn2"` — must match the pickle directory from `preprocess_data.py`
- `RESTART_ML_MODEL`: `1` loads existing weights; `0` trains from scratch
- `TRAIN_DNN`, `TRAIN_WDA`, `TRAIN_DNN_REP`: toggle which components are optimised
- `USE_DBH_DIAMETER`: `1` during Barker–Henderson-aware training
- `epochs`, `train_T`, `good_N`: training schedule

Outputs trained weights to `data/ml_model/ml_dicts/` and diagnostic plots to `output/plot_train/`.

#### `compute_bulk_tmd.py`

Computes mean-field and neural liquid–vapor coexistence densities and the critical point, comparing against MD reference data. Requires trained weights in `data/ml_model/ml_dicts/` when `RESTART_ML_MODEL = 1`.

Outputs to `output/bulk_tmd/`.

#### `lg_continuation.py`

Solves planar liquid–gas isotherms at fixed particle number. Set `WALL = "lg"` for the free-interface setup. Computes coexistence densities on the fly via `batched_coexistence_temperatures`; run `compute_bulk_tmd.py` first if you need an updated phase curve on disk.

Writes `data/dataset/pkl/profiles_lg/isotherm_profiles_lg_lg.pkl` and plots under `output/plot_lg_isotherm/`.

#### `compute_gamma_lg.py`

Estimates \(\gamma_{lg}\) from isotherm trajectories produced by `lg_continuation.py`. Populates missing isotherm frames if needed, then integrates \(\omega(x) - \omega_b\) across the interface.

Expects:

- `data/dataset/pkl/profiles_lg/isotherm_profiles_lg_lg.pkl`
- `data/dataset/pkl/bulk_tmd/pc_nn_operative.pkl`

Outputs plots to `output/plot_lg_isotherm/gamma_lg/` and updates `data/dataset/pkl/bulk_tmd/pc_md_operative_with_gamma.pkl`.

#### `adsoprtion_continuation.py`

Wall–liquid adsorption isotherms for `WALL = "wc"` (weakly attractive wall) or `"wn2"`. Uses the same Newton + continuation solver as the LG workflow but with a wall external potential.

Writes `data/dataset/pkl/profiles_wl_<wall>/isotherm_profiles_wl_<wall>.pkl` and plots to `output/plot_adsorption_isotherm/`.

---

### Troubleshooting

- **`ModuleNotFoundError: No module named 'libs'`** — run the script from `src/`, not the repo root.
- **Missing `dnn_fn.dict` or similar** — ensure `data/ml_model/ml_dicts/` is present, or train with `adj_train.py`.
- **CUDA out of memory** — set `DEVICE_KIND = "cpu"` or `export TORCH_DEVICE=cpu`, or reduce batch sizes / mesh resolution in the script.
- **MPS / float64 issues on Apple Silicon** — use `torch.float32` for `TORCH_DTYPE` and `TORCH_DEVICE=mps`.
- **Stale phase-curve pickle in `compute_gamma_lg.py`** — copy `output/bulk_tmd/pc_nn_operative.pkl` to `data/dataset/pkl/bulk_tmd/` after re-running `compute_bulk_tmd.py`.
