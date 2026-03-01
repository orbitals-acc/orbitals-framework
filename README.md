
# Orbitals: Target-aware Segmentation for Long-term Effect Estimation

Orbitals: A target-aware segmentation framework for estimating Limited LTV (lLTV) 
from short-term A/B tests using interpretable behavioral archetypes and transition dynamics.


(Reproduction code for *“Orbitals: A Target-aware Segmentation for Long-Term Effect Estimation in A/B Testing”* (SIGIR ’26))



## Installation

Requires Python 3.13.7 (pinned in `pyproject.toml`).


## Project structure

Each project lives under `projects/<project_name>/` and follows `project_template.yaml`:

```text
projects/<project_name>/
├── source/                          # raw session-level CSVs
├── configs/                         # project YAML configs
├── artifacts/
│   ├── learning/
│   │   ├── data/                    # feature + target CSVs
│   │   └── info/
│   │       ├── catboost_info/       # CatBoost training info
│   │       ├── catboost_logs/       # training logs
│   │       └── catboost_model/      # saved model
│   ├── polynomial_form/             # Monoforest monomials (active.yaml, …)
│   ├── compact/                     # compacted table rules (active.yaml, …)
│   └── evaluation/
│       └── stream/
│           ├── data/<batch>/        # per-timestamp feature CSVs
│           └── potentials/<batch>/  # orbital V_c estimates (active.yaml)
```


## Data


All datasets are fully anonymized and placed in `projects/main/source/`.
After unpacking, you should have:

| File | Description | Users |
| :-- | :-- | :-- |
| `dayuses.csv` | Activity log for orbital training (Jan–Oct 2025) | ~830K |
| `exp_dayuses.csv` | BNPL experiment activity log (Mar–Jul 2025) | ~970K |


## Notebooks

Run in order:

1. `orbital.ipynb` — feature engineering, CatBoost training, Monoforest rule extraction, orbital definition and multiplier estimation.
2. `predictive.ipynb` — strong predictive baseline for lLTV.
2. `aa_test.ipynb` — A/A sanity check.
3. `bnpl_test.ipynb` — BNPL experiment.

## Citation

```bibtex
@inproceedings{orbitals2026,
  title     = {Orbitals: A Target-aware Segmentation for Long-Term
               Effect Estimation in A/B Testing},
  booktitle = {Proc. 49th International ACM SIGIR Conference},
  year      = {2026}
}
```
