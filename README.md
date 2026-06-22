# Patient-Adaptive Information Acquisition for Prehospital Mortality Prediction Using Reinforcement Learning

Research code for learning a patient-adaptive clinical information-acquisition
policy for prehospital mortality prediction.

The framework starts from the information available during a prehospital
emergency call. A Maskable Proximal Policy Optimization (Maskable PPO) agent
then decides whether to acquire one of 21 clinical feature groups or to stop.
The downstream mortality predictor and its decision threshold remain fixed, so
the learned policy optimizes information use rather than retraining the
classifier.

> **Research-use notice:** This repository contains retrospective research
> code. It is not a medical device and must not be used for clinical decisions
> without independent validation and the required regulatory and institutional
> approvals.

## Overview

- **Initial state:** partially observed clinical information reconstructed from
  prehospital emergency-call data
- **Prediction model:** frozen ensemble mortality predictor
- **Actions:** acquire one of 21 feature groups or select `STOP`
- **Policy:** Maskable PPO with state-dependent action masking
- **Objective:** preserve mortality-prediction performance while minimizing
  additional information acquisition
- **Evaluation:** call-derived, slot-derived, and all-feature reference settings

The 21 feature groups cover demographics, injury characteristics, prehospital
physiology, and transport or arrival information. Categorical variables use
predefined missing or indeterminate categories. Missing vital-sign variables
use the same encodings defined for the frozen prediction model.

## Repository structure

```text
.
├── data/
│   └── README.md
├── outputs/
│   └── README.md
├── src/
│   ├── train_rl.py
│   ├── env_feature_acq.py
│   ├── preprocess.py
│   ├── config.py
│   ├── callbacks.py
│   ├── oracle.py
│   ├── group_map.py
│   ├── missing_rules.py
│   ├── mask_utils.py
│   ├── call_align.py
│   ├── export_compare_excel.py
│   ├── export_death_pcall_pall_full.py
│   └── analyze_policy_importance.py
├── .gitignore
└── requirements.txt
```

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation can vary by operating system and CUDA version. If GPU
training is required, install the appropriate PyTorch build before installing
the remaining requirements.

## Private data and model files

The following required files are intentionally excluded from version control:

```text
data/
├── cb_ch_all.xlsx
├── cb_ch_call_ver1.1.xlsx
└── Prehospital_final_model.pkl
```

- `cb_ch_all.xlsx`: linked all-feature registry data
- `cb_ch_call_ver1.1.xlsx`: call-stage partial-observation data
- `Prehospital_final_model.pkl`: frozen ensemble mortality predictor

These files may contain protected clinical information or non-public model
artifacts. Store them only in an approved secure environment. Do not upload
patient-level data, identifiers, emergency-call recordings, transcripts,
slot-filling outputs, or derived patient-level spreadsheets to GitHub.

The expected identifiers and outcome columns include:

- `OCS등록번호`
- `내원일시`
- `Mortality`

Additional raw feature columns and their encodings are defined in
`src/preprocess.py` and `src/missing_rules.py`.

## Training

Review the paths and hyperparameters in `src/config.py`, then run:

```bash
python src/train_rl.py
```

The default configuration uses:

- 1,000,000 training timesteps
- random seed 42
- question cost 0.04
- Maskable PPO with dynamic valid-action masks
- validation-based model selection and early stopping

Training artifacts are written to `outputs/` and are ignored by Git.

## Evaluation and analysis

After training, set `RLConfig.POLICY_MODEL_PATH` in `src/config.py` to the
selected policy checkpoint.

```bash
python src/export_compare_excel.py
python src/analyze_policy_importance.py
python src/export_death_pcall_pall_full.py
```

These scripts generate patient-level spreadsheets and figures under
`outputs/`. Treat all patient-level exports as private, even when direct
identifiers have been removed.

## Reproducibility notes

- Data splitting is stratified and uses the configured random seed.
- The prediction model is loaded once and remains frozen during policy
  optimization.
- Already observed or acquired feature groups are excluded through action
  masking.
- The `STOP` action remains available after the configured minimum number of
  acquisitions.
- The all-feature condition is a complete-information reference, not
  necessarily a strict performance upper bound.

The clinical datasets and frozen predictor are not distributed because of
privacy, governance, and institutional restrictions. Consequently, the public
repository documents the implementation but does not provide a fully
standalone clinical reproduction package.

## Citation

A manuscript is in preparation. Citation information will be added after
publication.

## License

No open-source license has been selected yet. Unless a license file is added,
copyright is retained by the repository owner and no reuse rights are granted
beyond those provided by applicable law.
