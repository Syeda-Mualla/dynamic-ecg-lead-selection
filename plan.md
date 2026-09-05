# Disease-Specific Quality-Aware Dynamic ECG Lead Selection
## Complete Research and Implementation Plan for Codex

---

## 1. Project title

**Disease-Specific Quality-Aware Dynamic ECG Lead Selection with Uncertainty for Multi-Label ECG Classification**

---

## 2. Core research idea

Build a model that receives a standard 12-lead ECG and does not treat every lead as equally useful.

The system should:

1. estimate the quality of each ECG lead;
2. estimate which leads are important for each disease;
3. dynamically select the best available leads for each patient;
4. predict multiple cardiac conditions;
5. estimate uncertainty;
6. refer the case for expert review when the remaining leads are insufficient or the model is uncertain.

Final intended pipeline:

```text
12-lead ECG
      ↓
Per-lead feature encoder
      ↓
Lead-quality estimator
      ↓
Disease-specific lead scorer
      ↓
Dynamic top-k lead selector
      ↓
Multi-label disease prediction
      ↓
Uncertainty estimation
      ↓
Accept prediction or refer for expert review
```

---

## 3. Research motivation

Existing ECG-AI research shows several important limitations:

1. Strong internal performance does not guarantee performance on data from another hospital or country.
2. Models trained on clean ECGs often lose substantial performance on noisy ECGs.
3. Fixed lead subsets can generalize better than all 12 leads in some settings.
4. One universal lead subset may not be optimal for every disease, patient, dataset, or noise condition.
5. Standard data augmentation may not fully solve physiological ECG noise.
6. A model may still produce confident predictions when the input is poor or unfamiliar.
7. Uncertainty estimation can help identify cases that should be referred to a clinician.
8. Most prior work studies fixed lead selection, quality estimation, robustness, external validation, or uncertainty separately.

The intended contribution is to combine these ideas into one disease-specific, patient-specific, quality-aware, dynamic lead-selection system.

---

## 4. Main research question

> Can a disease-specific, quality-aware dynamic ECG lead-selection model preserve the performance of a 12-lead classifier while using fewer reliable leads, and outperform fixed-lead models when important leads are noisy or missing?

---

## 5. Secondary research questions

1. Do different cardiac conditions depend on different ECG leads?
2. Can fewer than 12 leads preserve most of the full-model performance?
3. Does one fixed four-lead subset work equally well for all disease classes?
4. Can a lead-quality estimator identify corrupted, missing, or unreliable leads?
5. Can the model dynamically replace poor-quality leads with other informative leads?
6. Does dynamic selection reduce false negatives when one or more important leads are corrupted?
7. Does dynamic selection improve robustness under missing-lead conditions?
8. Does dynamic selection generalize better to an external dataset?
9. Can uncertainty estimation identify cases where insufficient reliable information remains?
10. Can the system achieve a useful trade-off between:
   - number of leads used;
   - predictive performance;
   - robustness;
   - uncertainty;
   - referral rate?

---

## 6. Initial target classes

Start with the five PTB-XL diagnostic superclasses:

| Code | Meaning |
|---|---|
| `NORM` | Normal ECG |
| `MI` | Myocardial infarction |
| `STTC` | ST/T changes |
| `CD` | Conduction disturbance |
| `HYP` | Hypertrophy |

This is a multi-label problem because one ECG can contain more than one condition.

Example label:

```text
NORM MI STTC CD HYP
  0   1   1   0   0
```

Do not start with all detailed PTB-XL statements. First prove that the research idea works on the five superclasses.

---

## 7. Data source

Primary dataset:

- PTB-XL version `1.0.3`
- 12-lead ECG
- 10-second recordings
- 100 Hz signals for initial experiments
- expected waveform shape: `1000 × 12`
- official patient-separated folds
- public diagnostic labels

Official dataset page:

```text
https://physionet.org/content/ptb-xl/1.0.3/
```

Original benchmark repository:

```text
https://github.com/helme/ecg_ptbxl_benchmarking
```

Use the original repository as a reference for:

- label aggregation;
- fold usage;
- benchmark protocol;
- preprocessing logic;
- expected performance ranges.

Do not copy its old FastAI environment into the final project. Implement the new system in modern modular PyTorch.

---

## 8. High-level implementation phases

Implement the project in the following order:

```text
Phase 0: Local project setup
Phase 1: Reproduce a trusted 12-lead baseline
Phase 2: Fixed-lead experiments
Phase 3: Lead-quality corruption benchmark
Phase 4: Lead-quality estimator
Phase 5: Disease-specific soft lead scoring
Phase L-A: Literature and novelty verification before the dynamic method
Phase 6: Quality-aware disease-specific soft lead routing
Phase 7: Hard sparse ECG lead selection
Phase 8: Add uncertainty and referral
Phase 9: External validation
Phase 10: Statistical analysis and ablations
Phase L-B: Literature and novelty verification before paper preparation or submission
Phase 11: Research paper preparation
```

Important:

> Do not implement all phases at once.

Codex must stop after each phase, run tests, generate results, and wait for review before continuing.

---

# PHASE 0 — PROJECT SETUP

Use the current working directory as the project root:

```text
.
```

Do not create a nested `dynamic-ecg-lead-selection/` directory unless conflicting
files make the current directory unsafe. The directory name in the tree below is
the conceptual repository root.

## 9. Required repository structure

Create this structure:

```text
dynamic-ecg-lead-selection/
│
├── README.md
├── plan.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
├── docs/
│   ├── literature_gap_matrix.csv
│   └── novelty_assessment.md
│
├── configs/
│   ├── baseline.yaml
│   ├── fixed_leads.yaml
│   ├── corruption.yaml
│   ├── quality_estimator.yaml
│   ├── disease_specific.yaml
│   ├── dynamic_selector.yaml
│   ├── uncertainty.yaml
│   ├── external_validation.yaml
│   ├── label_mapping_external.yaml
│   └── final_evaluation.yaml
│
├── data/
│   ├── raw/
│   │   └── ptb-xl/
│   ├── processed/
│   ├── cache/
│   └── external/
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   ├── verify.py
│   │   ├── labels.py
│   │   ├── splits.py
│   │   ├── cache.py
│   │   ├── normalization.py
│   │   ├── dataset.py
│   │   ├── corruptions.py
│   │   └── external_mapping.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet1d.py
│   │   ├── shared_lead_encoder.py
│   │   ├── quality_estimator.py
│   │   ├── disease_lead_scorer.py
│   │   ├── dynamic_selector.py
│   │   ├── classifiers.py
│   │   └── uncertainty.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── losses.py
│   │   ├── callbacks.py
│   │   ├── checkpointing.py
│   │   └── thresholding.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── calibration.py
│   │   ├── selective_prediction.py
│   │   ├── robustness.py
│   │   ├── lead_analysis.py
│   │   ├── statistics.py
│   │   └── plots.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       ├── device.py
│       ├── paths.py
│       ├── logging.py
│       ├── config.py
│       ├── environment.py
│       └── reproducibility.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── inspect_dataset.py
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   ├── run_fixed_lead_experiments.py
│   ├── generate_corruptions.py
│   ├── train_quality_estimator.py
│   ├── train_disease_specific_model.py
│   ├── train_dynamic_selector.py
│   ├── evaluate_uncertainty.py
│   ├── evaluate_external.py
│   ├── final_evaluation.py
│   └── run_full_experiment_matrix.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_results.ipynb
│   ├── 03_fixed_lead_analysis.ipynb
│   ├── 04_quality_analysis.ipynb
│   ├── 05_dynamic_selector_analysis.ipynb
│   └── 06_uncertainty_analysis.ipynb
│
├── tests/
│   ├── test_download.py
│   ├── test_labels.py
│   ├── test_splits.py
│   ├── test_cache.py
│   ├── test_normalization.py
│   ├── test_dataset.py
│   ├── test_corruptions.py
│   ├── test_baseline_model.py
│   ├── test_quality_estimator.py
│   ├── test_dynamic_selector.py
│   ├── test_metrics.py
│   └── test_checkpoint_loading.py
│
└── outputs/
    ├── checkpoints/
    ├── metrics/
    ├── predictions/
    ├── figures/
    ├── logs/
    ├── tensorboard/
    └── experiment_registry/
```

Create the complete directory structure during Phase 0. Implement only Phase 0
and Phase 1 modules, scripts, tests, and configurations at this stage. Later-phase
files may be empty placeholders only when they are necessary to represent the
structure; do not add speculative or nonfunctional later-phase implementations.

---

## 10. Environment requirements

Codex must detect the available hardware:

1. NVIDIA CUDA GPU
2. Apple Silicon MPS
3. CPU fallback

Do not hard-code CUDA.

Provide a device utility that returns:

```text
cuda
mps
cpu
```

in that order of preference.

Required Python version for this project:

```text
Python 3.11 managed by uv
```

Do not use the currently installed system Python 3.13.3 for this project.

Core dependencies:

```text
torch
numpy
pandas
scipy
scikit-learn
matplotlib
wfdb
pyyaml
tqdm
tensorboard
pytest
pytest-cov
```

Optional later dependencies:

```text
optuna
captum
torchmetrics
statsmodels
seaborn
```

Avoid unnecessary packages during Phase 1.

---

## 11. Local installation commands

Project setup on macOS/Linux:

```bash
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Windows PowerShell:

```powershell
uv python install 3.11
uv venv --python 3.11
.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

Codex must add clear installation instructions to `README.md`.

Record the exact Python 3.11 patch version in experiment metadata.

Initialize Git before implementation and create meaningful commits at phase
boundaries. Recommended commits are:

```text
chore: initialize project structure and Python environment
feat: implement PTB-XL data preparation and validation
feat: implement baseline model and training pipeline
test: add Phase 1 tests and smoke-test support
results: complete Phase 1 baseline validation
```

Do not commit raw ECG data, waveform caches, trained model files, large prediction
arrays, virtual environments, or temporary logs. Add all of these to `.gitignore`.

---

## 12. Reproducibility requirements

Every experiment must record:

- random seed;
- Python version;
- PyTorch version;
- operating system;
- device type;
- GPU name, if present;
- package versions;
- configuration file;
- Git commit hash;
- start time;
- end time;
- training duration;
- checkpoint path;
- dataset version;
- dataset split;
- selected leads;
- corruption settings;
- metric results.

Use fixed seeds for:

```text
Python random
NumPy
PyTorch CPU
PyTorch CUDA
```

Run final important experiments with at least three random seeds.

Recommended initial seeds:

```text
42
123
2026
```

---

# PHASE 1 — TRUSTED 12-LEAD BASELINE

## 13. Objective

Build a correct, reproducible 12-lead multi-label classifier on PTB-XL.

Do not add:

- lead selection;
- artificial noise;
- uncertainty;
- external data;
- class-specific selectors.

The purpose is to create a trusted baseline.

---

## 14. Dataset preparation

Implement:

```bash
python scripts/prepare_data.py \
  --data-dir data/raw/ptb-xl \
  --sampling-rate 100
```

The script must:

1. detect whether the dataset already exists;
2. avoid redownloading existing files;
3. support local ZIP extraction;
4. support official PhysioNet download;
5. verify required files;
6. verify expected number of waveform records;
7. fail clearly if files are missing;
8. never silently skip corrupted records.

Required files:

```text
ptbxl_database.csv
scp_statements.csv
records100/
```

Prefer downloading/extracting the complete official archive locally instead of making tens of thousands of individual requests.

---

## 15. Dataset inspection

Implement:

```bash
python scripts/inspect_dataset.py \
  --data-dir data/raw/ptb-xl
```

The script must print:

- dataset version;
- total ECG count;
- total patient count;
- waveform header count;
- waveform data count;
- sampling rate;
- lead names;
- fold distribution;
- missing waveform count;
- invalid waveform count;
- first waveform shape;
- label distribution;
- multi-label sample count.

Expected waveform:

```text
1000 time points × 12 leads
```

Expected lead order must be verified and stored.

---

## 16. Label construction

Read:

```text
ptbxl_database.csv
scp_statements.csv
```

Steps:

1. parse `scp_codes`;
2. retain diagnostic statements;
3. map diagnostic statements to diagnostic superclasses;
4. create five multi-hot labels:
   - NORM
   - MI
   - STTC
   - CD
   - HYP
5. exclude ECGs with none of the five selected superclass labels;
6. retain every ECG with one or more selected labels without forcing a single class;
7. save label matrix;
8. save processed metadata;
9. keep original ECG ID and patient ID;
10. keep fold number;
11. keep file path.

Record:

- total records before filtering;
- total records after filtering;
- number of excluded records;
- positive count for each superclass;
- number of multi-label records.

Output files:

```text
data/processed/metadata_superclasses.csv
data/processed/labels.npy
data/processed/class_names.json
```

---

## 17. Official split

Use:

```text
Folds 1–8 → training
Fold 9     → validation and every model-development decision
Fold 10    → locked final test set
```

Never use a random split.

Fold 9 is used for threshold selection, early stopping, model comparison, lead
selection, uncertainty tuning, referral tuning, and all other development choices.
Do not calculate, display, or save fold-10 predictions or metrics during Phase 1
or any later model-development phase.

Fold 10 may be unlocked only after all of the following are frozen:

- final baseline architecture;
- lead-selection approach;
- model hyperparameters;
- disease thresholds;
- corruption protocol;
- quality-estimation approach;
- uncertainty method;
- referral thresholds;
- external-label mapping;
- final comparison models.

The final fold-10 evaluation must include at least:

1. the 12-lead baseline;
2. the strongest fixed universal lead subset;
3. the strongest disease-specific fixed model;
4. the final dynamic quality-aware model.

All final models must use checkpoints and thresholds selected without fold 10.
Test evaluation requires a separate explicit command:

```bash
python scripts/final_evaluation.py \
  --config configs/final_evaluation.yaml \
  --unlock-test
```

The script must print a clear warning before fold-10 access and append an auditable
record showing that the locked test set was accessed.

Required assertions:

- no patient overlap between train and validation;
- no patient overlap between train and test;
- no patient overlap between validation and test;
- every sample belongs to exactly one split;
- no ECG ID is duplicated.

Save split indices:

```text
data/processed/train_indices.npy
data/processed/validation_indices.npy
data/processed/test_indices.npy
```

---

## 18. Waveform cache

Convert separate WFDB files into one memory-mapped array:

```text
data/cache/ptbxl_100hz.npy
```

Expected shape:

```text
N × 1000 × 12
```

The cache process must:

- validate every waveform shape;
- validate lead count;
- validate NaN/Inf values;
- report missing files;
- report corrupt files;
- support restart or safe rebuild;
- avoid rebuilding a valid cache;
- save a cache manifest.

Save:

```text
data/cache/cache_manifest.json
```

The manifest should include:

- number of samples;
- shape;
- data type;
- dataset version;
- sampling rate;
- creation time;
- source directory.

---

## 19. Normalization

Calculate one mean and standard deviation per lead using only training data.

Do not use validation or test samples.

Save:

```text
data/processed/normalization.json
```

Format:

```json
{
  "lead_names": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
  "mean": [],
  "std": []
}
```

Apply normalization:

```text
normalized = (signal - training_mean) / training_std
```

---

## 20. Dataset class

Implement a PyTorch dataset that returns:

```text
signal: shape 12 × 1000
target: shape 5
metadata: optional ECG ID and patient ID
```

Requirements:

- memory-mapped reads;
- no full-dataset RAM loading;
- float32 signals;
- float32 targets;
- optional selected-lead mode for later phases;
- optional corruption transform for later phases;
- deterministic validation/test behavior.

---

## 21. Baseline architecture

Implement a modern 1D ResNet.

Input:

```text
batch × 12 × 1000
```

Output:

```text
batch × 5 logits
```

Recommended components:

- 1D convolution stem;
- batch normalization;
- ReLU;
- residual blocks;
- increasing channel widths;
- temporal downsampling;
- adaptive global average pooling;
- dropout;
- linear classification layer.

Do not use softmax.

Use:

```text
BCEWithLogitsLoss
```

because this is multi-label classification.

---

## 22. Baseline configuration

Example `configs/baseline.yaml`:

```yaml
seed: 42

data:
  root: data/raw/ptb-xl
  cache: data/cache/ptbxl_100hz.npy
  sampling_rate: 100
  input_length: 1000
  leads:
    - I
    - II
    - III
    - aVR
    - aVL
    - aVF
    - V1
    - V2
    - V3
    - V4
    - V5
    - V6

labels:
  classes:
    - NORM
    - MI
    - STTC
    - CD
    - HYP

split:
  train_folds: [1, 2, 3, 4, 5, 6, 7, 8]
  validation_folds: [9]
  test_folds: [10]

evaluation:
  allow_test_evaluation: false

model:
  name: resnet1d
  input_channels: 12
  output_classes: 5
  dropout: 0.3

training:
  batch_size: 64
  epochs: 30
  optimizer: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
  early_stopping_patience: 5
  gradient_clip_norm: 5.0
  num_workers: 4
  mixed_precision: true

selection_metric:
  name: validation_macro_auroc
  mode: max
```

Codex may adjust batch size according to available memory.

---

## 23. Training requirements

Training command:

```bash
python scripts/train_baseline.py \
  --config configs/baseline.yaml
```

Training must include:

- device detection;
- AdamW;
- learning-rate scheduler;
- early stopping;
- gradient clipping;
- optional mixed precision on CUDA;
- checkpoint saving;
- CSV logging;
- TensorBoard logging;
- best model selected by validation macro AUROC;
- latest checkpoint;
- resume training support.

After all Phase 1 unit tests and the smoke test pass, run the complete baseline on
folds 1–8 and perform full validation on fold 9. Do not merely document the command,
and do not evaluate fold 10.

Before full training, print and save:

- project root;
- exact Python version;
- PyTorch version;
- operating system;
- selected device;
- GPU or MPS information;
- available RAM;
- available disk space;
- dataset version;
- model parameter count;
- configuration path;
- Git commit hash.

Save:

```text
outputs/checkpoints/baseline_best.pt
outputs/checkpoints/baseline_last.pt
outputs/logs/baseline_training.csv
outputs/tensorboard/baseline/
```

The complete baseline run must also produce fold-9 validation predictions,
validation thresholds, validation metrics, validation figures, package and hardware
metadata, and an experiment-registry entry. It must not produce fold-10 outputs.

---

## 24. Threshold selection

Do not use a fixed threshold of `0.5` without comparison.

Select one disease-specific threshold using validation data only.

Recommended initial strategy:

- search thresholds from `0.05` to `0.95`;
- maximize validation F1 per class;
- freeze thresholds for subsequent development and final evaluation;
- never apply them to fold 10 until the final test protocol is explicitly unlocked.

Save:

```text
outputs/metrics/baseline_validation_thresholds.csv
```

Never tune thresholds on the test set.

---

## 25. Baseline metrics

Report:

### Overall metrics

- macro AUROC;
- macro AUPRC;
- macro F1;
- macro precision;
- macro sensitivity;
- macro specificity;
- micro F1;
- subset accuracy only as a secondary metric.

### Per-class metrics

For every class:

- AUROC;
- AUPRC;
- F1;
- precision;
- sensitivity;
- specificity;
- false-negative rate;
- support;
- selected threshold.

Save:

```text
outputs/metrics/baseline_summary.csv
outputs/metrics/baseline_per_class.csv
outputs/predictions/baseline_validation_probabilities.npy
outputs/predictions/baseline_validation_targets.npy
```

During Phase 1, every metric and prediction above is for fold 9. Do not calculate,
display, or save fold-10 metrics or predictions.

---

## 26. Baseline figures

Generate:

- class distribution;
- training loss;
- validation loss;
- validation macro AUROC;
- per-class AUROC;
- per-class F1;
- five binary confusion matrices;
- ROC curves;
- precision-recall curves.

Save in:

```text
outputs/figures/baseline/
```

---

## 27. Phase 1 acceptance criteria

Phase 1 is complete only when:

- dataset verification passes;
- patient leakage checks pass;
- input batch shape is correct;
- output shape is correct;
- loss remains finite;
- training loss decreases;
- validation AUROC improves;
- best checkpoint reloads correctly;
- complete fold-9 validation metrics are produced;
- the locked fold-10 evaluation pipeline is ready but has not been executed;
- all tests pass;
- results are saved;
- README commands work.

Codex must stop after Phase 1 and report:

1. files created;
2. tests run;
3. test results;
4. model parameter count;
5. best validation macro AUROC;
6. validation macro AUPRC;
7. validation macro F1;
8. validation macro precision, sensitivity, and specificity;
9. per-class validation metrics;
10. validation-selected disease thresholds;
11. training history;
12. checkpoint-loading verification;
13. test-pipeline readiness status;
14. known problems.

### Phase 1 completion status

**Accepted and complete.** The authoritative baseline is commit `04c1ed0`.

Accepted evidence includes:

- replacement audit completed;
- 22/22 tests passing;
- checkpoint reproducibility verified;
- complete timing recorded;
- validation predictions verified finite;
- experiment-registry handling verified;
- fold 10 locked and untouched.

Authoritative fold-9 baseline:

| Metric | Value |
|---|---:|
| Macro AUROC | 0.923710 |
| Macro AUPRC | 0.802822 |
| Macro F1 | 0.743789 |
| Macro precision | 0.722661 |
| Macro sensitivity | 0.774599 |
| Macro specificity | 0.900852 |
| Micro F1 | 0.772979 |

Authoritative per-class fold-9 sensitivity:

| Class | Sensitivity |
|---|---:|
| NORM | 0.8890 |
| MI | 0.7759 |
| STTC | 0.8371 |
| CD | 0.7030 |
| HYP | 0.6679 |

---

# PHASE 2 — FIXED-LEAD EXPERIMENTS

## 28. Objective and scope

Proceed with Phase 2 only. Do not implement signal corruption, the lead-quality
estimator, dynamic lead selection, uncertainty, external validation, or fold-10
evaluation. Keep fold 10 fully locked.

Determine:

1. whether fewer ECG leads can preserve most of the 12-lead baseline performance;
2. whether the Paper 8 subset works well on PTB-XL;
3. which leads are most useful for each disease;
4. whether different diseases show different lead dependencies;
5. which fixed subsets should be retained as baselines for the later dynamic selector.

Provisional performance-preservation targets:

| Measure | Target |
|---|---:|
| Macro AUPRC | >= 0.762681 (`0.802822 × 0.95`) |
| Macro F1 | >= 0.723789 (`0.743789 - 0.02`) |
| NORM sensitivity | >= 0.8390 |
| MI sensitivity | >= 0.7259 |
| STTC sensitivity | >= 0.7871 |
| CD sensitivity | >= 0.6530 |
| HYP sensitivity | >= 0.6179 |

These are research targets, not reasons to hide or discard negative results.

### Required experimental distinction

Implement and report two clearly separated experiment types:

**A. Masked-input screening**

Use the existing trained 12-lead baseline. Set excluded channels to zero after
normalization. Because zero is the normalized mean, this is preferable to inserting
arbitrary raw values. Do not retrain the baseline for these experiments.

This measures how dependent the existing 12-lead model is on particular leads. It
is only a fast screening and ranking method. Do not describe masked-input results
as the performance of a model trained using fewer leads.

**B. Retrained subset models**

Train a new model from scratch using only the selected leads. The model input
channel count must match the number of selected leads.

These results represent the actual performance of reduced-lead models and must be
used for final Phase 2 comparisons. Never mix masked-input and retrained-subset
results in the same table without an explicit `experiment_type` column.

---

## 29. Step 1 — Generalize lead configuration

Update the dataset and model pipeline to accept a selected-leads list from YAML:

```yaml
data:
  selected_leads:
    - II
    - aVR
    - V1
    - V4
```

Use this canonical lead order:

```text
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

Validate that:

1. every requested lead exists;
2. no lead is duplicated;
3. selected leads are returned in canonical order;
4. model `input_channels` equals the selected-lead count;
5. saved checkpoints record the exact lead names and indices.

Do not change preprocessing, folds, normalization values, labels, or evaluation
definitions. Use the corresponding training-derived normalization values for every
selected lead.

---

## 30. Step 2 — Add Phase 2 configurations

Create configurations for:

1. all 12 leads;
2. Paper 8 subset: `II, aVR, V1, V4`;
3. one-lead experiments;
4. leave-one-lead-out experiments;
5. two-lead subset screening;
6. four-lead subset screening;
7. retrained shortlisted subsets.

Paper 8 citation:

> Lai C, Zhou S, and Trayanova NA. “Optimal ECG-lead selection increases
> generalizability of deep learning on ECG abnormality classification.” DOI:
> `10.1098/rsta.2020.0258`. PMID: `34689629`.

Do not assume that `II, aVR, V1, V4` are universally optimal.

---

## 31. Step 3 — Masked-input screening

Using the authoritative 12-lead checkpoint, run fold-9 inference for:

1. each individual lead alone;
2. leave-one-lead-out for all 12 leads;
3. Paper 8's four-lead subset;
4. all 66 two-lead combinations;
5. all 495 four-lead combinations.

These are inference-only experiments.

- Single-lead-only: keep one normalized lead and replace the other 11 normalized
  channels with zero.
- Leave-one-lead-out: replace only the removed normalized lead with zero.
- Subset screening: keep the selected normalized channels and replace all other
  normalized channels with zero.
- Do not reselect or modify the authoritative model checkpoint.

Use threshold-free metrics as the primary screening criteria, in this order:

1. macro AUPRC;
2. macro AUROC;
3. per-class AUPRC;
4. per-class AUROC.

Thresholded F1 and sensitivity may be reported, but do not use them as the primary
ranking criteria because fold 9 is used for development comparisons.

Save:

```text
outputs/metrics/phase2/masked_single_lead_results.csv
outputs/metrics/phase2/masked_leave_one_out_results.csv
outputs/metrics/phase2/masked_two_lead_results.csv
outputs/metrics/phase2/masked_four_lead_results.csv
outputs/metrics/phase2/masked_paper8_subset.csv
```

Each row must include:

```text
experiment_id
experiment_type
selected_leads
excluded_leads
number_of_selected_leads
macro_AUROC
macro_AUPRC
macro_F1
macro_sensitivity
NORM_AUPRC
MI_AUPRC
STTC_AUPRC
CD_AUPRC
HYP_AUPRC
NORM_sensitivity
MI_sensitivity
STTC_sensitivity
CD_sensitivity
HYP_sensitivity
checkpoint_commit
seed
```

---

## 32. Step 4 — Create disease-specific lead rankings

For every disease, combine evidence from:

1. single-lead AUPRC;
2. single-lead AUROC;
3. leave-one-lead-out performance drop;
4. pair-subset performance;
5. four-lead-subset performance.

Create:

```text
outputs/metrics/phase2/disease_specific_lead_rankings.csv
```

Required columns:

```text
disease
lead
single_lead_AUPRC
single_lead_AUROC
leave_one_out_AUPRC_drop
leave_one_out_AUROC_drop
selection_frequency_in_top_pairs
selection_frequency_in_top_four_subsets
combined_rank
```

Do not interpret one metric alone as clinical lead importance. The purpose is to
produce candidate rankings for later experiments.

---

## 33. Step 5 — Shortlist subsets for retraining

Do not train all 66 two-lead or 495 four-lead models. Use masked screening only to
shortlist candidates.

Shortlist at least:

1. Paper 8 subset: `II, aVR, V1, V4`;
2. the best universal four-lead subset by masked macro AUPRC;
3. two additional high-performing four-lead subsets that are not simple duplicates
   of the best subset;
4. the best universal two-lead subset by masked macro AUPRC;
5. two additional high-performing two-lead subsets;
6. the best single lead.

Prefer diversity when selecting candidates. Do not choose three four-lead subsets
that differ by only one lead unless the performance difference justifies it.

Save the shortlist and its selection rationale in:

```text
outputs/metrics/phase2/retraining_shortlist.csv
docs/phase2_shortlist_rationale.md
```

---

## 34. Step 6 — Retrain shortlisted models

Train each shortlisted subset from scratch using seed 42.

Reuse the same official folds, training budget, optimizer, learning-rate scheduler,
early-stopping rule, checkpoint-selection metric, label construction, normalization
protocol, metrics, logging, and experiment registry. The model may change only
where required by `input_channels`.

Do not reuse the 12-lead model weights for reduced-lead training.

For each retrained model:

1. train on folds 1–8;
2. select the checkpoint using fold-9 macro AUROC;
3. calculate fold-9 macro AUPRC and per-class metrics;
4. select model-specific class thresholds using fold 9;
5. save thresholds for future final fold-10 evaluation;
6. do not access fold 10.

The primary comparison metric is macro AUPRC. Secondary comparison metrics are:

- macro AUROC;
- macro F1;
- macro sensitivity;
- per-class AUPRC;
- per-class F1;
- per-class sensitivity;
- false-negative rate;
- number of leads;
- training time;
- inference time;
- parameter count.

Save:

```text
outputs/metrics/phase2/retrained_subset_comparison.csv
outputs/checkpoints/phase2/<subset-name>_best.pt
outputs/checkpoints/phase2/<subset-name>_last.pt
```

---

## 35. Step 7 — Model-specific thresholds

Every retrained subset model must receive its own validation-selected thresholds.
Do not reuse the 12-lead baseline thresholds for a separately trained model.

Store:

```text
outputs/metrics/phase2/thresholds/<subset-name>_validation_thresholds.csv
```

Use macro AUPRC and AUROC, not optimized validation F1, as the main basis for
selecting the strongest model. The saved thresholds will be frozen for eventual
fold-10 evaluation.

---

## 36. Step 8 — Phase 2 figures

Generate under `outputs/figures/phase2/`:

1. single-lead per-class AUPRC heatmap;
2. leave-one-lead-out performance-drop heatmap;
3. top 20 two-lead subsets;
4. top 20 four-lead subsets;
5. macro AUPRC versus number of leads;
6. retrained-model comparison;
7. per-class sensitivity comparison;
8. Paper 8 subset versus the best discovered subset;
9. disease-specific lead-ranking heatmap.

---

## 37. Step 9 — Tests

Add tests for:

1. selected-lead name validation;
2. canonical lead ordering;
3. duplicate-lead rejection;
4. correct channel extraction;
5. masked-input mode changes only excluded channels;
6. masking occurs after normalization;
7. variable input-channel model shapes;
8. subset checkpoint save and reload;
9. checkpoint records selected leads;
10. fold-10 access remains refused;
11. full 12-lead behavior is unchanged;
12. experiment registry includes selected leads and experiment type.

Run the entire test suite, not only new Phase 2 tests.

---

## 38. Step 10 — Compute and seed policy

Use seed 42 for Phase 2 screening and initial retraining. Do not run seeds 123 and
2026 for every screened subset.

After the shortlist has been reduced to the strongest models, report which models
should later be repeated with all three seeds.

Do not repeat the authoritative 12-lead baseline unless a code change alters its
behavior. Confirm with a regression test that the generalized selected-lead code
produces identical validation predictions for the all-12-lead configuration.

---

## 39. Step 11 — Phase 2 interpretation

At the end, answer:

1. What is the strongest retrained four-lead model?
2. Does it reach macro AUPRC >= 0.762681?
3. Does it keep macro F1 >= 0.723789?
4. Which disease loses the most performance?
5. Which disease benefits most from particular leads?
6. Does Paper 8's subset remain competitive on PTB-XL?
7. Are the best leads different across diseases?
8. Are masked-screening rankings consistent with retrained results?
9. Which models should proceed to three-seed evaluation?
10. Is there enough evidence to continue toward disease-specific dynamic selection?

A model failing the provisional preservation targets must still be reported.

---

## 40. Step 12 — Phase 2 stop condition and report

Stop after:

- masked screening is complete;
- shortlisted reduced-lead models are retrained;
- Phase 2 metrics and figures are saved;
- all tests pass;
- experiment registry is updated;
- results are committed;
- the Phase 2 completion report is produced.

Do not begin corruption generation, the quality estimator, disease-specific soft
weighting, dynamic top-k selection, uncertainty, external validation, or fold-10
evaluation.

The Phase 2 completion report must include:

1. Git commit hash;
2. files created and changed;
3. tests run and results;
4. masked single-lead ranking;
5. leave-one-lead-out ranking;
6. top two-lead subsets;
7. top four-lead subsets;
8. Paper 8 subset results;
9. retraining shortlist;
10. retrained subset comparison;
11. performance-preservation ratios;
12. per-class changes from the 12-lead baseline;
13. training and inference times;
14. differences between masked and retrained rankings;
15. models recommended for three-seed evaluation;
16. whether the evidence supports continuing to dynamic lead selection;
17. confirmation that fold 10 remained locked and untouched;
18. confirmation that Phase 3 has not started.

### Phase 2 execution status

**Accepted and complete.** The authoritative Phase 2 commit is
`9242bdde2aaa5fcd12c6b7da9c0891d43eacc389`. Masked screening covered all 586 required fold-9
experiments, eight shortlisted reduced-lead models were retrained from scratch, all
required metrics and figures were saved, and the all-12 generalized path reproduced
all 2,146 authoritative validation probabilities exactly. The strongest retrained
four-lead subset was `aVR,V1,V3,V5` with macro AUPRC `0.789837`, macro AUROC
`0.915940`, macro AUPRC preservation `0.983827` (98.38%), and macro F1 `0.730720`.
Paper 8's `II,aVR,V1,V4` scored macro AUPRC `0.786692`, macro AUROC `0.918068`,
and macro F1 `0.732078`. The best two-lead subset, `V1,V5`, scored macro AUPRC
`0.761447`. The full suite passed 40/40 tests. Fold 10 remained locked and untouched.

---

# PHASE 3 — CONTROLLED ECG CORRUPTION AND FIXED-MODEL ROBUSTNESS BENCHMARK

## 41. Purpose and scope

Phase 3 is an evaluation and data-generation phase. It must answer:

1. How much does the 12-lead model deteriorate when individual leads become noisy or unavailable?
2. How much do the best reduced-lead models deteriorate when one selected lead becomes noisy or unavailable?
3. Are reduced-lead models more fragile because they have fewer alternative leads?
4. Which corruption types and severities cause the largest performance losses?
5. Which disease classes are most sensitive to lead corruption?
6. Is there sufficient evidence that a lead-quality estimator and dynamic replacement mechanism are necessary?

Proceed with Phase 3 only. Do not train a new diagnostic classifier or implement the
lead-quality estimator, disease-specific soft weighting, dynamic top-k selection,
uncertainty, external validation, or fold-10 evaluation.

## 42. Frozen models

Evaluate these four existing best checkpoints using their existing clean-validation thresholds:

1. full 12-lead baseline: `I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6`;
2. best discovered four-lead model: `aVR,V1,V3,V5`;
3. Paper 8 four-lead model: `II,aVR,V1,V4`;
4. best two-lead model: `V1,V5`.

Do not retrain them and do not reselect thresholds under noisy conditions. The clean
fold-9 thresholds remain frozen so noisy evaluation measures deployment-like degradation.

Record in the Phase 3 report that the best universal subset was not uniformly best
for all diseases. For `aVR,V1,V3,V5` relative to the 12-lead model, the approximate
AUPRC changes were NORM `-0.0054`, MI `-0.0661`, STTC `+0.0219`, CD `-0.0406`, and
HYP `+0.0253`. Paper 8 preserved MI and CD better but was substantially weaker for
HYP. This is preliminary evidence for disease-dependent combinations, not a clinical
lead-importance claim.

## 43. Corruption module

Implement pure or reproducibly seeded functions in `src/data/corruptions.py` with an
interface equivalent to:

```python
corrupt_signal(
    signal,
    lead_indices,
    corruption_type,
    severity,
    random_generator,
    parameters,
) -> corrupted_signal, corruption_metadata
```

Input shape is `1000 × 12` or the selected-lead equivalent. Return the same shape and
complete metadata. Never modify the original cached waveform or overwrite PTB-XL files.

Implement and test:

1. baseline wander;
2. muscle artefact or high-frequency noise;
3. electrode-motion artefact or low-frequency transient disturbance;
4. Gaussian noise as a control;
5. power-line interference;
6. flat-line lead;
7. completely missing lead;
8. partial temporal dropout;
9. signal clipping or saturation;
10. amplitude scaling;
11. combined corruption: one noisy lead plus one missing lead.

Each corruption affects only requested leads. Unaffected leads remain exactly unchanged.

## 44. Severity system and pilot

Use `clean`, `mild`, `moderate`, `severe`, and `unusable`, mapped to future quality classes:

```text
clean      → clean
mild       → degraded
moderate   → degraded
severe     → unusable
unusable   → unusable
```

For additive-noise-style pilot settings begin with no added noise, approximately
20 dB, 10 dB, 5 dB, and 0 dB SNR respectively. Do not force SNR definitions onto
missing leads, flat lines, clipping, partial dropout, or amplitude scaling; use
separate parameters for each corruption type.

Before complete fold-9 evaluation, run a small training-fold pilot, generate visual
examples and numerical distortion summaries, and verify strictly increasing visible
and numerical distortion from mild through unusable, finite signals, and exact
preservation of unaffected leads. Freeze the Phase 3 configuration after the pilot.
Do not tune severity to produce a desired model result.

## 45. Deterministic manifests and seeds

Prefer deterministic manifests to duplicate corrupted datasets. For every corrupted
sample record ECG ID, patient ID, fold, global seed, corruption seed, corruption type,
severity, corrupted lead names and indices, affected interval, parameters, per-lead
quality-class labels, missing-lead flags, and scenario name.

Save:

```text
data/processed/corruption_manifests/phase3_validation_manifest.csv
data/processed/corruption_manifests/phase3_pilot_manifest.csv
data/processed/corruption_manifests/phase3_manifest_metadata.json
```

Use dedicated `corruption_seed: 31415`. Derive per-sample seeds deterministically from
the global seed, ECG ID, corruption type, severity, and scenario using SHA-256, never
Python's unstable built-in hash. The same manifest must reproduce identical waveforms
and predictions.

## 46. Corruption scenarios

Create:

A. clean reference;
B. one deterministically selected random available lead corrupted;
C. two random available leads corrupted;
D. four random available leads corrupted;
E. one selected model-input lead corrupted;
F. one non-selected lead corrupted for reduced models, which must leave predictions exactly identical;
G. one selected lead missing;
H. two selected leads missing where the model has at least two inputs;
I. all selected leads degraded;
J. all selected leads missing;
K. mixed corruption: one selected lead receives noise and another is missing;
L. disease-ranking stress: corrupt a highly ranked Phase 2 lead for each disease.

Scenario L is analysis-only. If the true disease label selects the lead, label it
`oracle disease-specific stress test` and never present it as deployable.

## 47. Fair corruption and missing-lead rules

Apply corruption before normalization, then use the same frozen training-only
normalization statistics as the original model.

During the pilot compare:

1. raw zero before normalization;
2. normalized zero, representing the training mean;
3. flat constant using the lead's training mean.

Choose, justify, document, and freeze one missing representation for the main benchmark.
Do not silently mix representations.

## 48. Frozen-threshold evaluation and metrics

Never retune thresholds by corruption or severity. Report threshold-free macro/per-class
AUROC and AUPRC, plus frozen-threshold macro F1, sensitivity, specificity, micro F1,
and per-class F1, sensitivity, specificity, and false-negative rate.

For every model, corruption type, severity, and scenario, also report absolute and
relative change from clean, corrupted-lead count, and affected model-input-lead count.
Calculate robustness AUC across severity, worst-severity performance, maximum absolute
drop, average non-clean drop, per-disease robustness ranking, and per-corruption ranking.

## 49. Required outputs

Save under `outputs/metrics/phase3/`:

```text
phase3_full_results.csv
phase3_clean_reference.csv
phase3_corruption_summary.csv
phase3_model_robustness_ranking.csv
phase3_disease_robustness_ranking.csv
phase3_severity_summary.csv
phase3_missing_lead_results.csv
phase3_selected_vs_nonselected_corruption.csv
phase3_robustness_auc.csv
phase3_pilot_distortion_summary.csv
```

Save only useful probabilities under `outputs/predictions/phase3/`, at minimum clean
reference, severe selected-lead corruption, missing selected-lead, and mixed-corruption
scenarios.

Generate under `outputs/figures/phase3/`:

1. clean versus corrupted ECG examples;
2. examples for each corruption type;
3. severity progression examples;
4. macro AUPRC versus severity;
5. macro F1 versus severity;
6. per-class AUPRC-drop heatmap;
7. per-class sensitivity-drop heatmap;
8. missing-lead robustness heatmap;
9. selected- versus non-selected-lead corruption comparison;
10. 12-lead versus best-four robustness comparison;
11. best-four versus Paper 8 comparison;
12. robustness AUC by model and corruption type.

## 50. Required tests and regressions

Add tests for:

1. deterministic corruption with a fixed seed;
2. different seeds changing stochastic corruption;
3. clean severity leaving signals unchanged;
4. only requested leads changing;
5. unaffected leads remaining exactly identical;
6. unchanged waveform shape;
7. finite outputs;
8. monotonically increasing severity parameters;
9. correct flat-line corruption;
10. correct missing-lead corruption;
11. partial dropout affecting only its requested interval;
12. clipping respecting limits;
13. corruption occurring before normalization;
14. identical manifest regeneration;
15. non-selected corruption producing identical reduced-model predictions;
16. clean predictions exactly matching authoritative saved clean predictions;
17. existing checkpoints reloading correctly;
18. fold-10 access remaining refused;
19. no test artifact being created;
20. registry records including corruption configuration and manifest hash.

Run the complete test suite. Before accepting Phase 3 confirm clean prediction equality
for every model (subject only to unavoidable saved precision), exact non-selected-lead
prediction equality, exact deterministic-repeat prediction equality, continued fold-10
refusal, and unchanged diagnostic weights.

## 51. Interpretation questions

Answer:

1. Which model is most robust overall and has the highest robustness AUC?
2. Does the 12-lead model benefit from redundant leads?
3. Are four-lead models more fragile when one selected lead fails?
4. Is the two-lead model unacceptably fragile?
5. Which corruption causes the largest macro AUPRC and sensitivity losses?
6. Which disease is most sensitive?
7. Do the best-four and Paper 8 models fail differently?
8. Does non-selected corruption leave reduced-model predictions unchanged?
9. Is aggregate degradation monotonic with severity?
10. Is a trainable lead-quality estimator justified?
11. Which corruption types should Phase 4 use?
12. How should severities map to clean/degraded/unusable?
13. Which definitions remain unrealistic or unstable?

## 52. Acceptance, reporting, and stop condition

Phase 3 is complete only when the system is deterministic, all corruption types and
frozen severities are documented, manifests are saved, all four frozen models are
evaluated, regressions and sanity checks pass, metrics and figures are saved, the full
test suite passes, fold 10 stays locked, no diagnostic model is retrained, results are
committed, and a completion report is produced.

Report:

1. final Git commit hash;
2. files created and changed;
3. corruption types and exact frozen parameters;
4. manifest sizes and hashes;
5. full test-suite result;
6. clean regression result;
7. deterministic-repeat result;
8. selected versus non-selected sanity result;
9. four-model robustness ranking;
10. per-corruption and per-severity drops;
11. per-class robustness;
12. missing-lead, selected-lead-failure, and mixed-corruption results;
13. figures generated;
14. recommended Phase 4 corruptions and three-class quality mapping;
15. unresolved limitations;
16. confirmation that fold 10 remained locked, no diagnostic model was retrained, and Phase 4 did not start.

Stop after the robustness benchmark. Do not implement the lead-quality neural network,
shared per-lead encoders, disease-specific scoring, soft weighting, hard top-k selection,
uncertainty, external validation, or fold-10 evaluation. Phase 4 requires explicit approval.

### Phase 3 execution status

**Accepted and complete.** The authoritative Phase 3 commit is
`6a84981c755d980ad9f285286d896fa63ca8d24b`. The frozen benchmark evaluated four accepted checkpoints
over 272 fold-9 conditions using a 583,712-row deterministic validation manifest. The
validation-manifest SHA-256 is
`097a9043ad68cd6fe8802245c593a3bb46ed4a798f30c8bd0bd653b37ca1c2ea`, generated with
master corruption seed `31415`. The full suite passed 61/61 tests. The
12-lead model ranked first by clean-normalized macro-AUPRC robustness AUC (`0.997672`),
followed by `aVR,V1,V3,V5` (`0.973380`), Paper 8 (`0.958656`), and `V1,V5`
(`0.953473`). All four clean regressions and deterministic repeats were exact, all 12
non-selected corruption checks were exact, and checkpoint hashes were unchanged. Fold
10 remained locked, no diagnostic model was retrained, and Phase 4 had not started at
acceptance. Flat-line corruption, amplitude failure, and partial dropout caused major
degradation; MI had the largest average degradation; and the two-lead model was
particularly fragile when one selected lead failed. These accepted results support a
trainable per-lead quality estimator. See `docs/phase3_completion_report.md`.

---

# PHASE 4 — TRAINABLE PER-LEAD ECG QUALITY ESTIMATION

## 53. Purpose, scope, and research questions

Build a shared model that receives an individual ECG lead and predicts `clean`,
`degraded`, or `unusable`. Full-ECG inference must return:

```text
quality_logits:        batch × 12 × 3
quality_probabilities: batch × 12 × 3
quality_scores:        batch × 12
```

Phase 4 must determine whether the estimator distinguishes all three classes across
all leads and corruption families, generalizes to unseen strengths, detects missing,
flat, clipped, noisy, and partially dropped-out signals, produces a severity-monotonic
score, correlates with Phase 3 diagnostic degradation, responds to natural PTB-XL
quality flags, and is stable enough for a later dynamic selector.

Proceed with Phase 4 only. Do not implement disease-specific lead scoring or weighting,
dynamic top-k selection, diagnostic-prediction uncertainty, external validation, or
fold-10 evaluation.

## 54. Step 1 — Plan gate

Before source-code changes, mark Phase 3 accepted and complete, record commit
`6a84981c755d980ad9f285286d896fa63ca8d24b`, accepted results, manifest hash, and this
complete Phase 4 protocol in `plan.md`; record the 50 Hz/Nyquist limitation; show the
exact diff; then implement. Only a new contradiction should pause execution.

## 55. Step 2 — Frozen quality labels

Freeze class indices and severity mapping as:

```text
0 = clean:     clean severity
1 = degraded:  mild or moderate severity
2 = unusable:  severe or unusable severity
```

Save the mapping in `configs/quality_estimator.yaml` and
`data/processed/quality_label_mapping.json`. Never change it after training begins.

## 56. Step 3 — Power-line correction

Exactly 50 Hz at 100 Hz sampling is Nyquist-, phase-, and alias-sensitive. Keep the
frozen 50 Hz implementation only as a documented Phase 3 artifact, but configure the
primary Phase 4 synthetic power-line corruption at **49 Hz**. Do not use 60 Hz directly
at 100 Hz without explicit alias handling. Evaluate real power-line noise later using
real recordings or a higher sampling rate. Add a test that Phase 4 uses 49 Hz, not 50 Hz.

## 57. Step 4 — Data splits and safety

Use folds 1–8 for quality-estimator training and fold 9 for validation, checkpoint
selection, calibration, and every Phase 4 development decision. Keep fold 10 locked.
Generate training corruptions only from folds 1–8 and validation corruptions only from
fold 9; never train on Phase 3 validation outputs. Assert patient separation, fold-10
refusal, and absence of fold-10 manifests and prediction artifacts.

## 58. Step 5 — Individual-lead design

The core model input is `batch × 1 × 1000` and output is `batch × 3` logits. A wrapper
accepts `batch × 12 × 1000`, reshapes to `(batch × 12) × 1 × 1000`, applies exactly the
same shared estimator, and restores `batch × 12 × 3`. Never train 12 independent models.

## 59. Step 6 — Deterministic on-the-fly training corruption

Generate folds 1–8 training corruption on the fly rather than saving a duplicated
dataset. SHA-256-derived seeds must depend on master seed, epoch, ECG ID, lead name,
requested quality class, and corruption type; never use Python's built-in hash. The
same run/configuration/seed must reproduce the corruption sequence while epochs may
change it reproducibly. Save the corruption-configuration hash, code version, Git
commit, master seed, and per-epoch seed policy.

## 60. Step 7 — Balanced quality sampling

Sample approximately equal lead instances from clean, degraded, and unusable classes.
Degraded uses mild and moderate corruptions. Unusable uses severe/unusable corruption,
missing and flat leads, severe clipping/dropout, and severe amplitude failure. Balance
corruption families within each class. Report the training distribution by class,
corruption type, severity, and lead.

## 61. Step 8 — Training corruption families

Use all frozen Phase 3 functions: baseline wander, muscle artefact, electrode motion,
Gaussian noise, power-line interference configured at 49 Hz, flat line, missing lead,
partial dropout, clipping/saturation, amplitude scaling/failure, and combined
corruption. Retain Phase 3 strengths except the documented 49 Hz adjustment. Never
change strength to obtain desired metrics.

## 62. Step 9 — Fixed fold-9 validation manifests

Create:

```text
data/processed/quality_manifests/quality_validation_balanced.csv
data/processed/quality_manifests/quality_validation_severity.csv
data/processed/quality_manifests/quality_validation_unseen_parameters.csv
data/processed/quality_manifests/quality_validation_combined.csv
data/processed/quality_manifests/quality_validation_clean.csv
data/processed/quality_manifests/quality_manifest_metadata.json
```

The sets respectively cover balanced classes/all training families, matched severity
progressions, within-family parameter values absent from training, uncommon/absent
training combinations, clean fold-9 leads, and an audit definition for available
PTB-XL signal-quality metadata. Record row counts and SHA-256 hashes. Never create a
fold-10 manifest.

## 63. Step 10 — Primary architecture

Implement `SharedLeadQualityCNN`: Conv1d normalization/activation/downsampling/dropout
blocks with residual connections where appropriate, adaptive global average pooling,
a compact lead embedding, and a linear three-class head. Expose:

```text
encode_lead(signal) -> embedding
forward(signal) -> quality_logits
predict_quality_probabilities(signal) -> probabilities
```

Do not use disease labels or disease prediction.

## 64. Step 11 — Baselines

Compare the neural model with (A) a rule detector using standard deviation, range,
constant fraction, derivative energy, clipping fraction, and missing/flat indicators;
and (B) a lightly tuned classical classifier such as logistic regression, random forest,
or gradient boosting on signal-quality features.

## 65. Step 12 — Optional lead-identity ablation

Only after the primary model succeeds, optionally compare the signal-only shared model
with a shared model augmented by a learned lead-identity embedding. Retain lead identity
only if generalization improves without large inter-lead inconsistency. Never create
12 encoders. This ablation is not required for Phase 4 acceptance.

## 66. Step 13 — Training protocol

Create `configs/quality_estimator.yaml`, `scripts/train_quality_estimator.py`, and
`scripts/evaluate_quality_estimator.py`. Use PyTorch, AdamW, cross-entropy, balanced
sampling, early stopping, LR scheduling, gradient clipping, checkpoint saving, exact
resume, CSV and TensorBoard logging, and MPS/CUDA/CPU support. Select checkpoints by
fold-9 macro F1, breaking ties with unusable-class recall. Save best and last checkpoints
under `outputs/checkpoints/phase4/`. Never select with diagnostic AUROC.

## 67. Step 14 — Loss

Begin with ordinary three-class cross-entropy plus balanced sampling. Use focal loss
only as a separately recorded ablation if unusable recall is poor, degraded confusion
persists, and balanced sampling is insufficient.

## 68. Step 15 — Probabilities and continuous score

Produce calibrated `P(clean)`, `P(degraded)`, and `P(unusable)`. Store this formula once
in configuration:

```text
quality_score = 1.0 * P(clean) + 0.5 * P(degraded) + 0.0 * P(unusable)
```

Test the `[0,1]` bound and correct monotonic response to clean/unusable probability.

## 69. Step 16 — Calibration

Report ECE, multiclass Brier score, reliability diagrams, and per-class calibration.
If needed, fit temperature scaling using fold 9 only and save
`outputs/checkpoints/phase4/quality_temperature_scaling.json`. Report before and after;
never calibrate on fold 10.

## 70. Step 17 — Safety-oriented unusable detection

In addition to argmax, evaluate unusable versus not unusable: AUROC, AUPRC, recall,
precision, and false-negative rate. On fold 9, choose a threshold targeting unusable
recall at least `0.95`; among satisfying thresholds maximize precision. If infeasible
without unreasonable false positives, report the trade-off. Save
`outputs/metrics/phase4/unusable_detection_threshold.csv`.

## 71. Step 18 — Required metrics

Report overall macro/weighted F1, balanced accuracy, secondary overall accuracy,
multiclass cross-entropy, ECE, and Brier score. Report per-class precision, recall, F1,
support, one-vs-rest AUROC/AUPRC; per-lead macro F1 and all recalls plus unusable FNR;
per-corruption macro F1, severity accuracy, unusable recall, and mean score; and
per-severity predicted class distribution plus score mean, median, and standard deviation.

## 72. Step 19 — Monotonicity

For each corruption family and lead, assess aggregate `clean > mild > moderate > severe
> unusable` behavior using mean and median quality score, Spearman correlation, and
monotonicity violations. Do not require perfect sample-level monotonicity.

## 73. Step 20 — Diagnostic-degradation relation

Using frozen Phase 3 outputs only, compare scenario-level mean quality and predicted
unusable count with Phase 3 macro-AUPRC and macro-sensitivity drops. Report Spearman and
secondary Pearson correlations plus appropriate confidence-interval plots. Expected
direction: lower quality corresponds to larger diagnostic loss. Never train on these
drops. Save `outputs/metrics/phase4/quality_vs_diagnostic_degradation.csv`.

## 74. Step 21 — Real PTB-XL metadata audit

Use PTB-XL noise/drift/electrode metadata only as weak real-data audit evidence, never
as primary training ground truth. Document parsing, usable/ambiguous/excluded records,
quality distributions, effect sizes, and uncertainty. If lead-level mapping is unreliable,
perform an ECG-level audit and state the limitation. Save
`outputs/metrics/phase4/ptbxl_real_quality_metadata_audit.csv`.

## 75. Step 22 — Development and final seeds

Develop and freeze the architecture/protocol with seed `42`; then train the frozen model
with seeds `42`, `123`, and `2026`. Report major-metric mean and standard deviation.
Do not conduct an extensive three-seed hyperparameter search.

## 76. Step 23 — Provisional targets

Targets are macro F1 at least `0.85`, clean recall at least `0.90`, argmax unusable
recall at least `0.90`, binary unusable recall at least `0.95`, per-lead macro F1
preferably at least `0.75`, aggregate severity-monotonic score, finite signals and
predictions, significant improvement over the rule baseline, and exact prediction
reproducibility after checkpoint reload. Report misses honestly; targets do not permit
metric-driven corruption changes.

## 77. Step 24 — Required files

Save under `outputs/metrics/phase4/`:

```text
quality_validation_summary.csv
quality_validation_per_class.csv
quality_validation_per_lead.csv
quality_validation_per_corruption.csv
quality_validation_per_severity.csv
quality_validation_unseen_parameters.csv
quality_validation_combined.csv
quality_calibration_metrics.csv
quality_monotonicity_analysis.csv
quality_vs_diagnostic_degradation.csv
ptbxl_real_quality_metadata_audit.csv
quality_model_seed_summary.csv
quality_baseline_comparison.csv
unusable_detection_threshold.csv
```

Under `outputs/predictions/phase4/`, save probabilities, labels, ECG IDs, and lead IDs
for balanced, severity, unseen-parameter, and clean-only validation at minimum.

## 78. Step 25 — Required figures

Generate under `outputs/figures/phase4/`: raw and normalized confusion matrices;
per-lead and per-corruption macro-F1 plots; per-severity score distributions; score
versus severity; pre/post-calibration reliability diagrams; unusable precision-recall;
quality versus Phase 3 AUPRC and sensitivity drops; neural-versus-rule comparison;
representative clean/degraded/unusable examples with probabilities; and metadata-flagged
versus unflagged quality distributions.

## 79. Step 26 — Tests

Test label mapping and indices; train/validation/fold-10 boundaries; absence of fold-10
manifests; deterministic epoch-dependent corruption; class and corruption balancing;
single-lead and 12-lead wrapper shapes; `batch × 12 × 3` output; probability sums;
score bounds/formula; shared weights; checkpoint reload and identical logits; fold-9-only
calibration and threshold selection; Phase 4 49 Hz configuration; unchanged diagnostic
checkpoints; no diagnostic retraining; reproducible manifest hashes; registry hashes;
and continued passage of all Phase 1–3 tests. Run the complete suite.

## 80. Step 27 — Regressions and safety

Before acceptance verify unchanged diagnostic checkpoint hashes, no diagnostic optimizer
or training execution, fold-10 inaccessibility and artifact absence, exact quality-logit
reproduction after checkpoint reload, exact repeated-validation probabilities, identical
regenerated manifest hashes, and labels derived solely from frozen recipes.

## 81. Step 28 — Interpretation

Answer final macro F1 and class recalls; binary unusable-target attainment; hardest lead,
corruption, and severity transition; monotonicity; neural-versus-baseline improvement;
unseen/combined generalization; diagnostic-degradation correlation; metadata evidence;
optional lead-identity effect if run; seed stability; selector readiness; and limitations
that must be addressed before Phase 5.

## 82. Step 29 — Acceptance criteria

Phase 4 is complete only when folds 1–8 train and fold 9 validates with fold 10 locked;
mapping is frozen; the shared model is trained; rule and classical baselines, calibration,
unusable detection, monotonicity, unseen parameters, combined corruptions, Phase 3
correlation, and real-metadata audit are complete; the frozen model has three final seeds;
all outputs/figures are saved; all tests pass; diagnostic checkpoints are unchanged;
results are committed; and a completion report is produced.

## 83. Step 30 — Stop condition and report

Stop after training and evaluating the quality estimator. Do not implement disease-specific
importance, dynamic quality × importance weighting, soft/hard selection, diagnostic
retraining with quality inputs, disease uncertainty, external validation, or fold 10.
Phase 5 requires explicit approval.

The completion report must include: final commit; changed files; exact mapping; training
generation; manifest sizes/hashes; architecture/parameter count; per-seed time; full
test result; overall/class/lead/corruption/severity metrics; calibration; unusable threshold;
unseen and combined results; monotonicity; neural/rule/classical comparison; Phase 3
correlation; metadata audit; three-seed mean/SD; limitations; recommendation about
disease-specific scoring; confirmation of locked fold 10, unchanged diagnostic checkpoints,
no diagnostic retraining, and no Phase 5 work.

### Phase 4 execution status

**Accepted and complete.** The authoritative Phase 4 commit is
`f801743cbcf74a0bef337ad05cc70e2432bd1a44`. The shared 225,763-parameter estimator was trained with
the frozen protocol at seeds 42, 123, and 2026 using only folds 1–8 and evaluated only
on fixed fold-9 manifests. The primary seed-42 model reached macro F1 `0.819473`, with
clean/degraded/unusable recalls `0.943167`/`0.613583`/`0.911250`. Binary unusable
detection met the 0.95 recall target at threshold `0.255058`. Aggregate quality score
decreased from `0.840311` clean to `0.051969` unusable and correlated with Phase 3
diagnostic degradation. The neural model beat the rule baseline but not the classical
gradient-boosting baseline (`0.857202` macro F1). It is accepted as a useful but imperfect
component, particularly for unusable-lead detection, but must not be the sole future
quality source. All 76 tests pass, diagnostic checkpoint hashes are unchanged, and fold
10 remained locked.

Freeze four quality sources for later Phase 6 ablations, without combining them during
Phase 5: (1) neural probabilities, continuous score, and learned embedding; (2) classical
gradient-boosting probabilities, score, feature pipeline, and trained artifact; (3) the
rule detector as a weak baseline; and (4) oracle synthetic quality labels as an
analysis-only upper bound. Later comparisons must include no quality, neural quality,
classical quality, oracle quality, and optionally fused neural/classical quality. See
`docs/phase4_completion_report.md`.

---

# PHASE 5 — DISEASE-SPECIFIC SOFT ECG LEAD SCORING

## 84. Purpose and scope

Train a clean diagnostic model that returns five independent disease probabilities and
`batch × 5 diseases × 12 leads` soft lead-weight distributions for NORM, MI, STTC, CD,
and HYP. Determine whether learned lead preferences differ meaningfully by disease.
Phase 5 must not consume neural, classical, rule, or oracle quality; primary training
must not use synthetic corruption; and no hard selection is permitted.

## 85. Step 1 — Plan gate

Before implementation, record accepted Phase 4 commit
`f801743cbcf74a0bef337ad05cc70e2432bd1a44`, metrics, the decision that neural quality
will not be the sole future gate, the four preserved quality sources, and this complete
Phase 5 protocol in `plan.md`; show the diff; then implement.

## 86. Step 2 — Data protocol

Use PTB-XL v1.0.3 at 100 Hz, existing five-superclass labels, official patient-separated
folds 1–8 for training, fold 9 for validation/checkpoint selection/thresholds, existing
training-only normalization, and clean original ECGs for primary training. Keep fold 10
locked. Only after freezing the clean model may selected Phase 3 manifests support a
small corruption analysis without retraining.

## 87. Step 3 — Shared encoder

Implement `SharedDiagnosticLeadEncoder`. Reshape `B×12×1000` to `(B×12)×1×1000`, apply
one practical MPS-compatible Conv1d/residual/normalization/activation/downsampling/global
pool encoder, and return `B×12×D`. Never create 12 encoders.

## 88. Step 4 — Lead identity

Add a learned canonical-lead identity embedding and combine it with signal embeddings
before scoring. Compare shared encoding with and without identity. Retain identity in
the primary model only when validation performance and ranking stability support it.

## 89. Step 5 — Disease-specific scorer

Return `B×5×12` raw scores and apply softmax independently across 12 leads for each
disease. Every disease distribution must sum to one; never substitute one universal
distribution in the proposed model.

## 90. Step 6 — Disease representations and heads

For each disease compute the weighted sum of 12 lead embeddings, yielding `B×5×D`, and
apply separate lightweight disease heads to return `B×5` logits. Train with
`BCEWithLogitsLoss`; never softmax the five multi-label diseases.

## 91. Step 7 — Required comparisons

Compare: the saved authoritative Phase 1 12-lead ResNet; equal-weight shared encoding;
one global learned soft distribution; disease-specific scoring without identity; and
disease-specific scoring with identity. Also report existing Phase 2 references
`aVR,V1,V3,V5`, `II,aVR,V1,V4`, and `V1,V5`, clearly labeled as prior-phase results.

## 92. Step 8 — Soft weighting only

Do not implement top-k, Gumbel-Softmax, straight-through estimators, hard masks, quality
multiplication, or quality thresholds. Phase 5 uses continuous soft weights only.

## 93. Step 9 — Sparsity analysis

Compare no entropy regularization with a small fold-9-selected entropy coefficient.
Entropy is `-sum(w*log(w))`; effective leads are `exp(entropy)`. Never use L1 on normalized
weights or force one-hot selection. Report performance, entropy, effective leads, and
ranking stability.

## 94. Step 10 — Training protocol

Create `configs/disease_specific_lead_scorer.yaml`,
`scripts/train_disease_specific_model.py`, and
`scripts/evaluate_disease_specific_model.py`. Use AdamW, BCEWithLogitsLoss, early
stopping, LR scheduling, gradient clipping, atomic checkpoints, exact resume, CSV and
TensorBoard logs, and MPS/CUDA/CPU. Select by fold-9 macro AUPRC, tie-breaking with macro
AUROC. Develop at seed 42; after architecture/regularization freeze, train the primary
model with seeds 42, 123, and 2026.

## 95. Step 11 — Validation thresholds

Select separate fold-9 disease thresholds for every newly trained model using the
existing validation-only protocol. Never reuse another model's thresholds. Save them
under `outputs/metrics/phase5/thresholds/`; never access fold 10.

## 96. Step 12 — Predictive metrics and targets

Report overall macro AUROC/AUPRC/F1/precision/sensitivity/specificity and micro F1, plus
per-disease AUROC/AUPRC/F1/precision/sensitivity/specificity/FNR/support. Compare against
authoritative `0.802822` macro AUPRC, `0.923710` macro AUROC, and `0.743789` macro F1.
Targets are macro AUPRC at least `0.762681`, macro F1 at least `0.723789`, and per-class
sensitivity drop no greater than `0.05`; report misses honestly.

## 97. Step 13 — Lead-importance artifacts

For each fold-9 ECG save IDs, true labels, disease probabilities, `5×12` weights,
entropy, and effective leads under `outputs/predictions/phase5/`. Save mean/median
weights, rankings, selection frequency, entropy, and effective-lead summaries under
`outputs/metrics/phase5/`.

## 98. Step 14 — Intervention validation

For every disease compare neutralizing its highest-weight lead, top two, a deterministic
random lead, lowest-weight lead, shuffling weights, and equal weights. Measure disease
AUPRC/AUROC/sensitivity and overall macro AUPRC. Only intervention evidence may support
importance interpretation; top removal should generally hurt more than low/random.

## 99. Step 15 — Phase 2 comparison

Compare learned disease rankings with Phase 2 single-lead, leave-one-out, pair, and
four-lead evidence using Spearman correlation, top-k overlap, frequency overlap, and
disease-specific disagreements. Do not assume masked-screening ordering is reliable.

## 100. Step 16 — Ranking stability

Report mean rank, rank SD, top-3/top-4 frequency, and pairwise seed Spearman correlations
across seeds, validation patients, positive/negative labels, single/multi-label ECGs, and
common disease combinations.

## 101. Step 17 — Multi-label analysis

Compare disease lead weights when a disease occurs alone, with one additional disease,
and with multiple additional diseases; include useful named combinations such as
MI+STTC, sample counts, and small-group cautions.

## 102. Step 18 — Limited Phase 3 analysis

After freezing the clean model, optionally analyze clean, one selected lead missing, one
selected lead severely flat, and mixed corruption without retraining. Determine whether
weights move away from corrupt leads and whether soft attention provides natural
robustness. This is analysis only, never quality-aware integration.

## 103. Step 19 — Figures

Generate 14 figures under `outputs/figures/phase5/`: mean/median disease×lead heatmaps;
seed ranking stability; effective leads; entropy; global-versus-disease performance;
identity and entropy ablations; top-lead and random-versus-top interventions; Phase 2
comparison; selected multi-label examples; selected corruption scenarios; and
per-disease predictive performance.

## 104. Step 20 — Tests

Test shared shapes/weights, canonical lead identity, `B×5×12` scorer shapes and independent
unit-sum distributions, `B×5×D` representations, `B×5` logits, multi-label BCE, entropy,
effective leads, equal/global/disease modes, checkpoint identity and metadata, fold-9
thresholds, fold-10 refusal/artifact absence, unchanged Phase 1–4 checkpoints, registry
architecture/seed fields, intervention scope, and all existing tests. Run the full suite.

## 105. Step 21 — Safety regressions

Verify no fold-10 access/artifact, unchanged diagnostic and Phase 4 checkpoint hashes,
exact checkpoint-reload logits, deterministic validation, unit-sum disease weights,
finite values, and canonical saved lead indices/names.

## 106. Step 22 — Interpretation

Answer whether disease scoring beats equal/global models; identity helps; mild entropy
helps concentration without unacceptable loss; top leads and rankings are stable and
different; interventions validate rankings; Phase 2 agrees; effective lead counts and
multi-label context differ; soft weights naturally avoid corruption; baseline performance
is preserved; Phase 6 quality integration is justified; and which model should freeze.

## 107. Step 23 — Acceptance

Phase 5 requires the shared encoder; equal/global/disease models; identity and entropy
ablations; three primary seeds; saved rankings; intervention, Phase 2, multi-label, and
documented limited corruption analyses; 14 figures; passing full suite; locked fold 10;
unchanged prior checkpoints; committed results; and a completion report.

## 108. Step 24 — Stop condition and report

Stop after disease-specific soft scoring. Do not integrate any quality source, hard
selection, Gumbel-Softmax, disease uncertainty, external validation, or fold 10. Phase 6
requires explicit approval.

The completion report must include final commit; changed files; architectures/parameters;
per-seed times; complete tests; equal/global/disease results; identity and entropy
ablations; seed mean/SD; disease rankings/stability; interventions; Phase 2 comparison;
multi-label and limited corruption results; performance preservation; recommended Phase 6
model; limitations; and confirmations of locked fold 10, unchanged checkpoints, no
quality integration, and no Phase 6 work.

### Phase 5 execution status

**Accepted and complete.** The authoritative local Phase 5 commit is
`474de34b923d74775d1ef2301862c497ab52a7bd`; do not push this commit to GitHub unless
explicitly instructed. The 82,090-parameter primary
`disease_identity_mild_entropy` model was trained on clean folds 1–8 and selected only
on fold 9. At seed 42 it reached macro AUROC/AUPRC/F1
`0.921732`/`0.810602`/`0.743206`; across seeds 42, 123, and 2026, macro AUPRC was
`0.808556 ± 0.003769`. Equal, global, no-identity, identity/no-entropy, and
identity/mild-entropy comparisons are complete. Identity improved macro AUPRC by
`0.030654`; mild entropy improved it by `0.000664` while reducing mean effective leads
from `6.962` to `4.574`.

Top-lead neutralization reduced disease AUPRC by `0.049910` on average, compared with
`0.002984` for deterministic random leads and `0.000032` for lowest-weight leads. This
supports the learned importance ordering. Phase 2 ranking agreement was weak (mean
Spearman `0.3047`, mean top-3 overlap `1.0/3`). The limited corruption analysis was a
negative result: attention increased toward corrupted selected leads by `0.031669` on
average while disease AUPRC fell by `0.051049`; the clean scorer is not naturally
quality-aware.

Attention alone is not a quality estimator. The macro AUPRC and macro F1 preservation
targets passed, but the per-class sensitivity target missed because HYP sensitivity
changed by `-0.059701`. All 98 tests pass; exact
checkpoint reload, repeated validation, resume-state presence, finite/unit-sum weights,
canonical metadata, and prior checkpoint hashes pass. Exactly 14 Phase 5 figures were
generated. Fold 10 remained locked, no quality source or hard selection was integrated,
and Phase 6 has not started. Accepted Phase 5 conclusions are that disease-specific
scoring outperformed equal weighting and global lead weighting, lead identity improved
performance, and mild entropy regularization improved the final model. The recommended
Phase 6 diagnostic checkpoint is the
seed-42 primary model, with equal/global controls retained and all four preserved quality
sources compared only after separate Phase 6 approval. See
`docs/phase5_completion_report.md`.

---

# PHASE L — LITERATURE AND NOVELTY VERIFICATION

## L.1 Timing

Perform this phase twice:

1. **Phase L-A:** after Phase 5 and before implementing the dynamic quality-aware
   method in Phase 6.
2. **Phase L-B:** after final experiments and before writing or submitting the paper
   in Phase 10.

## L.2 Required review scope

Review recent work on:

- disease-specific ECG lead selection;
- patient-specific ECG lead selection;
- reduced-lead classification;
- arbitrary- or missing-lead models;
- lead-quality assessment;
- dynamic routing;
- differentiable top-k selection;
- robust classification under physiological noise;
- uncertainty-aware ECG classification;
- selective prediction and abstention;
- uncertainty-based referral;
- external validation of reduced-lead models.

## L.3 Required outputs

Create:

```text
docs/literature_gap_matrix.csv
docs/novelty_assessment.md
```

The gap matrix must include:

```text
paper
year
dataset
diseases
fixed_or_dynamic_selection
disease_specific_selection
patient_specific_selection
quality_awareness
missing_lead_testing
noise_testing
uncertainty
external_validation
code_availability
main_limitation
```

Do not make “first study” or similar novelty claims until Phase L has been completed
and the evidence supports the claim.

---

# PHASE 6 — QUALITY-AWARE DISEASE-SPECIFIC SOFT ECG LEAD ROUTING

## 51. Purpose

Combine the disease-specific importance learned in Phase 5 with the lead-quality
information learned in Phase 4. The system should reduce the contribution of leads that
are noisy, degraded, flat, missing, clipped, partially dropped out, or otherwise
predicted to be unusable.

The intended Phase 6 pipeline is:

```text
12-lead ECG
      ↓
Shared diagnostic lead encoder
      ↓
Disease-specific importance scores
      +
Per-lead quality scores
      ↓
Quality-aware disease-specific soft weights
      ↓
Disease-specific representations
      ↓
Five disease predictions
```

Phase 6 must remain a soft-routing phase. Do not implement hard top-k selection,
Gumbel-Softmax, straight-through hard masks, disease-prediction uncertainty,
accept/refer decisions, external validation, or fold-10 evaluation.

## 52. Primary research questions

Phase 6 must answer whether quality-aware routing improves robustness compared with the
clean Phase 5 scorer; whether it outperforms corruption training without quality
information; which quality source works best among neural, classical, fused, rule, and
oracle; whether routing moves weight away from corrupted leads; whether robustness can
improve without materially damaging clean performance; whether classical quality beats
neural quality downstream; whether neural probabilities or embeddings add value; how
large the predicted-quality-to-oracle gap is; which diseases benefit most; whether HYP
sensitivity improves; whether MI performance is preserved; and whether the result is
ready for hard lead selection in Phase 7.

## 53. Step 1 — Plan gate

Before implementation, mark Phase 5 complete; record authoritative Phase 5 commit
`474de34b923d74775d1ef2301862c497ab52a7bd`; record accepted Phase 5 results; record the
negative corruption result that corrupted-lead attention increased by `0.031669`; record
that attention alone is not a quality estimator; add this complete Phase 6 protocol; show
the exact `plan.md` diff; then begin implementation. Do not push to GitHub unless
explicitly instructed.

## 54. Step 2 — Data and fold protocol

Use PTB-XL v1.0.3, 100 Hz signals, existing five diagnostic superclass labels, existing
patient-separated folds, existing training-only per-lead normalization, frozen Phase 3
corruption implementations, and the 49 Hz Phase 4 power-line configuration. Use folds
1–8 for training and fold 9 for validation, checkpoint selection, threshold selection,
architecture comparison, and all Phase 6 decisions. Keep fold 10 locked.

Add explicit assertions that Phase 6 training uses only folds 1–8; Phase 6 validation
uses only fold 9; no fold-10 manifest, prediction, or metric is generated; and fold-10
evaluation remains refused.

## 55. Step 3 — Freeze authoritative input components

Before Phase 6 training, preserve and hash the Phase 5 disease-specific scorer
checkpoints for seeds 42, 123, and 2026; Phase 4 neural quality-estimator checkpoints;
the Phase 4 classical gradient-boosting quality model; the Phase 4 rule-based quality
baseline; and Phase 1–3 diagnostic checkpoints. Save hashes under
`outputs/metrics/phase6/pre_phase6_checkpoint_hashes.json`.

Quality estimators must remain frozen during Phase 6. Do not backpropagate into the
neural quality estimator, the classical quality estimator, or the rule-based estimator.
The diagnostic routing and classification model may be trained or fine-tuned only
according to the controlled comparisons in this protocol.

## 56. Step 4 — Quality sources

Implement these quality sources:

1. No quality: use quality value `1.0` for every lead, reproducing disease-specific soft
   scoring without quality gating.
2. Neural quality: use the frozen Phase 4 neural model outputs `P(clean)`,
   `P(degraded)`, `P(unusable)`, continuous quality score, and optionally frozen neural
   lead embedding. Primary scalar quality is `1.0*P(clean) + 0.5*P(degraded)`.
3. Classical quality: use the frozen gradient-boosting model calibrated or accepted class
   probabilities and the same scalar score, `1.0*P(clean) + 0.5*P(degraded)`.
4. Rule quality: use the frozen rule detector as a weak baseline.
5. Fused quality: start with `q_fused = lambda*q_classical + (1-lambda)*q_neural`; evaluate
   `lambda` values `0.25`, `0.50`, and `0.75` on fold 9 only.
6. Oracle quality: use true synthetic corruption labels only for controlled Phase 3-style
   synthetic validation and training experiments. Map clean to `1.0`, degraded to `0.5`,
   and unusable to `0.0`. Oracle quality is an upper-bound research reference, not a real
   deployment claim.

Do not create a complex fusion network unless the simple combination is clearly
inadequate.

## 57. Step 5 — Quality-aware routing formula

Let `s(i,d,l)` be the raw disease-specific importance logit for sample `i`, disease `d`,
and lead `l`, and let `q(i,l)` be the quality score. Define:

```text
g(i,d,l) = s(i,d,l) + alpha * log(q(i,l) + epsilon)
quality_aware_weights(i,d,:) = softmax(g(i,d,:))
epsilon = 1e-6
```

This is equivalent to multiplying the original importance contribution by quality raised
to `alpha` and then renormalizing. Evaluate provisional `alpha` values `0.0`, `0.5`,
`1.0`, and `2.0`; select `alpha` using fold 9 only and save it in configuration. Do not
directly hard-mask leads in the primary Phase 6 model.

## 58. Step 6 — Required mathematical tests

Add tests proving that `alpha = 0` reproduces original Phase 5 weights exactly; all-one
quality reproduces original Phase 5 weights exactly; reducing one lead's quality lowers
its normalized routing weight when all else is equal; near-zero quality strongly
suppresses the lead; routing weights sum to one for every disease; zero quality produces
no NaN or Inf; and all-zero or nearly all-zero quality uses a documented safe fallback.
The recommended fallback is to replace all-zero quality with a uniform minimum quality
before normalization and mark the sample as insufficient-quality for later Phase 7/8 work.
Do not implement referral decisions yet.

## 59. Step 7 — Staged experiment design

Stage A is post-hoc quality gating: use the frozen Phase 5 diagnostic model without
changing its weights, apply the quality-routing formula during inference, and evaluate no
quality, neural quality, classical quality, fused quality, rule quality, and oracle
quality. This isolates quality gating, tests whether quality helps without diagnostic
retraining, identifies the strongest quality source, and measures whether post-hoc
routing damages clean performance. Do not modify Phase 5 checkpoints.

Stage B is quality-aware robust training: train or fine-tune controlled diagnostic models
using a mixture of clean and corrupted training ECGs. Compare no-quality robust training,
neural-quality robust training, classical-quality robust training, fused-quality robust
training, and oracle-quality robust training. All variants must use the same diagnostic
architecture, same initial Phase 5 seed-specific checkpoint, same corruption manifests or
deterministic corruption sequence, same clean/corrupted sample ratio, same optimizer,
same training budget, same early-stopping rule, same folds, and same diagnostic labels.
Only the quality source and gating behavior may differ.

## 60. Step 8 — Training data mixture

Use on-the-fly deterministic corruption on folds 1–8 with an initial mixture of 40%
clean ECGs, 40% single-corruption ECGs, and 20% combined-corruption ECGs. For corrupted
samples, balance approximately across mild, moderate, severe, and unusable corruption.
Include baseline wander, muscle artefact, electrode-motion artefact, Gaussian noise,
49 Hz power-line interference, flat-line, missing lead, partial dropout, clipping,
amplitude failure, and combined corruption. Do not use disease labels to choose corrupted
leads during primary training; select corrupted leads deterministically but independently
of the diagnostic label. Run a small seed-42 pilot, then freeze the mixture after
confirming finite signals, stable training, and reasonable clean retention.

## 61. Step 9 — Fair initialization

For seed 42, initialize Phase 6 models from the authoritative seed-42 Phase 5 checkpoint.
For seeds 123 and 2026, initialize from the corresponding Phase 5 checkpoints. For all
quality-source comparisons within one seed, use identical initial diagnostic weights,
training corruption order, minibatch order, optimizer settings, and initialization
checkpoint hash.

## 62. Step 10 — Quality feature options

For neural quality, compare scalar neural quality only, neural class probabilities through
a small routing adapter, and frozen neural quality embedding through a small trainable
adapter. Do not modify the frozen quality encoder. Keep this ablation limited. The primary
comparison must remain interpretable through the scalar quality-routing formula. For the
classical model, use scalar quality score and optionally its three probabilities. Do not
attempt end-to-end differentiation through the classical model.

## 63. Step 11 — No-quality robust baseline

The no-quality corruption-trained model is mandatory. It uses the same clean/corrupted
training mixture, `alpha = 0`, and no quality input. This model determines how much
improvement comes from ordinary robust training alone. Quality-aware models must be
compared primarily against this model, not only against the clean Phase 5 checkpoint.

## 64. Step 12 — Validation conditions

Evaluate on fold 9 under clean original ECGs; one randomly corrupted lead; two randomly
corrupted leads; four randomly corrupted leads; one high-importance lead corrupted; one
high-importance lead missing; two high-importance leads missing; flat-line selected lead;
partial dropout selected lead; amplitude-failure selected lead; mixed corruption; all
important leads degraded; all leads severely degraded; and oracle disease-specific stress
scenarios clearly labeled as non-deployable analysis. Use fixed deterministic manifests
and do not select validation corruption based on desired outcome.

## 65. Step 13 — Frozen diagnostic thresholds

For direct clean-versus-corrupt robustness comparison, evaluate each Phase 6 model using
its own fold-9 thresholds selected from clean validation data only. Do not reselect
thresholds separately for corruption type, corruption severity, or missing-lead scenario.
Also report threshold-free metrics. Save thresholds under
`outputs/metrics/phase6/thresholds/`. Do not access fold 10.

## 66. Step 14 — Primary metrics

Report clean macro AUROC, macro AUPRC, macro F1, macro sensitivity, macro specificity,
and per-class metrics. Report robustness AUC across severity, macro AUPRC under
corruption, macro F1 under corruption, macro sensitivity under corruption, worst-case
macro AUPRC, maximum performance drop, average non-clean performance drop, per-disease
robustness, and per-corruption robustness. Compare clean performance to Phase 5 reference
macro AUROC `0.921732`, macro AUPRC `0.810602`, and macro F1 `0.743206`.

## 67. Step 15 — Routing-behavior metrics

For every corruption scenario, report weight assigned to corrupted leads before and after
quality gating; mean corrupted-lead weight change; mean clean-lead weight change;
fraction of corrupted leads whose weight decreased; fraction of unusable leads remaining
in the top three weights; effective number of leads before and after quality gating;
weight transferred from corrupted leads to clean leads; disease-specific redistribution;
and quality-weight Spearman correlation. The Phase 5 reference corrupted-lead attention
change was `+0.031669`; the desired Phase 6 direction is corrupted-lead weight change
below zero. Report the exact value.

## 68. Step 16 — Quality-source comparison

Create a direct comparison table containing no quality, neural scalar quality, neural
probability adapter, neural embedding adapter if run, classical quality, fused quality,
rule quality, and oracle quality. Required columns are `quality_source`, `training_mode`,
`alpha`, `fusion_lambda`, `clean_macro_AUPRC`, `corrupted_macro_AUPRC`, `robustness_AUC`,
`clean_macro_F1`, `corrupted_macro_F1`, `clean_macro_sensitivity`,
`corrupted_macro_sensitivity`, `corrupted_lead_weight_change`,
`fraction_corrupted_weights_reduced`, `mean_effective_leads`, `training_time`,
`inference_time`, and `parameter_count`.

## 69. Step 17 — Oracle gap

Calculate `oracle_gap = oracle_quality_robustness - best_predicted_quality_robustness`
overall and by disease. A small gap suggests the quality estimator is sufficient; a large
gap suggests quality estimation remains the bottleneck. Do not present oracle performance
as deployable performance.

## 70. Step 18 — Clean-performance safeguards

Use provisional targets of clean macro AUPRC at least `0.800602`, clean macro F1 at least
`0.723206`, and per-class sensitivity drop no larger than `0.05` from Phase 5. Track HYP
sensitivity carefully because it already missed the previous provisional target. These
targets must not be used to hide negative outcomes.

## 71. Step 19 — Robustness success targets

The best predicted-quality model should improve robustness AUC over the no-quality
corruption-trained model, preferably by at least `0.01` absolute; mean corrupted-lead
routing weight should decrease; at least 80% of unusable corrupted leads should receive
lower weight after quality gating; missing-lead scenarios should improve relative to
no-quality robust training; the model should outperform the clean Phase 5 scorer under
severe corruption; and clean performance should remain within safeguards. These are
provisional research targets.

## 72. Step 20 — Disease-specific analysis

Report separately for NORM, MI, STTC, CD, and HYP. Answer which disease benefits most,
which has the largest robustness improvement, whether MI remains the most
corruption-sensitive class, whether HYP sensitivity recovers, whether one quality source
works better for one disease than another, whether disease-specific rankings are
preserved after quality gating, and whether the model replaces a corrupted important lead
with a clinically different but informative lead. Do not make clinical claims solely from
learned weights.

## 73. Step 21 — Classical versus neural decision

The classical model had higher three-class quality macro F1 than the neural model, but
the neural model may provide useful learned representations. The final Phase 6 decision
must be based on downstream diagnostic robustness, not standalone quality-classification
macro F1. Possible outcomes are freezing classical quality, neural quality, or simple
fusion for Phase 7; reporting quality estimation as the main bottleneck if oracle remains
substantially better; or reconsidering the dynamic-quality design if no quality source
improves over robust training.

## 74. Step 22 — Three-seed policy

Use seed 42 for alpha selection, fusion-lambda selection, adapter selection, the
training-mixture pilot, and architecture development. After freezing the Phase 6
protocol, run final selected variants with seeds 42, 123, and 2026. At minimum, repeat
the no-quality robust model, best neural-quality model, classical-quality model, best
fused-quality model, and oracle-quality upper bound. If one source is clearly inferior
during seed-42 development, it may be excluded from three-seed repetition only with a
documented reason.

## 75. Step 23 — Output files

Save required metrics under `outputs/metrics/phase6/`: `phase6_clean_performance.csv`,
`phase6_corruption_performance.csv`, `phase6_quality_source_comparison.csv`,
`phase6_routing_behavior.csv`, `phase6_robustness_auc.csv`,
`phase6_per_disease_results.csv`, `phase6_per_corruption_results.csv`,
`phase6_missing_lead_results.csv`, `phase6_oracle_gap.csv`,
`phase6_alpha_ablation.csv`, `phase6_fusion_ablation.csv`,
`phase6_training_mode_comparison.csv`, `phase6_three_seed_summary.csv`,
`phase6_checkpoint_hash_audit.json`, and `phase6_run_summary.json`.

Save predictions under `outputs/predictions/phase6/`, including at minimum clean
predictions, severe corruption predictions, missing-important-lead predictions,
mixed-corruption predictions, disease-specific weights before quality, disease-specific
weights after quality, quality scores, and sample identifiers.

## 76. Step 24 — Required figures

Generate at least 16 figures under `outputs/figures/phase6/`: clean performance by
quality source; robustness AUC by quality source; macro AUPRC versus severity; macro
sensitivity versus severity; corrupted-lead weight before and after quality gating;
fraction of corrupted leads downweighted; disease-by-quality-source robustness heatmap;
no-quality versus neural quality; no-quality versus classical quality; neural versus
classical quality; fused versus individual quality sources; predicted quality versus
oracle quality; oracle gap by disease; missing-lead performance; effective number of
leads under corruption; and representative ECG examples showing weight redistribution.

## 77. Step 25 — Required tests

Add tests for all quality sources returning `batch × 12` quality scores; quality values
within `[0, 1]`; frozen neural and classical quality models; no gradients entering frozen
quality models; `alpha = 0` reproducing Phase 5 routing; all-one quality reproducing
Phase 5 routing; lower quality reducing lead weight; routing weights summing to one;
zero-quality numerical safety; all-zero-quality fallback; fused-quality formula; oracle
quality mapping; canonical lead ordering; corruption applied before normalization;
identical corruption sequence across compared variants; identical minibatch order across
compared variants; checkpoint save/reload; reload producing identical logits; saved
checkpoints recording quality source, alpha, and fusion lambda; fold-10 refusal; no
fold-10 artifact; Phase 1–5 checkpoint hashes unchanged; diagnostic thresholds using
fold 9 only; clean alpha-zero predictions matching Phase 5; no hard top-k behavior;
experiment registry recording quality source and corruption hash; and all existing tests
continuing to pass.

## 78. Step 26 — Regression and safety audit

Before accepting Phase 6, verify all pre-Phase-6 checkpoint hashes, frozen quality models
were not modified, Phase 5 checkpoints were not overwritten, fold 10 was never accessed,
no fold-10 output exists, alpha-zero clean predictions reproduce Phase 5, repeated
evaluation is deterministic, corruption inputs are identical across quality-source
comparisons, no hard lead selection was implemented, and every saved lead weight is
finite and sums to one.

## 79. Step 27 — Phase 6 interpretation

At completion, answer whether post-hoc quality gating helps; whether quality-aware robust
training helps beyond ordinary corruption training; which quality source gives the best
downstream diagnostic robustness; whether classical quality remains superior downstream;
whether neural embedding adds value; whether fusion beats individual sources; how large
the oracle gap is; whether corrupted-lead weight change becomes negative; what percentage
of corrupted leads are downweighted; whether clean diagnostic performance is preserved;
which disease benefits most; whether HYP sensitivity improves; whether MI robustness
improves; whether missing important leads are handled better; whether quality estimation
is still the main bottleneck; whether the system is ready for hard top-k selection; and
which exact Phase 6 checkpoint and quality source should be frozen for Phase 7.

## 80. Step 28 — Acceptance criteria

Phase 6 is complete only when all quality sources are implemented; post-hoc gating is
evaluated; no-quality, neural-quality, classical-quality, fused-quality, and oracle
robust training are evaluated; alpha and fusion ablations are complete; clean retention,
corruption robustness, routing redistribution, oracle gap, and disease-specific effects
are measured; final selected variants are repeated across three seeds; all required
artifacts and figures are saved; the complete test suite passes; all previous
checkpoints remain unchanged; fold 10 remains locked; no hard top-k selection is
implemented; results are committed locally; and a completion report is produced.

## 81. Step 29 — Stop condition and required report

Stop after quality-aware soft lead routing. Do not implement top-4, top-3, top-2 hard
selection; Gumbel-Softmax; straight-through estimators; sparse hard masks;
disease-prediction uncertainty; accept/refer decisions; external validation; or fold-10
evaluation. Do not push to GitHub unless explicitly instructed. Phase 7 must not start
without explicit approval.

The Phase 6 report must include final local Git commit hash; whether the commit was
pushed; files created and changed; checkpoint-hash audit; quality sources implemented;
selected alpha; selected fusion lambda; training-data mixture; model architectures and
parameter counts; training times; complete test-suite results; post-hoc gating results;
no-quality, neural-quality, classical-quality, fused-quality, and oracle-quality results;
clean-performance retention; robustness-AUC comparison; routing redistribution results;
corrupted-lead downweighting percentage; per-disease results; missing-lead results;
oracle gap; three-seed mean and standard deviation; recommended quality source and
checkpoint for Phase 7; unresolved limitations; confirmation that fold 10 remained
locked; confirmation that prior checkpoints remained unchanged; confirmation that no hard
selection was implemented; and confirmation that Phase 7 has not started.

### Phase 6 execution status

**Complete as a controlled negative research result.** The authoritative Phase 5 commit
is `474de34b923d74775d1ef2301862c497ab52a7bd`; it has not been pushed. All declared
quality sources, the exact soft-routing formula, Stage A post-hoc gating, Stage B robust
training, alpha/fusion/adapter ablations, 14 validation scenarios, routing analysis,
three-seed final comparisons, required tables, predictions, audits, and 16 figures are
complete. The selected non-zero alpha is `0.5`, the selected fusion lambda is `0.75`,
and the unconstrained post-hoc optimum is alpha `0.0` (no gating).

The complete suite passes with **129 tests**. Phase 6-owned Python files pass Ruff and
`git diff --check` passes. Pre-Phase-6 hashes remain unchanged, alpha-zero clean
predictions exactly reproduce Phase 5, evaluation is deterministic, quality encoders are
frozen, compared sources use identical corruptions, saved weights are finite/unit-sum,
and no fold-10 artifact exists.

The main three-seed result is negative: the best predicted route, neural probabilities,
has robustness AUC `0.808453 ± 0.002676`, versus `0.808499 ± 0.002637` for identical
no-quality corruption training. Classical, fused, and oracle routing reduce robustness
further. Neural probability changes corrupted-lead weight by `-0.001174`; classical and
fused produce stronger changes of `-0.088809` and `-0.089649`, but the redistribution
does not improve aggregate diagnosis. The oracle gap is negative (`-0.012512`), so the
current routing assumption, rather than quality-estimator accuracy alone, is the likely
bottleneck.

Clean safeguards are preserved, and classical/fused quality modestly improves the two
missing-important-lead subsets, but neither effect changes the primary conclusion. No
quality source is recommended for Phase 7. The seed-42 no-quality robust checkpoint is
retained only as the safest Phase 6 reference, not as evidence for hard selection. Fold
10 remains locked, no hard selection is implemented, Phase 7 has not started, and no push
is authorized. See `docs/phase6_completion_report.md` for the complete results and audit.

---

# PHASE 7 — HARD SPARSE ECG LEAD SELECTION

## Phase 7 accepted baseline and scope

Phase 6 is accepted at authoritative local commit
`b376eaedfb554a12e333db7b7cd1c33ce70ce1de`; do not push it unless explicitly
instructed. Its accepted three-seed no-quality robustness AUC is
`0.808499 ± 0.002637`, best predicted-quality robustness AUC is
`0.808453 ± 0.002676`, predicted-quality corrupted-lead weight change is `-0.001174`,
and oracle gap is `-0.012512`. Quality-aware routing did not outperform ordinary
corruption training, although clean safeguards passed. The routing integration appears
to be the bottleneck, so predicted quality is not the primary Phase 7 selector input.

Phase 7 tests whether accurate disease predictions can be retained with hard budgets of
four, three, and two selected leads. Top one is an analysis baseline. Phase 7 is limited
to hard lead-budget evaluation: do not implement uncertainty, referral, external
validation, or final fold-10 evaluation.

## Step 1 — Plan and repository gate

Before implementation, record the accepted Phase 6 commit and findings, add this complete
Phase 7 protocol, show the exact `plan.md` diff, and verify that the worktree is clean at
the authoritative commit. Preserve and hash all accepted Phase 1–6 checkpoints before
training. Do not push to GitHub unless explicitly instructed.

## Step 2 — Data, folds, and locked decisions

Use PTB-XL v1.0.3 at 100 Hz with the existing five diagnostic superclasses,
patient-separated folds, canonical lead order, training-only normalization, and frozen
Phase 3 corruption implementations. Use folds 1–8 for training and fold 9 for every
Phase 7 development decision, checkpoint selection, threshold selection, and comparison.
Keep fold 10 locked and refuse any Phase 7 command that requests it.

Use seed 42 for budget, architecture, fallback, and optional quality-ablation development.
After the protocol is frozen, repeat the selected final models with seeds 42, 123, and
2026. Initialize each learned selector from the corresponding strongest Phase 6
no-quality robust checkpoint and use the same 40% clean / 40% single-corruption / 20%
combined-corruption sequence, optimizer, minibatch order, budget, and early-stopping rule
within each seed.

## Step 3 — Primary comparisons

Compare:

1. the authoritative all-12-lead diagnostic baseline;
2. the accepted Phase 5 soft disease-specific model;
3. the accepted Phase 6 no-quality robust soft model;
4. learned disease-specific hard top-4 selection;
5. learned disease-specific hard top-3 selection;
6. learned disease-specific hard top-2 selection;
7. hard top-1 selection as an analysis baseline;
8. the Phase 2 fixed four-lead model using `aVR, V1, V3, V5`;
9. the Paper 8 fixed subset `II, aVR, V1, V4`;
10. oracle-quality hard selection under controlled synthetic corruption;
11. predicted-quality hard selection only as a clearly labeled secondary ablation.

Use existing accepted fixed-subset checkpoints when the exact subset exists. If a required
fixed comparator does not exist, train it with the established Phase 2 from-scratch
protocol and fold policy; do not substitute a masked all-lead model without labeling it.

## Step 4 — Deterministic disease-specific top-k

For each sample and disease, rank the 12 learned disease-specific importance scores with
a stable canonical-lead tie break, select the highest `k`, mask every other lead
representation, and renormalize the selected weights to sum to one. Only selected lead
representations may contribute to that disease output. Return logits, the boolean
`B×5×12` selection mask, renormalized selected weights, selected indices, and the
insufficient-lead flag.

The primary selector uses disease importance only. For oracle or predicted-quality
ablations, rank the Phase 6 quality-adjusted logits and label the result separately.
Predicted quality must never replace the no-quality primary selector because Phase 6 did
not support that decision.

## Step 5 — Inference-only and selection-aware training

First apply deterministic top-k to frozen Phase 5 and Phase 6 models at inference, for
budgets `k ∈ {4, 3, 2, 1}`. This isolates pruning loss without retraining.

Then fine-tune controlled top-k-aware models initialized from seed-matched Phase 6
no-quality robust checkpoints. Deterministic `torch.topk` membership is allowed to be
non-differentiable; gradients must flow through the selected representations, selected
scores/weights, encoder, and classifier. Use straight-through estimation or
Gumbel-Softmax only if deterministic top-k fine-tuning is demonstrably impossible, and
report either method as a separate experiment rather than silently changing the primary
method.

Use fold-9 clean thresholds selected once per trained model. Freeze those thresholds for
all corruption and missing-lead evaluations; never select scenario-specific thresholds.

## Step 6 — Missing/unusable-lead safety fallback

For controlled corruptions, mark selected leads that are explicitly missing or unusable.
If all initially selected leads are unavailable, set `insufficient_lead=true`. Then use a
deterministic highest-importance available-lead fallback and renormalize its weights. If
no lead is available, retain the deterministic top-k mask with uniform `1/k` selected
weights so the output remains finite, preserve `insufficient_lead=true`, and record that
the prediction contains no usable signal. Do not implement a referral action in Phase 7.

The normal path must select exactly `k` leads. The fallback path must be deterministic,
finite, explicit in saved predictions, and included in the selected-lead failure rate.

## Step 7 — Validation conditions and fairness

Evaluate clean fold 9 and the same frozen 14 Phase 6 validation scenarios, including
random one/two/four-lead corruptions, high-importance corruption and missingness,
flat-line, partial dropout, amplitude failure, mixed corruption, all-important degraded,
all-leads severe, and oracle stress. Reuse identical manifests across budgets and methods.

Do not use disease labels to choose training corruptions. Do not retune corruptions to
favor hard selection. Report synthetic oracle-quality comparisons as non-deployable
upper-bound analyses.

## Step 8 — Metrics and research questions

Report macro AUROC, macro AUPRC, macro F1, macro sensitivity, macro specificity,
per-disease metrics, robustness AUC, worst-case corruption performance, average selected
lead count, lead-selection frequency, patient- and disease-specific selection variation,
seed stability, selected-lead failure rate, inference time, parameter count, and clean
performance retention. Track HYP sensitivity explicitly.

Answer:

1. how many leads are needed to preserve performance;
2. whether top four preserves at least 95% of full-model macro AUPRC;
3. the loss at top three and top two;
4. whether learned top four beats both fixed four-lead subsets;
5. whether selected leads differ by disease and patient;
6. whether selections remain stable across seeds;
7. whether corruption causes selection of damaged leads;
8. whether oracle quality improves hard selection;
9. whether predicted quality helps or hurts as a secondary ablation;
10. which disease loses most at small budgets;
11. whether the result is ready for uncertainty and referral testing.

## Step 9 — Required artifacts

Save configuration, pre-Phase-7 hashes, thresholds, checkpoints, predictions, and an
append-only registry record. Save metrics under `outputs/metrics/phase7/`, predictions
under `outputs/predictions/phase7/`, figures under `outputs/figures/phase7/`, and
checkpoints under `outputs/checkpoints/phase7/`.

At minimum, save model/budget comparisons, clean and corruption performance, robustness
AUC, per-disease results, fixed-versus-learned comparisons, lead-selection frequencies,
patient-specific selections, seed stability, selected-lead failures, oracle/predicted
quality ablations, checkpoint hashes, and a machine-readable run summary. Prediction
artifacts must include sample/patient identifiers, targets, probabilities, selected lead
indices/masks/weights, corruption status, and insufficient-lead flags.

Generate figures for performance versus lead budget, robustness versus lead budget,
learned versus fixed top four, per-disease budget effects, HYP sensitivity, lead-selection
frequencies, disease-by-lead heatmaps, patient variation, cross-seed stability, corrupted
selected-lead failure, oracle/predicted-quality ablations, and representative selections.

## Step 10 — Required tests

Add tests proving that exactly `k` leads are selected on the normal path; only selected
leads contribute; selected weights renormalize to one; top-k is deterministic with a
canonical tie break; disease-specific selections may differ; outputs contain no NaN or
Inf; all-missing fallback is finite, deterministic, and flagged; folds 1–8/fold 9 are
enforced; fold 10 remains refused; accepted checkpoints remain unchanged; all-12 clean
behavior remains reproducible; checkpoint reload preserves logits and masks; thresholds
come only from clean fold 9; registry rows record budget and corruption hash; and the
complete existing test suite still passes.

## Step 11 — Acceptance and stop condition

Phase 7 is complete only after inference-only and selection-aware comparisons, all four
budgets, both fixed four-lead comparators, the oracle hard-selection analysis, the
predicted-quality secondary decision, disease/patient/seed stability, failure/fallback
analysis, required metrics/artifacts/figures, the full test suite, checkpoint-hash audit,
and three-seed final models are complete and committed locally.

Stop after hard lead-budget evaluation. Do not implement diagnostic uncertainty,
accept/refer decisions, external validation, or final fold-10 evaluation. Do not push to
GitHub unless explicitly instructed. Phase 8 must not start without explicit approval.

### Phase 7 execution status

**Complete and accepted.** The authoritative local Phase 7 commit is
`45b37e01c609e41e29c492240760d269d911ec32`. The
three-seed fine-tuned top-4 model reaches clean macro AUPRC `0.809583 ± 0.002750`,
preserving `99.73%` of the matching Phase 6 soft mean and beating both required fixed
four-lead subsets. Top-3 and top-2 clean AUPRC are `0.804556 ± 0.002212` and
`0.795888 ± 0.004540`. Top four is the selected sparse budget and the seed-2026 top-4
checkpoint is the strongest individual reference. Prior checkpoint hashes are unchanged,
fold 10 remains locked and unevaluated, nothing was pushed, and Phase 8 has not started.
See `docs/phase7_completion_report.md` for the complete results and limitations.

---

# PHASE 8 — UNCERTAINTY-AWARE ECG PREDICTION AND SELECTIVE REFERRAL

## Phase 8 accepted baseline and scope

Phase 7 is accepted at authoritative local commit
`45b37e01c609e41e29c492240760d269d911ec32`. Its selected budget is disease-specific
hard top four, with three-seed clean macro AUPRC `0.809583 ± 0.002750`, retaining
`99.73%` of the Phase 6 soft mean. Hard top three and top two reach
`0.804556 ± 0.002212` and `0.795888 ± 0.004540`. Learned top four beats both fixed
four-lead comparators. Its main limitation is substantial degradation when all available
leads are severely corrupted. The accepted suite has 156 passing tests, earlier
checkpoint hashes are unchanged, fold 10 is locked, nothing was pushed, and Phase 8 had
not started at acceptance.

Phase 8 estimates when unchanged diagnostic predictions are unreliable. It may retain a
prediction or flag it for expert review; it must not change the diagnostic task, create a
real clinical workflow, retrain a diagnostic model, access fold 10, perform external
validation, begin final evaluation, or push to GitHub.

## Step 1 — Plan and repository gate

Before implementation, mark Phase 7 accepted, record its commit and results, add this
complete protocol, show the exact `plan.md` diff, and verify a clean worktree at the
accepted commit. Hash all Phase 7 and earlier checkpoints before Phase 8. Never overwrite
them.

## Step 2 — Primary models and data protocol

Use final Phase 7 hard top-4 checkpoints from seeds 42, 123, and 2026 both individually
and as a mean-probability three-model ensemble. The ensemble is the primary uncertainty
candidate. Do not run a diagnostic optimizer or alter diagnostic weights.

Folds 1–8 remain historical diagnostic-training folds. Split fold 9 deterministically at
patient level into disjoint uncertainty-calibration and uncertainty-evaluation subsets.
Use calibration only for probability calibration, uncertainty-method selection,
referral-threshold selection, optional error-predictor fitting, and operating-point
selection. Use evaluation only for final reporting. Keep earlier diagnostic thresholds
frozen and document that they were selected in the preceding fold-9 development
protocol. Refuse fold 10 and create no fold-10 artifact.

## Step 3 — Required uncertainty methods

Evaluate at least:

1. maximum predicted-probability confidence;
2. per-disease binary predictive entropy;
3. distance from each frozen disease threshold;
4. probability variance across the three seeds;
5. ensemble predictive entropy;
6. ensemble mutual information or an equivalent disagreement measure;
7. thresholded prediction disagreement across seeds;
8. the Phase 7 insufficient-lead flag;
9. the number of missing or unusable selected leads;
10. optional frozen neural/classical quality summaries used only for referral.

Phase 4 quality models must remain frozen. Predicted quality may not modify diagnostic
routing.

## Step 4 — Probability calibration

On the calibration subset only, compare raw probabilities, one global temperature, and
per-disease temperature scaling. Report negative log-likelihood, Brier score, expected
calibration error, and per-disease reliability. Retain calibrated probabilities only if
calibration improves without damaging discrimination. Never fit calibration on the
evaluation subset or fold 10.

## Step 5 — Disease-level uncertainty

For every sample × five-disease prediction, save probability, binary prediction,
predictive entropy, ensemble variance, ensemble disagreement, distance from the frozen
threshold, correctness, false-negative status, and false-positive status. Evaluate
uncertainty detection of any error, false negatives, and false positives. Report
false-negative detection separately as the primary safety concern.

## Step 6 — ECG-level uncertainty

Compare maximum disease uncertainty, mean disease uncertainty, top-two mean uncertainty,
maximum ensemble variance, any-disease disagreement, and the insufficient-lead rule.
Select the primary ECG referral score on calibration data only, prioritizing any
false-negative prediction, any incorrect prediction, and insufficient selected leads.

## Step 7 — Optional error-prediction model

A small interpretable logistic-regression or gradient-boosting error predictor may be fit
using calibration data only. Allowed inputs are ensemble uncertainty, probability
margins, disagreement, usable-selected-lead count, and optional frozen quality summaries.
Raw waveforms and diagnostic retraining are forbidden. Compare it with simple scores on
held-out evaluation and do not retain it if it does not improve.

## Step 8 — Referral policies and safety override

Evaluate disease-level abstention, where only an uncertain disease prediction is
withheld, and ECG-level referral, where the full ECG is flagged. Always flag when all
selected leads are missing or unusable, Phase 7 reports insufficient leads, or any
prediction/uncertainty is non-finite. Use `retained prediction`, `flagged for review`, and
`selective prediction`; do not imply an autonomous clinical action.

## Step 9 — Coverage levels and selective performance

Report 100%, 95%, 90%, 80%, and 70% automatic coverage. At each level report referral
rate, macro AUPRC, macro F1, macro sensitivity, macro specificity, false-negative rate,
retained error rate, referred error rate, and sample counts.

## Step 10 — Selective-prediction metrics

Calculate risk-coverage curves, AURC, excess AURC where appropriate, error-detection
AUROC/AUPRC, false-negative-detection AUROC/AUPRC, referral enrichment, retained error
rate, retained false-negative rate, and coverage at the chosen safety target. Referral
enrichment is referred error rate divided by retained error rate.

## Step 11 — Clean and corrupted conditions

Evaluate identical deterministic inputs for every method under:

1. clean ECGs;
2. one selected lead corrupted;
3. one selected lead missing;
4. two selected leads missing;
5. selected-lead flat line;
6. selected-lead partial dropout;
7. selected-lead amplitude failure;
8. mixed corruption;
9. all selected leads degraded;
10. all leads severely corrupted.

Reuse Phase 7/3 manifests and implementations where possible. Never generate different
corruption inputs by uncertainty method.

## Step 12 — Required comparisons and selection rule

Compare individual hard top-4 seeds 42, 123, and 2026; their mean-probability ensemble;
the best simple uncertainty score; any optional learned error predictor; the
insufficient-lead rule alone; and ensemble uncertainty plus the insufficient-lead safety
rule. Choose the primary candidate using calibration only. Base final conclusions solely
on the held-out evaluation subset.

## Step 13 — Clean-performance preservation

Uncertainty/referral must not alter underlying disease probabilities except for accepted
calibration. At 100% coverage, verify that the uncalibrated ensemble is exactly the mean
of the three seed probabilities. Report the Phase 7 hard top-4 reference
`0.809583 ± 0.002750`. Selective results are reliability results, not classifier
improvements.

## Step 14 — Provisional research targets

Treat these as targets, never as permission to hide a negative result:

1. referred ECGs have clearly higher error than retained ECGs;
2. retained false-negative rate falls meaningfully at 80% versus 100% coverage;
3. severe-corruption referral exceeds clean referral;
4. every insufficient-lead case is flagged;
5. the ensemble improves uncertainty estimation over individual models;
6. calibration improves or preserves Brier score and ECE;
7. disease-level referral imbalance is reported.

## Step 15 — Disease-specific analysis

Report NORM, MI, STTC, CD, and HYP separately with counts at every coverage. Identify the
highest-uncertainty disease, weakest error detection, greatest referral benefit, HYP
false-negative change, MI safety under corruption, and disproportionate referral.

## Step 16 — Required outputs

Save under `outputs/metrics/phase8/`:

- `phase8_calibration_summary.csv`;
- `phase8_uncertainty_method_comparison.csv`;
- `phase8_disease_level_uncertainty.csv`;
- `phase8_ecg_level_uncertainty.csv`;
- `phase8_error_detection_metrics.csv`;
- `phase8_false_negative_detection_metrics.csv`;
- `phase8_risk_coverage.csv`;
- `phase8_selective_performance.csv`;
- `phase8_per_disease_selective_results.csv`;
- `phase8_corruption_referral_results.csv`;
- `phase8_referral_enrichment.csv`;
- `phase8_operating_point.csv`;
- `phase8_three_seed_ensemble_summary.csv`;
- `phase8_checkpoint_hash_audit.json`;
- `phase8_run_summary.json`.

Save predictions under `outputs/predictions/phase8/` with raw/calibrated/ensemble
probabilities, uncertainty, referral/retention decisions, identifiers, corruption,
targets, and insufficient-lead flags.

## Step 17 — Required figures

Save at least 14 figures under `outputs/figures/phase8/`: risk-coverage; macro F1,
sensitivity, and false-negative rate versus coverage; error-detection ROC and PR;
false-negative-detection PR; calibration before/after; uncertainty for correct versus
incorrect predictions; uncertainty by clean/corruption; referral by corruption severity;
referral by disease; individuals versus ensemble; and retained versus referred error.

## Step 18 — Required tests

Test deterministic patient splitting and no patient overlap; calibration-only fitting;
calibration-only referral/method/operating-point selection; evaluation isolation; exact
ensemble mean; entropy, variance, disagreement, mutual information, and threshold-distance
calculations; increasing uncertainty with disagreement; risk/coverage calculations;
safety overrides for insufficient/non-finite cases; deterministic referral; checkpoint
reload equality; unchanged Phase 7 and quality checkpoints; fold-10 refusal/absence; no
diagnostic training; and the complete existing suite.

## Step 19 — Safety audit

Before acceptance, verify Phase 7 and all earlier hashes are unchanged; no diagnostic
optimizer ran; fold 10 was neither accessed nor mentioned by an artifact; calibration
and thresholds use only calibration patients; final metrics use only evaluation patients;
every insufficient case is flagged; all outputs are finite; and referral never silently
replaces disease predictions.

## Step 20 — Required interpretation

At completion answer: the best uncertainty method; ensemble benefit; error and
false-negative detection accuracy; sensitivity-versus-coverage behavior; recommended
operating point and referral proportion; error enrichment; severe-corruption referral;
insufficient-case coverage; disease benefit; HYP false-negative effect; calibration
utility; optional error-predictor result; and readiness for external validation.

## Step 21 — Acceptance and stop condition

Phase 8 is complete only after the patient split, calibration, method selection,
individual/ensemble comparison, disease/ECG referral policies, all ten conditions,
coverage and safety analysis, required artifacts/figures/tests, completion report,
checkpoint audit, and local commit are complete. Stop afterward. Do not access fold 10,
perform external validation, retrain diagnostics, modify Phase 7 checkpoints, begin final
evaluation, or push.

### Phase 8 execution status

**Complete and accepted.** The authoritative local Phase 8 commit is
`c8ef21a7aa0db5164dd7e1d2f2bb063fa6a41277`. Fold 9
was split into 958 calibration patients (1,081 ECGs) and 959 held-out evaluation patients
(1,065 ECGs), with no patient overlap. Per-disease temperature scaling was retained.
Threshold distance was selected for disease-level abstention, while mean disease entropy
plus the insufficient-lead safety rule was selected for ECG-level referral.

At the calibration-selected 80% target, held-out clean coverage is `0.825352` and the
referral rate is `0.174648`. Referred ECG error is `0.801075` versus `0.301479` retained
error (`2.657×` enrichment); retained false-negative rate falls from `0.193362` at full
coverage to `0.156827`. Error-detection AUROC/AUPRC are `0.818952/0.728517`, and
false-negative-detection AUROC/AUPRC are `0.701663/0.377776`. The ensemble improves
error-detection AUPRC over all three comparable individual scores. All 1,065 severe
insufficient-lead ECGs are flagged. Every frozen checkpoint hash is unchanged, all
outputs are finite where defined, no diagnostic optimizer ran, fold 10 stayed locked,
external validation did not start, and nothing was pushed. See
`docs/phase8_completion_report.md` for results and limitations.

---

# PHASE 9 — EXTERNAL DATASET VALIDATION

## Phase 9 accepted baseline and scope

Phase 8 is accepted at authoritative local commit
`c8ef21a7aa0db5164dd7e1d2f2bb063fa6a41277`. Its 21-step protocol and 179-test suite
passed. The disjoint fold-9 calibration/evaluation split contains 958/959 patients.
Per-disease temperature scaling, threshold-distance disease uncertainty, and mean disease
entropy plus the insufficient-lead override are frozen. The 80% target yields 82.54%
actual clean evaluation coverage, 17.46% referral, 2.657× error enrichment, and retained
false-negative rate 15.68% versus 19.34% at full coverage. Error-detection AUROC/AUPRC
are 0.818952/0.728517, false-negative-detection AUROC/AUPRC are
0.701663/0.377776, and all 1,065 all-leads-severe ECGs are flagged. All 57 checkpoint
hashes are unchanged; no diagnostic retraining or external validation occurred; fold 10
remained locked.

Phase 9 evaluates zero-shot generalization to a different dataset, institution,
population, and recording environment. It must not access fold 10, retrain or fine-tune
diagnostic models, tune architecture from external results, optimize external diagnostic
thresholds, fit external calibration, alter Phase 8 referral thresholds, begin final test
evaluation, or push without explicit authorization.

## Step 1 — Plan and repository gate

Before implementation, mark Phase 8 accepted, record its authoritative commit and
results, add this complete protocol, and show the exact `plan.md` diff. Verify a clean
worktree at the accepted commit before beginning the external-dataset audit.

## Step 2 — External dataset selection audit

Audit Chapman-Shaoxing, CPSC2018, PhysioNet/CinC 2020 source datasets, and the Georgia
ECG dataset where accessible. For every candidate record public accessibility, download
method, license and redistribution restrictions, ECG/patient counts, 12-lead
availability, duration, sampling rate, amplitude units, lead order, patient identifiers,
diagnostic labels and overlap with NORM/MI/STTC/CD/HYP, missing-lead and quality
information, population/institution differences, and zero-shot suitability.

Save `docs/external_dataset_audit.md`, `docs/external_dataset_selection.md`, and
`outputs/metrics/phase9/external_dataset_audit.csv`. Select scientifically rather than
by expected performance. Stop before evaluation if no candidate supports a defensible
mapping for enough classes.

## Step 3 — Label-mapping protocol

Create `configs/external_label_mapping.yaml` and
`docs/external_label_mapping_rationale.md`. For each external diagnosis record the
original label/description, mapped PTB-XL superclass, retained/excluded status, mapping
confidence, and rationale. Reject ambiguity, preserve multi-label records, and exclude
unmappable diagnoses only from the corresponding class analysis. Report original,
retained, excluded, patient, multi-label, unmapped-label, and per-class positive counts.

## Step 4 — External data integrity audit

Before preprocessing verify file completeness, waveform readability and finite values,
patient identifiers, duplicate records/patients, lead names/order, durations, sampling
rates, amplitude units, missing/flat/invalid leads, and possible PTB-XL overlap using
identifiers, metadata, and signal hashes where practical. Save
`outputs/metrics/phase9/external_data_integrity_summary.json`.

## Step 5 — Frozen preprocessing

Convert external ECGs to canonical order I, II, III, aVR, aVL, aVF, V1–V6; 100 Hz;
10 seconds; the PTB-XL physical amplitude unit; and existing PTB-XL training
normalization. Fit no external normalization. Use one deterministic label-independent
segment policy for long signals and one documented padding/exclusion policy for short
signals. Document every transformation.

## Step 6 — Separate external cache

Do not modify the PTB-XL cache. Save
`data/processed/external/waveform_cache.dat`,
`data/processed/external/waveform_metadata.csv`,
`data/processed/external/labels.csv`,
`data/processed/external/patient_ids.csv`, and
`data/processed/external/preprocessing_metadata.json`. Record hashes for raw metadata,
processed labels, waveform cache, and preprocessing configuration.

## Step 7 — Frozen models

Evaluate the Phase 1 full 12-lead baseline; Phase 2 aVR/V1/V3/V5 and Paper 8
II/aVR/V1/V4 fixed models; Phase 6 strongest no-quality robust soft route; Phase 7 hard
top-four seeds 42, 123, and 2026; their mean-probability ensemble; and the frozen Phase 8
uncertainty/referral system. Retraining and checkpoint overwrites are forbidden. Record
all checkpoint hashes before and after evaluation.

## Step 8 — Frozen thresholds and calibration

Use each model's existing PTB-XL fold-9 diagnostic thresholds. For Phase 8 reuse its
per-disease temperatures, uncertainty method, referral threshold/operating point, and
insufficient-lead override. Fit nothing externally and report the actual external
coverage rather than forcing 80%.

## Step 9 — Primary external metrics

Report overall macro AUROC/AUPRC/F1/precision/sensitivity/specificity, micro F1, and
subset accuracy. Per class report AUROC, AUPRC, F1, precision, sensitivity, specificity,
false-negative rate, and support. Use 2,000-repetition patient-level 95% bootstrap
confidence intervals with seed 31415. Mark classes with too few positives or negatives
as not reliably estimable.

## Step 10 — Performance-transfer analysis

For every model compare external results with PTB-XL fold-9 results: absolute and
relative metric differences, macro AUPRC preservation, macro F1 difference, per-class
sensitivity differences, and confidence intervals. Interpret dataset and label shift,
not only model failure.

## Step 11 — Dataset-shift analysis

Compare PTB-XL and external class prevalence, available age/sex distributions, recording
duration, preprocessed sampling rate, amplitude distribution, missing/flat-lead
frequency, basic signal statistics, and quality metadata. Never alter predictions using
protected/demographic attributes. Save
`outputs/metrics/phase9/external_dataset_shift_summary.csv`.

## Step 12 — Lead-selection analysis

For Phase 7 hard top four report selected leads by disease and frequency, usable selected
lead count, insufficient frequency, PTB-XL frequency differences, cross-seed ranking
stability, and fixed-subset overlap. Determine cross-dataset similarity without claiming
anatomical importance solely from learned selection.

## Step 13 — External uncertainty and referral

Apply the frozen Phase 8 system. Report actual coverage/referral, retained/referred error
and false-negative rates, enrichment, error- and false-negative-detection AUROC/AUPRC,
referral by disease/available quality indicators, and insufficient-lead safety. Compare
with Phase 8 internal evaluation.

## Step 14 — Natural quality and missing leads

Where available, separately evaluate naturally noisy, missing/flat-lead, short, clipped,
or acquisition-problem ECGs. Treat uncertain metadata as imperfect and report counts and
limitations.

## Step 15 — Optional synthetic robustness

Only after freezing clean external evaluation, optionally apply the same limited
one-selected-missing, flat-line, partial-dropout, amplitude-failure, and mixed
corruptions across models. This is secondary and cannot replace natural-data evaluation.

## Step 16 — Subgroup analysis

Where sample sizes permit, report patient-level results with counts and confidence
intervals by age group, sex, single/multi-label status, common disease combinations, and
available site/source. Draw no conclusion from very small groups.

## Step 17 — Required outputs

Save under `outputs/metrics/phase9/`:

- `external_dataset_audit.csv`;
- `external_data_summary.csv`;
- `external_label_mapping.csv`;
- `external_model_comparison.csv`;
- `external_per_class_metrics.csv`;
- `external_performance_transfer.csv`;
- `external_dataset_shift_summary.csv`;
- `external_lead_selection_summary.csv`;
- `external_uncertainty_results.csv`;
- `external_selective_performance.csv`;
- `external_referral_enrichment.csv`;
- `external_missing_lead_results.csv`;
- `external_subgroup_results.csv`;
- `external_bootstrap_confidence_intervals.csv`;
- `external_checkpoint_hash_audit.json`;
- `phase9_run_summary.json`.

Save predictions under `outputs/predictions/phase9/` with ECG/patient identifiers,
mapped labels, probabilities, binary predictions, selected leads, uncertainty/referral
decisions, and insufficient flags.

## Step 18 — Required figures

Save at least 14 figures under `outputs/figures/phase9/`: PTB-XL versus external
prevalence; internal/external macro AUPRC, per-class AUPRC, and sensitivity; external
model comparison; external lead-selection heatmap; internal/external selection
frequency; external risk-coverage; retained/referred errors; disease referral;
calibration diagram; signal-statistics shift; suitable subgroup performance; and
representative external ECGs.

## Step 19 — Required tests

Test label mapping, ambiguous-label rejection, multi-label preservation, canonical lead
ordering, resampling, deterministic segmentation, duration and amplitude handling,
PTB-XL normalization reuse, cache reproducibility, absence of external threshold tuning
or calibration fitting, checkpoint reload/hash preservation, ensemble calculation,
frozen referral threshold, patient-level bootstrap, insufficient override, fold-10
refusal/artifact absence, absence of diagnostic training, and the complete existing
suite.

## Step 20 — Safety and leakage audit

Before acceptance verify fold 10 was never accessed; no model was retrained; external
labels did not drive model selection; no external threshold/calibration was fitted; all
checkpoint hashes are unchanged; preprocessing is deterministic; confidence intervals
are patient-level; label mapping is documented; and no clinical-deployment claim is
made.

## Step 21 — Required interpretation

At completion identify the best externally generalizing model; overall and per-disease
transfer losses; best/worst disease; learned top-four versus fixed models; selection
stability; external error detection and referral enrichment; actual frozen-policy
coverage; insufficient-lead safety; important shifts; subgroup consistency; readiness
for final locked fold-10 evaluation; and paper limitations.

## Step 22 — Acceptance and stop condition

Stop after external validation and the completion report. Do not access fold 10, retrain
on external data, optimize external thresholds/calibration, modify final architecture,
begin final test evaluation, or push. Commit Phase 9 locally. The report must include the
local commit, push status, dataset selection/license/access, raw/retained/patient counts,
mapping and preprocessing, integrity/duplication and checkpoint audits, full tests,
primary/per-class/transfer/shift/selection/uncertainty/subgroup results, optional
corruption results, limitations, final-evaluation readiness, and all prohibitions.

### Phase 9 execution status

**Complete; local commit reported in the handoff.** All 22 steps are complete. The
selected Chapman–Shaoxing primary cohort contains 21,997 unique-signal patients after
conservative mapping and integrity exclusions. The hard top-four ensemble is best by
external macro AUPRC at `0.630693` (95% CI `0.618205–0.645848`), versus `0.834981`
internally. The frozen Phase 8 policy yields `74.60%` actual external coverage, `25.40%`
referral, and `1.720×` error enrichment. All required metric files, predictions, 14
figures, the completion report, and the complete **201-test** suite are complete. All 57 checkpoint hashes remain
unchanged; fold 10 remained locked; no model was retrained; no external threshold or
calibration was fitted; optional synthetic external corruption was not run; final test
evaluation was not begun; and nothing was pushed.

---

# PHASE 10 — FINAL LOCKED PTB-XL FOLD-10 EVALUATION

## Phase 10 accepted baseline and one-time rule

Phase 9 is accepted at authoritative local commit
`95b81e034055d7a0ca46951f9b443e948f29089f`. Its 22-step zero-shot validation on
Chapman–Shaoxing retained 21,997 unique-signal patients. The Phase 7 hard top-four
three-seed ensemble was best externally at macro AUPRC `0.630693` (patient-bootstrap
95% CI `0.618205–0.645848`) versus `0.834981` internally, preserving `75.53%`. Frozen
Phase 8 coverage/referral/error enrichment were `74.60%`/`25.40%`/`1.720×`; STTC
transferred best and MI worst; lead rankings were highly stable; calibration and
threshold-dependent performance degraded. The complete suite passed 201 tests, Phase 9
lint passed, all 57 checkpoint hashes were unchanged, no diagnostic model was retrained,
no external threshold/calibration was fitted, fold 10 remained locked, and nothing was
pushed.

This phase is the explicitly authorized **final one-time locked fold-10 evaluation**, not
another development phase. After results are revealed, do not change architecture,
retrain, alter lead budgets, select another checkpoint, modify diagnostic thresholds,
calibration, uncertainty/referral methods or thresholds, change corruption parameters,
or rerun model selection. All decisions must be frozen before access. Do not push unless
explicitly instructed.

## Step 1 — Plan and repository gate

Record the accepted Phase 9 commit/results, this complete protocol, the one-time rule,
and show the exact `plan.md` diff before any fold-10 access.

## Step 2 — Freeze final models

Freeze: (A) authoritative Phase 1 all-12 checkpoint and fold-9 thresholds; (B) Phase 2
aVR/V1/V3/V5 checkpoint and thresholds; (C) Phase 2 Paper 8 II/aVR/V1/V4 checkpoint and
thresholds; (D) authoritative Phase 5 disease-specific soft-routing model/ensemble from
existing results; (E) strongest authoritative Phase 6 no-quality robust soft-routing
model; (F) Phase 7 hard top-four checkpoints for seeds 42, 123, and 2026; (G) their
arithmetic-mean probability ensemble as the primary diagnostic model; and (H) the Phase
8 selective-referral system using the same ensemble, per-disease temperatures/disease
thresholds, ECG uncertainty method, referral operating point, and insufficient-lead
override. Add or remove no model after access.

## Step 3 — Pre-specify primary model, system, metric, and claims

The primary diagnostic model is the Phase 7 hard top-four three-seed ensemble. The
primary safety system is frozen Phase 8 referral applied to it. The primary metric is
macro AUPRC. Secondary metrics are macro AUROC/F1/sensitivity/specificity, micro F1,
per-disease AUPRC/sensitivity/FNR, coverage, and referral enrichment. Preservation is
`hard_top4_macro_AUPRC / full_12_lead_macro_AUPRC`, with a pre-specified target of at
least `0.95`. Do not change the primary model after results are seen.

## Step 4 — Pre-specify final hypothesis family

- H1: hard top-four ensemble preserves at least 95% of full-12 macro AUPRC.
- H2: learned hard top-four ensemble exceeds fixed aVR/V1/V3/V5 macro AUPRC.
- H3: learned hard top-four ensemble exceeds Paper 8 II/aVR/V1/V4 macro AUPRC.
- H4: referred ECGs have a higher error rate than retained ECGs.
- H5: retained false-negative rate is below the 100%-coverage false-negative rate.
- H6: every test ECG triggering the insufficient-lead rule is flagged.

This is the limited primary family. Apply Holm–Bonferroni where formal p-values are
calculated; do not add post-result exploratory significance tests.

## Step 5 — Checkpoint and configuration audit

Before unlock, create `outputs/metrics/final/final_checkpoint_hash_audit.json`,
`final_configuration_manifest.json`, `final_threshold_manifest.json`, and
`final_model_manifest.csv`. Record checkpoint path/hash, model/seed/leads/parameter count,
disease thresholds, calibration, uncertainty/referral settings, normalization and label
mapping hashes, code commit, and configuration hash. Confirm all 57 hashes are unchanged.

## Step 6 — Dedicated final evaluator

The only test command is `python scripts/final_evaluation.py --config
configs/final_evaluation.yaml --unlock-test`. It must refuse without the flag, warn that
access is permanent, require a clean committed tree, verify checkpoints/thresholds/
calibration/configuration, refuse if an access record exists, evaluate every frozen model
in one controlled session, and create a permanent access record. No ordinary development
command may silently access fold 10.

## Step 7 — Fold-9 preflight without test access

Run the complete evaluator with `--preflight-validation`. It must load every frozen
artifact, produce finite outputs, verify the exact ensemble mean, lead names/top-four
budget, deterministic referral, output writability, and metrics/figures while producing
no fold-10 access record or test artifact. Compare with authoritative fold-9 results and
resolve every unexplained mismatch before unlock.

## Step 8 — Tests and clean preparation commit

Before access, run the full suite and Ruff on Phase 10-owned code; verify refusal without
unlock, no access log/predictions/metrics, a clean tree, and commit all preparation code/
configuration (recommended `feat: prepare final locked fold-10 evaluation`). Run the
final evaluation only from that clean commit. Do not refactor the 67 historical Ruff
violations because unrelated changes could alter frozen behavior.

## Step 9 — Permanent access record

On unlock create, never delete or overwrite,
`outputs/test_access/fold10_access_record.json`, recording timestamp, authorization,
command/protocol, commit/working-tree status, configuration/checkpoint/threshold/
calibration hashes, fold-10 ECG/patient counts, evaluator version, device, Python, and
PyTorch versions. Refuse every future fold-10 run once it exists.

## Step 10 — Failure handling

Complete preflight before test predictions. If failure occurs before any prediction,
save failure state, do not retry silently, and request recovery authorization. If any
test prediction/metric exists, preserve all partial artifacts, document failure, change
nothing, and do not rerun silently.

## Step 11 — Clean final metrics

For every frozen model report overall macro AUROC/AUPRC/F1/precision/sensitivity/
specificity, micro F1, and subset accuracy; per disease report AUROC/AUPRC/F1/precision/
sensitivity/specificity/FNR/support. Use only frozen fold-9 thresholds.

## Step 12 — Three-seed ensemble

Load seeds independently, save each probability set, compute the arithmetic mean, apply
frozen ensemble thresholds, report seed/ensemble results, and verify equality within
floating-point tolerance. Save the four named prediction CSVs under
`outputs/predictions/final/`.

## Step 13 — Performance preservation

For each reduced/selected model report macro-AUPRC preservation versus all 12 leads,
macro-F1 and sensitivity differences, per-class AUPRC/sensitivity differences, and lead
count. State honestly whether the ensemble reaches 95%.

## Step 14 — Frozen selective referral

Apply unchanged Phase 8 temperatures, uncertainty score, referral threshold, and
insufficient override. Report actual coverage/referral, retained/referred error and FNR,
enrichment, retained macro F1/sensitivity/specificity, error/FN detection AUROC/AUPRC,
and per-disease referral. Do not force 80% coverage.

## Step 15 — Lead-selection analysis

Report selected leads by ECG/disease, selection frequency, mean effective pre-hard lead
count, cross-seed top-four stability, insufficient count, overlap with both fixed
subsets, and differences from fold 9 and external frequencies. Selection alone does not
establish anatomical or clinical importance; four leads do not necessarily mean four
electrodes.

## Step 16 — Limited secondary robustness

After clean predictions and within the same authorized session, run only: one selected
lead missing, one selected lead flat, selected-lead partial dropout, selected-lead
amplitude failure, mixed corruption, and all selected leads severely corrupted. Use
frozen Phase 3/7 parameters and seed 31415 on the all-12 baseline, fixed four-lead model,
hard top-four ensemble, and selective system. Keep this separate from primary clean
analysis and tune nothing.

## Step 17 — Statistical analysis

Use 2,000 patient-level bootstrap repetitions, 95% CIs, and seed 31415 for macro
AUPRC/AUROC/F1/sensitivity, preservation, paired model differences, referral enrichment,
and retained-FNR difference. Use paired patient bootstrap and Holm–Bonferroni for the
limited hypothesis family.

## Step 18 — Required outputs

Under `outputs/metrics/final/` save: `final_model_comparison.csv`,
`final_per_class_metrics.csv`, `final_three_seed_summary.csv`,
`final_performance_preservation.csv`, `final_paired_model_comparisons.csv`,
`final_bootstrap_confidence_intervals.csv`, `final_lead_selection_summary.csv`,
`final_selective_performance.csv`, `final_referral_enrichment.csv`,
`final_uncertainty_results.csv`, `final_corruption_results.csv`,
`final_hypothesis_results.csv`, `final_checkpoint_hash_audit.json`,
`final_configuration_manifest.json`, `final_threshold_manifest.json`, and
`final_run_summary.json`. Prediction artifacts must contain IDs, labels, raw
probabilities, frozen binary predictions, relevant selected leads, uncertainty/referral,
and insufficient flags.

## Step 19 — Required figures

Under `outputs/figures/final/` save at least: macro-AUPRC model comparison, per-disease
AUPRC and sensitivity, performance versus lead count, preservation ratios, fixed versus
learned comparison, seed stability, test selection heatmap, fold-9 versus fold-10
selection, internal/external/final performance, risk-coverage, retained/referred errors,
referral by disease, and corruption comparison.

## Step 20 — Final safety audit

Verify pre-unlock refusal; exclusive dedicated-command access; permanent access record;
no test threshold/calibration fitting, checkpoint selection, retraining, or post-access
architecture change; unchanged hashes; finite predictions/metrics; exactly four hard
leads except documented fallback; all insufficient cases referred; and only expected
result artifacts after the run.

## Step 21 — Final interpretation

Answer full-12 and hard-top-four macro AUPRC, preservation and target attainment,
comparisons with fixed subsets, best/worst disease and HYP sensitivity, lead stability,
coverage/referral/error enrichment/retained FNR/insufficient safety, fold-9/external
comparisons, supported and unsupported conclusions, paper limitations, and clinical
readiness. The expected conclusion remains research-only unless evidence clearly shows
otherwise.

## Step 22 — Final report

Create `docs/final_fold10_evaluation_report.md` with protocol/access details, hashes,
metrics/CIs/hypotheses/comparisons, lead/referral/robustness analyses, internal/external
comparison, limitations, reproducibility commands, and confirmation of no post-test
tuning. Mark this phase complete in `plan.md` only after all audits pass.

## Step 23 — Final commits and stop

Commit only expected tracked results, documentation, manifests, and summaries
(recommended `results: complete final locked fold-10 evaluation`); keep intentionally
ignored large checkpoints/predictions out of Git; report preparation/results hashes and
push status; then stop. Do not begin post-test tuning, architecture experiments, external
fine-tuning, threshold/calibration changes, or clinical-deployment work.

The completion handoff must report both commits, push status, exact command/access time
and record, final Git state/tests/hash/configuration audits, fold-10 counts, every model's
metrics, preservation/paired/bootstrap/selection/referral/corruption results, fold-9 and
external comparisons, supported/unsupported conclusions and limitations, and all safety
confirmations.

### Phase 10 execution status

**Complete; final results commit reported in the handoff.** All 23 steps are complete.
Preparation was frozen at commit `9b0d2de5b49199e29a883a77bd09d647b2552366` after the
full fold-9 preflight reproduced 40 authoritative checks within `1e-7`. The single
authorized fold-10 session evaluated 2,158 ECGs from 1,877 patients and permanently
recorded access at `outputs/test_access/fold10_access_record.json`.

Full-12 and hard top-four ensemble macro AUPRC are `0.810182` and `0.813377`; preservation
is `100.39%`, meeting the pre-specified 95% target. The ensemble exceeds both fixed
four-lead models. The unchanged referral system yields `80.58%` coverage, `19.42%`
referral, `2.226×` error enrichment, and retained FNR `0.172869` versus `0.195917` at
full coverage. All five formal hypotheses survive Holm correction. There are no clean
insufficient cases; all 445 insufficient all-selected-severe cases are referred. All 57
checkpoint hashes remain unchanged, no model or setting was changed after access, no
training/fitting occurred, all 218 tests pass, the report and required artifacts are complete, nothing was
pushed, and the conclusion remains research-only.

---

# PHASE 11 — PAPER PREPARATION

## 76. Paper should be written only after useful results

Do not begin with a claim that the method works.

The paper should be written only if experiments show meaningful results.

Examples of meaningful outcomes:

1. dynamic selection preserves most of the 12-lead performance with fewer leads;
2. dynamic selection beats fixed subsets under corruption;
3. dynamic selection lowers false negatives when key leads are missing;
4. uncertainty identifies high-risk cases;
5. external performance improves;
6. selected lead patterns differ by disease;
7. referral improves accepted-case reliability.

---

## 77. Possible paper title

```text
Disease-Specific Quality-Aware Dynamic Lead Selection
for Robust Multi-Label ECG Classification
```

Alternative:

```text
Adaptive ECG Lead Selection Under Signal Degradation:
Disease-Specific Dynamic Routing with Uncertainty-Aware Referral
```

---

## 78. Possible contribution statement

> We propose a disease-specific, quality-aware dynamic ECG lead-selection framework that estimates per-lead quality, learns disease-dependent lead importance, adapts lead usage for each patient, and refers uncertain cases when reliable evidence is insufficient.

Do not claim novelty until the final literature review confirms it.

---

## 79. Suggested paper structure

```text
1. Introduction
2. Related Work
3. Dataset and Label Mapping
4. Baseline Model
5. Lead-Quality Estimation
6. Disease-Specific Dynamic Selection
7. Uncertainty and Referral
8. Experimental Setup
9. Internal Results
10. Robustness Results
11. External Validation
12. Ablation Study
13. Discussion
14. Limitations
15. Ethical and Clinical Considerations
16. Conclusion
```

---

## 80. Main experiment tables for the paper

### Table A: Baseline comparison

```text
Model
Number of leads
Macro AUROC
Macro AUPRC
Macro F1
Macro sensitivity
```

### Table B: Robustness

```text
Model
Noise type
Noise severity
Missing leads
Macro F1
Performance drop
```

### Table C: Disease-specific performance

```text
Model
NORM
MI
STTC
CD
HYP
```

### Table D: Uncertainty

```text
Method
ECE
Brier score
Coverage
Selective risk
Referral rate
```

### Table E: External validation

```text
Model
Internal AUROC
External AUROC
Internal F1
External F1
Performance drop
```

---

## 81. Required figures

1. complete model architecture;
2. data split diagram;
3. example clean and corrupted ECG leads;
4. quality-estimator confusion matrix;
5. disease-specific lead heatmap;
6. selected-lead frequency plot;
7. performance versus number of leads;
8. robustness curves;
9. calibration curves;
10. risk-coverage curve;
11. internal versus external results;
12. example accepted and referred cases.

---

# DEVELOPMENT RULES FOR CODEX

## 82. Strict implementation rules

Codex must follow these rules:

1. Do not implement all phases at once.
2. Complete one phase, run tests, and stop.
3. Do not change the official PTB-XL folds.
4. Never tune using the test set.
5. Never calculate normalization using validation or test data.
6. Keep data, models, training, and evaluation modular.
7. Use type hints.
8. Use docstrings.
9. Add meaningful logging.
10. Add clear exceptions.
11. Do not silently skip bad files.
12. Make long-running scripts resumable.
13. Store settings in YAML.
14. Save predictions, not only final metrics.
15. Save exact thresholds.
16. Save package versions and hardware details.
17. Add unit tests before moving to the next phase.
18. Use patient-level separation.
19. Use multi-label loss and metrics.
20. Do not use softmax for the five labels.
21. Do not report accuracy alone.
22. Do not claim clinical deployment.
23. Do not claim disease-specific medical importance from attention alone.
24. Verify lead importance using removal or intervention experiments.
25. Record every experiment in an experiment registry.
26. Keep the test set untouched until final evaluation.
27. For external validation, document label mapping carefully.
28. Use multiple seeds for final results.
29. Include confidence intervals.
30. Prefer simpler stable methods before complex methods.

---

## 83. Experiment registry

Every experiment must append one row to:

```text
outputs/experiment_registry/experiments.csv
```

Columns:

```text
experiment_id
date
git_commit
phase
config_path
seed
device
selected_leads
corruption_setting
model_name
checkpoint_path
best_validation_metric
test_macro_auroc
test_macro_f1
notes
```

Leave test metric fields blank for development experiments. Populate them only
during the explicitly unlocked final fold-10 evaluation. Test-set access must also
be recorded in an append-only audit log.

---

## 84. Required tests

At minimum:

### Data tests

- required files exist;
- metadata parses;
- waveform shape is correct;
- lead order is correct;
- no NaN/Inf;
- label matrix shape is correct;
- fold assignment is correct;
- patient leakage is absent;
- cache shape is correct.

### Model tests

- baseline input/output shape;
- quality-estimator input/output shape;
- selector score shape;
- top-k output shape;
- missing-lead mask handling;
- checkpoint loading.

### Metric tests

- threshold selection uses validation data;
- sensitivity calculation;
- specificity calculation;
- AUROC calculation;
- calibration calculation;
- selective risk calculation.

### Corruption tests

- corruption changes only intended leads;
- deterministic corruption with fixed seed;
- missing lead is represented correctly;
- severity levels differ measurably.

---

## 85. Local and Colab compatibility

The project should support both:

```text
local execution
Google Colab execution
```

Local execution should be the main modular implementation.

Colab can use the same scripts by cloning the repository and mounting Google Drive.

Do not maintain two separate codebases.

---

## 86. Recommended execution order for Codex

Before beginning the 20-step implementation ledger, update `plan.md` with all
approved protocol decisions and show its exact diff. Wait only if that update
introduces a new contradiction; otherwise proceed directly.

Codex should execute in this exact order:

```text
1. Show proposed repository structure.
2. Initialize Git, configure the uv-managed Python 3.11 environment, create the
   complete directory structure and Phase 0 files, and add .gitignore—in that order.
3. Create requirements and configuration.
4. Implement hardware, environment, seed, logging, configuration, and
   reproducibility utilities.
5. Implement dataset verification.
6. Implement label construction.
7. Implement official split handling.
8. Implement waveform cache.
9. Implement normalization.
10. Implement dataset class.
11. Implement baseline ResNet.
12. Implement trainer.
13. Implement evaluation.
14. Implement tests.
15. Run tests.
16. Fix failures.
17. Run a tiny smoke test.
18. Run the full baseline on folds 1–8 only after the smoke test passes and perform
    complete validation on fold 9.
19. Save validation results, verify fold-10 remains locked, and verify the final
    test pipeline is ready but unexecuted.
20. Stop and report Phase 1.
```

Report completion status for every step using a table with:

```text
step_number
task
status
files_created_or_changed
command_executed
result
unresolved_issue
```

Do not mark a step complete unless it was actually performed.

---

## 87. Smoke-test mode

Before full training, support:

```bash
python scripts/train_baseline.py \
  --config configs/baseline.yaml \
  --smoke-test
```

Smoke test should:

- use a small subset;
- run one or two epochs;
- verify the full pipeline;
- save a temporary checkpoint;
- finish quickly.

Do not start full training until smoke test passes.

---

## 88. Stop conditions

Codex must stop and report instead of continuing when:

- dataset files are incomplete;
- patient leakage exists;
- waveform shapes are inconsistent;
- label construction fails;
- NaN/Inf appears;
- loss becomes NaN;
- checkpoint cannot reload;
- tests fail;
- test data was accidentally used for tuning;
- external label mapping is ambiguous;
- dynamic selection does not outperform simple baselines;
- results are unstable across seeds.

For correctable implementation failures, attempt the planned fixes first. Stop
when a failure remains unresolved after reasonable fixes or when continuing could
invalidate the research protocol.

---

## 89. Success criteria for the full research project

The primary preservation metric is macro AUPRC because the five labels are
imbalanced:

```text
preservation_ratio =
dynamic_model_macro_AUPRC / full_12_lead_baseline_macro_AUPRC
```

The primary preservation target is:

```text
preservation_ratio >= 0.95
```

Also require:

- macro F1 absolute drop no greater than `0.02`;
- per-class sensitivity absolute drop no greater than `0.05`.

For example, if the 12-lead validation macro AUPRC is `0.80`, the reduced or
dynamic model must reach at least `0.76` to meet the preservation target.

Continue using validation macro AUROC for initial checkpoint selection to remain
compatible with the PTB-XL benchmark. Final reduced-lead comparisons should
emphasize, in order:

1. macro AUPRC;
2. macro F1;
3. per-class sensitivity;
4. false-negative rate;
5. average number of selected leads.

These are research targets, not guaranteed outcomes. Report failures honestly.

The project is considered successful if the final method achieves at least one strong result such as:

1. meets the defined preservation target while using fewer average leads;
2. significantly outperforms fixed subsets under missing or noisy leads;
3. reduces false negatives under corruption;
4. improves external generalization;
5. produces useful uncertainty ranking;
6. lowers error among accepted cases through referral;
7. learns stable disease-specific lead patterns;
8. demonstrates that the best leads vary by disease and patient quality.

No publication is guaranteed.

The value of the paper depends on:

- novelty;
- correctness;
- external validation;
- robustness;
- reproducibility;
- statistical evidence;
- clear limitations.

---

# INITIAL CODEX TASK

## 90. Copy-paste starting instruction

```text
Read plan.md completely before writing code.

Implement only Phase 0 and Phase 1.

Do not implement fixed-lead experiments, signal corruption,
lead-quality estimation, disease-specific scoring,
dynamic lead selection, uncertainty, or external validation yet.

Your immediate task is to build a reproducible local PTB-XL
12-lead baseline with:

- PTB-XL v1.0.3
- 100 Hz waveforms
- five diagnostic superclasses
- official folds 1–8 / 9 / 10
- folds 1–8 for training, fold 9 for all development decisions, and fold 10 locked
- patient leakage checks
- memory-mapped waveform cache
- training-only per-lead normalization
- modern PyTorch 1D ResNet
- BCEWithLogitsLoss
- AdamW
- early stopping
- learning-rate scheduling
- checkpointing
- validation-only threshold selection
- macro and per-class metrics
- tests
- logging
- README instructions
- smoke-test mode
- experiment registry
- `evaluation.allow_test_evaluation: false`

First show the repository structure and exact implementation steps.
Then create the files.
Run all tests.
Fix failures.
Run a smoke test.
Run complete baseline training on folds 1–8.
Perform complete validation on fold 9.
Do not calculate, display, or save fold-10 predictions or metrics.
Stop after the Phase 1 pipeline is complete and report:

1. files created;
2. tests run;
3. test results;
4. smoke-test result;
5. full-training result;
6. best validation macro AUROC;
7. validation macro AUPRC;
8. validation macro F1;
9. validation precision, sensitivity, and specificity;
10. per-class validation metrics;
11. validation-selected thresholds;
12. checkpoint-loading verification;
13. fold-10 pipeline-readiness status;
14. any assumptions or unresolved issues.

Do not begin Phase 2 until explicitly instructed.
```

---

## 91. Final principle

The project should prove each part independently:

```text
First prove the baseline works.
Then prove lead choice matters.
Then prove quality can be estimated.
Then prove dynamic selection helps.
Then prove uncertainty identifies risky cases.
Then prove the system generalizes externally.
```

Do not claim success before the experiments support it.
