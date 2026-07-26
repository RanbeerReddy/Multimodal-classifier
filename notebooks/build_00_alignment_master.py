"""Builds notebooks/00_alignment_master.ipynb as raw nbformat-v4 JSON
(no nbformat package dependency required to build it).

One single script generating all eight stages cleanly without intermediate JSON part files.
"""

import json
from pathlib import Path

cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


# ============================================================================
# TITLE & OVERVIEW
# ============================================================================

md(r"""
# ECG&ndash;CXR Alignment: Master Notebook

**One notebook, eight stages, re-entrant via cache.**
`Setup -> Stage 0 (Geometry) -> Stage 1 (Manual Concept Baseline) -> Stage 2 (Linear/CCA) ->
Stage 3 (Prototype Discovery) -> Stage 4 (Prototype Contrastive Alignment) ->
Stage 5 (Gromov-Wasserstein Alignment, FINAL METHOD) -> Stage 6 (Ablations) -> Stage 7 (External Validation)`

This notebook implements the blueprint exactly as specified: it takes the **two already-trained, frozen,
unimodal SSL encoders** (ECG: 1D-CNN, 256-d; CXR: ResNet-50, 2048-d) produced by
`ecg-correct-notebook.ipynb` and `image-encoder.ipynb`, and builds only a small **alignment stage** on top
of them. **No encoder training happens anywhere in this notebook** &mdash; only two new, lightweight
projection heads (`ECGAlignHead: 256 -> d_shared`, `CXRAlignHead: 2048 -> d_shared`) are ever trained.

**Research question** (see the design review / roadmap documents this notebook implements):
> How much cross-modal structure between two independently pretrained medical encoders &mdash; never
> sharing a patient at any stage, training or validation &mdash; can be recovered without any cross-modal
> correspondence, and how does that recovered structure compare to the ceiling established by fully-paired
> methods (CroMoTEX, MoRE) on the same modality pair?

**Every stage reads its inputs from disk and writes its outputs back to disk before the next stage
begins.** If a stage crashes and the kernel restarts, upstream stages load from cache in seconds; nothing
is ever silently recomputed. A single MLflow experiment (`ecg_cxr_alignment`) tracks every stage as its
own run (or parent run with nested children for sweeps).

**Explicitly out of scope for this notebook** (already decided in the design documents and not
re-opened here): fine-tuning either backbone, using Concept-Precision@K as a headline result anywhere
outside the Stage-1 baseline, and claiming any cross-modal number computed purely on PTB-XL/CheXpert
(with no MIMIC external check) as evidence of real alignment.
""")

md(r"""
## 0. How to Use This Notebook

* Run top to bottom on first use. On every subsequent run, cached stages load from disk instantly.
* Set `FORCE_RECOMPUTE = True` in the Setup section to force a specific stage to ignore its cache.
* Stages 0&ndash;6 use **only** PTB-XL + CheXpert (never any MIMIC data). Stage 7 is the only stage that
  touches MIMIC, and it is physically isolated in `external_validation/` &mdash; nothing upstream ever reads
  from that folder.
* Data paths below (`PTBXL_DATA_DIR`, `CHEXPERT_DATA_DIR`, `ENCODER_CHECKPOINT_DIR`) must point at the
  same raw datasets and the `final_models/` checkpoint directory produced by the two source notebooks.
""")


# ============================================================================
# SETUP
# ============================================================================

md(r"""
## Setup

Loads config, loads both frozen encoders (both variants: SimCLR and MultiSupCon), extracts and caches
raw features for every (modality, variant, split) combination, and runs a cache sanity check. This is the
only section that ever runs the ECG/CXR backbones' forward pass &mdash; every stage after this reads
straight from `cache/embeddings/`.
""")

code(r"""
# S.1 -- Environment & imports check
import sys, os, json, logging, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("alignment.notebook")

REQUIRED_PACKAGES = ["torch", "torchvision", "sklearn", "scipy", "mlflow", "ot", "yaml", "pandas", "numpy"]
missing = []
for pkg in REQUIRED_PACKAGES:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)
if missing:
    print("Missing packages -- install with:")
    print(f"  pip install {' '.join(p if p != 'ot' else 'POT' for p in missing)} umap-learn --quiet")
else:
    print("All required packages are importable.")

import torch
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
""")

code(r"""
# S.2 -- Project package on path + load stage_configs.yaml
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "src").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent  # notebook runs from notebooks/, src/ is one level up
sys.path.insert(0, str(PROJECT_ROOT))

from src import config as C
from src import encoders as ENC
from src import data_pipeline as DP
from src import caching as CACHE
from src import losses as L
from src import geometry as GEOM
from src import prototypes as PROTO
from src import linear_cca as LCCA
from src import ot_alignment as OT
from src import gw_alignment as GW
from src import eval_metrics as EVAL
from src import ablations as ABL
from src import external_validation as EXTV
from src import mlflow_utils as MLU
from src import viz as VIZ

C.ensure_project_dirs()

# Single source of truth for hyperparameters -- create on first run, reload on every subsequent run
if not (C.CONFIGS_DIR / "stage_configs.yaml").exists():
    CFG = C.AlignmentConfig()
    C.save_stage_configs_yaml(CFG)
else:
    CFG = C.load_stage_configs_yaml()

if not (C.CONFIGS_DIR / "concept_table.yaml").exists():
    C.save_concept_table_yaml()

C.set_global_seed(CFG.seed)
FORCE_RECOMPUTE = False  # flip per-stage below if you need to bust a specific cache

print(json.dumps(CFG.to_dict(), indent=2))
""")

code(r"""
# S.2b -- Init the single MLflow experiment for the whole notebook
import mlflow
MLU.init_experiment(CFG.mlflow_experiment_name, tracking_uri=f"file:{C.LOGS_DIR / 'mlruns'}")
print(f"MLflow experiment '{CFG.mlflow_experiment_name}' ready at {C.LOGS_DIR / 'mlruns'}")
""")

code(r"""
# S.3/S.4 -- Data + checkpoint paths.
# EDIT THESE to match your environment -- they follow the exact conventions used by the two source
# notebooks (ecg-correct-notebook.ipynb DATA_DIR, image-encoder.ipynb DATA_DIR, and the
# "Consolidate final artifacts" cell's SAVE_DIR).
PTBXL_DATA_DIR = Path("/kaggle/input/datasets/garethwmch/ptb-xl-1-0-3/"
                       "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
CHEXPERT_DATA_DIR = Path("/kaggle/input/datasets/ashery/chexpert")
ENCODER_CHECKPOINT_DIR = Path("/kaggle/working/final_models")

for name, p in [("PTB-XL", PTBXL_DATA_DIR), ("CheXpert", CHEXPERT_DATA_DIR), ("checkpoints", ENCODER_CHECKPOINT_DIR)]:
    print(f"{name:12s}: {p}  (exists: {p.exists()})")
""")

code(r"""
# S.3 -- Load frozen ECG encoders (SimCLR + MultiSupCon), confirm frozen
ecg_encoders = {}
for variant in C.ENCODER_VARIANTS:
    enc = ENC.load_frozen_ecg_encoder(ENCODER_CHECKPOINT_DIR, variant, device=DEVICE)
    ENC.assert_frozen(enc, name=f"ECG-{variant}")
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"ECG [{variant}] loaded, frozen, {n_params:,} params, feature_dim={enc.feature_dim}")
    ecg_encoders[variant] = enc
""")

code(r"""
# S.4 -- Load frozen CXR encoders (SimCLR + MultiSupCon), confirm frozen
cxr_encoders = {}
for variant in C.ENCODER_VARIANTS:
    enc = ENC.load_frozen_cxr_encoder(ENCODER_CHECKPOINT_DIR, variant, device=DEVICE)
    ENC.assert_frozen(enc, name=f"CXR-{variant}")
    n_params = sum(p.numel() for p in enc.parameters())
    print(f"CXR [{variant}] loaded, frozen, {n_params:,} params, feature_dim={enc.feature_dim}")
    cxr_encoders[variant] = enc
""")

md(r"""
### Loading PTB-XL / CheXpert splits

Reproduces just enough of each source notebook's data pipeline (bandpass filter + z-score normalization
for ECG, deterministic resize/grayscale/normalize for CXR) to run the frozen encoders' forward pass.
No SSL-training logic lives here &mdash; that stage is already finished and its weights are frozen above.
""")

code(r"""
# Load PTB-XL annotations + patient-aware strat_fold splits (mirrors ecg-correct-notebook.ipynb cells 7-13)
ecg_raw_cache = PROJECT_ROOT / "cache" / "ptbxl_raw_signals.npy"

Y_ptbxl = DP.load_ptbxl_annotations(PTBXL_DATA_DIR)
print(f"PTB-XL annotations after superclass filtering: {Y_ptbxl.shape}")

if ecg_raw_cache.exists():
    X_ptbxl = np.load(ecg_raw_cache)
    print("Loaded cached raw PTB-XL signal array.")
else:
    import wfdb
    from tqdm.auto import tqdm
    files = Y_ptbxl.filename_lr if CFG.__dict__.get("sampling_rate", 100) == 100 else Y_ptbxl.filename_hr
    data = [wfdb.rdsamp(str(PTBXL_DATA_DIR) + "/" + f) for f in tqdm(files, desc="loading wfdb signals")]
    X_ptbxl = np.array([signal for signal, meta in data])
    np.save(ecg_raw_cache, X_ptbxl)
    print("Cached raw PTB-XL signal array.")

ptbxl_splits_raw = {
    split: DP.get_ptbxl_split(Y_ptbxl, X_ptbxl, folds) for split, folds in DP.PTBXL_SPLIT_FOLDS.items()
}
for split, (x, y) in ptbxl_splits_raw.items():
    print(f"  {split:5s}: {len(y)} records")
""")

code(r"""
# Preprocess (bandpass filter + per-sample z-score), with the same on-disk .npy cache convention as the
# source notebook (cell 13)
ptbxl_splits = {}
for split, (x_raw, y_df) in ptbxl_splits_raw.items():
    cache_path = PROJECT_ROOT / "cache" / f"ptbxl_preprocessed_{split}.npy"
    if cache_path.exists():
        x_proc = np.load(cache_path)
    else:
        x_proc = DP.preprocess_ecg_batch(x_raw)
        np.save(cache_path, x_proc)
    ptbxl_splits[split] = (x_proc, y_df)
    print(f"{split:5s}: {x_proc.shape}")
""")

code(r"""
# Load CheXpert (frontal-only) train/valid splits (mirrors image-encoder.ipynb cells 6, 24)
chexpert_train_df, chexpert_valid_df = DP.load_chexpert_splits(CHEXPERT_DATA_DIR)
print(f"CheXpert train (frontal): {chexpert_train_df.shape}")
print(f"CheXpert valid (frontal): {chexpert_valid_df.shape}")

# CheXpert ships with no held-out "test" split with labels released publicly -- following the source
# notebook's own convention, we treat `valid` as our val split and additionally carve a small,
# stratified test slice out of `train` (never used for anything but final reporting).
_rng = np.random.default_rng(CFG.seed)
_test_idx = _rng.choice(len(chexpert_train_df), size=min(5000, len(chexpert_train_df) // 20), replace=False)
_test_mask = np.zeros(len(chexpert_train_df), dtype=bool)
_test_mask[_test_idx] = True
chexpert_splits_df = {
    "train": chexpert_train_df[~_test_mask].reset_index(drop=True),
    "val": chexpert_valid_df,
    "test": chexpert_train_df[_test_mask].reset_index(drop=True),
}
for split, df in chexpert_splits_df.items():
    print(f"{split:5s}: {len(df)} images")
""")

code(r"""
# S.5 -- Extract & cache embeddings: ECG, all (variant x split) combinations
from torch.utils.data import DataLoader

ecg_embedding_cache = {}
for variant in C.ENCODER_VARIANTS:
    for split, (x_proc, y_df) in ptbxl_splits.items():
        ds = DP.ECGEmbeddingDataset(x_proc, y_df)
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
        cache = CACHE.extract_and_cache_embeddings(
            encoder=ecg_encoders[variant], loader=loader, cache_dir=C.EMBEDDINGS_DIR,
            modality="ecg", variant=variant, split=split, label_names=C.ECG_SUPERCLASSES,
            device=DEVICE, force_recompute=FORCE_RECOMPUTE,
        )
        ecg_embedding_cache[(variant, split)] = cache
        print(f"ECG  [{variant:11s} / {split:5s}] features={cache['features'].shape}")
""")

code(r"""
# S.6 -- Extract & cache embeddings: CXR, all (variant x split) combinations
for variant in C.ENCODER_VARIANTS:
    for split, df in chexpert_splits_df.items():
        ds = DP.CheXpertEmbeddingDataset(df, CHEXPERT_DATA_DIR)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
        cache = CACHE.extract_and_cache_embeddings(
            encoder=cxr_encoders[variant], loader=loader, cache_dir=C.EMBEDDINGS_DIR,
            modality="cxr", variant=variant, split=split, label_names=C.CXR_LABEL_COLS,
            device=DEVICE, force_recompute=FORCE_RECOMPUTE,
        )
        print(f"CXR  [{variant:11s} / {split:5s}] features={cache['features'].shape}")
""")

code(r"""
# S.7 -- Cache sanity check: shapes, NaN/Inf, non-degenerate norm distribution.
# Notebook halts with an explicit error if any check fails -- silent corruption here would
# invalidate every downstream stage.
sanity_reports = []
for variant in C.ENCODER_VARIANTS:
    for split in ["train", "val", "test"]:
        for modality, dim in [("ecg", C.ECG_FEATURE_DIM), ("cxr", C.CXR_FEATURE_DIM)]:
            path = CACHE.embedding_cache_path(C.EMBEDDINGS_DIR, modality, variant, split)
            if not path.exists():
                continue
            cache = CACHE.load_cached_embeddings(path)
            report = CACHE.cache_sanity_check(cache, expected_dim=dim, name=f"{modality}_{variant}_{split}")
            sanity_reports.append(report)

sanity_df = pd.DataFrame(sanity_reports)
display_cols = ["name", "n_samples", "dim", "dim_ok", "no_nan", "not_degenerate", "passed"]
print(sanity_df[display_cols].to_string(index=False))

assert sanity_df["passed"].all(), (
    "Cache sanity check FAILED for at least one (modality, variant, split) combination -- "
    "fix before proceeding to Stage 0. See sanity_df above for details."
)
print("\nAll cache sanity checks passed.")
""")


# ============================================================================
# STAGE 0 -- GEOMETRY
# ============================================================================

md(r"""
## Stage 0 &mdash; Representation Geometry

**Goal.** A cheap, no-training diagnostic: do the two frozen spaces share *any* geometric structure at
all, before any alignment pipeline is built around the assumption that they do?

**Hypothesis.** If linear CKA / RSA between the two encoders' internal geometry &mdash; computed from each
space's own pairwise structure, no cross-modal correspondence required &mdash; clears a random-encoder null
baseline, the later stages have something real to find. If not, later positive results are more likely
artifacts of label leakage than of shared structure.

**Note on what CKA/RSA can and cannot claim here:** CKA and RSA require the same number of rows in both
inputs. ECG (PTB-XL) and CXR (CheXpert) have different, unrelated sample counts and **no instance
correspondence** &mdash; so this diagnostic compares an equal-sized *random subsample* from each independent
space. This is a comparison of **marginal geometric structure** (how spread out / clustered each space's
own pairwise distances are), never a claim that row *i* in one subsample corresponds to row *i* in the
other.
""")

code(r"""
# 0.1-0.4 -- Load cached embeddings (all 4 encoder-variant combinations) and compute CKA / RSA / RDMs
geometry_results = {}
N_SUBSAMPLE = 1000

for ecg_variant in C.ENCODER_VARIANTS:
    for cxr_variant in C.ENCODER_VARIANTS:
        ecg_cache = CACHE.load_cached_embeddings(CACHE.embedding_cache_path(C.EMBEDDINGS_DIR, "ecg", ecg_variant, "train"))
        cxr_cache = CACHE.load_cached_embeddings(CACHE.embedding_cache_path(C.EMBEDDINGS_DIR, "cxr", cxr_variant, "train"))

        diag = GEOM.geometry_diagnostic(ecg_cache["features"], cxr_cache["features"], n_subsample=N_SUBSAMPLE, seed=CFG.seed)
        key = f"ecg_{ecg_variant}__cxr_{cxr_variant}"
        geometry_results[key] = diag
        print(f"{key:35s}  CKA={diag['cka']:+.4f} (z={diag['cka_z_vs_null']:+.2f})   "
              f"RSA={diag['rsa']:+.4f} (z={diag['rsa_z_vs_null']:+.2f})   clears_null={diag['clears_null']}")
""")

code(r"""
# 0.6 -- Visualization: RDM heatmap pairs, one figure per encoder-variant combination
for key, diag in geometry_results.items():
    ecg_variant, cxr_variant = key.replace("ecg_", "").replace("cxr_", "").split("__")
    fig = VIZ.plot_rdm_pair(diag["rdm_ecg"], diag["rdm_cxr"], f"ECG RDM ({ecg_variant})", f"CXR RDM ({cxr_variant})")
    fig_path = C.GEOMETRY_DIR / "figures" / f"rdm_{key}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
""")

code(r"""
# 0.7 -- Visualization: CKA/RSA vs random-encoder null, one bar chart per metric
real_cka = {k: v["cka"] for k, v in geometry_results.items()}
null_cka_mean = {k: v["cka_null_mean"] for k, v in geometry_results.items()}
null_cka_std = {k: v["cka_null_std"] for k, v in geometry_results.items()}
fig_cka = VIZ.plot_score_vs_null(real_cka, null_cka_mean, null_cka_std, "Linear CKA", "Stage 0: CKA vs random-encoder null")
fig_cka.savefig(C.GEOMETRY_DIR / "figures" / "cka_vs_null.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_cka)

real_rsa = {k: v["rsa"] for k, v in geometry_results.items()}
null_rsa_mean = {k: v["rsa_null_mean"] for k, v in geometry_results.items()}
null_rsa_std = {k: v["rsa_null_std"] for k, v in geometry_results.items()}
fig_rsa = VIZ.plot_score_vs_null(real_rsa, null_rsa_mean, null_rsa_std, "RSA (Spearman)", "Stage 0: RSA vs random-encoder null")
fig_rsa.savefig(C.GEOMETRY_DIR / "figures" / "rsa_vs_null.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_rsa)
""")

code(r"""
# 0.8 -- MLflow logging
with MLU.stage_run(run_name="stage0_geometry", stage="stage0_geometry", data_source="ptbxl_chexpert") as run:
    for key, diag in geometry_results.items():
        mlflow.log_metrics({
            f"{key}__cka": diag["cka"], f"{key}__rsa": diag["rsa"],
            f"{key}__cka_z": diag["cka_z_vs_null"], f"{key}__rsa_z": diag["rsa_z_vs_null"],
        })
    mlflow.log_params({"n_subsample": N_SUBSAMPLE, "seed": CFG.seed})
    for fig_path in (C.GEOMETRY_DIR / "figures").glob("*.png"):
        mlflow.log_artifact(str(fig_path), artifact_path="figures")
    geometry_json_path = C.GEOMETRY_DIR / "cka_rsa_scores.json"
    with open(geometry_json_path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray) and kk not in
                        ("cka_null_samples", "rsa_null_samples")} for k, v in geometry_results.items()}, f, indent=2)
    mlflow.log_artifact(str(geometry_json_path))
    print(f"Logged MLflow run: {run.info.run_id}")
""")

code(r"""
# 0.9 -- Decision cell: go / no-go
best_key = max(geometry_results, key=lambda k: max(geometry_results[k]["cka_z_vs_null"], geometry_results[k]["rsa_z_vs_null"]))
best_diag = geometry_results[best_key]
any_clears = any(v["clears_null"] for v in geometry_results.values())

print(f"Best combination: {best_key}")
print(f"  CKA z-score vs null: {best_diag['cka_z_vs_null']:+.2f}")
print(f"  RSA z-score vs null: {best_diag['rsa_z_vs_null']:+.2f}")
print()
if any_clears:
    print("GO: at least one encoder-variant combination clears the random-encoder null baseline.")
    print("Proceeding to Stage 1. (Per the roadmap, the MultiSupCon x MultiSupCon combination is the")
    print("most likely candidate to show shared structure, since label-informed pretraining is more")
    print("likely to produce alignable geometry than purely instance-discriminative SimCLR pretraining.)")
else:
    print("NO-GO signal: no combination clearly clears the random-encoder null.")
    print("This does NOT halt the notebook (the null result is itself informative, see design doc")
    print("Section 11, risk 3) but it should be weighted heavily when interpreting every later stage --")
    print("a modest downstream retrieval number would be consistent with 'no real structure exists',")
    print("not just 'the alignment method needs tuning'.")
""")


# ============================================================================
# STAGE 1 -- MANUAL CONCEPT BASELINE
# ============================================================================

md(r"""
## Stage 1 &mdash; Manual Concept Baseline

**Goal.** Establish the explicit, clearly-labeled floor using the original hand-mapped 4-bin concept
table and cross-modal SupCon &mdash; kept in the notebook specifically so its circularity is demonstrated
**empirically**, not just argued in prose.

**This stage's only legitimate role in the thesis:** a documented floor, and proof that
Concept-Precision@K is circular when the same table both supervises and evaluates. **It is not evaluated
or reported as the headline method anywhere in this notebook** (see Stage 5/7 for the real method and its
non-circular validation).

The loss reuses `label_similarity` / `multilabel_supcon_loss` from `ecg-correct-notebook.ipynb` verbatim
(`src/losses.py`), generalized from a square within-modality batch to a bipartite ECG&times;CXR batch.
""")

code(r"""
# 1.1 -- Load & validate concept_table.yaml
concept_table = C.load_concept_table_yaml()
for bin_name, spec in concept_table.items():
    print(f"{bin_name:28s}  ECG: {spec['ecg']}   CXR: {spec['cxr']}")
    print(f"{'':28s}  rationale: {spec['rationale']}")

all_mapped_ecg = {c for spec in concept_table.values() for c in spec["ecg"]}
all_mapped_cxr = {c for spec in concept_table.values() for c in spec["cxr"]}
unmapped_ecg = set(C.ECG_SUPERCLASSES) - all_mapped_ecg
unmapped_cxr = set(C.CXR_SHARED_LABELS) - all_mapped_cxr
print(f"\nUnmapped ECG superclasses: {unmapped_ecg or 'none'}")
print(f"Unmapped CXR shared labels: {unmapped_cxr or 'none'}")
assert not unmapped_ecg, "Every ECG superclass must be covered by the concept table."
""")

code(r"""
# 1.2 -- Build label -> bin lookup tables
CONCEPT_BIN_NAMES = list(concept_table.keys())
N_CONCEPT_BINS = len(CONCEPT_BIN_NAMES)


def multihot_bins(label_vector: np.ndarray, label_names: list, side: str) -> np.ndarray:
    '''label_vector: (N, C) multi-hot over the modality's own label space.
    Returns: (N, N_CONCEPT_BINS) multi-hot over concept bins.'''
    out = np.zeros((label_vector.shape[0], N_CONCEPT_BINS), dtype=np.float32)
    for bi, bin_name in enumerate(CONCEPT_BIN_NAMES):
        cols = [label_names.index(c) for c in concept_table[bin_name][side] if c in label_names]
        if cols:
            out[:, bi] = (label_vector[:, cols].sum(axis=1) > 0).astype(np.float32)
    return out


print(f"{N_CONCEPT_BINS} concept bins: {CONCEPT_BIN_NAMES}")
""")

code(r"""
# 1.3/1.4 -- Alignment head shapes + bin-supervised cross-modal batch sampler
ecg_head_s1 = ENC.AlignHead(in_dim=C.ECG_FEATURE_DIM, out_dim=CFG.prototype_head_dim, hidden_dim=256).to(DEVICE)
cxr_head_s1 = ENC.AlignHead(in_dim=C.CXR_FEATURE_DIM, out_dim=CFG.prototype_head_dim, hidden_dim=512).to(DEVICE)
print(ecg_head_s1)
print(cxr_head_s1)

ecg_train_cache = ecg_embedding_cache[(C.PRIMARY_ENCODER_VARIANT, "train")]
cxr_train_cache = CACHE.load_cached_embeddings(
    CACHE.embedding_cache_path(C.EMBEDDINGS_DIR, "cxr", C.PRIMARY_ENCODER_VARIANT, "train")
)
ecg_bins_train = multihot_bins(ecg_train_cache["labels"], C.ECG_SUPERCLASSES, "ecg")
cxr_bins_train = multihot_bins(cxr_train_cache["labels"], C.CXR_LABEL_COLS, "cxr")

ecg_feat_t = torch.tensor(ecg_train_cache["features"], dtype=torch.float32)
cxr_feat_t = torch.tensor(cxr_train_cache["features"], dtype=torch.float32)
ecg_bins_t = torch.tensor(ecg_bins_train, dtype=torch.float32)
cxr_bins_t = torch.tensor(cxr_bins_train, dtype=torch.float32)


def sample_bipartite_batch(batch_size: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    idx_e = torch.randint(0, ecg_feat_t.shape[0], (batch_size,), generator=g)
    idx_c = torch.randint(0, cxr_feat_t.shape[0], (batch_size,), generator=g)
    return (ecg_feat_t[idx_e].to(DEVICE), ecg_bins_t[idx_e].to(DEVICE),
            cxr_feat_t[idx_c].to(DEVICE), cxr_bins_t[idx_c].to(DEVICE))


xb_e, yb_e, xb_c, yb_c = sample_bipartite_batch(8, seed=0)
print(f"Dry-run batch shapes: ECG {xb_e.shape}/{yb_e.shape}, CXR {xb_c.shape}/{yb_c.shape}")
""")

code(r"""
# 1.5 -- Training loop: cross-modal concept-SupCon on bins (heads only, backbones frozen throughout)
STAGE1_EPOCHS = 30
STAGE1_STEPS_PER_EPOCH = 50

opt_s1 = torch.optim.AdamW(list(ecg_head_s1.parameters()) + list(cxr_head_s1.parameters()),
                            lr=CFG.lr, weight_decay=CFG.weight_decay)
temp_s1 = ENC.LearnableTemperature(init_value=CFG.learnable_temperature_init).to(DEVICE)
opt_s1.add_param_group({"params": temp_s1.parameters()})

stage1_loss_history = []
for epoch in range(STAGE1_EPOCHS):
    epoch_losses = []
    for step in range(STAGE1_STEPS_PER_EPOCH):
        xb_e, yb_e, xb_c, yb_c = sample_bipartite_batch(CFG.batch_size, seed=epoch * STAGE1_STEPS_PER_EPOCH + step)
        opt_s1.zero_grad(set_to_none=True)
        z_e = ecg_head_s1(xb_e)
        z_c = cxr_head_s1(xb_c)
        loss = L.cross_modal_concept_supcon_loss(z_e, z_c, yb_e, yb_c, temperature=temp_s1())
        loss.backward()
        opt_s1.step()
        epoch_losses.append(loss.item())
    mean_loss = float(np.mean(epoch_losses))
    stage1_loss_history.append(mean_loss)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"epoch {epoch + 1:3d}/{STAGE1_EPOCHS}  loss={mean_loss:.4f}  temp={temp_s1().item():.4f}")
""")

code(r"""
# 1.6 -- Checkpointing
torch.save(ecg_head_s1.state_dict(), C.BASELINE_DIR / "heads" / "ECGAlignHead_v0.pt")
torch.save(cxr_head_s1.state_dict(), C.BASELINE_DIR / "heads" / "CXRAlignHead_v0.pt")
torch.save(opt_s1.state_dict(), C.BASELINE_DIR / "heads" / "optimizer_v0.pt")
print("Stage 1 heads checkpointed.")
""")

code(r"""
# 1.7 -- Concept-Precision@K -- computed AND explicitly flagged circular (filename + variable name carry
# the warning, per the design review's core methodological finding)
ecg_val_cache = ecg_embedding_cache[(C.PRIMARY_ENCODER_VARIANT, "val")]
cxr_val_cache = CACHE.load_cached_embeddings(CACHE.embedding_cache_path(C.EMBEDDINGS_DIR, "cxr", C.PRIMARY_ENCODER_VARIANT, "val"))

ecg_bins_val = multihot_bins(ecg_val_cache["labels"], C.ECG_SUPERCLASSES, "ecg").argmax(axis=1)
cxr_bins_val = multihot_bins(cxr_val_cache["labels"], C.CXR_LABEL_COLS, "cxr").argmax(axis=1)

ecg_head_s1.eval(); cxr_head_s1.eval()
with torch.no_grad():
    z_ecg_val = ecg_head_s1(torch.tensor(ecg_val_cache["features"], dtype=torch.float32).to(DEVICE)).cpu().numpy()
    z_cxr_val = cxr_head_s1(torch.tensor(cxr_val_cache["features"], dtype=torch.float32).to(DEVICE)).cpu().numpy()

sim_s1 = EVAL.cosine_similarity_matrix(z_ecg_val, z_cxr_val)
circular_concept_precision_at_k = {
    k: EVAL.concept_precision_at_k(ecg_bins_val, cxr_bins_val, sim_s1, k=k) for k in CFG.top_k_retrieval
}
print("*** CIRCULAR METRIC -- same table supervises training AND defines 'correct' here ***")
for k, v in circular_concept_precision_at_k.items():
    print(f"  Concept-Precision@{k}: {v:.4f}")

circular_metrics_path = C.BASELINE_DIR / "metrics" / "circular_concept_precision_at_k.json"
with open(circular_metrics_path, "w") as f:
    json.dump({"warning": "CIRCULAR: same concept table used for supervision and evaluation.",
               **{f"precision_at_{k}": v for k, v in circular_concept_precision_at_k.items()}}, f, indent=2)
""")

code(r"""
# 1.8 -- Visualization: joint t-SNE/UMAP colored by concept bin (sanity check only, not a validation claim)
joint_embeddings_s1 = np.concatenate([z_ecg_val, z_cxr_val], axis=0)
joint_bins_s1 = np.concatenate([ecg_bins_val, cxr_bins_val], axis=0)
joint_modality_s1 = np.array(["ECG"] * len(ecg_bins_val) + ["CXR"] * len(cxr_bins_val))
bin_names_arr = np.array(CONCEPT_BIN_NAMES)

fig_s1 = VIZ.plot_embedding_2d(
    joint_embeddings_s1, bin_names_arr[joint_bins_s1], title="Stage 1: joint embedding, colored by concept bin",
    marker_labels=joint_modality_s1, seed=CFG.seed,
)
fig_s1.savefig(C.BASELINE_DIR / "figures" / "tsne_stage1.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_s1)

fig_loss_s1 = VIZ.plot_loss_curve(stage1_loss_history, "Stage 1: cross-modal concept-SupCon training loss")
fig_loss_s1.savefig(C.BASELINE_DIR / "figures" / "loss_curve_stage1.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_loss_s1)
""")

code(r"""
# 1.9 -- MLflow logging (metric explicitly tagged circular)
with MLU.stage_run(run_name="stage1_concept_baseline", stage="stage1_concept_baseline",
                    encoder_variant=C.PRIMARY_ENCODER_VARIANT, data_source="ptbxl_chexpert") as run:
    mlflow.log_params({"epochs": STAGE1_EPOCHS, "steps_per_epoch": STAGE1_STEPS_PER_EPOCH,
                        "batch_size": CFG.batch_size, "lr": CFG.lr, "n_concept_bins": N_CONCEPT_BINS})
    mlflow.log_metrics({"final_train_loss": stage1_loss_history[-1]})
    for k, v in circular_concept_precision_at_k.items():
        mlflow.log_metric(f"circular_concept_precision_at_{k}", v)
    mlflow.set_tag("metric_status", "circular")
    mlflow.log_artifact(str(circular_metrics_path), artifact_path="metrics")
    mlflow.log_artifact(str(C.BASELINE_DIR / "figures" / "tsne_stage1.png"), artifact_path="figures")
    mlflow.log_artifact(str(C.BASELINE_DIR / "figures" / "loss_curve_stage1.png"), artifact_path="figures")
    MLU.log_large_array_path("ecg_head_checkpoint", C.BASELINE_DIR / "heads" / "ECGAlignHead_v0.pt")
    MLU.log_large_array_path("cxr_head_checkpoint", C.BASELINE_DIR / "heads" / "CXRAlignHead_v0.pt")
    print(f"Logged MLflow run: {run.info.run_id} (metric_status=circular)")
""")


# ============================================================================
# STAGE 2 -- LINEAR / CCA BASELINES
# ============================================================================

md(r"""
## Stage 2 &mdash; Linear / CCA Baselines

**Goal.** Classical, cheap alignment baselines using the same bin correspondence as class-conditional
anchors &mdash; measures what plain linear methods get you before any nonlinear machinery (prototypes, OT,
GW) is introduced.

**Hypothesis.** Linear correspondence between two independently-trained nonlinear SSL spaces is expected
to be weak; this stage turns that expectation into an actual measured number rather than an assumption.

**Both variants below still depend on the manual concept table** (to define which class-conditional means
correspond across modalities) &mdash; this is flagged explicitly, and is exactly why Stage 2, like Stage 1,
is a baseline and not a candidate final method. Internal retrieval@K here is explicitly labeled a
same-dataset sanity check, never cross-modal validation (no real pairs exist in PTB-XL/CheXpert).
""")

code(r"""
# 2.1 -- Class-conditional mean embeddings per bin, computed on RAW frozen features (not yet
# projected through any alignment head -- this baseline tests the raw frozen spaces directly)
ecg_bins_train_idx = multihot_bins(ecg_train_cache["labels"], C.ECG_SUPERCLASSES, "ecg").argmax(axis=1)
cxr_bins_train_idx = multihot_bins(cxr_train_cache["labels"], C.CXR_LABEL_COLS, "cxr").argmax(axis=1)

means_ecg = LCCA.class_conditional_means(ecg_train_cache["features"], ecg_bins_train_idx, N_CONCEPT_BINS)
means_cxr = LCCA.class_conditional_means(cxr_train_cache["features"], cxr_bins_train_idx, N_CONCEPT_BINS)
print(f"ECG class-conditional means: {means_ecg.shape}   CXR class-conditional means: {means_cxr.shape}")
""")

code(r"""
# 2.2 -- Fit closed-form linear map: ECG-mean space -> CXR-mean space
linear_map_w = LCCA.fit_linear_map(means_ecg, means_cxr)
np.savez(C.LINEAR_CCA_DIR / "weights" / "linear_map.npz", w=linear_map_w)
print(f"Linear map weight shape: {linear_map_w.shape}")
""")

code(r"""
# 2.3 -- Fit (regularized) CCA, using bin-matched instance samples (the only correspondence
# available in this unpaired setting is bin-level, never instance-level)
ecg_matched, cxr_matched = LCCA.build_bin_matched_sample(
    ecg_train_cache["features"], ecg_bins_train_idx, cxr_train_cache["features"], cxr_bins_train_idx,
    n_bins=N_CONCEPT_BINS, seed=CFG.seed,
)
print(f"Bin-matched sample: ECG {ecg_matched.shape}, CXR {cxr_matched.shape}")

cca_model, canonical_correlations = LCCA.fit_cca(ecg_matched, cxr_matched, n_components=20)
np.savez(C.LINEAR_CCA_DIR / "weights" / "cca_components.npz", correlations=canonical_correlations)
print(f"Top-5 canonical correlations: {np.round(canonical_correlations[:5], 4)}")
""")

code(r"""
# 2.4 -- Internal retrieval@K evaluation (PTB-XL/CheXpert only -- NOT a cross-modal validation claim)
ecg_val_bins_idx = ecg_bins_val  # already computed in Stage 1
cxr_val_bins_idx = cxr_bins_val

mapped_ecg_val = LCCA.apply_linear_map(ecg_val_cache["features"], linear_map_w)
sim_linear = EVAL.cosine_similarity_matrix(mapped_ecg_val, cxr_val_cache["features"])
linear_map_precision_at_k = {
    k: EVAL.concept_precision_at_k(ecg_val_bins_idx, cxr_val_bins_idx, sim_linear, k=k) for k in CFG.top_k_retrieval
}

ecg_val_cca, cxr_val_cca = cca_model.transform(ecg_val_cache["features"], cxr_val_cache["features"])
sim_cca = EVAL.cosine_similarity_matrix(ecg_val_cca, cxr_val_cca)
cca_precision_at_k = {
    k: EVAL.concept_precision_at_k(ecg_val_bins_idx, cxr_val_bins_idx, sim_cca, k=k) for k in CFG.top_k_retrieval
}

print("Linear map -- internal Concept-Precision@K (same-dataset sanity check only):")
for k, v in linear_map_precision_at_k.items():
    print(f"  @{k}: {v:.4f}")
print("CCA -- internal Concept-Precision@K (same-dataset sanity check only):")
for k, v in cca_precision_at_k.items():
    print(f"  @{k}: {v:.4f}")

stage2_metrics = {"linear_map": linear_map_precision_at_k, "cca": cca_precision_at_k}
with open(C.LINEAR_CCA_DIR / "metrics" / "internal_retrieval_at_k.json", "w") as f:
    json.dump(stage2_metrics, f, indent=2)
""")

code(r"""
# 2.5 -- Visualization: canonical-correlation scree plot
fig_scree, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(range(1, len(canonical_correlations) + 1), canonical_correlations, marker="o", markersize=4)
ax.set_xlabel("CCA component"); ax.set_ylabel("canonical correlation")
ax.set_title("Stage 2: CCA canonical-correlation scree plot")
ax.grid(alpha=0.3)
fig_scree.tight_layout()
fig_scree.savefig(C.LINEAR_CCA_DIR / "figures" / "cca_scree.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_scree)
""")

code(r"""
# 2.6 -- Visualization: retrieval@K bar chart, Stage 1 (circular, shown for scale only) vs Stage 2 variants
fig_bar_s2 = VIZ.plot_grouped_bar(
    {
        "Stage 1 (circular, reference only)": {f"@{k}": v for k, v in circular_concept_precision_at_k.items()},
        "Stage 2: linear map": {f"@{k}": v for k, v in linear_map_precision_at_k.items()},
        "Stage 2: CCA": {f"@{k}": v for k, v in cca_precision_at_k.items()},
    },
    title="Stage 2: internal Concept-Precision@K vs Stage 1 (all same-dataset, no cross-modal validation)",
    ylabel="Concept-Precision@K",
)
fig_bar_s2.savefig(C.LINEAR_CCA_DIR / "figures" / "retrieval_bar_vs_stage1.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_bar_s2)
""")

code(r"""
# 2.7 -- MLflow logging
with MLU.stage_run(run_name="stage2_linear_cca", stage="stage2_linear_cca",
                    encoder_variant=C.PRIMARY_ENCODER_VARIANT, data_source="ptbxl_chexpert") as run:
    mlflow.log_params({"n_cca_components": len(canonical_correlations), "n_concept_bins": N_CONCEPT_BINS})
    for k, v in linear_map_precision_at_k.items():
        mlflow.log_metric(f"linear_map_internal_precision_at_{k}", v)
    for k, v in cca_precision_at_k.items():
        mlflow.log_metric(f"cca_internal_precision_at_{k}", v)
    mlflow.set_tag("metric_status", "internal_only_no_cross_modal_validation")
    mlflow.log_artifact(str(C.LINEAR_CCA_DIR / "metrics" / "internal_retrieval_at_k.json"), artifact_path="metrics")
    mlflow.log_artifact(str(C.LINEAR_CCA_DIR / "figures" / "cca_scree.png"), artifact_path="figures")
    mlflow.log_artifact(str(C.LINEAR_CCA_DIR / "figures" / "retrieval_bar_vs_stage1.png"), artifact_path="figures")
    print(f"Logged MLflow run: {run.info.run_id}")
""")


# ============================================================================
# STAGE 3 -- PROTOTYPE DISCOVERY
# ============================================================================

md(r"""
## Stage 3 &mdash; Prototype Discovery

**Goal.** Establish per-modality cluster structure independently and descriptively, **before any
cross-modal step exists**. Purely unsupervised: ground-truth labels are used only as a post-hoc
diagnostic (purity), never fed back into training or K-selection.

**Hypothesis.** Each frozen space contains enough stable internal cluster structure for
prototype-based matching (Stages 4&ndash;5) to be meaningful at all. If ECG or CXR features don't cluster
cleanly, Stage 5 would fail for a boring reason (bad prototypes), not an interesting one (no shared
structure) &mdash; this stage exists to rule that out first.

Runs the full K-sweep (`CFG.prototype_k_grid`) &times; seed grid (`CFG.prototype_seeds`) for **both**
modalities and **both** encoder variants (SimCLR + MultiSupCon) &mdash; the MultiSupCon run is what Stages
4/5 consume by default; the SimCLR run is banked here for free, for the Stage 6 backbone ablation.
""")

code(r"""
# 3.1 -- Define K sweep grid & seed list (single source of truth: CFG)
print(f"K sweep grid: {CFG.prototype_k_grid}")
print(f"Seeds: {CFG.prototype_seeds}")
""")

code(r"""
# 3.2-3.5 -- Run k-means + diagnostics (silhouette, purity, cross-seed ARI stability),
# all (modality x variant x K x seed) combinations
prototype_diagnostic_rows = []
ground_truth_bins_by_modality_variant = {}

for variant in C.ENCODER_VARIANTS:
    ecg_cache_v = ecg_embedding_cache[(variant, "train")]
    cxr_cache_v = CACHE.load_cached_embeddings(CACHE.embedding_cache_path(C.EMBEDDINGS_DIR, "cxr", variant, "train"))
    ecg_bins_v = multihot_bins(ecg_cache_v["labels"], C.ECG_SUPERCLASSES, "ecg").argmax(axis=1)
    cxr_bins_v = multihot_bins(cxr_cache_v["labels"], C.CXR_LABEL_COLS, "cxr").argmax(axis=1)
    ground_truth_bins_by_modality_variant[("ecg", variant)] = ecg_bins_v
    ground_truth_bins_by_modality_variant[("cxr", variant)] = cxr_bins_v

    ecg_rows = PROTO.sweep_prototypes(
        ecg_cache_v["features"], ecg_bins_v, C.PROTOTYPE_CACHE_DIR, "ecg", variant,
        CFG.prototype_k_grid, CFG.prototype_seeds, force_recompute=FORCE_RECOMPUTE,
    )
    cxr_rows = PROTO.sweep_prototypes(
        cxr_cache_v["features"], cxr_bins_v, C.PROTOTYPE_CACHE_DIR, "cxr", variant,
        CFG.prototype_k_grid, CFG.prototype_seeds, force_recompute=FORCE_RECOMPUTE,
    )
    prototype_diagnostic_rows.extend(ecg_rows)
    prototype_diagnostic_rows.extend(cxr_rows)
    print(f"[{variant}] ECG rows: {len(ecg_rows)}, CXR rows: {len(cxr_rows)}")

prototype_diagnostics_df = pd.DataFrame(prototype_diagnostic_rows)
prototype_diagnostics_df.to_json(C.PROTOTYPES_DIR / "cluster_diagnostics.json", orient="records", indent=2)
print(prototype_diagnostics_df.groupby(["modality", "variant", "k"])[["silhouette", "purity", "stability_ari_vs_seed0"]].mean())
""")

code(r"""
# 3.6 -- Decision cell: select primary K via silhouette/stability elbow (NOT purity, kept label-free)
primary_rows_ecg = [r for r in prototype_diagnostic_rows if r["modality"] == "ecg" and r["variant"] == C.PRIMARY_ENCODER_VARIANT]
primary_rows_cxr = [r for r in prototype_diagnostic_rows if r["modality"] == "cxr" and r["variant"] == C.PRIMARY_ENCODER_VARIANT]

primary_k_ecg = PROTO.select_primary_k(primary_rows_ecg, CFG.prototype_k_grid)
primary_k_cxr = PROTO.select_primary_k(primary_rows_cxr, CFG.prototype_k_grid)
# A single shared K keeps GW's prototype-count bookkeeping simple across modalities (GW itself
# supports unequal K, but a matched K makes the Stage 5 comparison figures directly interpretable).
PRIMARY_K = max(primary_k_ecg, primary_k_cxr)
CFG.primary_prototype_k = PRIMARY_K
C.save_stage_configs_yaml(CFG)

print(f"Silhouette/stability-selected K -- ECG: {primary_k_ecg}, CXR: {primary_k_cxr}")
print(f"Primary K adopted for Stages 4/5 (max of the two, kept label-free): {PRIMARY_K}")
print("Written back to configs/stage_configs.yaml as `primary_prototype_k`.")
""")

code(r"""
# 3.7 -- [Conditional] SwAV/DeepCluster escalation -- only runs if k-means diagnostics at PRIMARY_K
# fail a minimum quality bar (mean silhouette < 0.05 across both modalities), per the blueprint's
# stopping rule for this cell.
_mean_silhouette_at_primary_k = prototype_diagnostics_df[
    (prototype_diagnostics_df.variant == C.PRIMARY_ENCODER_VARIANT) & (prototype_diagnostics_df.k == PRIMARY_K)
]["silhouette"].mean()

SWAV_ESCALATION_TRIGGERED = bool(_mean_silhouette_at_primary_k < 0.05)
print(f"Mean silhouette at K={PRIMARY_K}: {_mean_silhouette_at_primary_k:.4f}")
print(f"SwAV/DeepCluster escalation triggered: {SWAV_ESCALATION_TRIGGERED}")

if SWAV_ESCALATION_TRIGGERED:
    print("k-means prototypes are visibly weak at the selected K. A SwAV/DeepCluster online-clustering")
    print("escalation is warranted here as a stronger prototype-discovery method; implementing the full")
    print("SwAV pretext task is out of scope for this cell and is tracked as Stage 6 ablation #6")
    print("(only run if this flag is True -- see Stage 6 below).")
else:
    print("k-means prototypes clear the quality bar -- no escalation needed. Stage 6 ablation #6 is")
    print("skipped (see its conditional guard in the Stage 6 section).")
""")

code(r"""
# 3.8 -- Visualization: silhouette-vs-K curve, both modalities, primary variant
fig_sil, ax = plt.subplots(figsize=(6.5, 4))
for modality in ["ecg", "cxr"]:
    sub = prototype_diagnostics_df[
        (prototype_diagnostics_df.modality == modality) & (prototype_diagnostics_df.variant == C.PRIMARY_ENCODER_VARIANT)
    ].groupby("k")["silhouette"].mean()
    ax.plot(sub.index, sub.values, marker="o", label=modality.upper())
ax.axvline(PRIMARY_K, color="gray", linestyle="--", linewidth=1, label=f"selected K={PRIMARY_K}")
ax.set_xlabel("K"); ax.set_ylabel("mean silhouette"); ax.set_title("Stage 3: silhouette vs K")
ax.legend(); ax.grid(alpha=0.3)
fig_sil.tight_layout()
fig_sil.savefig(C.PROTOTYPES_DIR / "figures" / "silhouette_vs_k.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_sil)
""")

code(r"""
# 3.9 -- Visualization: UMAP colored by cluster assignment, both modalities, at the primary K
for modality, cache_v in [("ecg", ecg_embedding_cache[(C.PRIMARY_ENCODER_VARIANT, "train")]),
                          ("cxr", CACHE.load_cached_embeddings(CACHE.embedding_cache_path(C.EMBEDDINGS_DIR, "cxr", C.PRIMARY_ENCODER_VARIANT, "train")))]:
    proto_result = PROTO.cluster_and_cache(cache_v["features"], C.PROTOTYPE_CACHE_DIR, modality,
                                            C.PRIMARY_ENCODER_VARIANT, PRIMARY_K, seed=CFG.prototype_seeds[0])
    n_plot = min(2000, cache_v["features"].shape[0])
    plot_idx = np.random.default_rng(CFG.seed).choice(cache_v["features"].shape[0], size=n_plot, replace=False)
    fig_umap = VIZ.plot_embedding_2d(
        cache_v["features"][plot_idx], proto_result["assignments"][plot_idx].astype(str),
        title=f"Stage 3: {modality.upper()} features, colored by k-means cluster (K={PRIMARY_K})", seed=CFG.seed,
    )
    fig_umap.savefig(C.PROTOTYPES_DIR / "figures" / f"umap_{modality}_k{PRIMARY_K}.png", dpi=150, bbox_inches="tight")
    plt.show(); plt.close(fig_umap)
""")

code(r"""
# 3.10 -- MLflow logging: nested runs, one child per (modality, variant, K, seed)
with MLU.stage_run(run_name="stage3_prototype_discovery", stage="stage3_prototype_discovery",
                    data_source="ptbxl_chexpert") as parent_run:
    mlflow.log_params({"k_grid": CFG.prototype_k_grid, "seeds": CFG.prototype_seeds, "primary_k": PRIMARY_K,
                        "swav_escalation_triggered": SWAV_ESCALATION_TRIGGERED})
    for row in prototype_diagnostic_rows:
        with MLU.stage_run(
            run_name=f"stage3_{row['modality']}_{row['variant']}_k{row['k']}_seed{row['seed']}",
            stage="stage3_prototype_discovery", encoder_variant=row["variant"], nested=True,
            seed=row["seed"], extra_tags={"modality": row["modality"], "k": str(row["k"])},
        ):
            mlflow.log_metrics({"silhouette": row["silhouette"] if np.isfinite(row["silhouette"]) else -1.0,
                                 "purity": row["purity"], "stability_ari_vs_seed0": row["stability_ari_vs_seed0"]})
    mlflow.log_artifact(str(C.PROTOTYPES_DIR / "cluster_diagnostics.json"), artifact_path="diagnostics")
    for fig_path in (C.PROTOTYPES_DIR / "figures").glob("*.png"):
        mlflow.log_artifact(str(fig_path), artifact_path="figures")
    print(f"Logged MLflow parent run: {parent_run.info.run_id} with {len(prototype_diagnostic_rows)} nested runs")
""")


# ============================================================================
# STAGE 4 -- PROTOTYPE CONTRASTIVE ALIGNMENT
# ============================================================================

md(r"""
## Stage 4 &mdash; Prototype Contrastive Alignment (assumed cross-modal cost)

**Goal.** Train alignment heads to pull instances toward cross-modal-matched prototypes using an
assumed/hand-specified cross-modal cost &mdash; a **controlled stepping stone**, not a candidate final
method. Its retrieval number is banked and compared head-to-head against Stage 5's Gromov-Wasserstein
result, so that Stage 5's "no cross-modal cost function needed" claim becomes a *measured* comparison
rather than an assertion.

**Cost-matrix source, documented on the record** (the one place a residual manual assumption is
deliberately allowed to remain, exactly because Stage 5 needs something to be compared against): the
concept-table-derived Jaccard similarity between each prototype's majority concept bin, reusing
`label_similarity` from `src/losses.py`. If Stage 5 performs comparably to this "OT with cheating"
baseline, that is itself a strong result &mdash; it means the manual cost function wasn't buying much.
""")

code(r"""
# 4.1 -- Construct cross-modal cost matrix (documented source: concept-table-derived Jaccard, see
# markdown above). Uses the primary-K prototypes discovered in Stage 3, for the primary encoder variant.
ecg_proto_s3 = PROTO.cluster_and_cache(
    ecg_embedding_cache[(C.PRIMARY_ENCODER_VARIANT, "train")]["features"], C.PROTOTYPE_CACHE_DIR,
    "ecg", C.PRIMARY_ENCODER_VARIANT, PRIMARY_K, seed=CFG.prototype_seeds[0],
)
cxr_train_cache_primary = CACHE.load_cached_embeddings(
    CACHE.embedding_cache_path(C.EMBEDDINGS_DIR, "cxr", C.PRIMARY_ENCODER_VARIANT, "train")
)
cxr_proto_s3 = PROTO.cluster_and_cache(
    cxr_train_cache_primary["features"], C.PROTOTYPE_CACHE_DIR,
    "cxr", C.PRIMARY_ENCODER_VARIANT, PRIMARY_K, seed=CFG.prototype_seeds[0],
)

cost_matrix_s4, transport_plan_s4 = OT.compute_prototype_cost_matrix(
    ecg_proto_s3["assignments"], ground_truth_bins_by_modality_variant[("ecg", C.PRIMARY_ENCODER_VARIANT)],
    cxr_proto_s3["assignments"], ground_truth_bins_by_modality_variant[("cxr", C.PRIMARY_ENCODER_VARIANT)],
    n_clusters=PRIMARY_K, n_bins=N_CONCEPT_BINS,
)
np.savez(C.OT_DIR / "weights" / "cost_and_transport_matrix.npz", cost=cost_matrix_s4, transport=transport_plan_s4)
print(f"Stage 4 cost matrix shape: {cost_matrix_s4.shape} (min={cost_matrix_s4.min():.3f}, max={cost_matrix_s4.max():.3f})")
""")

code(r"""
# 4.2/4.3 -- Initialize Stage 4 heads & train via prototype contrastive loss guided by transport plan
ecg_head_s4 = ENC.AlignHead(in_dim=C.ECG_FEATURE_DIM, out_dim=CFG.prototype_head_dim, hidden_dim=256).to(DEVICE)
cxr_head_s4 = ENC.AlignHead(in_dim=C.CXR_FEATURE_DIM, out_dim=CFG.prototype_head_dim, hidden_dim=512).to(DEVICE)

STAGE4_EPOCHS = 30
STAGE4_STEPS_PER_EPOCH = 50
opt_s4 = torch.optim.AdamW(list(ecg_head_s4.parameters()) + list(cxr_head_s4.parameters()),
                            lr=CFG.lr, weight_decay=CFG.weight_decay)
temp_s4 = ENC.LearnableTemperature(init_value=CFG.learnable_temperature_init).to(DEVICE)
opt_s4.add_param_group({"params": temp_s4.parameters()})

ecg_proto_t = torch.tensor(ecg_proto_s3["assignments"], dtype=torch.long, device=DEVICE)
cxr_proto_t = torch.tensor(cxr_proto_s3["assignments"], dtype=torch.long, device=DEVICE)
transport_t = torch.tensor(transport_plan_s4, dtype=torch.float32, device=DEVICE)

stage4_loss_history = []
for epoch in range(STAGE4_EPOCHS):
    epoch_losses = []
    for step in range(STAGE4_STEPS_PER_EPOCH):
        xb_e, _, xb_c, _ = sample_bipartite_batch(CFG.batch_size, seed=epoch * STAGE4_STEPS_PER_EPOCH + step)
        # Get prototype assignments for batch samples
        g = torch.Generator().manual_seed(epoch * STAGE4_STEPS_PER_EPOCH + step)
        idx_e = torch.randint(0, ecg_feat_t.shape[0], (CFG.batch_size,), generator=g)
        idx_c = torch.randint(0, cxr_feat_t.shape[0], (CFG.batch_size,), generator=g)
        pb_e = ecg_proto_t[idx_e]
        pb_c = cxr_proto_t[idx_c]

        opt_s4.zero_grad(set_to_none=True)
        z_e = ecg_head_s4(xb_e)
        z_c = cxr_head_s4(xb_c)
        loss = L.prototype_contrastive_loss(z_e, z_c, pb_e, pb_c, transport_t, temperature=temp_s4())
        loss.backward()
        opt_s4.step()
        epoch_losses.append(loss.item())
    mean_loss = float(np.mean(epoch_losses))
    stage4_loss_history.append(mean_loss)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"epoch {epoch + 1:3d}/{STAGE4_EPOCHS}  loss={mean_loss:.4f}  temp={temp_s4().item():.4f}")
""")

code(r"""
# 4.4 -- Checkpointing Stage 4 heads
torch.save(ecg_head_s4.state_dict(), C.OT_DIR / "heads" / "ECGAlignHead_v4.pt")
torch.save(cxr_head_s4.state_dict(), C.OT_DIR / "heads" / "CXRAlignHead_v4.pt")
torch.save(opt_s4.state_dict(), C.OT_DIR / "heads" / "optimizer_v4.pt")
print("Stage 4 heads checkpointed.")
""")

code(r"""
# 4.5 -- Internal retrieval@K evaluation for Stage 4 (same-dataset sanity check only)
ecg_head_s4.eval(); cxr_head_s4.eval()
with torch.no_grad():
    z_ecg_val_s4 = ecg_head_s4(torch.tensor(ecg_val_cache["features"], dtype=torch.float32).to(DEVICE)).cpu().numpy()
    z_cxr_val_s4 = cxr_head_s4(torch.tensor(cxr_val_cache["features"], dtype=torch.float32).to(DEVICE)).cpu().numpy()

sim_s4 = EVAL.cosine_similarity_matrix(z_ecg_val_s4, z_cxr_val_s4)
stage4_precision_at_k = {
    k: EVAL.concept_precision_at_k(ecg_bins_val, cxr_bins_val, sim_s4, k=k) for k in CFG.top_k_retrieval
}
print("Stage 4 (Prototype OT with Jaccard cost) -- internal Concept-Precision@K:")
for k, v in stage4_precision_at_k.items():
    print(f"  @{k}: {v:.4f}")

with open(C.OT_DIR / "metrics" / "internal_retrieval_at_k.json", "w") as f:
    json.dump({f"precision_at_{k}": v for k, v in stage4_precision_at_k.items()}, f, indent=2)
""")

code(r"""
# 4.6 -- Visualizations: cost matrix heatmap, loss curve, joint UMAP, and bar comparison
fig_cost = VIZ.plot_matrix_heatmap(cost_matrix_s4, "Stage 4: Assumed Cross-Modal Cost Matrix (Jaccard)")
fig_cost.savefig(C.OT_DIR / "figures" / "cost_matrix_s4.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_cost)

fig_loss_s4 = VIZ.plot_loss_curve(stage4_loss_history, "Stage 4: Prototype Contrastive OT Training Loss")
fig_loss_s4.savefig(C.OT_DIR / "figures" / "loss_curve_stage4.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_loss_s4)

fig_bar_s4 = VIZ.plot_grouped_bar(
    {
        "Stage 1 (circular)": {f"@{k}": v for k, v in circular_concept_precision_at_k.items()},
        "Stage 2 (CCA)": {f"@{k}": v for k, v in cca_precision_at_k.items()},
        "Stage 4 (OT w/ assumed cost)": {f"@{k}": v for k, v in stage4_precision_at_k.items()},
    },
    title="Stage 4 vs Previous Stages: Internal Concept-Precision@K",
    ylabel="Concept-Precision@K",
)
fig_bar_s4.savefig(C.OT_DIR / "figures" / "retrieval_bar_vs_previous.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_bar_s4)
""")

code(r"""
# 4.7 -- MLflow logging
with MLU.stage_run(run_name="stage4_prototype_ot", stage="stage4_prototype_ot",
                    encoder_variant=C.PRIMARY_ENCODER_VARIANT, data_source="ptbxl_chexpert") as run:
    mlflow.log_params({"epochs": STAGE4_EPOCHS, "primary_k": PRIMARY_K, "batch_size": CFG.batch_size, "lr": CFG.lr})
    mlflow.log_metrics({"final_train_loss": stage4_loss_history[-1]})
    for k, v in stage4_precision_at_k.items():
        mlflow.log_metric(f"stage4_internal_precision_at_{k}", v)
    mlflow.log_artifact(str(C.OT_DIR / "metrics" / "internal_retrieval_at_k.json"), artifact_path="metrics")
    for fig_path in (C.OT_DIR / "figures").glob("*.png"):
        mlflow.log_artifact(str(fig_path), artifact_path="figures")
    MLU.log_large_array_path("ecg_head_checkpoint", C.OT_DIR / "heads" / "ECGAlignHead_v4.pt")
    MLU.log_large_array_path("cxr_head_checkpoint", C.OT_DIR / "heads" / "CXRAlignHead_v4.pt")
    print(f"Logged MLflow run: {run.info.run_id}")
""")


# ============================================================================
# STAGE 5 -- GROMOV-WASSERSTEIN ALIGNMENT (FINAL METHOD)
# ============================================================================

md(r"""
## Stage 5 &mdash; Gromov-Wasserstein Alignment (FINAL METHOD)

**Goal.** Fully unsupervised cross-modal alignment using **Gromov-Wasserstein (GW) optimal transport**.
Unlike Stage 4, this stage assumes **zero cross-modal cost matrix**, zero shared patients, and zero
manual concept correspondence during alignment.

**Mechanism.** Gromov-Wasserstein matches the two modality spaces purely by comparing their internal,
within-modality geometric structures (the pairwise distances between ECG prototypes vs the pairwise
distances between CXR prototypes). If a cluster of hypertrophy ECG heart beats relates to normal rhythms in
the same relative geometry that cardiomegaly CXRs relate to normal chests, GW discovers and aligns that
mapping automatically.

**Hypothesis.** GW alignment matches or closely approaches the performance of Stage 4 (which "cheats" by
using the manual concept table for cost estimation), proving that independent medical representation
spaces natively share robust topological structure across modalities.
""")

code(r"""
# 5.1 -- Compute within-modality prototype distance matrices (RDMs) and solve Gromov-Wasserstein transport plan
ecg_centroids = ecg_proto_s3["centroids"]
cxr_centroids = cxr_proto_s3["centroids"]

gw_transport_plan, gw_cost = GW.compute_gw_alignment(ecg_centroids, cxr_centroids, epsilon=CFG.gw_epsilon, seed=CFG.seed)
np.savez(C.GW_DIR / "weights" / "gw_transport_plan.npz", transport_plan=gw_transport_plan, gw_cost=gw_cost)
print(f"Gromov-Wasserstein alignment solved! Optimal GW cost: {gw_cost:.6f}")
print(f"Transport plan shape: {gw_transport_plan.shape} (sum={gw_transport_plan.sum():.4f})")
""")

code(r"""
# 5.2 -- Initialize final Stage 5 alignment heads & train via GW coupling matrix
ecg_head_s5 = ENC.AlignHead(in_dim=C.ECG_FEATURE_DIM, out_dim=CFG.prototype_head_dim, hidden_dim=256).to(DEVICE)
cxr_head_s5 = ENC.AlignHead(in_dim=C.CXR_FEATURE_DIM, out_dim=CFG.prototype_head_dim, hidden_dim=512).to(DEVICE)

STAGE5_EPOCHS = 40
STAGE5_STEPS_PER_EPOCH = 50
opt_s5 = torch.optim.AdamW(list(ecg_head_s5.parameters()) + list(cxr_head_s5.parameters()),
                            lr=CFG.lr, weight_decay=CFG.weight_decay)
temp_s5 = ENC.LearnableTemperature(init_value=CFG.learnable_temperature_init).to(DEVICE)
opt_s5.add_param_group({"params": temp_s5.parameters()})

gw_transport_t = torch.tensor(gw_transport_plan, dtype=torch.float32, device=DEVICE)

stage5_loss_history = []
for epoch in range(STAGE5_EPOCHS):
    epoch_losses = []
    for step in range(STAGE5_STEPS_PER_EPOCH):
        xb_e, _, xb_c, _ = sample_bipartite_batch(CFG.batch_size, seed=epoch * STAGE5_STEPS_PER_EPOCH + step)
        g = torch.Generator().manual_seed(epoch * STAGE5_STEPS_PER_EPOCH + step)
        idx_e = torch.randint(0, ecg_feat_t.shape[0], (CFG.batch_size,), generator=g)
        idx_c = torch.randint(0, cxr_feat_t.shape[0], (CFG.batch_size,), generator=g)
        pb_e = ecg_proto_t[idx_e]
        pb_c = cxr_proto_t[idx_c]

        opt_s5.zero_grad(set_to_none=True)
        z_e = ecg_head_s5(xb_e)
        z_c = cxr_head_s5(xb_c)
        loss = L.gw_contrastive_loss(z_e, z_c, pb_e, pb_c, gw_transport_t, temperature=temp_s5())
        loss.backward()
        opt_s5.step()
        epoch_losses.append(loss.item())
    mean_loss = float(np.mean(epoch_losses))
    stage5_loss_history.append(mean_loss)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"epoch {epoch + 1:3d}/{STAGE5_EPOCHS}  loss={mean_loss:.4f}  temp={temp_s5().item():.4f}")
""")

code(r"""
# 5.3 -- Checkpointing final Stage 5 heads
torch.save(ecg_head_s5.state_dict(), C.GW_DIR / "heads" / "ECGAlignHead_v5.pt")
torch.save(cxr_head_s5.state_dict(), C.GW_DIR / "heads" / "CXRAlignHead_v5.pt")
torch.save(opt_s5.state_dict(), C.GW_DIR / "heads" / "optimizer_v5.pt")
print("Stage 5 (FINAL METHOD) heads checkpointed.")
""")

code(r"""
# 5.4 -- Internal retrieval@K evaluation on Val and Test splits
ecg_head_s5.eval(); cxr_head_s5.eval()
with torch.no_grad():
    z_ecg_val_s5 = ecg_head_s5(torch.tensor(ecg_val_cache["features"], dtype=torch.float32).to(DEVICE)).cpu().numpy()
    z_cxr_val_s5 = cxr_head_s5(torch.tensor(cxr_val_cache["features"], dtype=torch.float32).to(DEVICE)).cpu().numpy()

sim_s5_val = EVAL.cosine_similarity_matrix(z_ecg_val_s5, z_cxr_val_s5)
stage5_precision_val = {
    k: EVAL.concept_precision_at_k(ecg_bins_val, cxr_bins_val, sim_s5_val, k=k) for k in CFG.top_k_retrieval
}

# Evaluate on test slice
ecg_test_cache = ecg_embedding_cache[(C.PRIMARY_ENCODER_VARIANT, "test")]
cxr_test_cache = CACHE.load_cached_embeddings(CACHE.embedding_cache_path(C.EMBEDDINGS_DIR, "cxr", C.PRIMARY_ENCODER_VARIANT, "test"))
ecg_bins_test = multihot_bins(ecg_test_cache["labels"], C.ECG_SUPERCLASSES, "ecg").argmax(axis=1)
cxr_bins_test = multihot_bins(cxr_test_cache["labels"], C.CXR_LABEL_COLS, "cxr").argmax(axis=1)

with torch.no_grad():
    z_ecg_test_s5 = ecg_head_s5(torch.tensor(ecg_test_cache["features"], dtype=torch.float32).to(DEVICE)).cpu().numpy()
    z_cxr_test_s5 = cxr_head_s5(torch.tensor(cxr_test_cache["features"], dtype=torch.float32).to(DEVICE)).cpu().numpy()

sim_s5_test = EVAL.cosine_similarity_matrix(z_ecg_test_s5, z_cxr_test_s5)
stage5_precision_test = {
    k: EVAL.concept_precision_at_k(ecg_bins_test, cxr_bins_test, sim_s5_test, k=k) for k in CFG.top_k_retrieval
}

print("Stage 5 (Gromov-Wasserstein FINAL METHOD) -- Internal Concept-Precision@K:")
print("  VAL SPLIT:")
for k, v in stage5_precision_val.items():
    print(f"    @{k}: {v:.4f}")
print("  TEST SPLIT:")
for k, v in stage5_precision_test.items():
    print(f"    @{k}: {v:.4f}")

stage5_metrics = {"val": stage5_precision_val, "test": stage5_precision_test, "gw_cost": float(gw_cost)}
with open(C.GW_DIR / "metrics" / "gw_retrieval_at_k.json", "w") as f:
    json.dump(stage5_metrics, f, indent=2)
""")

code(r"""
# 5.5 -- Visualizations: GW transport plan vs Stage 4 assumed cost, loss curve, and benchmark bar chart
fig_plan = VIZ.plot_matrix_heatmap(gw_transport_plan, "Stage 5: Unsupervised Gromov-Wasserstein Coupling Matrix")
fig_plan.savefig(C.GW_DIR / "figures" / "gw_transport_plan.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_plan)

fig_loss_s5 = VIZ.plot_loss_curve(stage5_loss_history, "Stage 5: Gromov-Wasserstein Alignment Training Loss")
fig_loss_s5.savefig(C.GW_DIR / "figures" / "loss_curve_stage5.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_loss_s5)

fig_bar_s5 = VIZ.plot_grouped_bar(
    {
        "Stage 2 (CCA)": {f"@{k}": v for k, v in cca_precision_at_k.items()},
        "Stage 4 (OT w/ assumed cost)": {f"@{k}": v for k, v in stage4_precision_at_k.items()},
        "Stage 5 (GW Unsupervised - Val)": {f"@{k}": v for k, v in stage5_precision_val.items()},
        "Stage 5 (GW Unsupervised - Test)": {f"@{k}": v for k, v in stage5_precision_test.items()},
    },
    title="Headline Method Comparison: Internal Concept-Precision@K",
    ylabel="Concept-Precision@K",
)
fig_bar_s5.savefig(C.GW_DIR / "figures" / "retrieval_bar_headline_comparison.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_bar_s5)
""")

code(r"""
# 5.6 -- MLflow logging
with MLU.stage_run(run_name="stage5_gw_alignment", stage="stage5_gw_alignment",
                    encoder_variant=C.PRIMARY_ENCODER_VARIANT, data_source="ptbxl_chexpert") as run:
    mlflow.log_params({"epochs": STAGE5_EPOCHS, "primary_k": PRIMARY_K, "gw_epsilon": CFG.gw_epsilon, "lr": CFG.lr})
    mlflow.log_metric("gw_optimal_cost", float(gw_cost))
    for k, v in stage5_precision_val.items():
        mlflow.log_metric(f"stage5_val_precision_at_{k}", v)
    for k, v in stage5_precision_test.items():
        mlflow.log_metric(f"stage5_test_precision_at_{k}", v)
    mlflow.log_artifact(str(C.GW_DIR / "metrics" / "gw_retrieval_at_k.json"), artifact_path="metrics")
    for fig_path in (C.GW_DIR / "figures").glob("*.png"):
        mlflow.log_artifact(str(fig_path), artifact_path="figures")
    MLU.log_large_array_path("ecg_head_checkpoint", C.GW_DIR / "heads" / "ECGAlignHead_v5.pt")
    MLU.log_large_array_path("cxr_head_checkpoint", C.GW_DIR / "heads" / "CXRAlignHead_v5.pt")
    print(f"Logged MLflow run: {run.info.run_id}")
""")


# ============================================================================
# STAGE 6 -- ABLATIONS
# ============================================================================

md(r"""
## Stage 6 &mdash; Ablation Studies

**Goal.** Systematically evaluate the sensitivity and robustness of the Stage 5 Gromov-Wasserstein
alignment method across key structural dimensions:
1. **Backbone Objective**: MultiSupCon (primary) vs SimCLR (pure instance contrastive).
2. **Projection Head Dimensionality**: Comparing `d_shared` &in; {128, 256, 512}.
3. **Number of Prototypes (K)**: Sensitivity of GW coupling to cluster granularity across `CFG.prototype_k_grid`.
4. **Conditional SwAV/DeepCluster Escalation**: Executed only if triggered in Stage 3.
""")

code(r"""
# 6.1 -- Run ablation grid across backbone variants and head dimensions
ablation_results = {}

for variant in C.ENCODER_VARIANTS:
    for head_dim in [128, 256, 512]:
        key = f"variant_{variant}__dim_{head_dim}"
        print(f"Running ablation: {key}...")
        res = ABL.run_gw_ablation(
            ecg_embedding_cache[(variant, "train")], CACHE.load_cached_embeddings(CACHE.embedding_cache_path(C.EMBEDDINGS_DIR, "cxr", variant, "train")),
            ecg_embedding_cache[(variant, "val")], CACHE.load_cached_embeddings(CACHE.embedding_cache_path(C.EMBEDDINGS_DIR, "cxr", variant, "val")),
            head_dim=head_dim, k=PRIMARY_K, epochs=15, device=DEVICE, seed=CFG.seed,
        )
        ablation_results[key] = res
        print(f"  -> Precision@5: {res['precision_at_5']:.4f} | Precision@10: {res['precision_at_10']:.4f}")

# Check conditional escalation from Stage 3
if SWAV_ESCALATION_TRIGGERED:
    print("\nExecuting Conditional SwAV/DeepCluster Ablation (#6)...")
    swav_res = ABL.run_swav_gw_ablation(
        ecg_embedding_cache[(C.PRIMARY_ENCODER_VARIANT, "train")],
        CACHE.load_cached_embeddings(CACHE.embedding_cache_path(C.EMBEDDINGS_DIR, "cxr", C.PRIMARY_ENCODER_VARIANT, "train")),
        ecg_embedding_cache[(C.PRIMARY_ENCODER_VARIANT, "val")],
        CACHE.load_cached_embeddings(CACHE.embedding_cache_path(C.EMBEDDINGS_DIR, "cxr", C.PRIMARY_ENCODER_VARIANT, "val")),
        k=PRIMARY_K, device=DEVICE, seed=CFG.seed,
    )
    ablation_results["swav_escalation"] = swav_res
else:
    print("\nSkipping SwAV/DeepCluster Ablation (#6) since k-means cleared quality thresholds in Stage 3.")

with open(C.ABLATIONS_DIR / "metrics" / "ablation_results.json", "w") as f:
    json.dump(ablation_results, f, indent=2)
""")

code(r"""
# 6.2 -- Visualization: Ablation comparison chart
ablation_bar_data = {
    k: {f"@{metric.split('_')[-1]}": val for metric, val in v.items() if metric.startswith("precision_at")}
    for k, v in ablation_results.items()
}
fig_abl = VIZ.plot_grouped_bar(ablation_bar_data, title="Stage 6: Ablations Concept-Precision@K", ylabel="Concept-Precision@K")
fig_abl.savefig(C.ABLATIONS_DIR / "figures" / "ablations_comparison.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_abl)
""")

code(r"""
# 6.3 -- MLflow logging for Stage 6 ablations
with MLU.stage_run(run_name="stage6_ablations", stage="stage6_ablations", data_source="ptbxl_chexpert") as parent_run:
    for ab_key, res in ablation_results.items():
        with MLU.stage_run(run_name=f"stage6_{ab_key}", stage="stage6_ablations", nested=True, extra_tags={"ablation": ab_key}):
            for m_k, m_v in res.items():
                if isinstance(m_v, (int, float)):
                    mlflow.log_metric(m_k, m_v)
    mlflow.log_artifact(str(C.ABLATIONS_DIR / "metrics" / "ablation_results.json"), artifact_path="metrics")
    mlflow.log_artifact(str(C.ABLATIONS_DIR / "figures" / "ablations_comparison.png"), artifact_path="figures")
    print(f"Logged MLflow parent run: {parent_run.info.run_id}")
""")


# ============================================================================
# STAGE 7 -- EXTERNAL VALIDATION
# ============================================================================

md(r"""
## Stage 7 &mdash; External Validation on MIMIC-CXR / External Clinical Cohorts

**Goal.** Non-circular, independent external validation. This is the **only** stage that touches external
clinical evaluation sets (physically isolated in `external_validation/`). Upstream stages trained on PTB-XL
and CheXpert never read from this directory.

**Method.** The frozen Stage 5 Gromov-Wasserstein alignment heads (`ECGAlignHead_v5` and `CXRAlignHead_v5`)
are applied **zero-shot** to external representations to test whether the unsupervised topological mapping
actually learned generalizable cross-modal medical physiology, rather than dataset-specific artifact co-occurrence.
""")

code(r"""
# 7.1 -- Load isolated external validation data and evaluate Stage 5 GW heads zero-shot
print("Loading physically isolated external validation cohort from external_validation/...")
external_cache = EXTV.load_external_validation_data(PROJECT_ROOT / "external_validation", device=DEVICE)

ecg_head_s5.eval(); cxr_head_s5.eval()
with torch.no_grad():
    z_ecg_ext = ecg_head_s5(external_cache["ecg_features"].to(DEVICE)).cpu().numpy()
    z_cxr_ext = cxr_head_s5(external_cache["cxr_features"].to(DEVICE)).cpu().numpy()

sim_external = EVAL.cosine_similarity_matrix(z_ecg_ext, z_cxr_ext)
external_precision = {
    k: EVAL.concept_precision_at_k(external_cache["ecg_bins"], external_cache["cxr_bins"], sim_external, k=k)
    for k in CFG.top_k_retrieval
}
external_auroc = EVAL.compute_retrieval_auroc(external_cache["ecg_bins"], external_cache["cxr_bins"], sim_external)

print("\n=== STAGE 7 HEADLINE EXTERNAL VALIDATION RESULTS ===")
print(f"External Zero-Shot Retrieval AUROC: {external_auroc:.4f}")
for k, v in external_precision.items():
    print(f"  External Concept-Precision@{k}: {v:.4f}")

external_results = {"external_auroc": float(external_auroc), "external_precision_at_k": external_precision}
with open(C.EXTERNAL_VAL_DIR / "metrics" / "external_mimic_results.json", "w") as f:
    json.dump(external_results, f, indent=2)
""")

code(r"""
# 7.2 -- Visualization: Generalization Gap (Internal Test vs External Validation)
fig_ext = VIZ.plot_grouped_bar(
    {
        "Internal Test (PTB-XL/CheXpert)": {f"@{k}": v for k, v in stage5_precision_test.items()},
        "External Zero-Shot Validation": {f"@{k}": v for k, v in external_precision.items()},
    },
    title="Headline Result: Internal Test vs External Validation Retrieval",
    ylabel="Concept-Precision@K",
)
fig_ext.savefig(C.EXTERNAL_VAL_DIR / "figures" / "external_generalization_gap.png", dpi=150, bbox_inches="tight")
plt.show(); plt.close(fig_ext)
""")

code(r"""
# 7.3 -- Final MLflow logging & Summary Report
with MLU.stage_run(run_name="stage7_external_validation", stage="stage7_external_validation",
                    encoder_variant=C.PRIMARY_ENCODER_VARIANT, data_source="external_cohort") as run:
    mlflow.log_metric("external_auroc", float(external_auroc))
    for k, v in external_precision.items():
        mlflow.log_metric(f"external_precision_at_{k}", v)
    mlflow.log_artifact(str(C.EXTERNAL_VAL_DIR / "metrics" / "external_mimic_results.json"), artifact_path="metrics")
    mlflow.log_artifact(str(C.EXTERNAL_VAL_DIR / "figures" / "external_generalization_gap.png"), artifact_path="figures")
    print(f"Logged MLflow run: {run.info.run_id}")

print("\n" + "="*70)
print("ALIGNED MASTER NOTEBOOK EXECUTION COMPLETE.")
print("="*70)
print(f"Primary Encoder Variant: {C.PRIMARY_ENCODER_VARIANT}")
print(f"Selected Prototype K   : {PRIMARY_K}")
print(f"Stage 5 Internal Test  : Precision@5 = {stage5_precision_test.get(5, 0.0):.4f}")
print(f"Stage 7 External Val   : Precision@5 = {external_precision.get(5, 0.0):.4f} | AUROC = {external_auroc:.4f}")
print("All artifacts, checkpoints, metrics, and MLflow records successfully archived.")
""")


# ============================================================================
# EXPORT TO RAW NBFORMAT-V4 JSON
# ============================================================================

out_dir = Path("notebooks")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "00_alignment_master.ipynb"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Successfully compiled and saved {len(cells)} cells directly to {out_path} without intermediate part files.")
