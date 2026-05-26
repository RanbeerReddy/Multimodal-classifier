# AI Agent Instructions for ECG Project

## Purpose
This repository is a Python research project for ECG representation learning and multimodal classification. Use this file to understand the project structure, development commands, and where to focus when writing or reviewing code.

## Key Commands
- Install dependencies: `pip install -r requirements.txt`
- Run tests: `pytest`
- CI uses `.github/workflows/ci.yml`

## Project Structure
- `Multimodal classifier/` - main model code, data pipeline, training loops, and losses.
  - `Multimodal classifier/models/ecg models/` - ECG encoder, SimCLR wrapper, classifier, projection head.
  - `Multimodal classifier/data/Ecg data/` - data loading, augmentation, transforms, and split logic.
  - `Multimodal classifier/training/ecg training/` - SSL training and encoder training code.
  - `Multimodal classifier/losses/` - contrastive loss implementations.
- `configs/dataset/ecgdata_config.py` - dataset configuration and PTB-XL metadata settings.
- `checkpoints/` - saved model weights and checkpoint organization.
- `data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/` - PTB-XL dataset files.
- `tests/` - unit tests for datasets, models, losses, training, and evaluation.
- `plan.md` - architecture and design notes; use for high-level context.

## Important Notes for AI Agents
- The repository uses a non-standard directory name with a space: `Multimodal classifier/`. Preserve this exact path when editing or importing.
- There is no package manifest (`setup.py`/`pyproject.toml`), so prefer running from the repository root and using direct module imports.
- `README.md` is currently empty; do not rely on it for project behavior.
- The CI workflow is simple: install `requirements.txt` and run `pytest`. Keep changes compatible with this workflow.

## Recommended Focus
- Use `config/dataset/ecgdata_config.py` and `Multimodal classifier/data/Ecg data/` for dataset and preprocessing changes.
- Use `Multimodal classifier/models/ecg models/` for encoder and classifier architecture changes.
- Use `tests/` to add or update coverage for any new behavior.

## Guidance
- Keep changes minimal and aligned with existing model/data/training patterns.
- Document high-level design changes in `plan.md` rather than in source code comments when appropriate.
- If adding new functionality, also add or update tests under `tests/`.
