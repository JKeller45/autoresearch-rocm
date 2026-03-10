# autoresearch

This repo is an experiment in autonomous training research on a single AMD Radeon GPU.

## Setup

To set up a new experiment, work with the user to:

1. Agree on a run tag and create `autoresearch/<tag>` from the current main branch.
2. Read `README.md`, `prepare.py`, and `train.py` for full context.
3. Verify the environment is the supported ROCm path:
   - running inside WSL2
   - `.venv` exists
   - `uv run python -c "import torch; print(torch.cuda.is_available(), bool(torch.version.hip))"` prints `True True`
4. If the runtime is not installed, tell the human to run `./scripts/setup_rocm_wsl.sh`.
5. Verify data exists under `~/.cache/autoresearch/`. If not, tell the human to run `uv run prepare.py`.
6. Initialize `results.tsv` with the header row if it does not exist.

## Platform rules

- This fork is ROCm-on-WSL only.
- Native Windows Radeon training is out of scope.
- PyTorch ROCm still uses the `torch.cuda` API surface. Do not try to rename devices to `rocm` or `hip`.
- CUDA-only assumptions are invalid here:
  - no TF32
  - no cuDNN tuning assumptions
  - no NVIDIA compute capability logic
  - no Triton or `torch.compile` fast path
- `prepare.py` remains read-only except for repo-maintainer work. Experimentation stays in `train.py`.

## Experimentation

Each experiment runs on a single GPU and uses a fixed 5-minute training budget. Launch it with:

```bash
uv run train.py
```

Recommended validation command:

```bash
uv run train.py --smoke-test
```

What you can do:

- Modify `train.py`.
- Change architecture, optimizer behavior, hyperparameters, batch size, checkpointing strategy, and model size.

What you cannot do:

- Change the evaluation metric in `prepare.py`.
- Reintroduce NVIDIA-only runtime logic.
- Assume unsupported PyTorch features will work on ROCm without a guarded fallback.

## ROCm-specific guidance

- Keep eager execution and SDPA.
- If you introduce grouped-query attention, preserve the ROCm-safe fallback that expands K/V heads instead of relying on backend-specific `enable_gqa` support.
- Prefer simple, stable operator choices over backend-fragile optimizations.
- Use `AUTORESEARCH_FORCE_AMP_DTYPE=fp16` if BF16 behavior looks unstable on a given GPU.
- Use `AUTORESEARCH_DISABLE_PINNED_MEMORY=1` if host-to-device staging becomes unstable.

## Output format

Runs print a summary block like:

```text
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     12034.5
mfu_percent:      n/a
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Use `val_bpb` as the primary metric. Lower is better.
