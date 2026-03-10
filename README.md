# autoresearch

> Convert your gaming PC into an autonomous AI researcher.

> This repository is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). This fork targets a single **AMD Radeon GPU via ROCm on WSL2**, with PyTorch HIP using the `torch.cuda` API surface where appropriate.

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## Fork scope

- Upstream source: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- Primary objective: run on a single AMD Radeon GPU through the officially supported ROCm-on-WSL path
- Scope of changes: ROCm runtime detection, WSL packaging/setup, and conservative single-GPU stability updates
- Non-goals: native Windows Radeon training, NVIDIA/CUDA, multi-GPU, FlashAttention/Triton fast paths, `torch.compile`

## Platform policy

- Supported path: Windows host + WSL2 + Ubuntu 24.04 + Python 3.12 + AMD ROCm WSL stack
  - Tested on RX 9070XT (16Gb) on Windows 11 + WSL2 + Ubuntu 24.04 + Python 3.12 + AMD ROCm WSL stack
  - True Linux and other ROCm supported GPUs should function with minimal to minor code changes
- Native Windows Radeon PyTorch is not supported here for training (this is a ROCm limitation)
- The runtime is AMD-only and hard-fails unless PyTorch reports a HIP backend via `torch.version.hip`
- PyTorch ROCm intentionally reuses the `torch.cuda` API surface, so this repo still uses `torch.device("cuda")`, `torch.cuda.synchronize()`, and related APIs

## Project structure

```text
prepare.py                  constants, data prep + runtime utilities
train.py                    model, optimizer, training loop
program.md                  agent instructions
requirements-rocm-wsl.txt   pinned ROCm wheel URLs
scripts/setup_rocm_wsl.sh   WSL setup/bootstrap script
```

## Setup (WSL2)

Prerequisites:

1. Install the current AMD Adrenalin driver with WSL support on the Windows host.
2. Create an Ubuntu 24.04 WSL2 instance.
3. Install the system ROCm WSL components inside Ubuntu so `/opt/rocm` is present.
4. Install `uv`.

Repository setup inside WSL:

```bash
git clone <this repo>
cd autoresearch-win-radeon
./scripts/setup_rocm_wsl.sh
```

The setup script:

- creates `.venv` with Python 3.12
- installs the pure-Python project dependencies via `uv sync`
- removes any stale `torch`/`triton` packages from the venv
- installs the pinned ROCm 7.2 `torch` wheel plus the matching `pytorch_triton_rocm` wheel from `repo.radeon.com`
- patches `libhsa-runtime64.so` inside the venv when `/opt/rocm/lib/libhsa-runtime64.so.1` is available
- verifies `torch.cuda.is_available()` and `torch.version.hip`

## Quick start

```bash
uv run prepare.py
uv run train.py --smoke-test
uv run train.py
```

Expected smoke-test output includes:

- `Backend: rocm`
- `TF32: disabled`
- a valid AMP dtype (`torch.bfloat16` or `torch.float16`)

## Runtime notes

- The runtime stays on eager execution plus PyTorch SDPA.
- `torch.compile` remains disabled.
- TF32 is always disabled on ROCm.
- BF16 is used only when PyTorch reports BF16 support; otherwise FP16 is used.
- `AUTORESEARCH_FORCE_AMP_DTYPE=bf16|fp16` overrides AMP selection.
- `AUTORESEARCH_DISABLE_PINNED_MEMORY=1` disables pinned host memory for train/eval loaders.
- Autotune remains enabled for documented WSL ROCm profiles and caches the chosen batch/checkpointing pair per backend + GPU fingerprint.

## Running the agent

Point your agent at `program.md` after setup. The repo remains intentionally small: agents should only experiment in `train.py`.

## License

MIT
