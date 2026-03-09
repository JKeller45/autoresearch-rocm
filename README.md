# autoresearch

> This repository is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The sole purpose of this fork is native support for NVIDIA RTX 3080 (10 GB) GPUs on Windows machines.

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## Fork scope

- Upstream source: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- Primary objective: run natively on Windows with consumer NVIDIA GPUs, specifically RTX 3080 (10 GB), without unofficial Triton-on-Windows stacks.
- Scope of changes: compatibility and stability updates required for that target platform.
- Linux/H100-oriented fast paths are kept where practical, but they are not the focus of this fork.

## How it works

The repo is deliberately kept small and only really has a three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

## Quick start

**Requirements:** A single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

- Linux fast path (FA3 + `torch.compile` when available) remains supported.
- Native Windows support targets consumer GPUs (e.g. RTX 3080 10 GB) with official PyTorch CUDA wheels and SDPA fallback.
- Default dataset is now TinyStories GPT-4 clean for practical consumer-GPU setup.

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time)
#    Default dataset: TinyStories GPT-4 clean
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

Use climbmix explicitly if you want the old large-dataset workflow:

```bash
uv run prepare.py --dataset climbmix --num-shards 10
```

Quick validation run (recommended after setup):

```bash
uv run train.py --smoke-test
```

### Windows setup (PowerShell)

```powershell
# 1. Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Install dependencies
uv sync

# 3. Prepare TinyStories + tokenizer
uv run prepare.py

# 4. Smoke test
uv run train.py --smoke-test
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This fork's platform policy is intentionally narrow and explicit.

- Primary supported platform is Windows 10/11 with a single NVIDIA RTX 3080 (10 GB) using official PyTorch CUDA wheels.
- Windows runtime path uses PyTorch SDPA attention, avoids a hard dependency on Linux-centric HF `kernels`, and keeps `torch.compile` disabled by default.
- Memory behavior is tuned for 10 GB-class cards so training can continue through OOM pressure using fallback logic.
- Linux CUDA remains supported as a compatibility path and can still use FA3 plus `torch.compile` fast paths when available.
- Non-goals for this fork include unofficial Triton-for-Windows setups, AMD/ROCm, Apple Metal, and multi-GPU training.
- Default dataset is `karpathy/tinystories-gpt4-clean` for consumer-GPU practicality.
- `climbmix` remains available by explicit override (`uv run prepare.py --dataset climbmix --num-shards 10`).
- Dataset changes reset metric comparability: TinyStories runs should not be compared directly to historical climbmix baselines.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx)

## License

MIT
