"""
One-time data preparation for autoresearch experiments.
Downloads data and trains a BPE tokenizer.

Usage:
    python prepare.py
    python prepare.py --dataset climbmix --num-shards 8

Data and tokenizer are stored in the cache directory (overridable with
AUTORESEARCH_CACHE_DIR). The active dataset can be pinned with
AUTORESEARCH_DATASET or by running this script with --dataset.
"""

import argparse
import math
import os
import pickle
import sys
import time
from multiprocessing import Pool

import pyarrow.parquet as pq
import requests
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048          # context length
TIME_BUDGET = 300           # training time budget in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288   # number of tokens for validation eval
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Dataset + cache configuration
# ---------------------------------------------------------------------------

DEFAULT_DATASET = "tinystories"
DATASET_CHOICES = ("tinystories", "climbmix")


def _default_cache_dir():
    env_cache = os.environ.get("AUTORESEARCH_CACHE_DIR")
    if env_cache:
        return os.path.expanduser(env_cache)

    legacy_cache = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
    if os.name != "nt":
        return legacy_cache

    if os.path.exists(legacy_cache):
        return legacy_cache

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return os.path.join(local_app_data, "autoresearch")
    return legacy_cache


CACHE_DIR = _default_cache_dir()
DATASETS_DIR = os.path.join(CACHE_DIR, "datasets")
ACTIVE_DATASET_PATH = os.path.join(CACHE_DIR, "active_dataset.txt")

DATASET_CONFIGS = {
    "tinystories": {
        "filename": "tinystories_gpt4-clean.parquet",
        "url": "https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean/resolve/main/tinystories_gpt4-clean.parquet",
        "splits": {
            "test": (0, 10_000),
            "val": (10_000, 20_000),
            "train": (20_000, None),
        },
    },
    "climbmix": {
        "base_url": "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main",
        "max_shard": 6542,
        "val_shard": 6542,
    },
}


def _normalize_dataset_name(dataset_name):
    if dataset_name is None:
        return None
    value = dataset_name.strip().lower()
    if value not in DATASET_CHOICES:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Expected one of {DATASET_CHOICES}.")
    return value


def _load_active_dataset_from_file():
    if not os.path.exists(ACTIVE_DATASET_PATH):
        return None
    with open(ACTIVE_DATASET_PATH, "r", encoding="utf-8") as f:
        value = f.read().strip().lower()
    if value in DATASET_CHOICES:
        return value
    return None


def _resolve_dataset_name(dataset_name=None):
    normalized = _normalize_dataset_name(dataset_name)
    if normalized is not None:
        return normalized

    env_dataset = _normalize_dataset_name(os.environ.get("AUTORESEARCH_DATASET"))
    if env_dataset is not None:
        return env_dataset

    file_dataset = _load_active_dataset_from_file()
    if file_dataset is not None:
        return file_dataset

    return DEFAULT_DATASET


def _set_active_dataset(dataset_name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(ACTIVE_DATASET_PATH, "w", encoding="utf-8") as f:
        f.write(dataset_name + "\n")


def _dataset_root(dataset_name=None):
    dataset = _resolve_dataset_name(dataset_name)
    return os.path.join(DATASETS_DIR, dataset)


def _data_dir(dataset_name=None):
    return os.path.join(_dataset_root(dataset_name), "data")


def _tokenizer_dir(dataset_name=None):
    return os.path.join(_dataset_root(dataset_name), "tokenizer")


def _tiny_parquet_path(dataset_name=None):
    dataset = _resolve_dataset_name(dataset_name)
    config = DATASET_CONFIGS[dataset]
    return os.path.join(_data_dir(dataset), config["filename"])


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def _download_climbmix_single_shard(task):
    index, data_dir, base_url = task
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        return True

    url = f"{base_url}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, OSError) as exc:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {exc}")
            for path in (temp_path if "temp_path" in locals() else filepath + ".tmp", filepath):
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def _download_tinystories_file(dataset_name):
    config = DATASET_CONFIGS[dataset_name]
    data_dir = _data_dir(dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    filename = config["filename"]
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f"Data: {filename} already downloaded at {data_dir}")
        return

    url = config["url"]
    print(f"Data: downloading {filename}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    temp_path = filepath + ".tmp"
    with open(temp_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    os.rename(temp_path, filepath)
    print(f"Data: downloaded {filename} to {filepath}")


def _download_climbmix_data(dataset_name, num_shards, download_workers):
    config = DATASET_CONFIGS[dataset_name]
    data_dir = _data_dir(dataset_name)
    os.makedirs(data_dir, exist_ok=True)

    max_shard = config["max_shard"]
    val_shard = config["val_shard"]
    base_url = config["base_url"]

    num_train = min(num_shards, max_shard)
    shard_ids = list(range(num_train))
    if val_shard not in shard_ids:
        shard_ids.append(val_shard)

    existing = sum(
        1
        for shard_idx in shard_ids
        if os.path.exists(os.path.join(data_dir, f"shard_{shard_idx:05d}.parquet"))
    )
    if existing == len(shard_ids):
        print(f"Data: all {len(shard_ids)} shards already downloaded at {data_dir}")
        return

    needed = len(shard_ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")
    workers = max(1, min(download_workers, needed))
    tasks = [(idx, data_dir, base_url) for idx in shard_ids]
    with Pool(processes=workers) as pool:
        results = pool.map(_download_climbmix_single_shard, tasks)
    ready = sum(1 for ok in results if ok)
    print(f"Data: {ready}/{len(shard_ids)} shards ready at {data_dir}")


def download_data(dataset_name, num_shards, download_workers=8):
    dataset = _resolve_dataset_name(dataset_name)
    if dataset == "tinystories":
        _download_tinystories_file(dataset)
        return
    _download_climbmix_data(dataset, num_shards=num_shards, download_workers=download_workers)


# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files(dataset_name=None):
    dataset = _resolve_dataset_name(dataset_name)
    data_dir = _data_dir(dataset)
    if not os.path.exists(data_dir):
        return []
    files = sorted(
        name for name in os.listdir(data_dir)
        if name.endswith(".parquet") and not name.endswith(".tmp")
    )
    return [os.path.join(data_dir, name) for name in files]


def _iter_tinystories_texts(split, dataset_name=None):
    dataset = _resolve_dataset_name(dataset_name)
    config = DATASET_CONFIGS[dataset]
    start_idx, end_idx = config["splits"][split]
    tiny_path = _tiny_parquet_path(dataset)

    if not os.path.exists(tiny_path):
        raise FileNotFoundError(
            f"TinyStories parquet not found at {tiny_path}. Run prepare.py first."
        )

    current_idx = 0
    parquet_file = pq.ParquetFile(tiny_path)
    for row_group_idx in range(parquet_file.num_row_groups):
        row_group = parquet_file.read_row_group(row_group_idx, columns=["text"])
        texts = row_group.column("text").to_pylist()
        for text in texts:
            if current_idx < start_idx:
                current_idx += 1
                continue
            if end_idx is not None and current_idx >= end_idx:
                return
            yield text
            current_idx += 1


def text_iterator(dataset_name=None, max_chars=1_000_000_000, doc_cap=10_000):
    dataset = _resolve_dataset_name(dataset_name)
    chars = 0

    if dataset == "tinystories":
        text_iter = _iter_tinystories_texts("train", dataset_name=dataset)
        for text in text_iter:
            doc = text[:doc_cap] if len(text) > doc_cap else text
            chars += len(doc)
            yield doc
            if chars >= max_chars:
                return
        return

    config = DATASET_CONFIGS[dataset]
    val_filename = f"shard_{config['val_shard']:05d}.parquet"
    parquet_paths = [p for p in list_parquet_files(dataset) if not p.endswith(val_filename)]
    for filepath in parquet_paths:
        parquet_file = pq.ParquetFile(filepath)
        for row_group_idx in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(row_group_idx, columns=["text"])
            for text in row_group.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                chars += len(doc)
                yield doc
                if chars >= max_chars:
                    return


def train_tokenizer(dataset_name=None):
    dataset = _resolve_dataset_name(dataset_name)
    tokenizer_dir = _tokenizer_dir(dataset)
    tokenizer_pkl = os.path.join(tokenizer_dir, "tokenizer.pkl")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")

    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {tokenizer_dir}")
        return

    os.makedirs(tokenizer_dir, exist_ok=True)

    parquet_files = list_parquet_files(dataset)
    if dataset == "climbmix" and len(parquet_files) < 2:
        print("Tokenizer: climbmix needs at least 2 shards (1 train + 1 val).")
        sys.exit(1)
    if dataset == "tinystories" and len(parquet_files) < 1:
        print("Tokenizer: TinyStories parquet is missing. Run prepare.py first.")
        sys.exit(1)

    print(f"Tokenizer: training BPE tokenizer ({dataset})...")
    t0 = time.time()
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(
        text_iterator(dataset_name=dataset),
        vocab_size_no_special,
        pattern=SPLIT_PATTERN,
    )

    pattern = tokenizer.get_pattern()
    mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
    token_offset = len(mergeable_ranks)
    special_tokens = {name: token_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )

    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)

    t1 = time.time()
    print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {token_bytes_path}")

    with open(os.path.join(tokenizer_dir, "dataset.txt"), "w", encoding="utf-8") as f:
        f.write(dataset + "\n")

    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc, dataset):
        self.enc = enc
        self.dataset = _resolve_dataset_name(dataset)
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=None, dataset=None):
        dataset_name = _resolve_dataset_name(dataset)
        resolved_dir = tokenizer_dir if tokenizer_dir is not None else _tokenizer_dir(dataset_name)
        with open(os.path.join(resolved_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc, dataset=dataset_name)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu", dataset=None):
    dataset_name = _resolve_dataset_name(dataset)
    path = os.path.join(_tokenizer_dir(dataset_name), "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, dataset=None, tokenizer_batch_size=128):
    dataset_name = _resolve_dataset_name(dataset)
    assert split in ("train", "val", "test")

    if dataset_name == "tinystories":
        epoch = 1
        while True:
            batch = []
            for text in _iter_tinystories_texts(split, dataset_name=dataset_name):
                batch.append(text)
                if len(batch) >= tokenizer_batch_size:
                    yield batch, epoch
                    batch = []
            if batch:
                yield batch, epoch
            epoch += 1
        return

    config = DATASET_CONFIGS[dataset_name]
    parquet_paths = list_parquet_files(dataset_name)
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."

    val_filename = f"shard_{config['val_shard']:05d}.parquet"
    val_path = os.path.join(_data_dir(dataset_name), val_filename)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
    elif split == "val":
        parquet_paths = [val_path]
    else:
        raise ValueError("climbmix does not define a test split")

    epoch = 1
    while True:
        for filepath in parquet_paths:
            parquet_file = pq.ParquetFile(filepath)
            for row_group_idx in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(row_group_idx, columns=["text"])
                batch = row_group.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i + tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, device="cuda", dataset=None, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    dataset_name = _resolve_dataset_name(dataset or getattr(tokenizer, "dataset", None))
    if split == "test":
        assert dataset_name == "tinystories", "Test split exists only for TinyStories."
    assert split in ("train", "val", "test")

    row_capacity = T + 1
    batches = _document_batches(split, dataset=dataset_name)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1
    resolved_device = torch.device(device)
    use_cuda = resolved_device.type == "cuda"

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)

    if use_cuda:
        gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=resolved_device)
        inputs = gpu_buffer[:B * T].view(B, T)
        targets = gpu_buffer[B * T:].view(B, T)
    else:
        gpu_buffer = None
        inputs = cpu_inputs
        targets = cpu_targets

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.as_tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.as_tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        if use_cuda:
            gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE METRIC DEFINITION)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size, device="cuda", dataset=None, eval_tokens=EVAL_TOKENS):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    """
    dataset_name = _resolve_dataset_name(dataset or getattr(tokenizer, "dataset", None))
    token_bytes = get_token_bytes(device=device, dataset=dataset_name)
    val_loader = make_dataloader(
        tokenizer,
        batch_size,
        MAX_SEQ_LEN,
        "val",
        device=device,
        dataset=dataset_name,
    )
    steps = max(1, eval_tokens // (batch_size * MAX_SEQ_LEN))
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    if total_bytes == 0:
        raise RuntimeError("Evaluation produced zero target bytes; cannot compute BPB.")
    return total_nats / (math.log(2) * total_bytes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default=None,
        help=(
            "Dataset profile to prepare. If omitted, resolves in order: "
            "AUTORESEARCH_DATASET, active_dataset.txt, then default tinystories."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=10,
        help="Number of climbmix training shards to download (-1 = all). Ignored for TinyStories.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Number of parallel download workers for climbmix.",
    )
    args = parser.parse_args()

    dataset_name = _resolve_dataset_name(args.dataset)
    num_shards = DATASET_CONFIGS["climbmix"]["max_shard"] if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Dataset: {dataset_name}")
    print()

    download_data(dataset_name, num_shards=num_shards, download_workers=args.download_workers)
    print()
    train_tokenizer(dataset_name)
    _set_active_dataset(dataset_name)
    print()
    print(f"Done! Ready to train. Active dataset is now '{dataset_name}'.")
