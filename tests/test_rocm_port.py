import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import prepare
import train


class DummyTokenizer:
    dataset = "tinystories"

    def get_bos_token_id(self):
        return 0

    def encode(self, texts, prepend=None, num_threads=8):
        del num_threads
        rows = []
        for idx, _ in enumerate(texts, start=1):
            rows.append([prepend, idx, idx + 1, idx + 2])
        return rows


class RocmPortTests(unittest.TestCase):
    def test_detect_runtime_requires_hip_backend(self):
        with mock.patch.object(train.torch.cuda, "is_available", return_value=True), \
             mock.patch.object(train.platform, "system", return_value="Linux"), \
             mock.patch.object(train.torch.version, "hip", None, create=True):
            with self.assertRaisesRegex(RuntimeError, "HIP runtime is required"):
                train.detect_runtime()

    def test_detect_runtime_reports_rocm_backend(self):
        props = SimpleNamespace(total_memory=16 * 1024 ** 3)
        with mock.patch.object(train.torch.cuda, "is_available", return_value=True), \
             mock.patch.object(train.platform, "system", return_value="Linux"), \
             mock.patch.object(train.platform, "uname", return_value=SimpleNamespace(release="6.6.0-microsoft-standard-WSL2")), \
             mock.patch.object(train.torch.version, "hip", "7.2.1", create=True), \
             mock.patch.object(train.torch.cuda, "get_device_properties", return_value=props), \
             mock.patch.object(train.torch.cuda, "get_device_name", return_value="AMD Radeon RX 7900 XTX"), \
             mock.patch.object(train.torch.cuda, "is_bf16_supported", return_value=True), \
             mock.patch.dict(train.os.environ, {}, clear=True):
            runtime = train.detect_runtime()

        self.assertEqual(runtime.backend_kind, "rocm")
        self.assertEqual(runtime.backend_version, "7.2.1")
        self.assertTrue(runtime.is_wsl)
        self.assertTrue(runtime.supports_bf16)
        self.assertFalse(runtime.supports_tf32)
        self.assertTrue(runtime.supports_pinned_memory)
        self.assertTrue(runtime.gpu_profile.is_documented_supported)

    def test_autotune_cache_key_uses_backend_version(self):
        runtime = SimpleNamespace(
            backend_kind="rocm",
            backend_version="7.2.1",
            gpu_name="AMD Radeon RX 7900 XTX",
            gpu_total_memory_bytes=24 * 1024 ** 3,
        )
        key = train._make_autotune_cache_key(runtime)
        self.assertIn("rocm", key)
        self.assertIn("7.2.1", key)
        self.assertNotIn("8.9", key)

    def test_profile_selection_is_vram_driven(self):
        small = train._resolve_accelerator_profile("AMD Radeon RX 7700 XT", 12.0, True)
        large = train._resolve_accelerator_profile("AMD Radeon RX 7900 XTX", 24.0, True)
        compat = train._resolve_accelerator_profile("Unknown GPU", 16.0, False)

        self.assertEqual(small.name, "rocm-12-15gb")
        self.assertEqual(large.name, "rocm-24gb-plus")
        self.assertFalse(compat.is_documented_supported)
        self.assertTrue(compat.is_compatibility_only)

    def test_make_dataloader_uses_explicit_pin_memory_flag(self):
        original_empty = prepare.torch.empty
        pin_memory_flags = []

        def wrapped_empty(*args, **kwargs):
            if "pin_memory" in kwargs:
                pin_memory_flags.append(kwargs["pin_memory"])
                kwargs = dict(kwargs)
                kwargs["pin_memory"] = False
            return original_empty(*args, **kwargs)

        def fake_batches(split, dataset=None, tokenizer_batch_size=128):
            del split, dataset, tokenizer_batch_size
            while True:
                yield ["doc-a", "doc-b"], 1

        tokenizer = DummyTokenizer()
        with mock.patch.object(prepare, "_document_batches", side_effect=fake_batches), \
             mock.patch.object(prepare.torch, "empty", side_effect=wrapped_empty):
            loader = prepare.make_dataloader(tokenizer, 1, 4, "train", device="cpu", pin_memory=True)
            next(loader)

        self.assertIn(True, pin_memory_flags)

    def test_rocm_attention_fallback_expands_gqa_heads(self):
        config = train.GPTConfig(
            sequence_len=8,
            vocab_size=32,
            n_layer=1,
            n_head=4,
            n_kv_head=2,
            n_embd=16,
            backend_kind="rocm",
        )
        attn = train.CausalSelfAttention(config, layer_idx=0)
        x = torch.randn(2, 3, 16)
        cos = torch.ones(1, 3, 1, 2)
        sin = torch.zeros(1, 3, 1, 2)
        call_args = {}

        def fake_sdpa(q, k, v, attn_mask=None, is_causal=False, enable_gqa=False):
            del attn_mask, is_causal
            call_args["k_shape"] = tuple(k.shape)
            call_args["v_shape"] = tuple(v.shape)
            call_args["enable_gqa"] = enable_gqa
            return torch.zeros_like(q)

        with mock.patch.object(train.F, "scaled_dot_product_attention", side_effect=fake_sdpa):
            out = attn(x, None, (cos, sin), window_size=(8, 0))

        self.assertEqual(out.shape, (2, 3, 16))
        self.assertEqual(call_args["k_shape"][1], 4)
        self.assertEqual(call_args["v_shape"][1], 4)
        self.assertFalse(call_args["enable_gqa"])


if __name__ == "__main__":
    unittest.main()
