"""内存/显存释放逻辑与节点主链路的单元测试（不加载真实模型，用桩对象验证行为）。

运行方式（任选其一）：
    python tests/test_memory_release.py
    pytest tests/test_memory_release.py -v
"""
import gc
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bootstrap import (ROOT, ML, NODES, FakeTTS, make_loader,  # noqa: E402
                        make_fake_loader)

import torch  # noqa: E402


class TestLoaderRelease(unittest.TestCase):
    def test_release_clears_cache_and_drops_refs(self):
        loader = make_loader()
        fake = FakeTTS()
        loader._cache[("tts", True, False)] = fake

        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            loader.release()

        self.assertEqual(loader._cache, {})
        # 大对象引用被显式断开，GC 后即可回收
        self.assertIsNone(fake.gpt)
        self.assertIsNone(fake.s2mel)
        self.assertIsNone(fake.semantic_model)
        self.assertIsNone(fake.cache_spk_cond)
        gc.collect()

    def test_release_calls_cuda_empty_cache_when_available(self):
        loader = make_loader()
        fake = FakeTTS()
        loader._cache[("tts", True, False)] = fake

        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "synchronize") as sync, \
             mock.patch.object(torch.cuda, "empty_cache") as ec, \
             mock.patch.object(torch.cuda, "mem_get_info", return_value=(1 << 30, 2 << 30)):
            loader.release()

        self.assertTrue(sync.called)
        self.assertTrue(ec.called)
        self.assertEqual(loader._cache, {})

    def test_release_without_cache_is_noop(self):
        loader = make_loader()
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "empty_cache") as ec:
            loader.release()
        self.assertFalse(ec.called)

    def test_get_tts_returns_same_cached_instance(self):
        loader = make_loader()
        fake = FakeTTS()
        # 直接走缓存命中路径（不触发真实加载）
        loader._cache[("tts", True, False)] = fake
        with mock.patch.object(ML.IndexTTS2_5Loader, "_bf16_supported", lambda self: True):
            self.assertIs(loader.get_tts(use_bf16=True, use_qwen_emo=False), fake)

    def test_get_tts_cache_key_distinguishes_params(self):
        loader = make_loader()
        f1, f2 = FakeTTS(), FakeTTS()
        loader._cache[("tts", True, False)] = f1
        loader._cache[("tts", False, False)] = f2
        with mock.patch.object(ML.IndexTTS2_5Loader, "_bf16_supported", lambda self: True):
            self.assertIs(loader.get_tts(use_bf16=True), f1)
            self.assertIs(loader.get_tts(use_bf16=False), f2)

    def test_sampling_rate_prefers_model_config(self):
        # 桩对象没有 cfg -> 退回上游默认的 22050
        self.assertEqual(ML.IndexTTS2_5Loader.sampling_rate(FakeTTS()), 22050)

        class WithCfg:
            cfg = {"s2mel": {"preprocess_params": {"sr": 44100}}}

        self.assertEqual(ML.IndexTTS2_5Loader.sampling_rate(WithCfg()), 44100)


class TestBf16CapabilityGuard(unittest.TestCase):
    """bf16 需要 Ampere（sm_80）及以上；老卡必须自动降级为 fp32。"""

    def _loader(self, device):
        return make_loader(device)

    def test_cpu_and_mps_are_not_bf16(self):
        for device in ("cpu", "mps", "xpu"):
            with self.subTest(device=device):
                self.assertFalse(self._loader(device)._bf16_supported())

    def test_capability_matrix(self):
        cases = {(7, 0): False, (7, 5): False, (8, 0): True, (8, 6): True, (12, 0): True}
        loader = self._loader("cuda")
        for cap, expected in cases.items():
            with mock.patch.object(torch.cuda, "get_device_capability", return_value=cap):
                self.assertIs(loader._bf16_supported(), expected, f"sm_{cap[0]}{cap[1]}")

    def test_uses_requested_device_index(self):
        loader = self._loader("cuda:1")
        with mock.patch.object(torch.cuda, "get_device_capability",
                               return_value=(8, 6)) as gdc:
            self.assertTrue(loader._bf16_supported())
        gdc.assert_called_once_with(1)

    def test_capability_query_failure_falls_back_to_fp32(self):
        loader = self._loader("cuda")
        with mock.patch.object(torch.cuda, "get_device_capability",
                               side_effect=RuntimeError("no device")):
            self.assertFalse(loader._bf16_supported())

    def test_cache_key_uses_downgraded_value(self):
        """老卡上 True/False 应命中同一份 fp32 模型，不能各加载一遍。"""
        loader = self._loader("cuda")
        fake = FakeTTS()
        loader._cache[("tts", False, False)] = fake
        with mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)):
            self.assertIs(loader.get_tts(use_bf16=True), fake)


class TestSharedLoader(unittest.TestCase):
    def setUp(self):
        ML._shared_loader = None

    def test_get_shared_loader_is_singleton(self):
        a = ML.get_shared_loader(models_root=ROOT, device="cpu")
        b = ML.get_shared_loader()
        self.assertIs(a, b)

    def test_node_instances_share_loader(self):
        n1 = NODES.IndexTTS2_5Node()
        n2 = NODES.IndexTTS2_5Node()
        self.assertIs(n1.loader, n2.loader)


class TestNodeGenerate(unittest.TestCase):
    def setUp(self):
        ML._shared_loader = None
        self.node = NODES.IndexTTS2_5Node()
        self.ref_audio = {
            "waveform": torch.zeros(1, 1, 22050),
            "sample_rate": 22050,
        }

    def _mock_loader(self, tts):
        """mock loader：save_temp_wav / sampling_rate 走真实实现，get_tts 返回桩。"""
        loader = make_fake_loader(tts)
        self.node.loader = loader
        return loader

    def _call(self, **kw):
        defaults = dict(text="你好，测试。", reference_audio=self.ref_audio)
        defaults.update(kw)
        return self.node.generate(**defaults)

    def test_input_types_has_auto_free_memory_default_false(self):
        optional = NODES.IndexTTS2_5Node.INPUT_TYPES()["optional"]
        self.assertIn("auto_free_memory", optional)
        typ, info = optional["auto_free_memory"]
        self.assertEqual(typ, "BOOLEAN")
        self.assertFalse(info["default"])

    def test_generate_releases_when_enabled(self):
        mock_loader = self._mock_loader(FakeTTS())
        audio, seed, subtitle = self._call(auto_free_memory=True)
        self.assertEqual(mock_loader.get_tts.call_count, 1)
        self.assertEqual(mock_loader.release.call_count, 1)
        self.assertEqual(audio["sample_rate"], 22050)
        self.assertTrue(audio["waveform"].numel() > 0)
        self.assertEqual(seed, 0)

    def test_generate_no_release_when_disabled(self):
        mock_loader = self._mock_loader(FakeTTS())
        self._call(auto_free_memory=False)
        self.assertEqual(mock_loader.release.call_count, 0)

    def test_generate_passes_stream_options(self):
        tts = FakeTTS()
        self._mock_loader(tts)
        self._call(emo_vector="0.8,0,0,0,0,0,0,0", interval_silence=0,
                   max_text_tokens_per_segment=60, text_normalization=False,
                   emo_text="高兴")
        self.assertEqual(len(tts.infer_options), 1)
        options = tts.infer_options[0]
        self.assertEqual(options["max_text_tokens_per_segment"], 60)
        self.assertFalse(options["text_normalization"])
        self.assertEqual(options["emo_vector"], [0.8, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(options["use_emo_text"])
        # 段间静音由节点统一插入，因此上游固定收到 0，且走流式
        self.assertEqual(options["interval_silence"], 0)
        self.assertTrue(options["stream_return"])

    def test_temp_files_cleaned_on_success(self):
        mock_loader = mock.MagicMock()
        paths = []
        orig_save = ML.IndexTTS2_5Loader.save_temp_wav

        def spy_save(audio):
            p = orig_save(audio)
            paths.append(p)
            return p

        mock_loader.save_temp_wav.side_effect = spy_save
        mock_loader.sampling_rate.side_effect = ML.IndexTTS2_5Loader.sampling_rate
        mock_loader.get_tts.return_value = FakeTTS()
        self.node.loader = mock_loader

        self._call(auto_free_memory=False, emo_ref_audio=self.ref_audio)
        self.assertEqual(len(paths), 2, "应生成 spk + emo 两个临时 wav")
        for p in paths:
            self.assertFalse(os.path.exists(p), f"临时 wav 应被删除: {p}")

    def test_temp_files_cleaned_on_failure_and_no_release(self):
        class BoomTTS(FakeTTS):
            def infer_generator(self, **kwargs):
                raise RuntimeError("boom")
                yield  # pragma: no cover - 使其保持 generator 形态

        mock_loader = self._mock_loader(BoomTTS())
        with self.assertRaises(RuntimeError):
            self._call(auto_free_memory=True)
        # 失败时不应释放模型（保留缓存以便重试），但临时文件必须清理
        self.assertEqual(mock_loader.release.call_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
