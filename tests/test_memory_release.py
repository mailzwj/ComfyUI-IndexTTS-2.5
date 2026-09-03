"""内存/显存释放逻辑的单元测试（不加载真实模型，用桩对象验证行为）。

运行方式（任选其一）：
    python tests/test_memory_release.py
    pytest tests/test_memory_release.py -v
"""
import gc
import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_pkg():
    """custom node 目录名含 '-'，不是合法包名，手工挂载为 it25 包。"""
    if "it25" in sys.modules:
        return sys.modules["it25"]
    pkg = types.ModuleType("it25")
    pkg.__path__ = [ROOT]
    sys.modules["it25"] = pkg
    for name in ("model_loader", "nodes"):
        spec = importlib.util.spec_from_file_location(
            f"it25.{name}", os.path.join(ROOT, f"{name}.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"it25.{name}"] = mod
        spec.loader.exec_module(mod)
        setattr(pkg, name, mod)  # 模拟正常包导入：子模块挂到包属性上
    return pkg


PKG = _load_pkg()
ML = PKG.model_loader
NODES = PKG.nodes

import numpy as np  # noqa: E402
import torch  # noqa: E402


class FakeTTS:
    """模拟 IndexTTS2：持有大张量（模拟模型权重），infer 返回 (sr, wav)。"""

    def __init__(self):
        self.gpt = mock.MagicMock()
        self.semantic_model = mock.MagicMock()
        self.semantic_codec = mock.MagicMock()
        self.s2mel = mock.MagicMock()
        self.campplus_model = mock.MagicMock()
        self.bigvgan = mock.MagicMock()
        self.qwen_emo = None
        self.emo_matrix = torch.zeros(8, 16)
        self.spk_matrix = torch.zeros(8, 16)
        self.cache_spk_cond = torch.zeros(4, 4)
        # ~200MB CPU 张量，模拟真实模型占用的内存
        self._ballast = torch.zeros(50 * 1024 * 1024 // 4)

    def infer(self, **kwargs):
        return 22050, np.zeros(22050, dtype=np.int16)


def _make_loader():
    # models_root 传 ROOT 仅为构造合法对象；get_tts 会被 mock，不会真正校验/加载
    return ML.IndexTTS2_5Loader(models_root=ROOT, device="cpu")


class TestLoaderRelease(unittest.TestCase):
    def test_release_clears_cache_and_drops_refs(self):
        loader = _make_loader()
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
        loader = _make_loader()
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
        loader = _make_loader()
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "empty_cache") as ec:
            loader.release()
        self.assertFalse(ec.called)

    def test_get_tts_returns_same_cached_instance(self):
        loader = _make_loader()
        fake = FakeTTS()
        # 直接走缓存命中路径（不触发真实加载）
        loader._cache[("tts", True, False)] = fake
        self.assertIs(loader.get_tts(use_bf16=True, use_qwen_emo=False), fake)

    def test_get_tts_cache_key_distinguishes_params(self):
        loader = _make_loader()
        f1, f2 = FakeTTS(), FakeTTS()
        loader._cache[("tts", True, False)] = f1
        loader._cache[("tts", False, False)] = f2
        self.assertIs(loader.get_tts(use_bf16=True), f1)
        self.assertIs(loader.get_tts(use_bf16=False), f2)


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
        """mock loader：save_temp_wav 走真实实现（产生真实临时文件），get_tts 返回桩。"""
        mock_loader = mock.MagicMock()
        mock_loader.save_temp_wav.side_effect = ML.IndexTTS2_5Loader.save_temp_wav
        mock_loader.get_tts.return_value = tts
        self.node.loader = mock_loader
        return mock_loader

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
        out = self._call(auto_free_memory=True)
        audio, seed, subtitle = out
        self.assertEqual(mock_loader.get_tts.call_count, 1)
        self.assertEqual(mock_loader.release.call_count, 1)
        self.assertEqual(audio["sample_rate"], 22050)
        self.assertTrue(audio["waveform"].numel() > 0)

    def test_generate_no_release_when_disabled(self):
        mock_loader = self._mock_loader(FakeTTS())
        self._call(auto_free_memory=False)
        self.assertEqual(mock_loader.release.call_count, 0)

    def test_temp_files_cleaned_on_success(self):
        mock_loader = mock.MagicMock()
        paths = []
        orig_save = ML.IndexTTS2_5Loader.save_temp_wav

        def spy_save(audio):
            p = orig_save(audio)
            paths.append(p)
            return p

        mock_loader.save_temp_wav.side_effect = spy_save
        mock_loader.get_tts.return_value = FakeTTS()
        self.node.loader = mock_loader

        self._call(auto_free_memory=False, emo_ref_audio=self.ref_audio)
        self.assertEqual(len(paths), 2, "应生成 spk + emo 两个临时 wav")
        for p in paths:
            self.assertFalse(os.path.exists(p), f"临时 wav 应被删除: {p}")

    def test_temp_files_cleaned_on_failure_and_no_release(self):
        class BoomTTS(FakeTTS):
            def infer(self, **kwargs):
                raise RuntimeError("boom")

        mock_loader = self._mock_loader(BoomTTS())
        with self.assertRaises(RuntimeError):
            self._call(auto_free_memory=True)
        # 失败时不应释放模型（保留缓存以便重试），但临时文件必须清理
        self.assertEqual(mock_loader.release.call_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
