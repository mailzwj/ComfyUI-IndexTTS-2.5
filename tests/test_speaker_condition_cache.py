"""说话人条件 LRU 缓存的测试（桩 tts 模拟上游「按路径命中条件」的语义）。

运行方式（任选其一）：
    python tests/test_speaker_condition_cache.py
    pytest tests/test_speaker_condition_cache.py -v
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bootstrap import EN, ML, make_stateful_loader  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

SR = 22050
CACHE_ATTRS = ("cache_spk_cond", "cache_s2mel_style", "cache_s2mel_prompt",
               "cache_mel", "cache_emo_cond")


class UpstreamStub:
    """只保留上游与缓存相关的字段与判断：路径不同就重算条件张量。"""

    def __init__(self):
        for attr in CACHE_ATTRS:
            setattr(self, attr, None)
        self.cache_spk_audio_prompt = None
        self.cache_emo_audio_prompt = None
        self.spk_extracts = 0
        self.emo_extracts = 0

    def regenerate(self, spk_path, emo_path):
        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_path:
            self.spk_extracts += 1
            marker = float(abs(hash(spk_path)) % 1000)
            for attr in ("cache_spk_cond", "cache_s2mel_style",
                         "cache_s2mel_prompt", "cache_mel"):
                setattr(self, attr, torch.full((2,), marker))
            self.cache_spk_audio_prompt = spk_path
        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_path:
            self.emo_extracts += 1
            self.cache_emo_cond = torch.full((2,), float(abs(hash(emo_path)) % 1000))
            self.cache_emo_audio_prompt = emo_path


class WavTempMixin(unittest.TestCase):
    def setUp(self):
        self._tmp_files = []

    def tearDown(self):
        for path in self._tmp_files:
            try:
                os.remove(path)
            except OSError:
                pass

    def wav(self, value, tag="a"):
        """写一个内容可区分的临时 wav（值域限制在 ±1，否则 soundfile 会削顶）。

        每次调用都落在新的临时路径上——这正是节点里的真实情形：同一音色
        的内容不变，但临时文件名每次都不同，所以只能靠内容指纹命中。
        """
        fd, path = tempfile.mkstemp(suffix=f"_{tag}.wav")
        os.close(fd)
        sf.write(path, np.full(SR // 10, float(value) / 100.0, dtype=np.float32), SR)
        self._tmp_files.append(path)
        return path


class TestSpeakerConditionCache(WavTempMixin):
    def setUp(self):
        super().setUp()
        self.loader = make_stateful_loader(None)
        self.tts = UpstreamStub()

    def store(self, spk_path, emo_path=None):
        self.loader.store_speaker_conditions(
            self.tts, self.loader.speaker_cache_key(self.tts, spk_path, emo_path))

    def test_bind_on_empty_cache_is_a_miss(self):
        path = self.wav(1.0)
        key = self.loader.speaker_cache_key(self.tts, path, None)
        self.assertFalse(self.loader.bind_speaker_conditions(self.tts, key, path, path))

    def test_same_content_under_new_path_hits(self):
        first, second = self.wav(1.0, "x"), self.wav(1.0, "y")
        self.tts.regenerate(first, first)
        snapshot = self.tts.cache_spk_cond.clone()
        self.store(first)

        # 上游只看路径：换个文件名必然 miss；绑定后路径与条件张量一起复原
        self.assertNotEqual(self.tts.cache_spk_audio_prompt, second)
        key = self.key_of(second)
        self.assertTrue(self.loader.bind_speaker_conditions(self.tts, key, second, second))
        self.assertEqual(self.tts.cache_spk_audio_prompt, second)
        self.assertEqual(self.tts.cache_emo_audio_prompt, second)
        self.assertTrue(torch.equal(self.tts.cache_spk_cond, snapshot))

    def key_of(self, spk_path, emo_path=None):
        return self.loader.speaker_cache_key(self.tts, spk_path, emo_path)

    def test_different_content_does_not_hit(self):
        a, b = self.wav(1.0, "a"), self.wav(2.0, "b")
        self.tts.regenerate(a, a)
        self.store(a)
        self.assertFalse(self.loader.bind_speaker_conditions(self.tts, self.key_of(b), b, b))

    def test_emo_reference_is_part_of_the_key(self):
        """同一说话人换情感参考：说话人条件仍命中，情感条件留空交给上游重算。"""
        spk, emo = self.wav(1.0, "spk"), self.wav(3.0, "emo")
        self.tts.regenerate(spk, spk)
        self.store(spk)
        self.assertTrue(self.loader.bind_speaker_conditions(
            self.tts, self.key_of(spk, emo), spk, emo))
        self.assertIsNone(self.tts.cache_emo_cond)

    def test_speaker_snapshot_survives_missing_emo(self):
        """情感条件还没算出来时，说话人快照照样值得缓存（两层分开存）。"""
        path = self.wav(1.0, "partial")
        self.tts.cache_spk_cond = torch.ones(2)
        self.store(path)
        self.assertIn(self.key_of(path)[0], self.tts._it25_spk_cache)
        self.assertFalse(hasattr(self.tts, "_it25_emo_cache"))

    def test_no_snapshot_without_speaker_condition(self):
        """上游一个条件都没算出来时没什么可缓存，连字典都不该建。"""
        path = self.wav(1.0, "empty")
        self.store(path)
        self.assertFalse(hasattr(self.tts, "_it25_spk_cache"))

    def test_lru_capacity_and_eviction(self):
        paths = [self.wav(float(i + 1), f"l{i}")
                 for i in range(self.loader.SPEAKER_CACHE_CAPACITY + 1)]
        fingerprints = []
        for path in paths:
            self.tts.regenerate(path, path)
            fingerprints.append(self.key_of(path)[0])
            self.store(path)
        self.assertEqual(len(self.tts._it25_spk_cache), self.loader.SPEAKER_CACHE_CAPACITY)
        self.assertNotIn(fingerprints[0], self.tts._it25_spk_cache)
        self.assertIn(fingerprints[-1], self.tts._it25_spk_cache)

    def test_hit_counts_as_recently_used(self):
        paths = [self.wav(float(i + 1), f"u{i}") for i in range(self.loader.SPEAKER_CACHE_CAPACITY)]
        keys = []
        for path in paths:
            self.tts.regenerate(path, path)
            keys.append(self.key_of(path))
            self.store(path)
        self.assertTrue(self.loader.bind_speaker_conditions(
            self.tts, keys[0], paths[0], paths[0]))
        extra = self.wav(99.0, "extra")
        self.tts.regenerate(extra, extra)
        self.store(extra)
        self.assertIn(keys[0][0], self.tts._it25_spk_cache)
        self.assertNotIn(keys[1][0], self.tts._it25_spk_cache)

    def test_missing_file_falls_back_to_path_fingerprint(self):
        key_a = self.loader.speaker_cache_key(self.tts, "/no/such/a.wav", None)
        key_b = self.loader.speaker_cache_key(self.tts, "/no/such/b.wav", None)
        self.assertNotEqual(key_a, key_b)
        self.assertTrue(key_a[0].startswith("path:"))

    def test_emo_snapshots_are_keyed_by_emo_content_alone(self):
        """同一情感参考配不同说话人也能复用，情感桶只按情感音频内容分键。"""
        emo = self.wav(5.0, "shared_emo")
        for level in (1.0, 2.0):
            spk = self.wav(level, f"spk{level}")
            self.tts.regenerate(spk, emo)
            self.store(spk, emo)
        self.assertEqual(len(self.tts._it25_spk_cache), 2)
        self.assertEqual(len(self.tts._it25_emo_cache), 1)

    def test_release_drops_snapshot_cache(self):
        path = self.wav(1.0, "rel")
        self.tts.regenerate(path, path)
        self.store(path)
        self.assertTrue(self.tts._it25_spk_cache)
        self.assertTrue(self.tts._it25_emo_cache)
        ML.IndexTTS2_5Loader._drop_tts_refs(self.tts)
        self.assertIsNone(self.tts._it25_spk_cache)
        self.assertIsNone(self.tts._it25_emo_cache)
        for attr in CACHE_ATTRS:
            self.assertIsNone(getattr(self.tts, attr))


class TestSynthesizeTurnCache(WavTempMixin):
    """engine.synthesize_turn：同一音色多句只提一次特征，换音色才重算。"""

    def setUp(self):
        super().setUp()
        self.loader = make_stateful_loader(None)
        self.tts = UpstreamStub()
        # 用桩替换真正的分段合成，只保留上游「按路径判断是否重算」这一段语义
        self.patcher = mock.patch.object(EN, "synthesize_segments",
                                         side_effect=self.fake_upstream)

    def fake_upstream(self, tts, spk_path, emo_path, text, language, infer_options,
                      progress_hook=None):
        emo_effective = spk_path if (infer_options.get("use_emo_text")
                                     or infer_options.get("emo_vector") is not None) \
            else (emo_path or spk_path)
        tts.regenerate(spk_path, emo_effective)
        return [np.zeros(100, dtype=np.int16)], [text], False

    def run_turn(self, spk_path, emo_path=None, speaker_cache=True, infer_options=None):
        with self.patcher:
            return EN.synthesize_turn(self.loader, self.tts, spk_path, emo_path,
                                      "文本", "zh", infer_options or EN.build_infer_options(),
                                      speaker_cache=speaker_cache)

    def test_alternating_speakers_extract_each_voice_once(self):
        a, b = self.wav(1.0, "A"), self.wav(2.0, "B")
        hits = [self.run_turn(s)[3] for s in (a, b, a, b, a)]
        self.assertEqual(self.tts.spk_extracts, 2, "两个音色各提一次特征")
        self.assertEqual(self.tts.emo_extracts, 2)
        self.assertEqual(hits, [False, False, True, True, True])

    def test_disabled_cache_leaves_the_upstream_thrash(self):
        """关掉缓存时我们完全不插手：换角色仍会重算，也不留下任何快照。"""
        a, b = self.wav(1.0, "off1"), self.wav(2.0, "off2")
        for spk in (a, b, a, b, a):
            self.assertEqual(self.run_turn(spk, speaker_cache=False)[3], False)
        self.assertEqual(self.tts.spk_extracts, 5)
        self.assertFalse(hasattr(self.tts, "_it25_spk_cache"))

    def test_emo_vector_turns_share_the_speakers_emo_condition(self):
        """上游在给了情感向量时会忽略外部情感音频，缓存也要按同一语义分键。

        每次调用都用新的临时文件名（相当于重跑一次工作流），命中只能靠内容指纹。
        """
        options = EN.build_infer_options(emo_vector=[0, 0, 0, 0, 0, 0, 0, 1])
        for _ in range(2):
            self.run_turn(self.wav(1.0, "vec"), self.wav(5.0, "emoA"),
                          infer_options=options)
        self.assertEqual(self.tts.spk_extracts, 1)
        self.assertEqual(self.tts.emo_extracts, 1)

    def test_different_emo_reference_is_a_separate_entry(self):
        for spk_level, emo_level in ((1.0, 5.0), (1.0, 6.0), (1.0, 5.0)):
            self.run_turn(self.wav(spk_level, "twoemo"), self.wav(emo_level, "emo"))
        self.assertEqual(self.tts.spk_extracts, 1, "说话人条件只提一次")
        self.assertEqual(self.tts.emo_extracts, 2, "两份情感参考各提一次，第三次靠快照复用")


if __name__ == "__main__":
    unittest.main(verbosity=2)
