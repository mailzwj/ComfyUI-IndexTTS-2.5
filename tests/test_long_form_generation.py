"""长文本流式合成与字幕对齐的测试（桩模型，不加载真实权重）。

运行方式（任选其一）：
    python tests/test_long_form_generation.py
    pytest tests/test_long_form_generation.py -v
"""
import os
import re
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bootstrap import ML, NODES, FakeTTS, make_fake_loader  # noqa: E402

import torch  # noqa: E402

SR = 22050


class ThreeSegmentTTS(FakeTTS):
    SEGMENT_SAMPLES = [SR // 2, SR, int(SR * 1.5)]
    SEGMENT_TEXTS = ["第一段。", "第二段。", "第三段。"]


class MismatchTTS(FakeTTS):
    """分段文本捕获与实际产出数量不一致（模拟上游改版）。"""

    SEGMENT_SAMPLES = [SR]
    SEGMENT_TEXTS = ["多余。", "还多。"]


class LowVramMultiChunkTTS(FakeTTS):
    SEGMENT_SAMPLES = [SR // 2, SR // 2]

    def __init__(self):
        super().__init__()
        self.low_vram = True

    def split_text_by_punctuation(self, text, max_chars=40):
        return [text[:20], text[20:]]

    def split_text_by_tokens(self, text, max_tokens, lang_prefix=""):
        return [text + "|A", text + "|B"]


class TestLongFormGenerate(unittest.TestCase):
    def setUp(self):
        ML._shared_loader = None
        self.node = NODES.IndexTTS2_5Node()
        self.ref_audio = {"waveform": torch.zeros(1, 1, SR), "sample_rate": SR}

    def _run(self, tts, **kw):
        self._loader = make_fake_loader(tts)
        self.node.loader = self._loader
        args = dict(text="测试文本。", reference_audio=self.ref_audio)
        args.update(kw)
        return self.node.generate(**args)

    def test_multi_segment_audio_and_exact_subtitle(self):
        audio, _, subtitle = self._run(ThreeSegmentTTS(), interval_silence=200,
                                       subtitle_format="srt")
        samples = audio["waveform"].shape[-1]
        # 0.5 + 1.0 + 1.5 段音频 + 2 段 0.2s 静音
        self.assertEqual(samples, int(3.0 * SR) + int(0.2 * SR) * 2)
        blocks = [b for b in subtitle.strip().split("\n\n") if b]
        self.assertEqual(len(blocks), 3)
        self.assertTrue(blocks[0].startswith("1\n00:00:00,000 --> 00:00:00,500"))
        # 段间静音让第 2 段从 0.7s 开始（0.5 + 0.2）
        self.assertTrue(blocks[1].startswith("2\n00:00:00,700 --> 00:00:01,700"))
        self.assertTrue(blocks[2].startswith("3\n00:00:01,900 --> 00:00:03,400"))

    def test_zero_interval_silence_concatenates_tightly(self):
        audio, _, _ = self._run(ThreeSegmentTTS(), interval_silence=0)
        self.assertEqual(audio["waveform"].shape[-1], int(3.0 * SR))

    def test_subtitle_text_can_use_original_wording(self):
        _, _, subtitle = self._run(ThreeSegmentTTS(), interval_silence=0,
                                   text="第一段。第二段。第三段。",
                                   subtitle_text="original", subtitle_format="plain")
        self.assertIn(">> 第一段。", subtitle)
        self.assertIn(">> 第三段。", subtitle)

    def test_split_recorder_is_restored_after_generate(self):
        tts = ThreeSegmentTTS()
        self._run(tts)
        bound = tts.split_text_by_tokens.__func__ if hasattr(
            tts.split_text_by_tokens, "__func__") else None
        self.assertIs(bound, FakeTTS.split_text_by_tokens)
        self.assertIsNone(tts.gr_progress)

    def test_alignment_mismatch_degrades_instead_of_crashing(self):
        audio, _, subtitle = self._run(MismatchTTS(), subtitle_format="srt")
        self.assertEqual(audio["waveform"].shape[-1], SR)
        self.assertIn("-->", subtitle)

    def test_low_vram_splits_into_chunks_and_keeps_alignment(self):
        tts = LowVramMultiChunkTTS()
        audio, _, subtitle = self._run(tts, text="一" * 60, interval_silence=200)
        self.assertEqual(len(tts.infer_options), 2, "低显存模式应逐块生成")
        # 2 块 x 2 段 x 0.5s + 3 个 0.2s 间隔
        self.assertEqual(audio["waveform"].shape[-1], int(2.0 * SR) + int(0.2 * SR) * 3)
        self.assertEqual(len([b for b in subtitle.strip().split("\n\n") if b]), 4)

    def test_truncation_warning_is_surfaced(self):
        class WarnTTS(ThreeSegmentTTS):
            WARN_ON_CALL = True

        with self.assertRaises(RuntimeError) as ctx:
            self._run(WarnTTS(), strict_max_mel_tokens=True, max_mel_tokens=1500)
        message = str(ctx.exception)
        self.assertIn("max_mel_tokens", message)
        self.assertIn("max_text_tokens_per_segment", message)

    def test_truncation_without_strict_still_returns_audio(self):
        class WarnTTS(ThreeSegmentTTS):
            WARN_ON_CALL = True

        audio, _, _ = self._run(WarnTTS(), strict_max_mel_tokens=False)
        self.assertEqual(audio["waveform"].shape[-1], int(3.0 * SR) + int(0.2 * SR) * 2)

    def test_progress_hook_installed_during_generate_then_restored(self):
        seen = []

        def fake_hook():
            def hook(value=0.0, desc=None):
                seen.append((value, desc))
            return hook

        tts = ThreeSegmentTTS()
        with mock.patch.object(NODES.IndexTTS2_5Node, "_make_progress_hook",
                               staticmethod(fake_hook)):
            self._run(tts)
        # FakeTTS 在生成时会回调一次 gr_progress
        self.assertEqual(seen, [(0.1, "starting")])
        # 生成结束后必须还原，避免污染共享的模型实例
        self.assertIsNone(tts.gr_progress)

    def test_without_comfy_progress_bar_generation_still_works(self):
        audio, _, _ = self._run(ThreeSegmentTTS())
        self.assertTrue(audio["waveform"].numel() > 0)

    def test_srt_timestamp_shape(self):
        _, _, subtitle = self._run(ThreeSegmentTTS())
        pattern = re.compile(
            r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n.+")
        for block in subtitle.strip().split("\n\n"):
            self.assertRegex(block, pattern)


# ---- 发音标注在字幕中的还原 -------------------------------------------------
_TAG_RE = re.compile(r"<([^<>|\n]+)\|([^<>|\n]+)>")


def _apply_annotations(text):
    """与上游 infer_v2_5.apply_pronunciation_annotations 同规则的最小实现。

    上游函数名/规则由 tests/test_upstream_contract.py 的源码断言钉住；
    这里只用于构造「模型实际看到的文本」。
    """
    def _sub(match):
        word, pron = match.group(1), match.group(2).upper()
        token = "SPECIAL_TOKEN_2" if re.search(r"[\u4e00-\u9fff]", word) else "SPECIAL_TOKEN_1"
        return f"<|{token}|>{pron}<|{token}|>"
    return _TAG_RE.sub(_sub, text)


class AnnotatedTTS(FakeTTS):
    """模拟上游：split_text_by_tokens 收到的是发音标注替换后的文本。"""

    LOWER_BEFORE_ANNOTATE = True

    def infer_generator(self, **kwargs):
        text = kwargs.get("text", "")
        if self.LOWER_BEFORE_ANNOTATE and re.search(r"[\u4e00-\u9fff]", text):
            text = text.lower()
        kwargs["text"] = _apply_annotations(text)
        return (yield from super().infer_generator(**kwargs))


def _fake_infer_module():
    module = types.ModuleType("indextts.infer_v2_5")
    module.apply_pronunciation_annotations = _apply_annotations
    return module


class TestSubtitlePronunciationRestore(unittest.TestCase):
    def setUp(self):
        ML._shared_loader = None
        self.node = NODES.IndexTTS2_5Node()
        self.ref_audio = {"waveform": torch.zeros(1, 1, SR), "sample_rate": SR}

    def _run(self, tts, **kw):
        self.node.loader = make_fake_loader(tts)
        args = dict(text="测试文本。", reference_audio=self.ref_audio)
        args.update(kw)
        with mock.patch.dict(sys.modules, {"indextts.infer_v2_5": _fake_infer_module()}):
            return self.node.generate(**args)

    def test_chinese_annotation_is_restored_to_the_word(self):
        _, _, subtitle = self._run(AnnotatedTTS(), text="长<朝|zhao1>夕。")
        self.assertNotIn("SPECIAL_TOKEN", subtitle)
        self.assertNotIn("ZHAO1", subtitle)
        self.assertIn("长朝夕", subtitle)

    def test_english_annotation_keeps_users_original_casing(self):
        tts = AnnotatedTTS()
        tts.LOWER_BEFORE_ANNOTATE = False
        _, _, subtitle = self._run(tts, text="Say <Going|G OW1 . IH0 NG> now.",
                                   language="en")
        self.assertNotIn("SPECIAL_TOKEN", subtitle)
        self.assertIn("Going", subtitle)

    def test_original_mode_strips_annotation_tags(self):
        _, _, subtitle = self._run(ThreeSegmentTTS(), text="长<朝|zhao1>夕。",
                                   subtitle_text="original", subtitle_format="srt")
        self.assertNotIn("SPECIAL_TOKEN", subtitle)
        self.assertNotIn("|", subtitle)
        # 原文被按段占比切片，逐帧拼回来应还原成去掉标注记号的可读文本
        body = "".join(line for line in subtitle.splitlines()
                       if line and "-->" not in line and not line.isdigit())
        self.assertEqual(body, "长朝夕。")

    def test_without_upstream_module_falls_back_to_readable_pronunciation(self):
        """拿不到上游函数时（极端情况），至少不能把 token 标记留在字幕里。"""
        tts = AnnotatedTTS()
        self.node.loader = make_fake_loader(tts)
        real_pop = sys.modules.pop("indextts.infer_v2_5", None)
        try:
            _, _, subtitle = self.node.generate(
                text="长<朝|zhao1>夕。", reference_audio=self.ref_audio)
        finally:
            if real_pop is not None:
                sys.modules["indextts.infer_v2_5"] = real_pop
        self.assertNotIn("SPECIAL_TOKEN", subtitle)
        self.assertIn("ZHAO1", subtitle)  # 兜底：保留读音，不保留标记


if __name__ == "__main__":
    unittest.main(verbosity=2)
