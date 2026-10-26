"""字幕时间轴与格式化的单元测试。

运行方式（任选其一）：
    python tests/test_subtitle_timeline.py
    pytest tests/test_subtitle_timeline.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bootstrap import SUB  # noqa: E402


def _close(a, b, eps=1e-6):
    return abs(a - b) < eps


class TestTimeline(unittest.TestCase):
    def test_segment_windows_are_exact_and_silence_is_gap(self):
        frames = SUB.build_timeline(
            ["第一段。", "第二段。"], [1.0, 2.0], interval_silence_ms=500)
        self.assertEqual([f["text"] for f in frames], ["第一段。", "第二段。"])
        self.assertTrue(_close(frames[0]["start"], 0.0))
        self.assertTrue(_close(frames[0]["end"], 1.0))
        # 段间静音 0.5s 计入下一段的起点，不算进任何一条字幕
        self.assertTrue(_close(frames[1]["start"], 1.5))
        self.assertTrue(_close(frames[1]["end"], 3.5))

    def test_sentences_split_within_segment_by_char_weight(self):
        frames = SUB.build_timeline(["你好，世界。再见。"], [3.0])
        self.assertEqual([f["text"] for f in frames], ["你好，", "世界。", "再见。"])
        self.assertTrue(_close(frames[0]["end"], 1.0))
        self.assertTrue(_close(frames[1]["end"], 2.0))
        self.assertTrue(_close(frames[2]["end"], 3.0))

    def test_original_text_is_mapped_back_by_char_ratio(self):
        # 合成文本每段长度 3 / 5，原文 8 个字符 -> 切片 "AB。" + "CD。EF"
        frames = SUB.build_timeline(
            ["一二。", "三四五六。"], [1.0, 2.0], original_text="AB。CD。EF")
        self.assertEqual([f["text"] for f in frames], ["AB。", "CD。", "EF"])
        # 第二段窗口 = [1.0, 3.0]，内部按字符权重（3:2）继续细分
        self.assertTrue(_close(frames[1]["start"], 1.0))
        self.assertTrue(_close(frames[1]["end"], 2.2))
        self.assertTrue(_close(frames[2]["start"], 2.2))
        self.assertTrue(_close(frames[-1]["end"], 3.0))

    def test_missing_segment_text_falls_back_to_proportional(self):
        frames = SUB.build_timeline([""], [4.0], original_text="第一句。第二句。")
        self.assertEqual([f["text"] for f in frames], ["第一句。", "第二句。"])
        self.assertTrue(_close(frames[0]["end"], 2.0))
        self.assertTrue(_close(frames[-1]["end"], 4.0))

    def test_empty_input(self):
        self.assertEqual(SUB.build_timeline([], []), [])

    def test_whitespace_only_segment_produces_no_frame(self):
        self.assertEqual(SUB.build_timeline([" "], [1.0]), [])


class TestFormats(unittest.TestCase):
    def setUp(self):
        self.frames = SUB.build_timeline(["你好，世界。"], [2.0])

    def test_srt(self):
        text = SUB.format_frames(self.frames, "srt")
        self.assertTrue(text.startswith("1\n00:00:00,000 --> 00:00:01,000\n你好，\n\n"))
        self.assertIn("2\n00:00:01,000 --> 00:00:02,000\n世界。", text)

    def test_vtt(self):
        text = SUB.format_frames(self.frames, "vtt")
        self.assertTrue(text.startswith("WEBVTT\n\n"))
        self.assertIn("00:00:00.000 --> 00:00:01.000", text)

    def test_plain_keeps_legacy_layout(self):
        text = SUB.format_frames(self.frames, "plain")
        self.assertEqual(text.splitlines()[0], ">> 0:00.000-0:01.000")
        self.assertEqual(text.splitlines()[1], ">> 你好，")

    def test_default_format_is_srt(self):
        out = SUB.build_subtitle(["测试。"], [1.0])
        self.assertIn("--> 00:00:01,000", out)

    def test_frame_text_never_spans_multiple_lines(self):
        out = SUB.build_subtitle(["你好\n世界。"], [1.0], subtitle_format="srt")
        self.assertIn("你好", out)
        self.assertIn("世界。", out)
        # 每条字幕只占一行：不应出现空行紧跟正文的畸形块
        self.assertNotIn("\n\n\n", out)


class TestCleanPronunciationMarkers(unittest.TestCase):
    """字幕里不能出现发音标注的 token 形态，必须还原成可读文本。"""

    def test_annotated_form_restores_the_word_via_mapping(self):
        text = "长<|SPECIAL_TOKEN_2|>ZHAO1<|SPECIAL_TOKEN_2|>夕。"
        mapping = {"<|SPECIAL_TOKEN_2|>ZHAO1<|SPECIAL_TOKEN_2|>": "朝"}
        self.assertEqual(SUB.clean_pronunciation_markers(text, mapping), "长朝夕。")

    def test_english_phoneme_form_restores_the_word(self):
        text = "Say <|SPECIAL_TOKEN_1|>G OW1 . IH0 NG<|SPECIAL_TOKEN_1|> now."
        mapping = {"<|SPECIAL_TOKEN_1|>G OW1 . IH0 NG<|SPECIAL_TOKEN_1|>": "Going"}
        self.assertEqual(SUB.clean_pronunciation_markers(text, mapping),
                         "Say Going now.")

    def test_unmapped_annotation_keeps_pronunciation_without_tokens(self):
        text = "长<|SPECIAL_TOKEN_2|>ZHAO1<|SPECIAL_TOKEN_2|>夕。"
        cleaned = SUB.clean_pronunciation_markers(text)
        self.assertNotIn("SPECIAL_TOKEN", cleaned)
        self.assertEqual(cleaned, "长ZHAO1夕。")

    def test_lone_token_marker_is_dropped(self):
        self.assertEqual(SUB.clean_pronunciation_markers("前后<|SPECIAL_TOKEN_2|>中间"),
                         "前后中间")

    def test_raw_tag_form_is_reduced_to_the_word(self):
        self.assertEqual(SUB.clean_pronunciation_markers("长<朝|zhao1>夕。"), "长朝夕。")

    def test_language_switch_tokens_are_preserved(self):
        """<|ZH|> / <|EN|> 是用户主动写的语种切换标记，不能当成发音标注吃掉。"""
        text = "中文<|ZH|>then English<|EN|>"
        self.assertEqual(SUB.clean_pronunciation_markers(text), text)

    def test_empty_and_none_input(self):
        self.assertEqual(SUB.clean_pronunciation_markers(""), "")
        self.assertIsNone(SUB.clean_pronunciation_markers(None))

    def test_longest_mapping_wins(self):
        mapping = {"<|SPECIAL_TOKEN_2|>A<|SPECIAL_TOKEN_2|>": "短",
                   "<|SPECIAL_TOKEN_2|>A<|SPECIAL_TOKEN_2|>X<|SPECIAL_TOKEN_2|>B<|SPECIAL_TOKEN_2|>": "长短"}
        self.assertEqual(
            SUB.clean_pronunciation_markers("<|SPECIAL_TOKEN_2|>A<|SPECIAL_TOKEN_2|>X<|SPECIAL_TOKEN_2|>B<|SPECIAL_TOKEN_2|>", mapping),
            "长短")


if __name__ == "__main__":
    unittest.main(verbosity=2)
