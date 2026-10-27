"""多人对话脚本解析器的测试（纯字符串，不加载模型）。

运行方式（任选其一）：
    python tests/test_dialogue_script_parser.py
    pytest tests/test_dialogue_script_parser.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bootstrap import DS  # noqa: E402


class TestSpeakerPrefix(unittest.TestCase):
    def test_colon_forms(self):
        turns = DS.parse_script("小雪: 你好。\n阿明：早上好！")
        self.assertEqual([(t.speaker, t.text) for t in turns],
                         [("小雪", "你好。"), ("阿明", "早上好！")])

    def test_bracket_form_and_optional_colon(self):
        turns = DS.parse_script("[旁白] 走廊很安静\n[主持人]: 欢迎回来。")
        self.assertEqual([(t.speaker, t.text) for t in turns],
                         [("旁白", "走廊很安静"), ("主持人", "欢迎回来。")])

    def test_names_with_spaces_and_latin_letters(self):
        turns = DS.parse_script("Dr. Lee: Hello there.\n小明 同学: 好久不见。")
        self.assertEqual([t.speaker for t in turns], ["Dr. Lee", "小明 同学"])
        self.assertEqual(turns[1].text, "好久不见。")

    def test_blank_lines_and_full_line_comments_are_skipped(self):
        script = "\n# 这是注释\n小雪: 你好。\n   # 缩进的注释也算注释\n阿明: 你好呀。\n"
        turns = DS.parse_script(script)
        self.assertEqual([t.speaker for t in turns], ["小雪", "阿明"])
        self.assertEqual([t.text for t in turns], ["你好。", "你好呀。"])

    def test_line_numbers_recorded(self):
        turns = DS.parse_script("# 说明\n小雪: 你好。\n阿明: 嗨。")
        self.assertEqual(turns[0].lines, [2])
        self.assertEqual(turns[1].lines, [3])


class TestContinuationLines(unittest.TestCase):
    def test_indented_line_merges_into_previous_turn(self):
        turns = DS.parse_script("阿明: 给你带了杯咖啡！\n  还是冰的，路上没化。\n小雪: 谢谢。")
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].text, "给你带了杯咖啡！还是冰的，路上没化。")
        self.assertEqual(turns[0].lines, [1, 2])

    def test_english_continuation_gets_a_space(self):
        turns = DS.parse_script("Ann: Hello there\n  how are you?")
        self.assertEqual(turns[0].text, "Hello there how are you?")

    def test_unindented_prefixless_line_is_rejected(self):
        with self.assertRaises(DS.DialogueScriptError) as ctx:
            DS.parse_script("小雪 你好。\n")
        self.assertIn("第 1 行", str(ctx.exception))
        self.assertIn("角色前缀", str(ctx.exception))

    def test_continuation_without_preceding_turn_reports_clearly(self):
        with self.assertRaises(DS.DialogueScriptError):
            DS.parse_script("# 注释\n  孤零零的续行")


class TestInlineOverrides(unittest.TestCase):
    def test_paren_emo_text_is_split_off_the_name(self):
        turns = DS.parse_script("阿明（高兴）: 太棒了！")
        self.assertEqual(turns[0].speaker, "阿明")
        self.assertEqual(turns[0].options["emo_text"], "高兴")

    def test_english_paren_emo_text(self):
        turns = DS.parse_script("Ann (surprised): Oh my god!")
        self.assertEqual(turns[0].speaker, "Ann")
        self.assertEqual(turns[0].options["emo_text"], "surprised")

    def test_options_before_colon(self):
        turns = DS.parse_script("[旁白] @lang=zh @speed=0.9: 走廊尽头的窗户开着一条缝。")
        self.assertEqual(turns[0].speaker, "旁白")
        self.assertEqual(turns[0].options["language"], "zh")
        self.assertEqual(turns[0].options["duration_factor"], 0.9)

    def test_bracket_form_without_colon_keeps_text_intact(self):
        turns = DS.parse_script("[旁白] @lang=zh 走廊尽头的窗户开着一条缝。")
        self.assertEqual(turns[0].options["language"], "zh")
        self.assertEqual(turns[0].text, "走廊尽头的窗户开着一条缝。")

    def test_bracket_form_at_in_text_is_not_an_option(self):
        turns = DS.parse_script("[小雪] 记得 @speed=1 只是文本")
        self.assertEqual(turns[0].options, {})
        self.assertEqual(turns[0].text, "记得 @speed=1 只是文本")

    def test_aliases_and_case_insensitive_keys(self):
        turns = DS.parse_script("小雪 @LANGUage=EN @duration_factor=1.2: Hello.")
        self.assertEqual(turns[0].options["language"], "en")
        self.assertEqual(turns[0].options["duration_factor"], 1.2)

    def test_emo_vector_override(self):
        turns = DS.parse_script("小雪 @emo_vector=0.8,0,0,0,0,0,0,0: 你好。")
        self.assertEqual(turns[0].options["emo_vector"], [0.8, 0, 0, 0, 0, 0, 0, 0])
        self.assertNotIn("emo_text", turns[0].options)

    def test_vector_wins_over_paren_emo_text(self):
        turns = DS.parse_script("小雪（高兴） @emo_vector=0,0,0,0,0,0,0,1: 你好。")
        self.assertNotIn("emo_text", turns[0].options)
        self.assertEqual(turns[0].options["emo_vector"][7], 1.0)

    def test_text_after_colon_is_never_parsed_as_options(self):
        turns = DS.parse_script("小雪: 记得 @speed=1 这种写法只是文本")
        self.assertEqual(turns[0].options, {})
        self.assertEqual(turns[0].text, "记得 @speed=1 这种写法只是文本")


class TestErrorMessages(unittest.TestCase):
    def _error(self, script, **kw):
        with self.assertRaises(DS.DialogueScriptError) as ctx:
            DS.parse_script(script, **kw)
        return str(ctx.exception)

    def test_empty_script(self):
        self.assertIn("对话脚本为空", self._error("   \n\n"))

    def test_unknown_speaker_lists_available(self):
        message = self._error("小雪: 你好。", known_speakers=["阿明", "旁白"])
        self.assertIn("小雪", message)
        self.assertIn("阿明, 旁白", message)

    def test_empty_line_text(self):
        self.assertIn("没有台词内容", self._error("小雪:"))

    def test_unknown_option_key(self):
        message = self._error("小雪 @pitch=3: 你好。")
        self.assertIn("pitch", message)
        self.assertIn("@emo_text=", message)

    def test_unsupported_language(self):
        self.assertIn("ko", self._error("小雪 @lang=ko: 你好。"))

    def test_speed_not_a_number(self):
        self.assertIn("@speed", self._error("小雪 @speed=fast: 你好。"))

    def test_speed_out_of_range(self):
        self.assertIn("语速范围", self._error("小雪 @speed=5: 你好。"))

    def test_vector_wrong_length(self):
        message = self._error("小雪 @emo_vector=0.5,0.5: 你好。")
        self.assertIn("8 个", message)
        self.assertIn("高兴/愤怒/悲伤", message)

    def test_conflicting_emo_text(self):
        self.assertIn("冲突", self._error("小雪（高兴） @emo_text=悲伤: 你好。"))


class TestKnownSpeakersOptional(unittest.TestCase):
    def test_without_registry_any_name_passes(self):
        turns = DS.parse_script("路人甲: 你好。")
        self.assertEqual(turns[0].speaker, "路人甲")


if __name__ == "__main__":
    unittest.main(verbosity=2)
