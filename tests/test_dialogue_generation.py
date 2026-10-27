"""多人对话节点的测试（桩模型，不加载真实权重）。

覆盖：整段音频长度 = 各 turn 之和 + 分档静音、字幕角色前缀与时间轴、
说话人条件缓存命中次数、行级覆盖透传到 infer 参数、错误路径与临时文件清理。

运行方式（任选其一）：
    python tests/test_dialogue_generation.py
    pytest tests/test_dialogue_generation.py -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bootstrap import DS, ML, ND, FakeTTS, VoiceFactoryMixin, make_stateful_loader  # noqa: E402

import torch  # noqa: E402

SR = 22050
TURN_GAP = 450
SAME_GAP = 200
INNER_GAP = 200


def _gap(ms):
    return int(SR * max(0, int(ms)) / 1000)


class OneSegmentTTS(FakeTTS):
    """每个 turn 产出一段 0.5s 音频，段文本就是这句台词。"""

    SEGMENT_SAMPLES = [SR // 2]


class TwoSegmentTTS(FakeTTS):
    """每个 turn 内部再拆两段，用来验证「句内静音」这一档。"""

    SEGMENT_SAMPLES = [SR // 4, SR // 4]
    SEGMENT_TEXTS = ["前半。", "后半。"]


THREE_VOICES = """小雪: 你终于来了。
阿明: 给你带了杯咖啡。
小雪: 谢谢。
"""


class DialogueNodeMixin(VoiceFactoryMixin):
    def setUp(self):
        super().setUp()
        ML._shared_loader = None
        self.tts = OneSegmentTTS()
        self.loader = make_stateful_loader(self.tts)
        self.temp_files = []

        def save_temp_wav(audio):
            path = ML.IndexTTS2_5Loader.save_temp_wav(audio)
            self.temp_files.append(path)
            return path

        self.loader.save_temp_wav = save_temp_wav
        self.get_tts_calls = []

        def get_tts(**kwargs):
            self.get_tts_calls.append(kwargs)
            return self.tts

        self.loader.get_tts = get_tts
        self.node = ND.IndexTTSDialogueNode()
        self.node.loader = self.loader

    def tearDown(self):
        for path in self.temp_files:
            if os.path.exists(path):
                os.remove(path)

    def run_dialogue(self, script=THREE_VOICES, tts=None, pack=None, **kwargs):
        if tts is not None:
            self.tts = tts
            self.loader.get_tts = lambda **kw: tts
        args = dict(dialogue_script=script,
                    speakers=pack if pack is not None else self.pack(
                        ("小雪", 1.0), ("阿明", 2.0)),
                    default_language="zh", turn_gap=TURN_GAP,
                    same_speaker_gap=SAME_GAP, interval_silence=INNER_GAP, seed=7)
        args.update(kwargs)
        return self.node.generate(**args)

    def blocks(self, subtitle):
        return [b for b in subtitle.strip().split("\n\n") if b]


class TestDialogueAudio(DialogueNodeMixin):
    def test_total_length_is_turns_plus_gaps(self):
        audio, seed, _, _ = self.run_dialogue()
        expected = 3 * (SR // 2) + 2 * _gap(TURN_GAP)
        self.assertEqual(audio["waveform"].shape[-1], expected)
        self.assertEqual(audio["sample_rate"], SR)
        self.assertEqual(seed, 7)

    def test_same_speaker_boundary_uses_same_speaker_gap(self):
        script = "小雪: 第一句。\n小雪: 第二句。\n阿明: 第三句。\n"
        audio, _, _, _ = self.run_dialogue(script)
        expected = 3 * (SR // 2) + _gap(SAME_GAP) + _gap(TURN_GAP)
        self.assertEqual(audio["waveform"].shape[-1], expected)

    def test_intra_turn_segments_use_interval_silence(self):
        audio, _, _, _ = self.run_dialogue(tts=TwoSegmentTTS())
        # 每句两段 -> 6 段；句内 3 个边界用 interval_silence，换角色 2 个用 turn_gap
        expected = 6 * (SR // 4) + 3 * _gap(INNER_GAP) + 2 * _gap(TURN_GAP)
        self.assertEqual(audio["waveform"].shape[-1], expected)

    def test_continuation_line_is_one_turn_not_two(self):
        script = "阿明: 给你带了杯咖啡！\n  还是冰的。\n小雪: 谢谢。\n"
        audio, _, _, report = self.run_dialogue(script)
        self.assertIn("台词 2 句 / 音频 2 段", report)
        self.assertEqual(audio["waveform"].shape[-1],
                         2 * (SR // 2) + _gap(TURN_GAP))


class TestDialogueSubtitle(DialogueNodeMixin):
    def test_speaker_prefix_and_timeline(self):
        _, _, subtitle, _ = self.run_dialogue()
        blocks = self.blocks(subtitle)
        self.assertEqual(len(blocks), 3)
        self.assertTrue(blocks[0].startswith("1\n00:00:00,000 --> 00:00:00,500"))
        self.assertIn("【小雪】你终于来了。", blocks[0])
        # 第 2 句从 0.5s + 0.45s 停顿后开始
        self.assertTrue(blocks[1].startswith("2\n00:00:00,950 --> 00:00:01,450"))
        self.assertIn("【阿明】给你带了杯咖啡。", blocks[1])

    def test_prefix_can_be_disabled(self):
        _, _, subtitle, _ = self.run_dialogue(subtitle_speaker=False)
        self.assertNotIn("【小雪】", subtitle)
        self.assertIn("你终于来了。", subtitle)

    def test_custom_prefix_template(self):
        _, _, subtitle, _ = self.run_dialogue(subtitle_speaker_prefix="{name}: ")
        self.assertIn("小雪: 你终于来了。", subtitle)

    def test_plain_format_lists_speakers(self):
        _, _, subtitle, _ = self.run_dialogue(subtitle_format="plain")
        self.assertIn(">> 【小雪】你终于来了。", subtitle.splitlines())


class TestDialogueInferOptions(DialogueNodeMixin):
    def test_line_overrides_reach_the_model(self):
        script = ("[旁白] @lang=en @speed=1.5: Hello there.\n"
                  "小雪（高兴）: 你来啦。\n"
                  "阿明 @emo_vector=0,0,0,0,0,0,0,1: 嗯。\n")
        pack = self.pack(("旁白", 3.0), ("小雪", 1.0), ("阿明", 2.0))
        _, _, _, _ = self.run_dialogue(script, pack=pack)
        calls = self.tts.infer_options
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["lang"], "en")
        self.assertEqual(calls[0]["duration_factor"], 1.5)
        self.assertTrue(calls[1]["use_emo_text"])
        self.assertEqual(calls[1]["emo_text"], "高兴")
        self.assertEqual(calls[2]["emo_vector"][7], 1.0)
        self.assertFalse(calls[2]["use_emo_text"])

    def test_settings_node_overrides_defaults(self):
        options = {"temperature": 0.3, "max_text_tokens_per_segment": 60,
                   "num_beams": 1, "glossary_yaml": ""}
        self.run_dialogue(options=options)
        first = self.tts.infer_options[0]
        self.assertEqual(first["temperature"], 0.3)
        self.assertEqual(first["max_text_tokens_per_segment"], 60)
        self.assertEqual(first["num_beams"], 1)

    def test_emo_text_turn_loads_qwen_model(self):
        self.run_dialogue("小雪（难过）: 你好。")
        self.assertTrue(self.get_tts_calls[0]["use_qwen_emo"])

    def test_no_emo_text_skips_qwen_model(self):
        self.run_dialogue()
        self.assertFalse(self.get_tts_calls[0]["use_qwen_emo"])

    def test_speaker_language_default_is_used(self):
        pack = self.pack(("小雪", 1.0, "en"), ("阿明", 2.0))
        self.run_dialogue("小雪: Hello.\n阿明: 你好。\n", pack=pack)
        self.assertEqual(self.tts.infer_options[0]["lang"], "en")
        self.assertEqual(self.tts.infer_options[1]["lang"], "zh")


class TestDialogueCacheAndReport(DialogueNodeMixin):
    def test_each_voice_extracts_once_and_hits_are_reported(self):
        _, _, _, report = self.run_dialogue()
        self.assertIn("说话人条件缓存：开启，3 句中命中 1 次", report)
        self.assertEqual(len(self.tts._it25_spk_cache), 2)

    def test_disabled_cache_reports_zero_hits(self):
        _, _, _, report = self.run_dialogue(speaker_cache=False)
        self.assertIn("说话人条件缓存：关闭，3 句中命中 0 次", report)
        self.assertFalse(hasattr(self.tts, "_it25_spk_cache"))

    def test_per_line_notes_list_speaker_and_duration(self):
        _, _, _, report = self.run_dialogue()
        lines = report.splitlines()
        self.assertIn("角色 2 个：小雪, 阿明", lines[0])
        self.assertTrue(lines[1].startswith("台词 3 句 / 音频 3 段 / 总时长 "))
        self.assertIn("[1/3] 行 1 小雪：lang=zh speed=1", lines[4])

    def test_continuation_lines_are_one_turn_in_the_subtitle(self):
        script = "小雪: 第一句。\n  还是冰的。\n阿明: 第二句。\n"
        _, _, subtitle, _ = self.run_dialogue(script)
        self.assertEqual(len(self.blocks(subtitle)), 3)
        self.assertIn("【小雪】第一句。", subtitle)
        self.assertIn("【小雪】还是冰的。", subtitle)


class TestDialogueDefaults(DialogueNodeMixin):
    def test_panel_default_script_runs_with_two_example_roles(self):
        """面板默认示例只用小雪/阿明：连上最常见的双人 Pack 就能直接出活。

        示例里曾经写过未注册的「旁白」，用户拖出新节点第一下就报错。
        """
        turns = DS.parse_script(ND.SCRIPT_EXAMPLE, known_speakers=["小雪", "阿明"])
        self.assertEqual([t.speaker for t in turns], ["小雪", "阿明", "小雪"])
        audio, _, subtitle, report = self.run_dialogue(ND.SCRIPT_EXAMPLE)
        self.assertIn("台词 3 句 / 音频 3 段", report)
        self.assertIn("【阿明】给你带了杯咖啡！", subtitle)

    def test_unknown_speaker_message_offers_a_remedy(self):
        with self.assertRaises(ValueError) as ctx:
            self.run_dialogue("旁白: 走廊尽头的窗户开着一条缝。\n")
        message = str(ctx.exception)
        self.assertIn("未注册", message)
        self.assertIn("IndexTTS Speaker", message)
        self.assertIn("改成已注册的角色", message)


class TestDialogueErrors(DialogueNodeMixin):
    def test_unknown_speaker_is_rejected_before_generation(self):
        with self.assertRaises(ValueError) as ctx:
            self.run_dialogue("路人: 你好。\n")
        self.assertIn("未注册", str(ctx.exception))
        self.assertEqual(self.tts.infer_options, [])

    def test_empty_pack_is_rejected(self):
        with self.assertRaises(ValueError):
            self.node.generate(dialogue_script="小雪: 你好。", speakers={},
                               default_language="zh")

    def test_temp_wavs_are_cleaned_up(self):
        self.run_dialogue()
        self.assertTrue(self.temp_files)
        for path in self.temp_files:
            self.assertFalse(os.path.exists(path))

    def test_temp_wavs_cleaned_up_on_error(self):
        class BoomTTS(FakeTTS):
            def infer_generator(self, **kwargs):
                raise RuntimeError("上游炸了")

        with self.assertRaises(RuntimeError):
            self.run_dialogue(tts=BoomTTS())
        self.assertTrue(self.temp_files)
        for path in self.temp_files:
            self.assertFalse(os.path.exists(path))


class TestSpeakerNodes(unittest.TestCase):
    def test_speaker_node_returns_voice(self):
        voice = ND.IndexTTSSpeakerNode().build(
            "小雪", {"waveform": torch.zeros(1, 1, 10), "sample_rate": SR},
            language="en", duration_factor=1.2)[0]
        self.assertEqual(voice.name, "小雪")
        self.assertEqual(voice.language, "en")
        self.assertEqual(voice.duration_factor, 1.2)

    def test_speaker_node_rejects_empty_name(self):
        with self.assertRaises(ValueError):
            ND.IndexTTSSpeakerNode().build("  ", {"waveform": torch.zeros(1, 1, 10),
                                                  "sample_rate": SR})

    def test_pack_chaining_and_duplicate_names(self):
        node = ND.IndexTTSSpeakerPackNode()
        audio = {"waveform": torch.zeros(1, 1, 10), "sample_rate": SR}
        first = node.build(s1=ND.IndexTTSSpeakerNode().build("小雪", audio)[0])[0]
        second = node.build(prev_pack=first,
                            s1=ND.IndexTTSSpeakerNode().build("阿明", audio)[0])[0]
        self.assertEqual(sorted(second), ["小雪", "阿明"])
        with self.assertRaises(ValueError):
            node.build(prev_pack=first,
                       s1=ND.IndexTTSSpeakerNode().build("小雪", audio)[0])

    def test_empty_pack_is_rejected(self):
        with self.assertRaises(ValueError):
            ND.IndexTTSSpeakerPackNode().build()

    def test_settings_node_drops_none_values(self):
        options = ND.IndexTTSSettingsNode().build(temperature=0.5, top_k=None)[0]
        self.assertEqual(options, {"temperature": 0.5})


if __name__ == "__main__":
    unittest.main(verbosity=2)
