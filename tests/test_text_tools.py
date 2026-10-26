"""文本前端工具的测试：发音标注校验 + 预处理序列复刻。

prepare_text 会对 indextts.infer_v2_5 做惰性 import，这里注入桩模块，
使测试不需要加载真实模型依赖即可验证调用序列。

运行方式（任选其一）：
    python tests/test_text_tools.py
    pytest tests/test_text_tools.py -v
"""
import os
import re
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bootstrap import TT  # noqa: E402


class TestPronunciationChecks(unittest.TestCase):
    def test_valid_annotations_are_quiet(self):
        self.assertEqual(TT.check_pronunciation_tags("这里读<行|XING2>。", "zh"), [])
        self.assertEqual(TT.check_pronunciation_tags("Read <going|G OW1 . IH0 NG>.", "en"), [])
        self.assertEqual(TT.check_pronunciation_tags("普通文本，没有标注。", "zh"), [])

    def test_language_tokens_are_not_treated_as_annotations(self):
        self.assertEqual(TT.check_pronunciation_tags("<|zh|> 你好。", "zh"), [])

    def test_missing_separator(self):
        warnings = TT.check_pronunciation_tags("这里<行>应该这样读。", "zh")
        self.assertTrue(any("缺少 '|'" in w for w in warnings))

    def test_chinese_pronunciation_without_tone(self):
        warnings = TT.check_pronunciation_tags("<行|hang>", "zh")
        self.assertTrue(any("声调" in w for w in warnings))

    def test_english_lowercase_pronunciation(self):
        warnings = TT.check_pronunciation_tags("<going|g ow1>", "en")
        self.assertTrue(any("ARPAbET" in w for w in warnings))

    def test_empty_pronunciation(self):
        warnings = TT.check_pronunciation_tags("<行|>", "zh")
        self.assertTrue(any("为空" in w for w in warnings))

    def test_unbalanced_brackets(self):
        warnings = TT.check_pronunciation_tags("小于号 < 大于号", "zh")
        self.assertTrue(any("括号不匹配" in w for w in warnings))

    def test_japanese_hint_without_annotation(self):
        warnings = TT.check_pronunciation_tags("漢字の読み方", "ja")
        self.assertTrue(any("仮名" in w or "假名" in w for w in warnings))


class StubTextProcess:
    def __init__(self):
        self.clean_pattern = re.compile(r"[‍]")
        self.char_rep_map = {}
        self.normalize_calls = []

    def normalize(self, text):
        self.normalize_calls.append(text)
        return text.replace("2026", "二零二六")


class StubEmbedding:
    emb = types.SimpleNamespace(num_embeddings=1024)


class StubGpt:
    text_pos_embedding = StubEmbedding()


class StubTTS:
    def __init__(self):
        self.text_process = StubTextProcess()
        self.gpt = StubGpt()

    def _token_len(self, text):
        return len(text)

    def split_text_by_tokens(self, text, max_tokens, lang_prefix=""):
        return [text[:5], text[5:]] if len(text) > 5 else [text]


def _install_indextts_stub():
    """注入 indextts 桩模块：发音标注函数 + NeMo 归一化函数。"""
    package = types.ModuleType("indextts")
    package.__path__ = []
    module = types.ModuleType("indextts.infer_v2_5")
    pattern = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")

    def apply_pronunciation_annotations(text):
        def _replace(match):
            word, pron = match.group(1), match.group(2).upper()
            token = "SPECIAL_TOKEN_2" if re.search(r"[\u4e00-\u9fff]", word) else "SPECIAL_TOKEN_1"
            return f"<|{token}|>{pron}<|{token}|>"
        return pattern.sub(_replace, text)

    module.apply_pronunciation_annotations = apply_pronunciation_annotations

    utils = types.ModuleType("indextts.utils")
    utils.__path__ = []
    nemo = types.ModuleType("indextts.utils.nemo_tn")
    nemo.normalize_text = lambda text, lang: text.upper()
    utils.nemo_tn = nemo

    package.infer_v2_5 = module
    package.utils = utils
    sys.modules["indextts"] = package
    sys.modules["indextts.infer_v2_5"] = module
    sys.modules["indextts.utils"] = utils
    sys.modules["indextts.utils.nemo_tn"] = nemo
    return module


class TestPrepareText(unittest.TestCase):
    def setUp(self):
        _install_indextts_stub()
        self.tts = StubTTS()

    def test_zh_flow_normalizes_lowercases_and_splits(self):
        info = TT.prepare_text(self.tts, "2026Year 你好。", language="zh")
        self.assertTrue(info["normalization_applied"])
        # 与上游 infer_generator 一致：先归一化，再统一小写
        self.assertEqual(self.tts.text_process.normalize_calls, ["2026Year 你好。"])
        self.assertIn("二零二六", info["processed_text"])
        self.assertIn("year", info["processed_text"])  # 中文分支会统一小写
        self.assertEqual(info["segment_count"], len(info["segments"]))
        self.assertEqual(info["token_budget"], 120)
        self.assertEqual(info["over_budget"], [])

    def test_pronunciation_annotation_is_expanded(self):
        info = TT.prepare_text(self.tts, "这里读<行|xing2>。", language="zh")
        self.assertIn("<|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>", info["processed_text"])

    def test_ar_skips_normalization_but_keeps_case(self):
        info = TT.prepare_text(self.tts, "مرحبا World", language="ar")
        self.assertFalse(info["normalization_applied"])
        self.assertIn("World", info["processed_text"])

    def test_es_uppercases(self):
        info = TT.prepare_text(self.tts, "hola mundo", language="es")
        self.assertTrue(info["normalization_applied"])
        self.assertIn("HOLA", info["processed_text"])

    def test_over_budget_is_reported(self):
        tts = StubTTS()
        tts.split_text_by_tokens = lambda text, max_tokens, lang_prefix="": [text]
        info = TT.prepare_text(tts, "一" * 200, language="zh",
                               max_text_tokens_per_segment=50)
        self.assertEqual(info["token_budget"], 50)
        self.assertEqual(info["over_budget"], [1])

    def test_normalization_disabled_does_not_call_normalize(self):
        info = TT.prepare_text(self.tts, "2026年", language="zh", text_normalization=False)
        self.assertEqual(self.tts.text_process.normalize_calls, [])
        self.assertFalse(info["normalization_applied"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
