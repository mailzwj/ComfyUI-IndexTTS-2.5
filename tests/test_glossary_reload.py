"""术语表（glossary）热重载与内联叠加的测试。

运行方式（任选其一）：
    python tests/test_glossary_reload.py
    pytest tests/test_glossary_reload.py -v
"""
import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bootstrap import ML, FakeTTS  # noqa: E402


class TestGlossaryReload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="it25_glossary_")
        self.models_root = os.path.join(self.tmp, "models")
        self.model_dir = os.path.join(self.models_root, "TTS", "IndexTTS-2.5")
        os.makedirs(self.model_dir)
        self.glossary_path = os.path.join(self.model_dir, "glossary.yaml")
        self.loader = ML.IndexTTS2_5Loader(models_root=self.models_root, device="cpu")
        self.tts = FakeTTS()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, content, mtime=None):
        with open(self.glossary_path, "w", encoding="utf-8") as f:
            f.write(content)
        if mtime is not None:
            os.utime(self.glossary_path, (mtime, mtime))

    def test_no_file_and_no_inline(self):
        self.assertEqual(self.loader.refresh_glossary(self.tts), 0)
        self.assertEqual(self.tts.text_process.term_glossary, {})

    def test_file_terms_are_applied(self):
        self._write("M.2:\n  zh: M 二\n  en: M dot two\nNVMe: N-V-M-E\n")
        count = self.loader.refresh_glossary(self.tts)
        self.assertEqual(count, 2)
        terms = self.tts.text_process.term_glossary
        self.assertEqual(terms["M.2"], {"zh": "M 二", "en": "M dot two"})
        self.assertEqual(terms["NVMe"], "N-V-M-E")

    def test_file_change_is_picked_up_without_reload(self):
        self._write("Old: 旧读法\n", mtime=time.time() - 60)
        self.loader.refresh_glossary(self.tts)
        self.assertIn("Old", self.tts.text_process.term_glossary)

        # 改内容并把 mtime 推到未来，模拟模型常驻期间编辑文件
        self._write("New: 新读法\n", mtime=time.time() + 60)
        count = self.loader.refresh_glossary(self.tts)
        self.assertEqual(count, 1)
        terms = self.tts.text_process.term_glossary
        self.assertIn("New", terms)
        self.assertNotIn("Old", terms, "整体替换，不应残留上一次运行的术语")

    def test_inline_overrides_and_extends_file(self):
        self._write("M.2:\n  zh: 文件读法\nOnly-File: 仅文件\n")
        inline = "M.2:\n  zh: 节点读法\nOnly-Inline: 仅节点\n"
        count = self.loader.refresh_glossary(self.tts, inline)
        self.assertEqual(count, 3)
        terms = self.tts.text_process.term_glossary
        self.assertEqual(terms["M.2"], {"zh": "节点读法"})
        self.assertIn("Only-File", terms)
        self.assertIn("Only-Inline", terms)

    def test_unchanged_state_is_not_reapplied(self):
        self._write("A: a读\n")
        first = self.loader.refresh_glossary(self.tts)
        self.tts.text_process.term_glossary["A"] = "被外部改过"
        second = self.loader.refresh_glossary(self.tts)
        self.assertEqual(first, second)
        # 内容未变 -> 认定已生效，不覆盖（幂等）
        self.assertEqual(self.tts.text_process.term_glossary["A"], "被外部改过")

    def test_broken_file_yaml_degrades_to_inline(self):
        self._write("M.2:\n\tbad: [unclosed\n")
        count = self.loader.refresh_glossary(self.tts, "Ok: 好的\n")
        self.assertEqual(count, 1)
        self.assertIn("Ok", self.tts.text_process.term_glossary)

    def test_broken_inline_yaml_keeps_file_terms(self):
        self._write("Ok: 好的\n")
        count = self.loader.refresh_glossary(self.tts, "Ok: [unclosed")
        self.assertEqual(count, 1)
        self.assertEqual(self.tts.text_process.term_glossary["Ok"], "好的")

    def test_illegal_entries_are_dropped(self):
        self._write('"": 空键\nNumber: 3\nGood: 好读法\nList:\n  - a\n')
        count = self.loader.refresh_glossary(self.tts)
        self.assertEqual(count, 1)
        self.assertEqual(self.tts.text_process.term_glossary, {"Good": "好读法"})

    def test_missing_text_process_is_noop(self):
        class Bare:
            pass

        self.assertEqual(self.loader.refresh_glossary(Bare()), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
