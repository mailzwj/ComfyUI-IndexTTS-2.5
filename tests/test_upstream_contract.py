"""上游契约守卫：本节点依赖 vendored indextts 的若干内部行为，
这里用源码断言把它们钉住。上游升版导致这些语句变化时，测试会先失败，
而不是等到线上生成出错或字幕错位。

运行方式（任选其一）：
    python tests/test_upstream_contract.py
    pytest tests/test_upstream_contract.py -v
"""
import ast
import os
import re
import unittest

NODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFER_25 = os.path.join(NODE_ROOT, "indextts", "infer_v2_5.py")
FRONT = os.path.join(NODE_ROOT, "indextts", "utils", "front.py")
def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


SOURCE = _read(INFER_25)
FRONT_SOURCE = _read(FRONT)

# 被说话人条件缓存快照的那批张量（见 model_loader.SPK_TENSOR_ATTRS / EMO_TENSOR_ATTRS）
_CONDITION_ATTRS = ("cache_spk_cond", "cache_s2mel_style", "cache_s2mel_prompt",
                    "cache_mel", "cache_emo_cond")
_CONDITION_USE_RE = re.compile(
    r"self\.(" + "|".join(_CONDITION_ATTRS) + r")\b(.*)$", re.MULTILINE)
_INPLACE_MUTATORS = ("[", ".copy_", ".add_", ".mul_", ".zero_", ".data")


def _count(literal, text=SOURCE):
    """按字面量计数（不做正则转义），用于统计源码里某调用了出现几次。"""
    return text.count(literal)


def _func_args(name, path=INFER_25):
    tree = ast.parse(_read(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node.args
    raise AssertionError(f"{os.path.basename(path)} 中找不到函数 {name}")


class TestInferGeneratorContract(unittest.TestCase):
    def test_module_present(self):
        self.assertTrue(os.path.exists(INFER_25))

    def test_single_split_call_site(self):
        """分段记录器依赖「infer_generator 里只有一处 split 调用」。"""
        self.assertEqual(_count("self.split_text_by_tokens("), 1)
        self.assertIn("segments = self.split_text_by_tokens(text, "
                      "max_text_tokens_per_segment, lang_prefix)", SOURCE)

    def test_stream_yields_one_tensor_per_segment_plus_silence(self):
        self.assertIn("if stream_return:\n                    yield wav.cpu()", SOURCE)
        self.assertIn("yield silence", SOURCE)

    def test_silence_helper_returns_list_when_disabled(self):
        """interval_silence<=0 时上游返回的是 list（不是张量），收集器要能跳过。"""
        self.assertIn("if not wavs or interval_silence <= 0:\n            return wavs", SOURCE)

    def test_truncation_warning_mentions_max_mel_tokens(self):
        self.assertIn("generation stopped due to exceeding `max_mel_tokens`", SOURCE)

    def test_progress_hook_shape(self):
        self.assertIn("def _set_gr_progress(self, value, desc):", SOURCE)
        self.assertIn("self.gr_progress(value, desc=desc)", SOURCE)
        self.assertIn("self.gr_progress = None", SOURCE)

    def test_sampling_rate_config_path(self):
        self.assertIn('self.cfg.s2mel["preprocess_params"]["sr"]', SOURCE)

    def test_low_vram_attributes(self):
        self.assertIn("self.low_vram = False", SOURCE)
        self.assertIn("def split_text_by_punctuation(text, max_chars=40)", SOURCE)

    def test_infer_generator_options_we_forward(self):
        """我们用 **infer_generator 的关键字透传参数，签名变了要第一时间发现。

        用 ast 静态解析而非 import：其它测试会往 sys.modules 注入 indextts 桩，
        且真实 import 会拉起整套依赖，守卫测试应当又快又与执行顺序无关。
        """
        args = _func_args("infer_generator")
        names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
        for name in ("spk_audio_prompt", "text", "output_path", "lang",
                     "emo_audio_prompt", "emo_alpha", "emo_vector", "use_emo_text",
                     "emo_text", "use_random", "interval_silence", "verbose",
                     "max_text_tokens_per_segment", "stream_return",
                     "duration_factor", "text_normalization"):
            self.assertIn(name, names, f"infer_generator 缺少参数 {name}")
        self.assertEqual(getattr(args.kwarg, "arg", None), "generation_kwargs",
                         "采样参数依赖 **generation_kwargs 透传")

    def test_speaker_conditioning_cached_by_path(self):
        """上游按参考音频「路径字符串」命中缓存：稳定路径才谈得上复用。"""
        self.assertIn("if self.cache_spk_cond is None or self.cache_spk_audio_prompt "
                      "!= spk_audio_prompt:", SOURCE)

    def test_condition_tensors_are_never_mutated_in_place(self):
        """说话人条件缓存的前提：上游对这些张量只整体赋值或只读，从不就地改。

        同一份张量会被回填给多个角色复用，一旦出现 ``self.cache_spk_cond[...] = ...``
        这类就地修改，就会串音色。上游升版改成就地写入时，本条要先失败。
        """
        for match in _CONDITION_USE_RE.finditer(SOURCE):
            tail = match.group(2)
            if tail.lstrip().startswith("="):  # 整体重新赋值（含置 None）
                continue
            self.assertFalse(
                any(token in tail for token in _INPLACE_MUTATORS),
                f"上游就地改写了条件张量：{match.group(0).strip()}")

    def test_emo_text_falls_back_to_main_text(self):
        self.assertIn("emo_text = text  # use main text prompt", SOURCE)

    def test_pronunciation_annotation_exported(self):
        self.assertIn("def apply_pronunciation_annotations(text: str) -> str:", SOURCE)

    def test_pronunciation_annotation_token_names(self):
        """字幕还原按这两个 token 名识别标注产物，改名要同步 subtitles.py。"""
        self.assertIn("token = 'SPECIAL_TOKEN_2' if has_chinese else 'SPECIAL_TOKEN_1'", SOURCE)
        self.assertIn("f'<|{token}|>{pronunciation}<|{token}|>'", SOURCE)


class TestTextProcessContract(unittest.TestCase):
    def test_glossary_enabled_and_loaded_at_init(self):
        self.assertIn("self.text_process = TextNormalizer(enable_glossary=True)", SOURCE)
        self.assertIn('self.glossary_path = os.path.join(self.model_dir, "glossary.yaml")',
                      SOURCE)

    def test_glossary_yaml_loader_replaces_dict(self):
        self.assertIn("self.term_glossary = external_glossary", FRONT_SOURCE)

    def test_glossary_only_applied_inside_normalize(self):
        """术语表在 normalize 内应用 -> 关闭 text_normalization 时不生效。"""
        self.assertGreaterEqual(_count("self.apply_glossary_terms(text, lang=", FRONT_SOURCE), 2)

    def test_token_budget_formula(self):
        self.assertIn("capacity = self.gpt.text_pos_embedding.emb.num_embeddings", SOURCE)
        self.assertIn("budget = min(max_tokens, capacity - 2) - self._token_len(lang_prefix)",
                      SOURCE)

    def test_preprocessing_sequence_we_replicate(self):
        for statement in (
            "text = self.text_process.clean_pattern.sub(lambda x: self.text_process.char_rep_map[x.group()], text)",
            "text = self.text_process.normalize(text)",
            "text = nemo_text_normalize(text, lang.lower())",
            "text = apply_pronunciation_annotations(text)",
            "text = self.ja_text_process.process_ja_text(text)",
            "lang_prefix = f'<|{lang.lower()}|> '",
        ):
            self.assertIn(statement, SOURCE, f"预处理语句变化：{statement}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
