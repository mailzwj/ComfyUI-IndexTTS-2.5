"""测试基座：custom node 目录名含 '-'，不是合法包名，手工挂载为 it25 包。

各测试文件统一 `from _bootstrap import ...`，既能 `python tests/xxx.py` 直跑，
也能 `pytest tests/` 收集。
"""
import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_NAMES = ("model_loader", "subtitles", "text_tools", "engine",
                "dialogue_script", "nodes", "nodes_dialogue")


def load_pkg():
    """把节点目录挂载成 `it25` 包并返回它（幂等）。"""
    if "it25" in sys.modules:
        return sys.modules["it25"]
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    pkg = types.ModuleType("it25")
    pkg.__path__ = [ROOT]
    sys.modules["it25"] = pkg
    for name in MODULE_NAMES:
        spec = importlib.util.spec_from_file_location(
            f"it25.{name}", os.path.join(ROOT, f"{name}.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"it25.{name}"] = mod
        spec.loader.exec_module(mod)
        setattr(pkg, name, mod)  # 模拟正常包导入：子模块挂到包属性上
    return pkg


PKG = load_pkg()
ML = PKG.model_loader
NODES = PKG.nodes
SUB = PKG.subtitles
TT = PKG.text_tools
EN = PKG.engine
DS = PKG.dialogue_script
ND = PKG.nodes_dialogue

import numpy as np  # noqa: E402
import torch  # noqa: E402


class StubTextProcess:
    """最小可用的文本前端桩，只承载 term_glossary。"""

    def __init__(self, terms=None):
        self.term_glossary = dict(terms or {})


class FakeTTS:
    """模拟 IndexTTS2：持有大张量（模拟模型权重），流式产出分段音频。"""

    # 各段采样点数，决定每段时长（sr=22050 时 22050 -> 1.0s）
    SEGMENT_SAMPLES = [22050]
    # split_text_by_tokens 的返回值；None 表示整段文本不拆分
    SEGMENT_TEXTS = None
    WARN_ON_CALL = False

    def __init__(self):
        self.gpt = mock.MagicMock()
        self.semantic_model = self.gpt
        self.semantic_codec = self.gpt
        self.s2mel = self.gpt
        self.campplus_model = self.gpt
        self.bigvgan = self.gpt
        self.qwen_emo = None
        self.emo_matrix = torch.zeros(8, 16)
        self.spk_matrix = torch.zeros(8, 16)
        self.cache_spk_cond = torch.zeros(4, 4)
        self.low_vram = False
        self.gr_progress = None
        self.text_process = StubTextProcess()
        self.infer_options = []
        # ~200MB CPU 张量，模拟真实模型占用的内存
        self._ballast = torch.zeros(50 * 1024 * 1024 // 4)

    # ---- 上游分段接口 ------------------------------------------------------
    def split_text_by_tokens(self, text, max_tokens, lang_prefix=""):
        if self.SEGMENT_TEXTS is not None:
            return list(self.SEGMENT_TEXTS)
        return [text]

    def split_text_by_punctuation(self, text, max_chars=40):
        return [text]

    def infer_generator(self, **kwargs):
        self.infer_options.append(kwargs)
        if self.WARN_ON_CALL:
            import warnings
            warnings.warn(
                f"WARN: generation stopped due to exceeding `max_mel_tokens` "
                f"({kwargs.get('max_mel_tokens')}).",
                category=RuntimeWarning)
        # 上游在逐段生成前按 token 预算切分文本，这里保留同样的调用点，
        # 让节点的分段记录器能被真实触发
        self.split_text_by_tokens(
            kwargs.get("text", ""),
            kwargs.get("max_text_tokens_per_segment", 120),
            f"<|{kwargs.get('lang', 'zh')}|> ")
        hook = getattr(self, "gr_progress", None)
        if callable(hook):
            hook(0.1, "starting")
        for samples in self.SEGMENT_SAMPLES:
            # 上游流式产出：float 张量（±32767 量级，shape [1, T]）
            yield torch.full((1, samples), 1000.0, dtype=torch.float32)
            # interval_silence<=0 时上游 interval_silence() 返回 list，非张量
            yield ["silence-placeholder"]


def make_loader(device="cpu"):
    """models_root 传 ROOT 仅为构造合法对象；get_tts 会被 mock，不会真正加载。"""
    return ML.IndexTTS2_5Loader(models_root=ROOT, device=device)


def make_fake_loader(tts):
    """返回 mock loader：save_temp_wav / sampling_rate 走真实实现，get_tts 返回桩。"""
    from unittest import mock

    loader = mock.MagicMock()
    loader.save_temp_wav.side_effect = ML.IndexTTS2_5Loader.save_temp_wav
    loader.sampling_rate.side_effect = ML.IndexTTS2_5Loader.sampling_rate
    loader.refresh_glossary.side_effect = lambda *args, **kwargs: 0
    loader.get_tts.return_value = tts
    return loader


def make_stateful_loader(tts):
    """真实 loader 实例（只把 get_tts/refresh_glossary 打桩）。

    说话人条件缓存这类「有状态」逻辑必须由真实现跑，MagicMock 会把返回值
    变成真值对象，测不出命中/回填是否正确。
    """
    loader = ML.IndexTTS2_5Loader(models_root=ROOT, device="cpu")
    loader.get_tts = lambda **kwargs: tts
    loader.refresh_glossary = lambda *args, **kwargs: 0
    return loader


SR = 22050


class VoiceFactoryMixin(unittest.TestCase):
    """为对话节点造多角色 pack：每个角色的参考音频内容互不相同。

    内容不同才谈得上「按内容指纹命中缓存」；level 决定波形的采样值。
    """

    def voice(self, name, level, language="inherit", duration_factor=None,
              emo_text="", emo_vector="", emo_ref_audio=None):
        wave = torch.full((1, 1, SR // 10), float(level) / 100.0)
        return ND.Voice(name=name, audio={"waveform": wave, "sample_rate": SR},
                        language=language, duration_factor=duration_factor,
                        emo_text=emo_text, emo_vector=emo_vector,
                        emo_ref_audio=emo_ref_audio)

    def pack(self, *specs):
        return {v.name: v for v in (self.voice(*spec) for spec in specs)}

