"""IndexTTS-2.5 模型加载与包装。

将官方 index-tts 仓库的 ``indextts`` 包 vendored 到本节点目录下，
在 ComfyUI 进程内以惰性方式加载 ``indextts.infer_v2_5.IndexTTS2``。

模型目录约定：``<ComfyUI>/models/TTS/IndexTTS-2.5``
"""
import os
import gc
import sys
import tempfile

import numpy as np
import torch
import soundfile as sf


class IndexTTS2_5Loader:
    """IndexTTS-2.5 的轻量加载器。

    职责：
    - 解析模型目录（models/TTS/IndexTTS-2.5）
    - 校验必需文件
    - 将 vendored 源码挂到 sys.path 并惰性构造 IndexTTS2
    """

    DEFAULT_DIRNAME = "IndexTTS-2.5"

    # 必需模型文件（其余辅助模型 w2v-bert-2.0 / campplus / bigvgan 会在
    # 首次运行时由 upstream 的 ensure_models_available 下载到 hf_cache/）。
    REQUIRED_FILES = [
        "config.yaml",
        "gpt.pth",
        "s2mel.pth",
        "codec.pth",
        "wav2vec2bert_stats.pt",
        "feat1.pt",
        "feat2.pt",
        "multilingual_zh_ja_yue_char_del.tiktoken",
    ]

    def __init__(self, models_root=None, device=None):
        self._models_root = models_root or self._default_models_root()
        # 模型放在 models/TTS/ 下
        self._model_dir = os.path.join(self._models_root, "TTS", self.DEFAULT_DIRNAME)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache = {}
        # vendored 源码根目录：<node>/indextts
        self._node_dir = os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def _default_models_root() -> str:
        # 本文件位于 .../ComfyUI/custom_nodes/ComfyUI-IndexTTS-2_5/model_loader.py
        # 向上 3 级得到 .../ComfyUI/，再拼接 models。
        here = os.path.abspath(__file__)
        for _ in range(3):
            here = os.path.dirname(here)
        return os.path.join(here, "models")

    @property
    def model_dir(self) -> str:
        return self._model_dir

    @property
    def device(self) -> str:
        return self._device

    def validate(self) -> None:
        missing = [f for f in self.REQUIRED_FILES
                   if not os.path.exists(os.path.join(self._model_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"IndexTTS-2.5 模型目录缺少文件 {self._model_dir}: {', '.join(missing)}"
            )

    def get_tts(self, use_bf16=True, use_qwen_emo=False):
        """返回缓存的 IndexTTS2 实例（按关键参数区分缓存）。"""
        key = ("tts", bool(use_bf16), bool(use_qwen_emo))
        if key in self._cache:
            return self._cache[key]

        self.validate()

        # 让 vendored 的 `indextts` 包可被 import（放在 sys.path 最前）
        if self._node_dir not in sys.path:
            sys.path.insert(0, self._node_dir)
        elif sys.path.index(self._node_dir) != 0:
            sys.path.remove(self._node_dir)
            sys.path.insert(0, self._node_dir)

        # 清除旧的 indextts 模块，避免与其它 vendored 版本冲突
        for k in list(sys.modules.keys()):
            if k == "indextts" or k.startswith("indextts."):
                del sys.modules[k]
        gc.collect()

        from indextts.infer_v2_5 import IndexTTS2  # noqa: E402

        on_cuda = self._device.startswith("cuda")
        tts = IndexTTS2(
            cfg_path=os.path.join(self._model_dir, "config.yaml"),
            model_dir=self._model_dir,
            use_bf16=bool(use_bf16) and on_cuda,
            # 保守起见默认关闭 CUDA kernel / flash-attn 加速，
            # 避免 torch2.9+cu130 下预编译扩展 ABI 不匹配；需要可再开启。
            use_cuda_kernel=False,
            use_deepspeed=False,
            use_accel=False,
            use_torch_compile=False,
            use_qwen_emo=bool(use_qwen_emo),
        )
        self._cache[key] = tts
        return tts

    @staticmethod
    def save_temp_wav(audio) -> str:
        """把 (wave, sr) 写入临时 wav，返回路径。wave 为 float32 单声道。"""
        wave, sr = audio
        wave = np.asarray(wave, dtype=np.float32)
        if wave.ndim == 2:
            wave = wave.mean(axis=0)
        wave = np.clip(wave, -1.0, 1.0)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, wave, int(sr))
        return path
