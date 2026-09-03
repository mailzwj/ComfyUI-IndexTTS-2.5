"""IndexTTS-2.5 模型加载与包装。

将官方 index-tts 仓库的 ``indextts`` 包 vendored 到本节点目录下，
在 ComfyUI 进程内以惰性方式加载 ``indextts.infer_v2_5.IndexTTS2``。

模型目录约定：``<ComfyUI>/models/TTS/IndexTTS-2.5``
"""
import os
import gc
import sys
import tempfile
import threading

import numpy as np
import torch
import soundfile as sf


class IndexTTS2_5Loader:
    """IndexTTS-2.5 的轻量加载器。

    职责：
    - 解析模型目录（models/TTS/IndexTTS-2.5）
    - 校验必需文件
    - 将 vendored 源码挂到 sys.path 并惰性构造 IndexTTS2
    - release() 释放全部缓存模型，归还内存/显存
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
        # 用户显式指定的 device（None 表示交给 IndexTTS2 自动探测 cuda/xpu/mps/cpu）
        self._device_requested = device
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._cache = {}
        # 保护 _cache 的并发访问（ComfyUI 多 prompt 并发执行时避免重复加载）
        self._lock = threading.Lock()
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
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        with self._lock:
            # 双检：并发下可能已有其它线程完成加载
            cached = self._cache.get(key)
            if cached is not None:
                return cached

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
                # None → 由 IndexTTS2 自动探测设备（保持原有行为）；显式指定时尊重之
                device=self._device_requested,
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

    def release(self):
        """释放全部缓存的 IndexTTS2 实例，归还内存/显存。

        - 从缓存中移除所有 TTS 实例并显式断开其大子模型引用（帮助 GC 回收循环引用）；
        - gc.collect() 回收 Python 层内存；
        - torch.cuda.empty_cache()（或 MPS 对应接口）把 PyTorch 缓存分配器
          预留的显存真正归还给系统，供 ComfyUI 中其它模型使用。

        释放后下次 get_tts() 会重新加载模型（有首次加载耗时）。
        """
        with self._lock:
            released = list(self._cache.values())
            self._cache.clear()

        if not released:
            return

        for tts in released:
            self._drop_tts_refs(tts)

        gc.collect()
        freed_vram_gb = None
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                free, total = torch.cuda.mem_get_info()
                freed_vram_gb = (total - free) / (1024 ** 3)
            except Exception as e:  # noqa: BLE001 - 释放尽力而为，不阻断主流程
                print(f"[IndexTTS-2.5] CUDA 显存回收失败（忽略）: {e!r}")
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception as e:  # noqa: BLE001
                print(f"[IndexTTS-2.5] MPS 显存回收失败（忽略）: {e!r}")

        if freed_vram_gb is not None:
            print(f"[IndexTTS-2.5] 已释放模型，当前 GPU 占用约 {freed_vram_gb:.2f} GB")
        else:
            print("[IndexTTS-2.5] 已释放模型（CPU 模式），内存已交还系统")

    @staticmethod
    def _drop_tts_refs(tts) -> None:
        """显式断开 IndexTTS2 内部大对象引用，帮助 GC 及时回收。"""
        attrs = (
            "gpt", "semantic_model", "semantic_codec", "s2mel",
            "campplus_model", "bigvgan", "qwen_emo",
            "emo_matrix", "spk_matrix",
            # infer 过程中按参考音频缓存的中间张量
            "cache_spk_cond", "cache_s2mel_style", "cache_s2mel_prompt",
            "cache_emo_cond", "cache_mel",
        )
        for attr in attrs:
            try:
                if hasattr(tts, attr):
                    setattr(tts, attr, None)
            except Exception:  # noqa: BLE001 - 单个属性失败不影响整体释放
                pass

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


# ---- 进程级共享加载器 -------------------------------------------------------
_shared_loader = None
_shared_loader_lock = threading.Lock()


def get_shared_loader(models_root=None, device=None) -> IndexTTS2_5Loader:
    """返回进程内共享的加载器单例。

    ComfyUI 会为工作流中每个节点实例分别构造类对象；若各自持有独立 loader，
    同一张图里放多个 IndexTTS 节点会重复加载整套模型（显存成倍增长）。
    共享后：相同参数的节点复用同一模型实例，release() 也能一次性全部释放。
    """
    global _shared_loader
    with _shared_loader_lock:
        if _shared_loader is None:
            _shared_loader = IndexTTS2_5Loader(models_root=models_root, device=device)
        return _shared_loader
