"""IndexTTS-2.5 模型加载与包装。

将官方 index-tts 仓库的 ``indextts`` 包 vendored 到本节点目录下，
在 ComfyUI 进程内以惰性方式加载 ``indextts.infer_v2_5.IndexTTS2``。

模型目录约定：``<ComfyUI>/models/TTS/IndexTTS-2.5``
"""
import os
import gc
import re
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
        # glossary.yaml 热重载缓存：(mtime, 解析后的术语字典)
        self._glossary_cache = None

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

    # ---- 术语表（glossary）--------------------------------------------------
    # 上游的词表里没有「拼音+声调」这种 token：把中文词直接替换成裸拼音（朝 -> zhao1）
    # 只会让模型逐字母拼读，所以要提示改用 <文字|读音> 标注形式。
    _CJK_RE = re.compile(r"[\u4e00-\u9fff]")
    _BARE_PINYIN_RE = re.compile(r"^(?:[a-zü]+[1-5])+$")

    @classmethod
    def _looks_like_bare_pinyin(cls, term, reading):
        """仅当术语本身是中文、读法是纯「拼音+声调」串时才判定为误用。"""
        if not cls._CJK_RE.search(term):
            return False
        return bool(cls._BARE_PINYIN_RE.match(reading.strip().lower().replace(" ", "")))

    @classmethod
    def _normalize_glossary(cls, data, source):
        """校验并规范化术语字典，忽略结构不合法的条目。"""
        terms = {}
        if not isinstance(data, dict):
            return terms
        for key, value in data.items():
            if not isinstance(key, str) or not key.strip():
                print(f"[IndexTTS-2.5] {source}: 术语键不合法，已忽略 -> {key!r}")
                continue
            if isinstance(value, str):
                terms[key] = value
                readings = [value]
            elif isinstance(value, dict):
                per_lang = {k: v for k, v in value.items()
                            if isinstance(k, str) and isinstance(v, str)}
                if not per_lang:
                    print(f"[IndexTTS-2.5] {source}: 术语 {key} 的读法不是字符串字典，已忽略")
                    continue
                terms[key] = per_lang
                readings = list(per_lang.values())
            else:
                print(f"[IndexTTS-2.5] {source}: 术语 {key} 的值类型不支持，已忽略")
                continue
            for reading in readings:
                if cls._looks_like_bare_pinyin(key, reading):
                    print(f"[IndexTTS-2.5] {source}: 术语 {key} 的读法 {reading!r} 是裸拼音，"
                          f"模型不会按拼音读（词表无此 token），建议写成 <{key}|{reading}>。")
                    break
        return terms

    def _read_glossary_file(self):
        """读取 model_dir/glossary.yaml，按 mtime 缓存解析结果。"""
        path = os.path.join(self._model_dir, "glossary.yaml")
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return {}, 0.0
        if self._glossary_cache and self._glossary_cache[0] == mtime:
            return self._glossary_cache[1], mtime
        terms = {}
        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            terms = self._normalize_glossary(loaded, path)
        except Exception as e:  # noqa: BLE001 - 术语表损坏不应阻断生成
            print(f"[IndexTTS-2.5] 术语表解析失败（忽略文件内容）: {e!r}")
        self._glossary_cache = (mtime, terms)
        return terms, mtime

    def refresh_glossary(self, tts, inline_yaml=""):
        """把「文件术语 + 节点内联术语」写入运行中的 text_process。

        上游只在模型初始化时读一次 glossary.yaml，模型常驻缓存时改文件不会生效，
        因此这里按 mtime 重新解析并整体替换（避免上一次运行残留的术语继续生效）。
        """
        text_process = getattr(tts, "text_process", None)
        if text_process is None:
            return 0
        base, _ = self._read_glossary_file()
        inline = {}
        raw = (inline_yaml or "").strip()
        if raw:
            try:
                import yaml
                inline = self._normalize_glossary(yaml.safe_load(raw), "节点 glossary_yaml")
            except Exception as e:  # noqa: BLE001 - 内联 YAML 写错时退回文件术语
                print(f"[IndexTTS-2.5] 节点内联术语表解析失败（忽略）: {e!r}")
        merged = dict(base)
        merged.update(inline)
        if getattr(tts, "_it25_glossary_state", None) == merged:
            return len(merged)
        text_process.term_glossary = merged
        # 存快照而不是同一个对象，否则外部改写 term_glossary 会被误判成「内容变了」
        tts._it25_glossary_state = {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in merged.items()
        }
        print(f"[IndexTTS-2.5] 术语表已生效：文件 {len(base)} 条 + 节点 {len(inline)} 条")
        return len(merged)

    def get_tts(self, use_bf16=True, use_qwen_emo=False):
        """返回缓存的 IndexTTS2 实例（按关键参数区分缓存）。

        ``use_bf16`` 会按显卡算力自动降级（见 :meth:`_bf16_supported`），缓存 key 用
        降级后的值，否则老卡上 True/False 会各加载一份完全相同的 fp32 模型。
        """
        effective_bf16 = bool(use_bf16) and self._bf16_supported()
        key = ("tts", effective_bf16, bool(use_qwen_emo))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        with self._lock:
            # 双检：并发下可能已有其它线程完成加载
            cached = self._cache.get(key)
            if cached is not None:
                return cached

            if use_bf16 and not effective_bf16:
                print(f"[IndexTTS-2.5] {self._device} 不支持 bf16（需 Ampere/sm_80 及以上），"
                      f"已按 fp32 加载；速度会慢于新卡。")

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

            tts = IndexTTS2(
                cfg_path=os.path.join(self._model_dir, "config.yaml"),
                model_dir=self._model_dir,
                # None → 由 IndexTTS2 自动探测设备（保持原有行为）；显式指定时尊重之
                device=self._device_requested,
                use_bf16=effective_bf16,
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

    # Ampere（sm_80）起才有 bf16 硬件支持；Turing/Pascal/Volta 走 bf16 只会更慢甚至报错。
    MIN_BF16_CAPABILITY = (8, 0)

    def _bf16_supported(self) -> bool:
        """当前设备能否安全使用 bf16。"""
        if not self._device.startswith("cuda"):
            return False
        try:
            index = int(self._device.rsplit(":", 1)[-1]) if ":" in self._device else 0
            return torch.cuda.get_device_capability(index) >= self.MIN_BF16_CAPABILITY
        except Exception as e:  # noqa: BLE001 - 问不到算力就按 fp32，宁可慢也不要崩
            print(f"[IndexTTS-2.5] 读取 GPU 算力失败，按 fp32 加载: {e!r}")
            return False

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
    def sampling_rate(tts) -> int:
        """输出采样率：读模型配置，失败时退回上游硬编码的 22050。"""
        try:
            # OmegaConf 的 DictConfig 与普通 dict 都支持下标访问
            return int(tts.cfg["s2mel"]["preprocess_params"]["sr"])
        except Exception:  # noqa: BLE001 - 配置结构变化不应阻断生成
            return 22050

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
