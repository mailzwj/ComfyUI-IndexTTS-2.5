"""ComfyUI 节点：Index TTS 2.5 文本转语音。

参考 ComfyUI-Index-TTS 的节点设计，适配 IndexTTS-2.5 的官方 API
（indextts.infer_v2_5.IndexTTS2.infer_generator）。
"""
import json
import os
import random
import sys
import threading
import warnings

import numpy as np
import torch

from . import subtitles, text_tools
from .model_loader import get_shared_loader

# 生成期间会对共享的 tts 实例打补丁（分段记录、进度钩子），必须串行执行。
# GPU 推理本身即串行，因此不会损失吞吐。
_GEN_LOCK = threading.Lock()


class IndexTTS2_5Node:
    """Index TTS 2.5 单节点，覆盖文本 / 语言 / 语速 / 情感 / 长文本 / 字幕控制。"""

    # 情感向量顺序：高兴 愤怒 悲伤 恐惧 反感 低落 惊讶 自然
    EMO_ORDER = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "你好，欢迎使用 IndexTTS 2.5 语音合成。"}),
                "reference_audio": ("AUDIO",),
                "language": (["zh", "en", "ja", "es", "ar"], {"default": "zh"}),
                "duration_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 32 - 1}),
            },
            "optional": {
                # ---- 情感 ----
                "emo_ref_audio": ("AUDIO", {
                    "tooltip": "情感参考音频：用一段带目标情绪的语音引导合成情感。\n提供 emo_vector 或 emo_text 后此输入会被忽略。",
                }),
                "emo_alpha": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "情感强度（0~1）：对 emo_vector 做整体缩放；\n或控制情感参考音频与说话人音色的混合比例。",
                }),
                "emo_vector": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": "8 维情感向量，顺序固定：高兴/愤怒/悲伤/恐惧/反感/低落/惊讶/自然。\n逗号分隔或 JSON 数组，如 0.8,0,0,0,0,0,0,0 或 [0,0,0,0,0,0,0,1]（自然）。\n必须恰好 8 个数字，否则忽略；提供后覆盖 emo_ref_audio。",
                }),
                "emo_text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "情感文本：用自然语言描述情绪（如「高兴」「悲伤」「愤怒」），由 QwenEmotion 自动转为情感向量。\n非空即自动启用，需 qwen0.6bemo4-merge 模型；提供后覆盖 emo_ref_audio。\n留空但勾选 use_emo_text 时，直接用正文内容推断情绪。",
                }),
                "use_emo_text": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "使用情感文本。emo_text 非空时自动启用；emo_text 留空则用正文内容推断情绪。",
                }),
                "use_qwen_emo": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "加载 QwenEmotion 情感模型（仅情感文本需要；使用 emo_text 时自动加载）。",
                }),
                "use_random": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "随机情感：随机挑选情感参考向量（需同时提供 emo_vector 或 emo_text 才生效）。",
                }),
                # ---- 采样参数 ----
                "do_sample": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "length_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "max_mel_tokens": ("INT", {"default": 1500, "min": 50, "max": 1815, "step": 5}),
                # ---- 长文本 / 分段 ----
                "max_text_tokens_per_segment": ("INT", {
                    "default": 120, "min": 10, "max": 300, "step": 5,
                    "tooltip": "每段最多合成的文本 token 数（上游按标点优先切分）。\n长文本被截断（话说一半就停）时调小此值，逐段生成更稳；\n调大则跨句韵律更连贯，但更容易撞上 max_mel_tokens。\n可用 Text Preview 节点预览实际分段与各段 token 数。",
                }),
                "interval_silence": ("INT", {
                    "default": 200, "min": 0, "max": 2000, "step": 10,
                    "tooltip": "分段之间插入的静音时长（毫秒）。0 表示无缝拼接，\n长文本听感更连贯；200~500 更接近自然句间停顿。",
                }),
                "strict_max_mel_tokens": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "某段因超出 max_mel_tokens 被截断时：\n勾选 = 直接报错（避免产出残缺音频还继续跑）；\n不勾选（默认）= 仅在控制台打印可操作的提示。",
                }),
                # ---- 文本前端 / 发音 ----
                "text_normalization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "文本归一化（读法转换）：中/英走 wetext 规则，日/西语走 NeMo，阿拉伯语不处理。\n「2026年3月5日」「M.2」这类内容依赖它才能读对，一般保持开启；\n已经手工写好读法、或归一化把专有名词改坏时再关掉。\n注意：术语表（glossary）只在归一化阶段生效，关闭后 glossary 不起作用。",
                }),
                "glossary_yaml": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "内联术语表（YAML），叠加在模型目录的 glossary.yaml 之上，键为术语、值为读法：\n  M.2:\n    zh: M 二\n    en: M dot two\n  NVMe: N-V-M-E\n同名条目以这里为准；留空则只用文件术语。模型常驻期间改文件也会自动热重载。",
                }),
                # ---- 字幕 ----
                "subtitle_format": (["srt", "vtt", "plain"], {
                    "default": "srt",
                    "tooltip": "subtitle 输出格式：\nsrt = 标准 SubRip（可直接导入剪映/PR/达芬奇或字幕节点）；\nvtt = WebVTT（网页播放器）；\nplain = 旧的「>> 0:02.130-0:05.870」文本行。",
                }),
                "subtitle_text": (["synthesized", "original"], {
                    "default": "synthesized",
                    "tooltip": "字幕文字来源：\nsynthesized = 模型实际朗读的文本（归一化后，如「二零二六年」），时间轴最准；\noriginal = 你的原始输入（措辞干净），段内按字符占比映射回原文，边界略有近似。",
                }),
                # ---- 模型加载 ----
                "use_bf16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "以 bf16 加载 GPT 主干（显存约省一半、速度更快），仅在 CUDA 上生效。\n"
                               "需要 Ampere（RTX 30 系 / A 系列）及以上；显卡算力不足时会自动按 fp32 加载并在控制台提示。",
                }),
                "auto_free_memory": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "生成完成后自动回收内存/显存（卸载整套 IndexTTS-2.5 模型并执行 gc + CUDA empty_cache）。\n"
                               "勾选后：本次生成结束即释放，下次生成需重新加载模型（约 10~60 秒），\n"
                               "适合需要把显存腾给图像/视频等其它模型的场景；\n"
                               "不勾选（默认）：模型常驻 GPU，连续多次生成更快。",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio"

    def __init__(self):
        # 使用进程级共享加载器：同一工作流中多个节点复用同一模型实例，
        # 避免重复加载整套模型导致显存成倍增长。
        self.loader = get_shared_loader()

    # ---- 输入处理 ----------------------------------------------------------
    @staticmethod
    def _process_audio_input(audio):
        """把 ComfyUI AUDIO 输入转成 (wave_1d_float32, sr)。"""
        if isinstance(audio, dict) and "waveform" in audio and "sample_rate" in audio:
            wave = audio["waveform"]
            sr = int(audio["sample_rate"])
            if isinstance(wave, torch.Tensor):
                if wave.dim() == 3:
                    wave = wave[0, 0].detach().cpu().numpy()
                elif wave.dim() == 1:
                    wave = wave.detach().cpu().numpy()
                else:
                    wave = wave.flatten().detach().cpu().numpy()
            elif isinstance(wave, np.ndarray):
                if wave.ndim == 3:
                    wave = wave[0, 0]
                elif wave.ndim == 2:
                    wave = wave[0]
            return wave.astype(np.float32), sr
        if isinstance(audio, tuple) and len(audio) == 2:
            wave, sr = audio
            if isinstance(wave, torch.Tensor):
                wave = wave.detach().cpu().numpy()
            return np.asarray(wave, dtype=np.float32), int(sr)
        raise ValueError("reference_audio 必须是 AUDIO 类型")

    @classmethod
    def _parse_emo_vector(cls, raw):
        """解析情感向量：逗号分隔或 JSON 数组，返回 8 维 list 或 None。"""
        if raw is None:
            return None
        raw = str(raw).strip()
        if not raw:
            return None
        try:
            if raw.startswith("["):
                vals = json.loads(raw)
            else:
                vals = [x.strip() for x in raw.split(",") if x.strip() != ""]
                vals = [float(x) for x in vals]
        except Exception:
            return None
        vals = [float(x) for x in vals]
        if len(vals) != 8:
            return None
        return vals

    # ---- 流式合成 -----------------------------------------------------------
    @staticmethod
    def _segment_to_int16(item):
        """上游流式产出的是 ±32767 量级的 float 张量，转成一维 int16。"""
        tensor = item.detach().float().flatten()
        tensor = torch.clamp(tensor, -32767.0, 32767.0).to(torch.int16)
        return tensor.numpy()

    @staticmethod
    def _make_progress_hook():
        """把上游的 gr_progress 回调桥接到 ComfyUI 进度条；不可用时返回 None。"""
        try:
            from comfy.utils import ProgressBar
        except Exception:  # noqa: BLE001 - 单测/离线环境没有 comfy 包
            return None
        try:
            bar = ProgressBar(1000)
        except Exception:  # noqa: BLE001
            return None

        def hook(value=0.0, desc=None):
            try:
                ratio = max(0.0, min(1.0, float(value)))
                bar.update_absolute(int(round(ratio * 1000)))
            except Exception:  # noqa: BLE001 - 进度显示失败不影响生成
                pass
            if desc:
                print(f"[IndexTTS-2.5] {desc}")

        return hook

    def _synthesize_segments(self, tts, spk_path, emo_path, text, language, infer_options):
        """逐段消费上游生成器，返回 (段音频列表, 段文本列表或 None, 是否被截断)。

        非流式 infer() 只会返回拼接好的整段音频，拿不到分段时长；这里走
        infer_generator(stream_return=True)，每合成完一段就 yield 一次，
        并临时记录 split_text_by_tokens 的返回值，得到「文本 <-> 音频」配对。
        """
        captured = []
        splitter = getattr(tts, "split_text_by_tokens", None)
        original_splitter = splitter
        record = None
        if callable(splitter):
            def record(seg_text, max_tokens, lang_prefix=""):
                segments = original_splitter(seg_text, max_tokens, lang_prefix)
                captured.append(list(segments))
                return segments
            tts.split_text_by_tokens = record

        previous_hook = getattr(tts, "gr_progress", None)
        hook = self._make_progress_hook()
        if hook is not None:
            tts.gr_progress = hook

        chunks = [text]
        low_vram = bool(getattr(tts, "low_vram", False))
        if low_vram and len(text) > 40 and hasattr(tts, "split_text_by_punctuation"):
            chunks = list(tts.split_text_by_punctuation(text, 40)) or [text]
            print(f"[IndexTTS-2.5] 低显存模式：文本按标点切成 {len(chunks)} 块逐块生成")

        wav_segments, segment_texts = [], []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for chunk_index, chunk in enumerate(chunks):
                    before = len(captured)
                    gen = tts.infer_generator(
                        spk_audio_prompt=spk_path,
                        text=chunk,
                        output_path=None,
                        lang=language,
                        # 段间静音由本节点统一插入，因此这里固定传 0，
                        # 避免上游产出 list/张量两种形态的静音对象
                        interval_silence=0,
                        stream_return=True,
                        **infer_options,
                    )
                    got = [self._segment_to_int16(item)
                           for item in gen if isinstance(item, torch.Tensor)]
                    if not got:
                        raise RuntimeError("IndexTTS-2.5 推理未返回音频数据")
                    wav_segments.extend(got)
                    if before < len(captured):
                        segment_texts.extend(captured[before])
                    else:
                        segment_texts.extend([""] * len(got))
                truncated = any("max_mel_tokens" in str(w.message) for w in caught)
        finally:
            if record is not None:
                try:
                    tts.split_text_by_tokens = original_splitter
                except Exception:  # noqa: BLE001
                    pass
            tts.gr_progress = previous_hook

        aligned = len(segment_texts) == len(wav_segments)
        if not aligned:
            print(f"[IndexTTS-2.5] 分段文本捕获不一致（{len(segment_texts)} vs "
                  f"{len(wav_segments)}），字幕退化为整段时长按字符均分")
        return wav_segments, (segment_texts if aligned else None), truncated

    @staticmethod
    def _assemble_audio(wav_segments, sample_rate, interval_silence):
        """拼接各段并在段间插入静音，返回 (waveform_tensor, 每段时长秒数)。"""
        durations = [len(seg) / float(sample_rate) for seg in wav_segments]
        gap = int(sample_rate * max(0, int(interval_silence)) / 1000)
        parts = []
        for index, seg in enumerate(wav_segments):
            parts.append(seg)
            if gap and index < len(wav_segments) - 1:
                parts.append(np.zeros(gap, dtype=np.int16))
        if not parts:
            waveform = torch.zeros((1, 1, 0), dtype=torch.float32)
        else:
            pcm = np.concatenate(parts)
            wave_f = (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
            waveform = torch.tensor(wave_f, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return waveform, durations

    @staticmethod
    def _pronunciation_mapping(text, glossary=None):
        """{上游标注后的形式: 应显示的原词}，用于把字幕里的 token 还原成可读文本。

        标注有两个来源：正文里用户手写的 ``<文字|读音>``，以及术语表读法里内嵌的标注
        （``朝: <朝|zhao1>`` 这种写法会让替换结果带着 token 进来）。
        直接复用上游的 ``apply_pronunciation_annotations``（规则会随版本变，复刻一次
        就漂移一次）；该模块在 get_tts() 后必然已在 sys.modules 中，这里不再 import。
        上游会在标注前对文本做大小写归一，所以三种大小写都登记一次。
        """
        annotate = getattr(sys.modules.get("indextts.infer_v2_5"),
                           "apply_pronunciation_annotations", None)
        if not callable(annotate):
            return {}

        sources = []
        for word, pron in subtitles.PRONUNCIATION_TAG_RE.findall(text or ""):
            sources.append((f"<{word}|{pron}>", word))
        for term, reading in (glossary or {}).items():
            readings = reading.values() if isinstance(reading, dict) else [reading]
            for item in readings:
                if isinstance(item, str) and "<" in item:
                    sources.append((item, term))

        mapping = {}
        for source, display in sources:
            for variant in {source, source.lower(), source.upper()}:
                annotated = annotate(variant)
                if annotated and annotated != variant:
                    mapping.setdefault(annotated, display)
        return mapping

    # ---- 主函数 ------------------------------------------------------------
    def generate(
        self,
        text,
        reference_audio,
        language="zh",
        duration_factor=1.0,
        seed=0,
        emo_ref_audio=None,
        emo_alpha=1.0,
        emo_vector="",
        emo_text="",
        use_emo_text=False,
        use_qwen_emo=False,
        use_random=False,
        do_sample=True,
        temperature=0.8,
        top_p=0.8,
        top_k=30,
        repetition_penalty=10.0,
        length_penalty=0.0,
        num_beams=3,
        max_mel_tokens=1500,
        max_text_tokens_per_segment=120,
        interval_silence=200,
        strict_max_mel_tokens=False,
        text_normalization=True,
        glossary_yaml="",
        subtitle_format="srt",
        subtitle_text="synthesized",
        use_bf16=True,
        auto_free_memory=False,
    ):
        text = (text or "").strip()
        if not text:
            raise ValueError("text 不能为空")

        for hint in text_tools.check_pronunciation_tags(text, language):
            print(f"[IndexTTS-2.5][发音提示] {hint}")

        # 参考音频 -> 临时 wav
        spk_path = self.loader.save_temp_wav(self._process_audio_input(reference_audio))

        # 情感参考音频
        emo_path = None
        if emo_ref_audio is not None:
            emo_path = self.loader.save_temp_wav(self._process_audio_input(emo_ref_audio))

        try:
            emo_vec = self._parse_emo_vector(emo_vector)
            use_emo_text = bool(use_emo_text) or bool((emo_text or "").strip())

            # 需要 QwenEmotion 时自动加载
            load_qwen = bool(use_qwen_emo) or use_emo_text

            # 设置随机种子（seed=0 时不固定，沿用全局状态）
            if seed != 0:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            tts = self.loader.get_tts(use_bf16=use_bf16, use_qwen_emo=load_qwen)
            self.loader.refresh_glossary(tts, glossary_yaml)

            infer_options = {
                "emo_audio_prompt": emo_path,
                "emo_alpha": float(emo_alpha),
                "emo_vector": emo_vec,
                "use_emo_text": use_emo_text,
                "emo_text": (emo_text.strip() or None) if use_emo_text else None,
                "use_random": bool(use_random),
                "duration_factor": float(duration_factor),
                "text_normalization": bool(text_normalization),
                "max_text_tokens_per_segment": int(max_text_tokens_per_segment),
                # 采样参数
                "do_sample": bool(do_sample),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "top_k": int(top_k),
                "repetition_penalty": float(repetition_penalty),
                "length_penalty": float(length_penalty),
                "num_beams": int(num_beams),
                "max_mel_tokens": int(max_mel_tokens),
            }

            sample_rate = self.loader.sampling_rate(tts)
            print(f"[IndexTTS-2.5] 开始推理 text_len={len(text)} lang={language} "
                  f"duration_factor={duration_factor} seed={seed} "
                  f"seg_tokens<= {max_text_tokens_per_segment} sr={sample_rate}")

            with _GEN_LOCK:
                wav_segments, segment_texts, truncated = self._synthesize_segments(
                    tts, spk_path, emo_path, text, language, infer_options)

            waveform, durations = self._assemble_audio(wav_segments, sample_rate, interval_silence)
            total_duration = waveform.shape[-1] / float(sample_rate)

            if truncated:
                hint = (f"有分段因超出 max_mel_tokens({max_mel_tokens}) 被截断（音频不完整）。"
                        f"请把 max_text_tokens_per_segment 从 {max_text_tokens_per_segment} 调小，"
                        f"或把 max_mel_tokens 调大。")
                if strict_max_mel_tokens:
                    raise RuntimeError("IndexTTS-2.5 " + hint)
                print(f"[IndexTTS-2.5][警告] {hint}")

            # 字幕展示用可读文本：把发音标注的产物（<|SPECIAL_TOKEN_*|>读音）还原成原词
            pron_map = self._pronunciation_mapping(
                text, getattr(getattr(tts, "text_process", None), "term_glossary", None))
            if segment_texts is not None:
                segment_texts = [subtitles.clean_pronunciation_markers(seg, pron_map)
                                 for seg in segment_texts]
            readable_text = subtitles.clean_pronunciation_markers(text, pron_map)

            if segment_texts is not None and subtitle_text == "original":
                subtitle = subtitles.build_subtitle(
                    segment_texts, durations, interval_silence,
                    original_text=readable_text, subtitle_format=subtitle_format)
            elif segment_texts is not None:
                subtitle = subtitles.build_subtitle(
                    segment_texts, durations, interval_silence,
                    subtitle_format=subtitle_format)
            else:
                # 段文本没抓到：退化为「整段时长 + 原文按字符均分」
                subtitle = subtitles.build_subtitle(
                    [""], [total_duration], 0,
                    original_text=readable_text, subtitle_format=subtitle_format)

            audio_dict = {"waveform": waveform, "sample_rate": sample_rate}
            print(f"[IndexTTS-2.5] 完成，{len(wav_segments)} 段 / 时长 {total_duration:.2f}s")

            # 生成成功后按需释放模型（归还内存/显存给 ComfyUI 其它节点）
            if auto_free_memory:
                self.loader.release()

            return (audio_dict, seed, subtitle)
        finally:
            # 清理临时 wav，避免系统临时目录堆积
            for path in (spk_path, emo_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass


class EmoSettingsNode:
    """8 维情感向量配置节点。

    将 8 种情绪作为可调滑块（0~1）展示，输出一个与 ``IndexTTS2_5Node`` 的
    ``emo_vector`` 输入兼容的逗号分隔字符串，可直接连线使用。
    """

    # 与 IndexTTS2_5Node.EMO_ORDER 保持同一顺序
    EMO_ORDER = IndexTTS2_5Node.EMO_ORDER
    EMO_LABELS = {
        "happy": "高兴",
        "angry": "愤怒",
        "sad": "悲伤",
        "afraid": "恐惧",
        "disgusted": "反感",
        "melancholic": "低落",
        "surprised": "惊讶",
        "calm": "自然",
    }

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            name: (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": f"{cls.EMO_LABELS[name]}（{name}）：情感强度，范围 0~1。",
                },
            )
            for name in cls.EMO_ORDER
        }
        return {"required": required}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("emo_vector",)
    FUNCTION = "build_emo_vector"
    CATEGORY = "audio"

    def build_emo_vector(self, happy, angry, sad, afraid, disgusted, melancholic, surprised, calm):
        """按 EMO_ORDER 顺序拼成逗号分隔字符串，供 emo_vector 使用。"""
        values = [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
        return (",".join(f"{float(v):g}" for v in values),)


class IndexTTSTextPreviewNode:
    """文本前端预览节点：不消耗 GPU 合成，直接看模型「会怎么读、会怎么分段」。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "2026年3月5日，M.2 接口的 NVMe 硬盘很快。"}),
                "language": (["zh", "en", "ja", "es", "ar"], {"default": "zh"}),
            },
            "optional": {
                "text_normalization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "与生成节点的 text_normalization 保持一致，用于对比开关前后的读法与分段差异。",
                }),
                "max_text_tokens_per_segment": ("INT", {
                    "default": 120, "min": 10, "max": 300, "step": 5,
                    "tooltip": "与生成节点取同一值；这里只做预览，不产出音频。",
                }),
                "glossary_yaml": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "与生成节点同格式的内联术语表，用于确认术语读法是否被应用。",
                }),
                "use_bf16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "预览需要用到分词器与文本前端，会触发模型加载（与生成节点共用同一份缓存）。",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("processed_text", "segments", "segment_count", "diagnostics")
    FUNCTION = "preview"
    CATEGORY = "audio"

    def __init__(self):
        self.loader = get_shared_loader()

    def preview(self, text, language="zh", text_normalization=True,
                max_text_tokens_per_segment=120, glossary_yaml="", use_bf16=True):
        text = (text or "").strip()
        if not text:
            raise ValueError("text 不能为空")

        loader = self.loader
        tts = loader.get_tts(use_bf16=use_bf16, use_qwen_emo=False)
        loader.refresh_glossary(tts, glossary_yaml)

        notes = []
        for hint in text_tools.check_pronunciation_tags(text, language):
            notes.append(f"[发音标注] {hint}")

        info = text_tools.prepare_text(
            tts, text, language=language, text_normalization=text_normalization,
            max_text_tokens_per_segment=max_text_tokens_per_segment)
        if info.get("error"):
            raise RuntimeError(info["error"])

        glossary_count = len(getattr(tts.text_process, "term_glossary", {}) or {})
        if not text_normalization:
            notes.append("[归一化] 已关闭：数字/缩写按原文朗读，术语表也不会生效。")
        elif info["normalization_applied"]:
            notes.append("[归一化] 已应用（processed_text 即模型实际朗读的文本）。")
        else:
            notes.append(f"[归一化] {language} 不做归一化，processed_text 仅做了大小写与标注转换。")
        if "<|SPECIAL_TOKEN" in info["processed_text"]:
            notes.append("[读音标注] <|SPECIAL_TOKEN_1|>…<|SPECIAL_TOKEN_1|>（英文音素）与 "
                         "<|SPECIAL_TOKEN_2|>…（中文拼音）是上游对 <文字|读音> 的转换结果，"
                         "即模型真正读的内容；生成节点的字幕输出会自动还原成原词。")
        notes.append(f"[术语表] 当前生效 {glossary_count} 条。")

        counts = info["token_counts"]
        notes.append(f"[分段] 共 {info['segment_count']} 段，"
                     f"token 数 {counts}，单段预算 {info['token_budget']}。")
        if info["segment_count"] > 1:
            notes.append("[分段] 段间会插入 interval_silence 静音；想让句子连读更顺，可调大 "
                         "max_text_tokens_per_segment（注意 max_mel_tokens 截断风险）。")
        if info["over_budget"]:
            notes.append(f"[分段] 第 {info['over_budget']} 段超出目标 token 数，"
                         "通常是单句过长或不含标点，建议断句。")
        changed = info["processed_text"] != text
        if changed:
            notes.append("[提示] processed_text 与原文不同，说明归一化改写了读法；"
                         "读法不对时用 <文字|读音> 或术语表修正。")

        diagnostics = "\n".join(notes)
        segments_text = "\n".join(f"{i + 1}. {seg}" for i, seg in enumerate(info["segments"]))
        return (info["processed_text"], segments_text, info["segment_count"], diagnostics)
