"""ComfyUI 节点：Index TTS 2.5 文本转语音。

参考 ComfyUI-Index-TTS 的节点设计，适配 IndexTTS-2.5 的官方 API
（indextts.infer_v2_5.IndexTTS2.infer）。
"""
import json
import re
import random

import numpy as np
import torch

from .model_loader import IndexTTS2_5Loader


class IndexTTS2_5Node:
    """Index TTS 2.5 单节点，覆盖文本 / 语言 / 语速 / 情感控制。"""

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
                # 情感参考音频（可选）
                "emo_ref_audio": ("AUDIO", {
                    "tooltip": "情感参考音频：用一段带目标情绪的语音引导合成情感。\n提供 emo_vector 或 emo_text 后此输入会被忽略。",
                }),
                "emo_alpha": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "情感强度（0~1）：对 emo_vector 做整体缩放；\n或控制情感参考音频与说话人音色的混合比例。",
                }),
                # 情感向量（可选）：8 维，逗号或 JSON 数组，顺序见 EMO_ORDER
                "emo_vector": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": "8 维情感向量，顺序固定：高兴/愤怒/悲伤/恐惧/反感/低落/惊讶/自然。\n逗号分隔或 JSON 数组，如 0.8,0,0,0,0,0,0,0 或 [0,0,0,0,0,0,0,1]（自然）。\n必须恰好 8 个数字，否则忽略；提供后覆盖 emo_ref_audio。",
                }),
                # 情感文本（可选）：非空即自动启用（自动加载 QwenEmotion）
                "emo_text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "情感文本：用自然语言描述情绪（如「高兴」「悲伤」「愤怒」），由 QwenEmotion 自动转为情感向量。\n非空即自动启用，需 qwen0.6bemo4-merge 模型；提供后覆盖 emo_ref_audio。",
                }),
                "use_emo_text": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "使用情感文本。emo_text 非空时自动启用，无需单独勾选。",
                }),
                "use_qwen_emo": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "加载 QwenEmotion 情感模型（仅情感文本需要；使用 emo_text 时自动加载）。",
                }),
                "use_random": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "随机情感：随机挑选情感参考向量（需同时提供 emo_vector 或 emo_text 才生效）。",
                }),
                # 采样参数
                "do_sample": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "length_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "max_mel_tokens": ("INT", {"default": 1500, "min": 50, "max": 1815, "step": 5}),
                # 模型加载
                "use_bf16": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle")
    FUNCTION = "generate"
    CATEGORY = "audio"

    def __init__(self):
        self.loader = IndexTTS2_5Loader()

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

    @staticmethod
    def _seconds_to_time(seconds):
        minutes = int(seconds // 60)
        sec = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{minutes}:{sec:02d}.{ms:03d}"

    def _make_subtitle(self, text, total_duration):
        """按标点粗略分句，均分时长，生成简化字幕。"""
        parts = re.split(r"([,，。！？!?；;：:])", text)
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
            if sentence.strip():
                sentences.append(sentence)
        if not sentences:
            sentences = [text]

        seg_dur = total_duration / len(sentences) if sentences else total_duration
        lines = []
        t = 0.0
        for s in sentences:
            lines.append(f">> {self._seconds_to_time(t)}-{self._seconds_to_time(t + seg_dur)}")
            lines.append(f">> {s.strip()}")
            t += seg_dur
        return "\n".join(lines)

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
        use_bf16=True,
    ):
        text = (text or "").strip()
        if not text:
            raise ValueError("text 不能为空")

        # 参考音频 -> 临时 wav
        spk_path = self.loader.save_temp_wav(self._process_audio_input(reference_audio))

        # 情感参考音频
        emo_path = None
        if emo_ref_audio is not None:
            emo_path = self.loader.save_temp_wav(self._process_audio_input(emo_ref_audio))

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

        print(f"[IndexTTS-2.5] 开始推理 text_len={len(text)} lang={language} "
              f"duration_factor={duration_factor} seed={seed}")

        result = tts.infer(
            spk_audio_prompt=spk_path,
            text=text,
            output_path=None,
            lang=language,
            emo_audio_prompt=emo_path,
            emo_alpha=float(emo_alpha),
            emo_vector=emo_vec,
            use_emo_text=use_emo_text,
            emo_text=(emo_text.strip() or None) if use_emo_text else None,
            use_random=bool(use_random),
            duration_factor=float(duration_factor),
            # 采样参数通过 generation_kwargs 传入
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            length_penalty=float(length_penalty),
            num_beams=int(num_beams),
            max_mel_tokens=int(max_mel_tokens),
        )

        if result is None:
            raise RuntimeError("IndexTTS-2.5 推理返回空结果")

        sr, wav = result
        sr = int(sr)
        wav = np.asarray(wav)
        if wav.ndim == 2:
            wav = wav[:, 0]  # [T, C] -> [T]
        # int16 -> float32 [-1, 1]
        wav_f = (wav.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        wave_tensor = torch.tensor(wav_f, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        audio_dict = {"waveform": wave_tensor, "sample_rate": sr}
        subtitle = self._make_subtitle(text, wave_tensor.shape[-1] / sr)
        print(f"[IndexTTS-2.5] 完成，时长 {wave_tensor.shape[-1] / sr:.2f}s")

        return (audio_dict, seed, subtitle)


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
