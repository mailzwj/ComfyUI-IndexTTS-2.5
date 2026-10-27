"""ComfyUI 节点：Index TTS 2.5 文本转语音。

参考 ComfyUI-Index-TTS 的节点设计，适配 IndexTTS-2.5 的官方 API
（indextts.infer_v2_5.IndexTTS2.infer_generator）。
"""
import os

from . import engine, subtitles, text_tools
from .model_loader import get_shared_loader

# 生成期间会对共享的 tts 实例打补丁（分段记录、进度钩子），必须串行执行。
# GPU 推理本身即串行，因此不会损失吞吐。
_GEN_LOCK = engine.GEN_LOCK


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
                "speaker_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "按参考音频内容缓存说话人/情感条件张量（LRU 6 份）：\n"
                               "同一音色再次生成时跳过 wav2vbert + s2mel + campplus 提特征，多人对话交替角色时提速最明显。\n"
                               "关闭可腾出这部分显存；模型卸载（auto_free_memory）时缓存随之释放。",
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

    # ---- 输入处理（实现见 engine.py，这里保留方法名作为可 patch 的入口）------
    @staticmethod
    def _process_audio_input(audio):
        return engine.process_audio_input(audio)

    @classmethod
    def _parse_emo_vector(cls, raw):
        return engine.parse_emo_vector(raw)

    # ---- 流式合成 -----------------------------------------------------------
    @staticmethod
    def _make_progress_hook():
        return engine.make_progress_hook()

    def _synthesize_segments(self, tts, spk_path, emo_path, text, language,
                             infer_options, speaker_cache=True):
        return engine.synthesize_turn(
            self.loader, tts, spk_path, emo_path, text, language, infer_options,
            progress_hook=self._make_progress_hook(), speaker_cache=speaker_cache)
    @staticmethod
    def _assemble_audio(wav_segments, sample_rate, interval_silence):
        return engine.assemble_audio(wav_segments, sample_rate, interval_silence)

    @staticmethod
    def _pronunciation_mapping(text, glossary=None):
        return engine.pronunciation_mapping(text, glossary)

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
        speaker_cache=True,
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
            engine.apply_seed(seed)

            tts = self.loader.get_tts(use_bf16=use_bf16, use_qwen_emo=load_qwen)
            self.loader.refresh_glossary(tts, glossary_yaml)

            infer_options = engine.build_infer_options(
                duration_factor=duration_factor,
                text_normalization=text_normalization,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                do_sample=do_sample, temperature=temperature, top_p=top_p,
                top_k=top_k, repetition_penalty=repetition_penalty,
                length_penalty=length_penalty, num_beams=num_beams,
                max_mel_tokens=max_mel_tokens, emo_alpha=emo_alpha,
                emo_vector=emo_vec, use_emo_text=use_emo_text,
                emo_text=emo_text, use_random=use_random)

            sample_rate = self.loader.sampling_rate(tts)
            print(f"[IndexTTS-2.5] 开始推理 text_len={len(text)} lang={language} "
                  f"duration_factor={duration_factor} seed={seed} "
                  f"seg_tokens<= {max_text_tokens_per_segment} sr={sample_rate}")

            with _GEN_LOCK:
                wav_segments, segment_texts, truncated, cache_hit = self._synthesize_segments(
                    tts, spk_path, emo_path, text, language, infer_options,
                    speaker_cache=speaker_cache)

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
            print(f"[IndexTTS-2.5] 完成，{len(wav_segments)} 段 / 时长 {total_duration:.2f}s"
                  f"{'（说话人条件命中缓存）' if cache_hit else ''}")

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
