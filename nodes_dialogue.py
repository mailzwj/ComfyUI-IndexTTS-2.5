"""ComfyUI 节点：多人对话（多角色）语音生成。

三个节点配合使用：
- IndexTTS Speaker      输出 SPEAKER     ：一个角色的音色与默认说话方式
- IndexTTS Speaker Pack 输出 SPEAKER_PACK：把若干角色打包（可链式扩展）
- IndexTTS Dialogue     消费脚本          ：逐 turn 生成并串成一条音频 + 带角色名的字幕
"""
import os

from . import dialogue_script as script_grammar
from . import engine, subtitles, text_tools
from .model_loader import get_shared_loader

_GEN_LOCK = engine.GEN_LOCK

# 节点面板里表示「不设默认，交给对话节点/行级覆盖」的取值
INHERIT = "inherit"
_LANG_CHOICES = [INHERIT] + list(script_grammar.LANGUAGES)

SCRIPT_EXAMPLE = (
    "小雪: 你终于来了。\n"
    "阿明（高兴）: 给你带了杯咖啡！\n"
    "  还是冰的，路上没化。\n"
    "小雪: 谢谢。\n"
    "# 旁白也要先在 Speaker Pack 里注册音色才能用，例如：\n"
    "# [旁白] @lang=zh @speed=0.9: 走廊尽头的窗户开着一条缝。"
)


class Voice:
    """一个角色的说话方式（音色 + 默认语言/语速/情感）。"""

    __slots__ = ("name", "audio", "language", "duration_factor",
                 "emo_text", "emo_vector", "emo_ref_audio")

    def __init__(self, name, audio, language=INHERIT, duration_factor=None,
                 emo_text="", emo_vector="", emo_ref_audio=None):
        self.name = name
        self.audio = audio
        self.language = language
        self.duration_factor = duration_factor
        self.emo_text = emo_text
        self.emo_vector = emo_vector
        self.emo_ref_audio = emo_ref_audio

    def __repr__(self):
        return f"Voice(name={self.name!r}, language={self.language!r})"


def _resolve_speaker_pack(speakers):
    """SPEAKER_PACK 允许是 {名字: Voice} 或 Voice 列表/单个 Voice，统一成字典。"""
    if speakers is None:
        return {}
    if isinstance(speakers, Voice):
        return {speakers.name: speakers}
    if isinstance(speakers, dict):
        return dict(speakers)
    if isinstance(speakers, (list, tuple)):
        pack = {}
        for item in speakers:
            if not isinstance(item, Voice):
                raise ValueError(f"speaker pack 里出现非 Voice 元素：{type(item).__name__}")
            pack[item.name] = item
        return pack
    raise ValueError("speakers 必须是 IndexTTS Speaker Pack 的输出")


class IndexTTSSpeakerNode:
    """定义一个对话角色：名字 + 参考音频 + 可选的默认语言/语速/情感。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {
                    "default": "小雪",
                    "tooltip": "角色名，对话脚本里用「角色名: 台词」引用它。\n"
                               "同一个 Speaker Pack 内不能重名。",
                }),
                "reference_audio": ("AUDIO", {
                    "tooltip": "该角色的音色参考音频（建议 5~15 秒干净人声）。\n"
                               "多人对话里每个角色一个，同一角色所有台词共用它。",
                }),
            },
            "optional": {
                "language": (_LANG_CHOICES, {
                    "default": INHERIT,
                    "tooltip": "该角色默认语言；inherit 表示跟随对话节点的 default_language。\n"
                               "脚本里还可用 @lang= 覆盖单句。",
                }),
                "duration_factor": ("FLOAT", {
                    "default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05,
                    "tooltip": "该角色默认语速（>1 更慢、<1 更快）；0.5~2.0。\n"
                               "脚本里可用 @speed= 覆盖单句。",
                }),
                "emo_text": ("STRING", {
                    "default": "",
                    "tooltip": "该角色默认情感文本（如「高兴」「悲伤」），需 qwen0.6bemo4-merge 模型。\n"
                               "脚本里用「名字（情感）:」或 @emo_text= 覆盖。",
                }),
                "emo_vector": ("STRING", {
                    "default": "",
                    "tooltip": "该角色默认 8 维情感向量（逗号分隔或 JSON 数组），"
                               "顺序：高兴/愤怒/悲伤/恐惧/反感/低落/惊讶/自然。\n"
                               "可由 Emo Settings 节点连线生成；提供后覆盖 emo_text。",
                }),
                "emo_ref_audio": ("AUDIO", {
                    "tooltip": "该角色默认的情感参考音频；脚本里改用 emo_text/emo_vector 时忽略。",
                }),
            },
        }

    RETURN_TYPES = ("SPEAKER",)
    RETURN_NAMES = ("speaker",)
    FUNCTION = "build"
    CATEGORY = "audio"

    def build(self, name, reference_audio, language=INHERIT, duration_factor=1.0,
              emo_text="", emo_vector="", emo_ref_audio=None):
        name = (name or "").strip()
        if not name:
            raise ValueError("name（角色名）不能为空")
        if duration_factor is None:
            duration_factor = 1.0
        return (Voice(name=name, audio=reference_audio, language=language or INHERIT,
                      duration_factor=float(duration_factor), emo_text=emo_text,
                      emo_vector=emo_vector, emo_ref_audio=emo_ref_audio),)


class IndexTTSSpeakerPackNode:
    """把多个角色收集成一份 speaker pack，供对话节点使用；可链式接续 prev_pack。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "prev_pack": ("SPEAKER_PACK", {
                    "tooltip": "接上一个 Speaker Pack 的输出，用于突破 8 个角色的上限：\n"
                               "把前一个包连到这里，再补 s1~s8 即可。",
                }),
                **{f"s{i}": ("SPEAKER", {
                    "tooltip": f"第 {i} 个角色（IndexTTS Speaker 的输出）。",
                }) for i in range(1, 9)},
            },
        }

    RETURN_TYPES = ("SPEAKER_PACK",)
    RETURN_NAMES = ("speakers",)
    FUNCTION = "build"
    CATEGORY = "audio"

    def build(self, prev_pack=None, **kwargs):
        pack = dict(_resolve_speaker_pack(prev_pack))
        for index in range(1, 9):
            voice = kwargs.get(f"s{index}")
            if isinstance(voice, Voice):
                if voice.name in pack:
                    raise ValueError(f"角色名重复：{voice.name}（重名会让台词分配到错误音色）")
                pack[voice.name] = voice
        if not pack:
            raise ValueError("Speaker Pack 里没有任何角色：请连线 s1~s8 或 prev_pack。")
        return (pack,)


class IndexTTSSettingsNode:
    """对话生成的细调参数包（采样 / 分段 / 文本前端 / 术语表），不连线则用默认值。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "do_sample": ("BOOLEAN", {"default": True}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 15.0, "step": 0.5}),
                "length_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "max_mel_tokens": ("INT", {"default": 1500, "min": 50, "max": 1815, "step": 5}),
                "max_text_tokens_per_segment": ("INT", {
                    "default": 120, "min": 10, "max": 300, "step": 5,
                    "tooltip": "单句内部再分段时的预算。对话里每句通常很短，"
                               "只有长独白才会用到；调小更稳，调大跨句韵律更连贯。",
                }),
                "strict_max_mel_tokens": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "某段因超出 max_mel_tokens 被截断时：勾选=直接报错；不勾选=仅控制台提示。",
                }),
                "text_normalization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "文本归一化（读法转换）。术语表（glossary）只在归一化阶段生效。",
                }),
                "glossary_yaml": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "内联术语表（YAML），叠加在模型目录 glossary.yaml 之上，对整段对话生效。",
                }),
                "emo_alpha": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "情感强度整体缩放；对 emo_vector 缩放，或控制情感参考音频与音色的混合比例。",
                }),
                "use_random": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "随机情感：从情感矩阵里随机取样（需同时提供 emo_vector 或 emo_text 才生效）。",
                }),
            },
        }

    RETURN_TYPES = ("TTS_OPTIONS",)
    RETURN_NAMES = ("options",)
    FUNCTION = "build"
    CATEGORY = "audio"

    def build(self, **kwargs):
        options = {key: value for key, value in kwargs.items() if value is not None}
        return (options,)


class IndexTTSDialogueNode:
    """按「角色名: 台词」脚本逐句合成，串成一条连续音频并输出带角色名的字幕。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dialogue_script": ("STRING", {
                    "multiline": True, "default": SCRIPT_EXAMPLE,
                    "tooltip": "对话脚本，一行一句：`角色名: 台词` 或 `[角色名] 台词`。\n"
                               "行首缩进表示同一角色继续说（并入上一句）；空行与整行 # 注释忽略。\n"
                               "单句可加行级覆盖：`阿明（高兴）: …`、`[旁白] @lang=zh @speed=0.9: …`、"
                               "`@emo_text=` / `@emo_vector=`（8 维）。",
                }),
                "speakers": ("SPEAKER_PACK", {
                    "tooltip": "IndexTTS Speaker Pack 的输出：脚本里用到的每个角色都要在包里，\n"
                               "旁白/无名字说话人也算一个角色，同样需要一段参考音频当音色。",
                }),
                "default_language": (list(script_grammar.LANGUAGES), {
                    "default": "zh",
                    "tooltip": "脚本与角色都没指定语言时用它。",
                }),
                "turn_gap": ("INT", {
                    "default": 450, "min": 0, "max": 3000, "step": 10,
                    "tooltip": "换角色时的停顿（毫秒）。对话听感主要靠它：\n"
                               "300~600 接近真人接话，0 会像连读。",
                }),
                "same_speaker_gap": ("INT", {
                    "default": 200, "min": 0, "max": 3000, "step": 10,
                    "tooltip": "同一角色连续两句之间的停顿（毫秒）。",
                }),
                "interval_silence": ("INT", {
                    "default": 200, "min": 0, "max": 2000, "step": 10,
                    "tooltip": "同一句内部被拆成多段时的段间静音（毫秒）。",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 32 - 1}),
            },
            "optional": {
                "options": ("TTS_OPTIONS", {
                    "tooltip": "IndexTTS Settings 的输出（采样 / 分段 / 文本前端 / 术语表）。\n"
                               "不连线则全部用默认值，日常对话不需要动。",
                }),
                "subtitle_format": (["srt", "vtt", "plain"], {"default": "srt"}),
                "subtitle_text": (["synthesized", "original"], {"default": "synthesized"}),
                "subtitle_speaker": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "字幕每行是否加角色名前缀。",
                }),
                "subtitle_speaker_prefix": ("STRING", {
                    "default": "【{name}】",
                    "tooltip": "角色名前缀模板，{name} 会替换成角色名；\n"
                               "没有 {name} 时按前缀直接拼在句首（如 `A: `）。",
                }),
                "speaker_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "按参考音频内容缓存说话人/情感条件张量（LRU）：\n"
                               "多人对话每换一次角色都要重算条件特征，这是主要开销；\n"
                               "开启后同一音色在整段对话里只提一次特征，重复角色越多提速越明显。\n"
                               "显存紧张或需要严格每次重算时关闭。",
                }),
                "use_bf16": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "以 bf16 加载 GPT 主干（显存约省一半、速度更快），仅在 CUDA 上生效。\n"
                               "需要 Ampere（RTX 30 系 / A 系列）及以上；算力不足时自动按 fp32 加载。",
                }),
                "auto_free_memory": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "整段对话生成完成后自动卸载模型并回收显存。",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "INT", "STRING", "STRING")
    RETURN_NAMES = ("audio", "seed", "subtitle", "report")
    FUNCTION = "generate"
    CATEGORY = "audio"

    def __init__(self):
        self.loader = get_shared_loader()

    # ---- 单句参数决议 --------------------------------------------------------
    @staticmethod
    def _resolve_turn(turn, voice, default_language, options):
        """行级覆盖 > 角色默认 > 对话节点默认。"""
        language = turn.options.get("language")
        if not language or language == INHERIT:
            language = voice.language if voice.language != INHERIT else default_language
        duration_factor = turn.options.get("duration_factor")
        if duration_factor is None:
            duration_factor = voice.duration_factor
        if duration_factor is None:
            duration_factor = float(options.get("duration_factor", 1.0))

        emo_vector = turn.options.get("emo_vector")
        if emo_vector is None:
            emo_vector = engine.parse_emo_vector(voice.emo_vector)
        emo_text = turn.options.get("emo_text")
        if emo_vector is None and not emo_text:
            emo_text = (voice.emo_text or "").strip() or None
        return {
            "language": language or "zh",
            "duration_factor": float(duration_factor),
            "emo_vector": emo_vector,
            "emo_text": emo_text,
        }

    def _prepare_voices(self, pack):
        """把每个角色的参考音频落盘一次（同一 turn 序列内路径稳定，供上游缓存命中）。"""
        paths = {}
        for name, voice in pack.items():
            paths[name] = {
                "spk": self.loader.save_temp_wav(engine.process_audio_input(voice.audio)),
                "emo": (self.loader.save_temp_wav(
                    engine.process_audio_input(voice.emo_ref_audio))
                    if voice.emo_ref_audio is not None else None),
            }
        return paths

    # ---- 主函数 -------------------------------------------------------------
    def generate(self, dialogue_script, speakers, default_language="zh",
                 turn_gap=450, same_speaker_gap=200, interval_silence=200, seed=0,
                 options=None, subtitle_format="srt", subtitle_text="synthesized",
                 subtitle_speaker=True, subtitle_speaker_prefix="【{name}】",
                 speaker_cache=True, use_bf16=True, auto_free_memory=False):
        pack = _resolve_speaker_pack(speakers)
        if not pack:
            raise ValueError("speaker pack 为空：请先连线 IndexTTS Speaker。")

        turns = script_grammar.parse_script(
            dialogue_script, known_speakers=list(pack))
        for hint in _pronunciation_hints(turns):
            print(f"[IndexTTS-2.5][发音提示] {hint}")

        options = options or {}
        resolved = [self._resolve_turn(turn, pack[turn.speaker], default_language, options)
                    for turn in turns]
        needs_qwen = any(item["emo_text"] for item in resolved)

        loader = self.loader
        engine.apply_seed(seed)
        tts = loader.get_tts(use_bf16=use_bf16, use_qwen_emo=needs_qwen)
        loader.refresh_glossary(tts, options.get("glossary_yaml", ""))
        sample_rate = loader.sampling_rate(tts)

        voice_paths = self._prepare_voices(pack)
        notes = []
        try:
            # 逐 turn 生成，收集「段音频 / 段文本 / 段所属角色与 turn」
            segments, seg_texts, seg_originals, seg_speakers, seg_turn = [], [], [], [], []
            truncated_any, cache_hits = False, 0
            bridge = engine.ProgressBridge(units=len(turns))

            with _GEN_LOCK:
                for index, turn in enumerate(turns):
                    bridge.set_unit(index)
                    item = resolved[index]
                    paths = voice_paths[turn.speaker]
                    wav_segments, segment_texts, truncated, hit = engine.synthesize_turn(
                        loader, tts, paths["spk"], paths["emo"], turn.text,
                        item["language"],
                        _build_infer_options(item, options),
                        progress_hook=bridge.hook, speaker_cache=speaker_cache)
                    cache_hits += 1 if hit else 0
                    truncated_any = truncated_any or truncated
                    if truncated and options.get("strict_max_mel_tokens"):
                        raise RuntimeError(
                            f"第 {turn.lines[0]} 行（{turn.speaker}）某段超出 "
                            f"max_mel_tokens（{options.get('max_mel_tokens', 1500)}），"
                            "音频不完整；请调小 max_text_tokens_per_segment 或调大 max_mel_tokens。")

                    if segment_texts is None:
                        # 上游分段文本没抓到：整句文本挂在第一段，后续段留空
                        segment_texts = [turn.text] + [""] * (len(wav_segments) - 1)
                    pron_map = engine.pronunciation_mapping(
                        turn.text,
                        getattr(getattr(tts, "text_process", None), "term_glossary", None))
                    shown = [subtitles.clean_pronunciation_markers(seg, pron_map)
                             for seg in segment_texts]
                    originals = (subtitles.slice_text_by_ratio(turn.text, shown)
                                 if subtitle_text == "original" else shown)

                    for seg_index, seg in enumerate(wav_segments):
                        segments.append(seg)
                        seg_texts.append(shown[seg_index] if seg_index < len(shown) else "")
                        seg_originals.append(originals[seg_index]
                                             if seg_index < len(originals) else "")
                        seg_speakers.append(turn.speaker)
                        seg_turn.append(index)

                    notes.append(
                        f"[{index + 1}/{len(turns)}] 行 {turn.lines[0]} {turn.speaker}："
                        f"lang={item['language']} speed={item['duration_factor']:g} "
                        f"emo={'向量' if item['emo_vector'] is not None else (item['emo_text'] or '默认')} "
                        f"{len(wav_segments)} 段 "
                        f"{sum(len(s) for s in wav_segments) / sample_rate:.2f}s"
                        f"{' · 缓存命中' if hit else ''}")

            # 相邻两段之间的静音：同句内 / 同角色续句 / 换角色
            gaps_ms = []
            for i in range(len(segments) - 1):
                if seg_turn[i] == seg_turn[i + 1]:
                    gaps_ms.append(interval_silence)
                elif seg_speakers[i] == seg_speakers[i + 1]:
                    gaps_ms.append(same_speaker_gap)
                else:
                    gaps_ms.append(turn_gap)

            gap_samples = [int(sample_rate * max(0, int(g)) / 1000) for g in gaps_ms]
            waveform, durations = engine.assemble_audio(segments, sample_rate, gap_samples)
            total_duration = waveform.shape[-1] / float(sample_rate)

            subtitle = subtitles.build_subtitle(
                seg_texts, durations, gaps_ms,
                originals=seg_originals if subtitle_text == "original" else None,
                speakers=seg_speakers,
                speaker_prefix=(subtitle_speaker_prefix if subtitle_speaker else None),
                subtitle_format=subtitle_format)

            report_lines = [
                f"角色 {len(pack)} 个：{', '.join(sorted(pack))}",
                f"台词 {len(turns)} 句 / 音频 {len(segments)} 段 / 总时长 {total_duration:.2f}s",
                f"停顿：换角色 {turn_gap}ms · 同角色 {same_speaker_gap}ms · 句内 {interval_silence}ms",
                f"说话人条件缓存：{'开启' if speaker_cache else '关闭'}，"
                f"{len(turns)} 句中命中 {cache_hits} 次",
            ]
            if truncated_any:
                report_lines.append("[警告] 有分段超出 max_mel_tokens（音频不完整），"
                                    "请调小 max_text_tokens_per_segment 或调大 max_mel_tokens。")
            report_lines.extend(notes)
            print(f"[IndexTTS-2.5] 多人对话完成：{len(turns)} 句 / {total_duration:.2f}s")

            if auto_free_memory:
                loader.release()

            return ({"waveform": waveform, "sample_rate": sample_rate}, seed,
                    subtitle, "\n".join(report_lines))
        finally:
            for paths in voice_paths.values():
                for path in paths.values():
                    if path and os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError:
                            pass


def _build_infer_options(item, options):
    """单句的 infer 参数：行级/角色级结果 + Settings 包的采样与文本前端参数。"""
    return engine.build_infer_options(
        duration_factor=item["duration_factor"],
        text_normalization=bool(options.get("text_normalization", True)),
        max_text_tokens_per_segment=int(options.get("max_text_tokens_per_segment", 120)),
        do_sample=bool(options.get("do_sample", True)),
        temperature=float(options.get("temperature", 0.8)),
        top_p=float(options.get("top_p", 0.8)),
        top_k=int(options.get("top_k", 30)),
        repetition_penalty=float(options.get("repetition_penalty", 10.0)),
        length_penalty=float(options.get("length_penalty", 0.0)),
        num_beams=int(options.get("num_beams", 3)),
        max_mel_tokens=int(options.get("max_mel_tokens", 1500)),
        emo_alpha=float(options.get("emo_alpha", 1.0)),
        emo_vector=item["emo_vector"],
        use_emo_text=bool(item["emo_text"]),
        emo_text=item["emo_text"],
        use_random=bool(options.get("use_random", False)),
    )


def _pronunciation_hints(turns):
    """逐句复用单文本节点的发音标注提示（带行号，便于在长脚本里定位）。"""
    hints = []
    for turn in turns:
        for hint in text_tools.check_pronunciation_tags(turn.text, "zh"):
            hints.append(f"第 {turn.lines[0]} 行（{turn.speaker}）：{hint}")
    return hints
