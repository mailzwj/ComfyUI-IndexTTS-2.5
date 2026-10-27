"""合成内核：把「tts 实例 + 参考音频路径 + 文本 -> 分段 int16 音频」的机械过程
集中在这里，供单文本节点与多人对话节点共用。

不包含 ComfyUI 面板定义，也不负责模型加载（见 model_loader.py）。
"""
import random
import sys
import threading
import warnings

import numpy as np
import torch

# 生成期间会对共享的 tts 实例打补丁（分段记录、进度钩子、说话人条件回填），
# 必须串行执行。GPU 推理本身即串行，因此不会损失吞吐。
GEN_LOCK = threading.Lock()


# ---- 输入处理 ---------------------------------------------------------------
def process_audio_input(audio):
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


def parse_emo_vector(raw):
    """解析情感向量：逗号分隔或 JSON 数组，返回 8 维 list 或 None。"""
    if raw is None:
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    import json

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


def apply_seed(seed):
    """设置随机种子（seed=0 时不固定，沿用全局状态）。"""
    if not seed:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_infer_options(duration_factor=1.0, text_normalization=True,
                        max_text_tokens_per_segment=120, do_sample=True,
                        temperature=0.8, top_p=0.8, top_k=30,
                        repetition_penalty=10.0, length_penalty=0.0,
                        num_beams=3, max_mel_tokens=1500, emo_alpha=1.0,
                        emo_vector=None, use_emo_text=False, emo_text=None,
                        use_random=False):
    """拼出 infer_generator 的透传参数（interval_silence 由内核固定为 0）。"""
    return {
        "emo_alpha": float(emo_alpha),
        "emo_vector": emo_vector,
        "use_emo_text": bool(use_emo_text),
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


# ---- 进度 -------------------------------------------------------------------
def _new_bar(total=1000):
    """创建 ComfyUI 进度条；离线/单测环境没有 comfy 包时返回 None。"""
    try:
        from comfy.utils import ProgressBar
    except Exception:  # noqa: BLE001 - 单测/离线环境没有 comfy 包
        return None
    try:
        return ProgressBar(total)
    except Exception:  # noqa: BLE001
        return None


def _report(bar, ratio, desc):
    if bar is not None:
        try:
            bar.update_absolute(int(round(max(0.0, min(1.0, float(ratio))) * 1000)))
        except Exception:  # noqa: BLE001 - 进度显示失败不影响生成
            pass
    if desc:
        print(f"[IndexTTS-2.5] {desc}")


def make_progress_hook():
    """把上游的 gr_progress 回调桥接到 ComfyUI 进度条（单次生成，0~1 全程）。"""
    bar = _new_bar()
    if bar is None:
        return None

    def hook(value=0.0, desc=None):
        _report(bar, value, desc)

    return hook


class ProgressBridge:
    """多次生成（多人对话的逐 turn）共用一条进度条。

    上游每次 infer_generator 都按 0~1 上报，这里把第 unit 次映射到
    [unit/units, (unit+1)/units) 区间，避免进度条来回回退。
    """

    def __init__(self, units=1):
        self.units = max(1, int(units))
        self.bar = _new_bar()
        self.unit = 0

    def set_unit(self, index):
        self.unit = max(0, min(self.units - 1, int(index)))

    def hook(self, value=0.0, desc=None):
        if self.bar is None:
            if desc:
                print(f"[IndexTTS-2.5] {desc}")
            return
        ratio = max(0.0, min(1.0, float(value)))
        _report(self.bar, (self.unit + ratio) / self.units, desc)


# ---- 逐段合成 ---------------------------------------------------------------
def segment_to_int16(item):
    """上游流式产出的是 ±32767 量级的 float 张量，转成一维 int16。"""
    tensor = item.detach().float().flatten()
    tensor = torch.clamp(tensor, -32767.0, 32767.0).to(torch.int16)
    return tensor.numpy()


def synthesize_segments(tts, spk_path, emo_path, text, language, infer_options,
                        progress_hook=None):
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
    if progress_hook is not None:
        tts.gr_progress = progress_hook

    chunks = [text]
    low_vram = bool(getattr(tts, "low_vram", False))
    if low_vram and len(text) > 40 and hasattr(tts, "split_text_by_punctuation"):
        chunks = list(tts.split_text_by_punctuation(text, 40)) or [text]
        print(f"[IndexTTS-2.5] 低显存模式：文本按标点切成 {len(chunks)} 块逐块生成")

    wav_segments, segment_texts = [], []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for chunk in chunks:
                before = len(captured)
                gen = tts.infer_generator(
                    spk_audio_prompt=spk_path,
                    text=chunk,
                    output_path=None,
                    lang=language,
                    # 段间静音由调用方统一插入，因此这里固定传 0，
                    # 避免上游产出 list/张量两种形态的静音对象
                    interval_silence=0,
                    stream_return=True,
                    emo_audio_prompt=emo_path,
                    **infer_options,
                )
                got = [segment_to_int16(item)
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


def synthesize_turn(loader, tts, spk_path, emo_path, text, language, infer_options,
                    progress_hook=None, speaker_cache=True):
    """带说话人条件缓存的单次合成，返回 (段音频, 段文本或 None, 是否截断, 是否命中缓存)。

    上游按「参考音频路径字符串」缓存条件张量，路径一变就重算并 empty_cache()；
    多人对话交替说话人时每句都会踩到。这里按音频内容指纹快照/回填，
    loader 不缓存时（离线或关闭开关）退化为直接生成。
    """
    cache_key = None
    hit = False
    if speaker_cache and hasattr(loader, "speaker_cache_key"):
        # 上游在 use_emo_text / emo_vector 存在时忽略外部情感音频，改用说话人音频；
        # 这里算出「上游实际会用的情感参考路径」，缓存键与回填路径才和上游的判断一致
        if infer_options.get("use_emo_text") or infer_options.get("emo_vector") is not None:
            effective_emo = None
        else:
            effective_emo = emo_path
        cache_key = loader.speaker_cache_key(tts, spk_path, effective_emo)
        hit = loader.bind_speaker_conditions(
            tts, cache_key, spk_path, effective_emo or spk_path)
    wav_segments, segment_texts, truncated = synthesize_segments(
        tts, spk_path, emo_path, text, language, infer_options,
        progress_hook=progress_hook)
    if cache_key is not None and not hit:
        loader.store_speaker_conditions(tts, cache_key)
    return wav_segments, segment_texts, truncated, hit


def assemble_audio(wav_segments, sample_rate, interval_silence):
    """拼接各段并在段间插入静音，返回 (waveform_tensor, 每段时长秒数)。"""
    durations = [len(seg) / float(sample_rate) for seg in wav_segments]
    gaps = interval_silence if isinstance(interval_silence, (list, tuple)) \
        else [int(sample_rate * max(0, int(interval_silence)) / 1000)] * max(0, len(wav_segments) - 1)
    parts = []
    for index, seg in enumerate(wav_segments):
        parts.append(seg)
        gap = gaps[index] if index < len(gaps) else 0
        if gap and index < len(wav_segments) - 1:
            parts.append(np.zeros(int(gap), dtype=np.int16))
    if not parts:
        waveform = torch.zeros((1, 1, 0), dtype=torch.float32)
    else:
        pcm = np.concatenate(parts)
        wave_f = (pcm.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        waveform = torch.tensor(wave_f, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return waveform, durations


def pronunciation_mapping(text, glossary=None):
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

    from . import subtitles

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
