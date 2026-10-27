"""字幕时间轴构建与格式化（plain / SRT / VTT）。

逐段合成能拿到「段」级精确时长；段内再按标点切句、按字符数分配时长，
误差被限制在单段之内。
"""
import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[，。！？、；：,\.!\?;:\n])")

# 用户写的发音标注 <文字|读音>
PRONUNCIATION_TAG_RE = re.compile(r"<([^<>|\n]+)\|([^<>\n|]+)>")
# 上游 apply_pronunciation_annotations 的产物：<|SPECIAL_TOKEN_1/2|>读音<|...|>
_ANNOTATED_PAIR_RE = re.compile(
    r"<\|SPECIAL_TOKEN_[12]\|>\s*(.*?)\s*<\|SPECIAL_TOKEN_[12]\|>", re.S)
_ANNOTATED_TOKEN_RE = re.compile(r"<\|SPECIAL_TOKEN_[12]\|>")


def clean_pronunciation_markers(text, mapping=None):
    """把发音标注的产物还原成人能读的文本，供字幕展示。

    合成文本里出现的是上游替换后的 ``<|SPECIAL_TOKEN_2|>ZHAO1<|SPECIAL_TOKEN_2|>``，
    直接展示没法当字幕用。

    mapping 是 ``{上游替换后的形式: 用户原词}``（由上游同一个函数算出，精确匹配）。
    未命中 mapping 的按顺序兜底：整对 token 保留其中的读音、孤立 token 标记删掉、
    还没被替换的 ``<文字|读音>`` 只留文字。``<|ZH|>`` 这类语言切换标记不受影响。
    """
    if not text:
        return text
    cleaned = text
    for annotated, word in sorted((mapping or {}).items(),
                                  key=lambda kv: len(kv[0]), reverse=True):
        cleaned = cleaned.replace(annotated, word)
    cleaned = _ANNOTATED_PAIR_RE.sub(r"\1", cleaned)
    cleaned = _ANNOTATED_TOKEN_RE.sub("", cleaned)
    return PRONUNCIATION_TAG_RE.sub(r"\1", cleaned)


def split_sentences(text):
    """按标点切句，标点跟随前文；返回非空句子列表。"""
    if not text:
        return []
    return [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p and p.strip()]


def _one_line(text):
    return re.sub(r"\s+", " ", text).strip()


def slice_text_by_ratio(original_text, segment_texts):
    """把原始文本按各段合成文本的字符占比切片，用于还原用户原始措辞。"""
    total = sum(max(0, len(t)) for t in segment_texts)
    if not original_text or total == 0:
        return [original_text or ""] + [""] * (len(segment_texts) - 1)
    limit = len(original_text)
    out, cursor, acc = [], 0, 0
    for i, seg in enumerate(segment_texts):
        acc += max(0, len(seg))
        if i == len(segment_texts) - 1:
            end = limit
        else:
            end = min(limit, max(cursor, int(round(acc / total * limit))))
        out.append(original_text[cursor:end])
        cursor = end
    return out


def build_timeline(segment_texts, segment_durations, interval_silence_ms=0,
                   original_text=None, originals=None, speakers=None):
    """构建字幕帧列表 [{'text', 'start', 'end', 'speaker'}]，单位为秒。

    segment_texts / segment_durations 一一对应，顺序即合成顺序。
    interval_silence_ms 可以是单个值（所有边界同宽），也可以是逐边界的列表
    （多人对话：换角色用 turn_gap、同角色续句用 same_speaker_gap）。
    original_text 不为 None 时，字幕文字改用原始输入（措辞干净，段内时间按比例映射）；
    originals 是「逐段原始文本」，多人对话按 turn 切片时使用，优先级高于 original_text；
    两者都不给则直接展示模型实际朗读的文本（归一化后）。
    speakers 为逐段角色名（可为 None），写入每帧的 speaker 字段。
    """
    n = len(segment_durations)
    if n == 0:
        return []

    if isinstance(interval_silence_ms, (list, tuple)):
        sil_list = [max(0, int(s)) / 1000.0 for s in interval_silence_ms]
        sil_s = 0.0
    else:
        sil_list = None
        sil_s = max(0, int(interval_silence_ms)) / 1000.0

    starts, ends, cursor = [], [], 0.0
    for i, dur in enumerate(segment_durations):
        starts.append(cursor)
        cursor += max(0.0, float(dur))
        ends.append(cursor)
        if i < n - 1:
            cursor += sil_list[i] if sil_list is not None else sil_s

    if originals is not None:
        sources = list(originals)
    elif original_text is not None:
        sources = slice_text_by_ratio(original_text, segment_texts)
    else:
        sources = list(segment_texts)

    frames = []
    for i in range(n):
        text = sources[i] if i < len(sources) else ""
        sentences = split_sentences(text)
        if not sentences and text.strip():
            sentences = [text.strip()]
        if not sentences:
            continue
        weights = [max(1, len(s)) for s in sentences]
        weight_total = sum(weights)
        window = ends[i] - starts[i]
        cur = starts[i]
        speaker = speakers[i] if speakers is not None and i < len(speakers) else None
        for sentence, weight in zip(sentences, weights):
            step = window * weight / weight_total
            frames.append({"text": _one_line(sentence),
                           "start": cur, "end": cur + step,
                           "speaker": speaker})
            cur += step
    return frames


def _clock(seconds, ms_separator=","):
    ms = int(round(max(0.0, float(seconds)) * 1000))
    hours, rem = divmod(ms, 3600000)
    minutes, rem = divmod(rem, 60000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{ms_separator}{millis:03d}"


def _clock_short(seconds):
    """历史 plain 格式：M:SS.mmm"""
    ms = int(round(max(0.0, float(seconds)) * 1000))
    minutes, rem = divmod(ms, 60000)
    secs, millis = divmod(rem, 1000)
    return f"{minutes}:{secs:02d}.{millis:03d}"


def _frame_text(frame, speaker_prefix=None):
    """按前缀模板加上角色名；模板里没有 {name} 时原样拼在句首。"""
    text = frame["text"]
    name = frame.get("speaker")
    if not speaker_prefix or not name:
        return text
    if "{name}" in speaker_prefix:
        return f"{speaker_prefix.replace('{name}', name)}{text}"
    return f"{speaker_prefix}{text}"


def format_frames(frames, subtitle_format="srt", speaker_prefix=None):
    fmt = (subtitle_format or "srt").lower()
    if fmt == "plain":
        lines = []
        for frame in frames:
            lines.append(f">> {_clock_short(frame['start'])}-{_clock_short(frame['end'])}")
            lines.append(f">> {_frame_text(frame, speaker_prefix)}")
        return "\n".join(lines)
    if fmt == "vtt":
        blocks = ["WEBVTT", ""]
        for idx, frame in enumerate(frames, start=1):
            blocks.append(f"{idx}")
            blocks.append(f"{_clock(frame['start'], '.')} --> {_clock(frame['end'], '.')}")
            blocks.append(_frame_text(frame, speaker_prefix))
            blocks.append("")
        return "\n".join(blocks).rstrip() + "\n"
    blocks = []
    for idx, frame in enumerate(frames, start=1):
        blocks.append(f"{idx}")
        blocks.append(f"{_clock(frame['start'])} --> {_clock(frame['end'])}")
        blocks.append(_frame_text(frame, speaker_prefix))
        blocks.append("")
    return "\n".join(blocks).rstrip() + "\n"


def build_subtitle(segment_texts, segment_durations, interval_silence_ms=0,
                   original_text=None, subtitle_format="srt",
                   originals=None, speakers=None, speaker_prefix=None):
    frames = build_timeline(segment_texts, segment_durations,
                            interval_silence_ms=interval_silence_ms,
                            original_text=original_text, originals=originals,
                            speakers=speakers)
    return format_frames(frames, subtitle_format, speaker_prefix=speaker_prefix)
