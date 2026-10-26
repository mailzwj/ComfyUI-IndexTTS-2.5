"""文本前端工具：发音标注校验 + 预处理复刻（供 Text Preview 节点使用）。

预处理序列与 indextts/infer_v2_5.py 的 infer_generator 保持一致，
使预览得到的分段与真实合成完全对齐。
"""
import re

_TAG_RE = re.compile(r"<([^<>\n]*)>")
_PRONUNCIATION_RE = re.compile(r"<([^|>\n]+)\|([^>\n]+)>")
_PINYIN_RE = re.compile(r"^(?:[A-Za-zÜü]+[1-5])+[\.]?$")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_LANG_TOKEN_RE = re.compile(r"<\|[^|<>]*\|>")


def check_pronunciation_tags(text, language="zh"):
    """检查 <文字|读音> 发音标注，返回人类可读的告警列表（不抛异常）。"""
    warnings = []
    if not text:
        return warnings

    stripped = _LANG_TOKEN_RE.sub("", text)
    if stripped.count("<") != stripped.count(">"):
        warnings.append("发音标注 <> 括号不匹配（可能漏写 > ），未闭合的标注不会生效。")

    for body in _TAG_RE.findall(stripped):
        if "|" not in body:
            if not body.strip():
                warnings.append("发现空标注 <>，应写作 <文字|读音> 形式。")
            else:
                warnings.append(
                    f"<{body}> 缺少 '|' 分隔符，不会按发音朗读；"
                    f"正确写法如 <{body.strip()[:4]}|ZHENG1 QUE4>。")
            continue
        word, _, pron = body.partition("|")
        if not word.strip() or not pron.strip():
            warnings.append(
                f"<{body}> 的正文或读音为空，不会按发音朗读，应写作 <文字|读音>。")

    for word, pron in _PRONUNCIATION_RE.findall(text):
        pron = pron.strip()
        if _KANA_RE.search(pron):
            continue
        has_cjk_word = bool(_CJK_RE.search(word))
        if has_cjk_word:
            if not _PINYIN_RE.match(pron.replace(" ", "")):
                warnings.append(
                    f"<{word}|{pron}>：中文词的读音应是带声调的拼音"
                    f"（如 {word[:1]}IN1），当前写法可能被按字面朗读。")
        elif pron != pron.upper():
            warnings.append(
                f"<{word}|{pron}>：英文读音建议用大写 ARPAbET 音素"
                f"（如 G OW1 . IH0 NG），小写会被统一转成大写。")

    if language and language.lower() in ("ja",) and _CJK_RE.search(text):
        if not _PRONUNCIATION_RE.search(text):
            warnings.append(
                "日语 mixed 汉字较多时可用 <漢字|カンジ> 假名标注消除读音歧义。")
    return warnings


def prepare_text(tts, text, language="zh", text_normalization=True,
                 max_text_tokens_per_segment=120):
    """复刻 infer_generator 的文本预处理，返回预览信息字典。

    依赖已加载的 tts 实例（text_process / tokenizer / split_text_by_tokens）。
    """
    from indextts.infer_v2_5 import apply_pronunciation_annotations

    lang = (language or "zh").lower()
    lang_prefix = f"<|{lang}|> "
    processed = tts.text_process.clean_pattern.sub(
        lambda x: tts.text_process.char_rep_map[x.group()], text)

    normalized_applied = False
    if text_normalization:
        if lang in ("zh", "zhen", "en"):
            processed = tts.text_process.normalize(processed)
            normalized_applied = True
        elif lang in ("ja", "es"):
            from indextts.utils.nemo_tn import normalize_text as nemo_normalize
            try:
                processed = nemo_normalize(processed, lang)
                normalized_applied = True
            except Exception as exc:  # noqa: BLE001 - 预览不应因规则库缺失而失败
                processed = ""
                normalized_applied = False
                return {"error": f"NeMo 文本归一化失败：{exc!r}"}
        # 其它语言（如 ar）上游不做归一化，此处保持一致

    if lang in ("ja", "zh", "zhen", "en"):
        processed = processed.lower()
    if lang == "es":
        processed = processed.upper()

    processed = apply_pronunciation_annotations(processed)
    if lang == "ja":
        processed = tts.ja_text_process.process_ja_text(processed)
    processed = re.sub(r"<\|([^|]+)\|>", lambda m: f"<|{m.group(1).upper()}|>", processed)

    segments = tts.split_text_by_tokens(processed, max_text_tokens_per_segment, lang_prefix)
    token_counts = [tts._token_len(lang_prefix + seg) for seg in segments]
    budget = min(max_text_tokens_per_segment,
                 tts.gpt.text_pos_embedding.emb.num_embeddings - 2)
    return {
        "processed_text": processed,
        "segments": list(segments),
        "segment_count": len(segments),
        "token_counts": token_counts,
        "token_budget": int(budget),
        "over_budget": [i + 1 for i, c in enumerate(token_counts) if c > max_text_tokens_per_segment],
        "normalization_applied": normalized_applied,
    }
