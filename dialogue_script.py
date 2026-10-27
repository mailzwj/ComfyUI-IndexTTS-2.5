"""多人对话脚本解析：把「角色名: 台词」的多行文本解析成有序的 turn 列表。

纯字符串处理，不依赖模型，可离线单测。语法：

    小雪: 你终于来了。
    阿明（高兴）: 给你带了杯咖啡！
      还是冰的，路上没化。          # 缩进续行，并入上一句
    [旁白] @lang=zh @speed=0.9: 走廊尽头的窗户开着一条缝。

- 角色前缀两种写法：`名字:` / `名字：`，或 `[名字] 台词`
- 覆盖项写在**角色名那一段**（冒号左侧）：紧跟名字的 `（…）`/`(...)` 是情感文本，
  `@key=value` 支持 `emo_text` / `emo_vector` / `lang` / `speed`
- 空行与 `#` 开头的整行注释被跳过
"""
import re

from . import engine

LANGUAGES = ("zh", "en", "ja", "es", "ar")

# 支持的覆盖键（同时用于节点 tooltip，避免两处说明各写一份）
OPTION_KEYS = ("emo_text", "emo_vector", "lang", "speed")
_KEY_ALIASES = {
    "emo_text": "emo_text",
    "emo_vector": "emo_vector",
    "lang": "language",
    "language": "language",
    "speed": "duration_factor",
    "duration_factor": "duration_factor",
}
# 情感向量的 8 个维度顺序，报错时提示用户
EMO_ORDER_HINT = "高兴/愤怒/悲伤/恐惧/反感/低落/惊讶/自然"
SPEED_RANGE = (0.1, 3.0)

_BRACKET_RE = re.compile(r"^\s*\[(?P<name>[^\]\n]+)\]\s*(?P<rest>.*)$")
_COLON_RE = re.compile(r"^\s*(?P<name>[^:：\n]+?)\s*[:：]\s*(?P<rest>.*)$")
# 名字尾部的情感括注：`阿明（高兴）`
_PAREN_RE = re.compile(r"^(?P<name>.*?)\s*[（(]\s*(?P<emo>[^）)]+?)\s*[）)]\s*$")
_OPTION_RE = re.compile(r"@(?P<key>[A-Za-z_]\w*)\s*=\s*(?P<value>[^\s@]+)")
# `[名字] @lang=zh @speed=0.9: 台词`：括号式写法里，覆盖项紧跟在 ] 之后
_LEADING_OPTION_RE = re.compile(r"^\s*@([A-Za-z_]\w*)\s*=\s*([^\s@]*)")
# 续行拼接时，前句以这些字符结尾则无需补空格
_SENTENCE_END = "。！？；，、：.!?,;:\"'”’》)）]】"


class DialogueScriptError(ValueError):
    """脚本不合法；消息始终带行号，便于在多行文本框里定位。"""


class Turn:
    """一个角色连续说完的一句话（可能由多行合并而成）。"""

    __slots__ = ("speaker", "text", "options", "lines")

    def __init__(self, speaker, text, options=None, lines=()):
        self.speaker = speaker
        self.text = text
        self.options = options or {}
        self.lines = list(lines)

    def __repr__(self):
        return (f"Turn(speaker={self.speaker!r}, text={self.text!r}, "
                f"options={self.options!r}, lines={self.lines!r})")


def _join_text(head, tail):
    """拼接续行：中文标点后直接连，英文单词之间补空格。"""
    if not head:
        return tail
    if head[-1] in _SENTENCE_END or head[-1].isspace() or tail[0].isspace():
        return head + tail
    if head[-1].isalnum() and tail[0].isalnum():
        return f"{head} {tail}"
    return head + tail


def _normalize_options(emo_from_paren, raw_options, line_no):
    """校验 `@key=value` 与括注情感，返回 {language?, duration_factor?, emo_text?, emo_vector?}。"""
    options = {}
    if emo_from_paren:
        options["emo_text"] = emo_from_paren

    for key, value in raw_options.items():
        canonical = _KEY_ALIASES.get(key.lower())
        if canonical is None:
            supported = " / ".join(f"@{k}=" for k in OPTION_KEYS)
            raise DialogueScriptError(
                f"第 {line_no} 行：不认识 @{key}=，支持的是 {supported}。")
        if canonical == "language":
            lang = value.lower()
            if lang not in LANGUAGES:
                raise DialogueScriptError(
                    f"第 {line_no} 行：@{key}={value!r} 不是支持的语言，"
                    f"可选 {', '.join(LANGUAGES)}。")
            options["language"] = lang
        elif canonical == "duration_factor":
            try:
                factor = float(value)
            except ValueError:
                raise DialogueScriptError(
                    f"第 {line_no} 行：@{key}={value!r} 不是数字，应写作如 @speed=0.9。"
                ) from None
            if not SPEED_RANGE[0] <= factor <= SPEED_RANGE[1]:
                raise DialogueScriptError(
                    f"第 {line_no} 行：@{key}={value!r} 超出语速范围 "
                    f"{SPEED_RANGE[0]}~{SPEED_RANGE[1]}。")
            options["duration_factor"] = factor
        elif canonical == "emo_vector":
            vec = engine.parse_emo_vector(value)
            if vec is None:
                raise DialogueScriptError(
                    f"第 {line_no} 行：@{key} 需要恰好 8 个逗号分隔的数字"
                    f"（顺序 {EMO_ORDER_HINT}），当前是 {value!r}。"
                    "含空格的话请改用引号包裹的整段或换用 @emo_text。")
            options["emo_vector"] = vec
        else:
            existing = options.get("emo_text")
            if existing and existing != value:
                raise DialogueScriptError(
                    f"第 {line_no} 行：情感写法冲突（括注 {existing!r} 与 "
                    f"@emo_text={value!r} 不一致）。")
            options["emo_text"] = value
    return options


def _split_options(name_part):
    """摘走角色名段里的 `@key=value`，返回 (去掉覆盖项的名字, {键: 值})。"""
    raw = {m.group("key"): m.group("value") for m in _OPTION_RE.finditer(name_part)}
    return _OPTION_RE.sub("", name_part).strip(), raw


def _split_leading_options(segment):
    """摘走段首连续的 `@key=value` 覆盖项，返回 (剩余台词, {键: 值})。

    只认段首，避免把台词正文里的 @speed=1 当参数吃掉；值末尾紧跟的冒号属于
    「名字: 台词」的分隔符，退回给剩余文本。
    """
    raw = {}
    rest = segment
    while True:
        match = _LEADING_OPTION_RE.match(rest)
        if match is None:
            break
        key, value = match.group(1), match.group(2)
        rest = rest[match.end():]
        if value and value[-1] in ":：":
            rest = value[-1] + rest
            value = value[:-1]
        raw[key] = value
        if not value:
            break
    return rest.strip(), raw


def _parse_header(line, line_no):
    """把一行解析为 turn 起始行；不是则返回 None（按续行处理）。"""
    match = _BRACKET_RE.match(line)
    if match is not None:
        name, raw_options = _split_options(match.group("name"))
        # `[名字] @lang=zh: 台词` —— 冒号可选，覆盖项也可以紧跟在 ] 之后
        rest, leading = _split_leading_options(match.group("rest"))
        raw_options.update(leading)
        rest = re.sub(r"^[:：]\s*", "", rest)
    else:
        match = _COLON_RE.match(line)
        if match is None:
            return None
        name, raw_options = _split_options(match.group("name"))
        rest = match.group("rest")

    if not name:
        return None
    emo_from_paren = None
    paren = _PAREN_RE.match(name)
    if paren:
        name = paren.group("name").strip()
        emo_from_paren = paren.group("emo").strip()
    if not name:
        return None
    options = _normalize_options(emo_from_paren, raw_options, line_no)
    return name, options, rest.strip()


def parse_script(raw_text, known_speakers=None):
    """解析脚本为 Turn 列表。

    known_speakers 非空时校验角色名，未知角色报错并附可用名单。
    写错一律抛异常而不是静默兜底：把一句台词塞进错误的音色里生成，
    比提前失败难排查得多。
    """
    if not (raw_text or "").strip():
        raise DialogueScriptError("对话脚本为空。")

    available = [str(s).strip() for s in (known_speakers or []) if str(s).strip()]
    turns = []
    for line_no, line in enumerate((raw_text or "").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        parsed = _parse_header(line, line_no)
        if parsed is None:
            if not turns:
                raise DialogueScriptError(
                    f"第 {line_no} 行缺少角色前缀：应写作 `角色名: 台词` 或 "
                    "`[角色名] 台词`；续行需要行首缩进并紧跟在对应角色那句之后。")
            turns[-1].text = _join_text(turns[-1].text, stripped)
            turns[-1].lines.append(line_no)
            continue

        speaker, options, text = parsed
        if available and speaker not in available:
            raise DialogueScriptError(
                f"第 {line_no} 行：角色 {speaker!r} 未注册（已注册："
                f"{', '.join(available)}）。二选一：给 Speaker Pack 补一个 "
                f"name={speaker} 的 IndexTTS Speaker（旁白也要一段参考音频当音色），"
                f"或把这一行的角色名改成已注册的角色。")
        if not text:
            raise DialogueScriptError(
                f"第 {line_no} 行：角色 {speaker!r} 后面没有台词内容。")
        turns.append(Turn(speaker, text, options, lines=[line_no]))

    if not turns:
        raise DialogueScriptError("对话脚本里没有任何台词行。")

    # 向量优先于文本（与单文本节点里 emo_vector 覆盖 emo_text 的语义一致）
    for turn in turns:
        if turn.options.get("emo_vector") is not None:
            turn.options.pop("emo_text", None)
    return turns
