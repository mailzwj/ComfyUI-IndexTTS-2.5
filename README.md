# ComfyUI-IndexTTS-2_5

IndexTTS-2.5 的 ComfyUI 自定义节点。零样本文本转语音，支持中文 / 英文 / 日语 /
西班牙语 / 阿拉伯语，具备情感控制与语速控制。

## 节点

- **Index TTS 2.5**（`IndexTTS2_5Node`）
  - 必填：`text`、`reference_audio`（AUDIO）、`language`、`duration_factor`、`seed`
  - 可选情感控制：`emo_ref_audio` / `emo_vector`（8 维）/ `emo_text`（需 QwenEmotion）
  - 长文本与分段：`max_text_tokens_per_segment` / `interval_silence` / `strict_max_mel_tokens`
  - 发音控制：`text_normalization` / `glossary_yaml`（详见「发音与术语表」）
  - 字幕：`subtitle_format`（`srt` / `vtt` / `plain`，默认 `srt`）/ `subtitle_text`
  - 采样参数：`do_sample` / `temperature` / `top_p` / `top_k` / `repetition_penalty` 等
  - 输出：`audio`（AUDIO）、`seed`、`subtitle`（真实时间轴字幕，见「长文本与字幕」）
- **Emo Settings**（`EmoSettingsNode`）
  - 8 个情感滑块：`happy` / `angry` / `sad` / `afraid` / `disgusted` / `melancholic` / `surprised` / `calm`，范围 0~1
  - 输出：`emo_vector`（STRING，逗号分隔的 8 维向量）
  - 连线到 `Index TTS 2.5` 的 `emo_vector` 输入即可，无需手写向量字符串
- **IndexTTS Text Preview**（`IndexTTSTextPreviewNode`）
  - 文本前端预览：不合成音频，直接给出模型**实际读法**（`processed_text`）、
    分段结果（`segments` / `segment_count`）与诊断信息（`diagnostics`）
  - 输入与生成节点同名（`language` / `text_normalization` / `max_text_tokens_per_segment` /
    `glossary_yaml`），用于在跑 GPU 之前先确认读音和分段是否符合预期

### 工作流预览

![Index TTS 2.5 工作流](preview.png)

参考音频：

<audio controls src="example/ref_audio.mp3"></audio>

生成结果：

<audio controls src="example/result.wav"></audio>

## 使用说明

### 情感控制

节点提供三种情感控制方式：情感参考音频 `emo_ref_audio`、情感向量 `emo_vector`、
情感文本 `emo_text`。三者互斥——提供 `emo_vector` 或 `emo_text` 时，`emo_ref_audio`
会被忽略。

#### emo_vector（情感向量）

8 个数字，顺序固定：

```
高兴 / 愤怒 / 悲伤 / 恐惧 / 反感 / 低落 / 惊讶 / 自然
```

支持逗号分隔或 JSON 数组两种写法：

- `0.8,0,0,0,0,0,0,0`
- `[0.8,0,0,0,0,0,0,0]`

规则：

- 必须**恰好 8 个**数字，否则被忽略（不生效）。
- 数值建议 0~1，越大该情绪越强；`0,0,0,0,0,0,0,1` 表示自然 / 平静。
- `emo_alpha`（0~1）会整体缩放向量强度。

示例：

- 高兴：`0.8,0,0,0,0,0,0,0`
- 愤怒：`0,0.9,0,0,0,0,0,0`
- 悲伤：`0,0,0.8,0,0,0,0,0`
- 自然：`0,0,0,0,0,0,0,1`

> 也可用 **Emo Settings** 节点：8 个滑块（0~1）可视化调节情绪，输出即 `emo_vector`，
> 直接连线到 `Index TTS 2.5` 的 `emo_vector` 输入，免去手写字符串。

#### emo_text（情感文本）

用自然语言描述情绪，由 QwenEmotion 模型自动分析为 8 维情感向量（效果等同
`emo_vector`）：

- 支持「高兴」「愤怒」「悲伤」「恐惧」「反感」「低落」「惊讶」「自然」，也支持完整句子。
- **非空即自动启用**：无需单独勾选 `use_emo_text` / `use_qwen_emo`。
- 需要 `qwen0.6bemo4-merge/` 模型（见「模型放置」），否则报错。

> 触发词「低落 / melancholy / depressed / gloomy」等会映射到「低落」，避免被误判为「悲伤」。

> **只勾选 `use_emo_text` 但不填 `emo_text`** 是一个有用的组合：此时情感**从正文本身
> 推断**（上游把主文本当作情感描述），适合「这段话自带情绪、不用再描述一遍」的场景。
> 需要 `qwen0.6bemo4-merge/` 模型（会自动加载）。

### 长文本与字幕

长文本不再是一次性塞给 `infer()`，而是走流式接口 `infer_generator()`：**逐段合成、
逐段计时**，最后拼成一条完整音频。这样每一段都有真实的时长，字幕时间轴是**真实**的
（旧版本按句子数量均分总时长，长文本会越往后越偏）。

#### 参数

- **`max_text_tokens_per_segment`（INT，默认 120，10~300）**
  单段允许的最大 token 数。文本会按标点/长度切成若干段，每段独立生成。
  - 调**小**：单段更短、更稳（不容易被截断），但段间拼接次数增多、整体稍慢。
  - 调**大**：段落更连贯，但一段超出 `max_mel_tokens` 就会被截断（音频不完整）。
- **`interval_silence`（INT，默认 200 ms，0~2000）**
  段与段之间插入的静音。只影响音频与时间轴，**不计入任何一帧字幕**（字幕窗口仍然
  贴合各自的人声）。设为 0 则段间无停顿。
- **`strict_max_mel_tokens`（BOOLEAN，默认 `false`）**
  上游在单段超出 `max_mel_tokens` 时会丢弃后半段并只发一条 `RuntimeWarning`，表现为
  「音频莫名变短」。
  - `false`：控制台打印明确提示（调小 `max_text_tokens_per_segment` 或调大
    `max_mel_tokens`），仍返回不完整音频。
  - `true`：直接报错中断，适合批量生产——宁可失败也不要拿到半句话的音频。
- **进度条**：在 ComfyUI 中运行时会自动挂上 `comfy.utils.ProgressBar`，按段推进；
  脱离 ComfyUI 运行（如单测）时自动跳过。

#### 低显存模式

模型自身在 `low_vram` 模式下会把过长的文本先按标点分块再逐块合成，本节点会沿用这一
行为（控制台打印「低显存模式：文本按标点切成 N 块逐块生成」），无需手动配置。

#### 字幕输出（`subtitle`）

- **`subtitle_format`（`srt` / `vtt` / `plain`，默认 `srt`）**
  - `srt`：`00:00:01,700 --> 00:00:03,100`（可直接存成 `.srt` 喂剪辑软件）
  - `vtt`：带 `WEBVTT` 头、`00:00:01.700` 小数点格式（网页 `<track>` 用）
  - `plain`：旧版 `>> M:SS.mmm-M:SS.mmm` + 文本行
- **`subtitle_text`（`synthesized` / `original`，默认 `synthesized`）**
  - `synthesized`：模型**实际朗读**的文本（数字、单位已被读法归一，如 `2026年` →
    `二零二六年`），和音频一一对应。
  - `original`：回到你输入的原文（按字符占比映射回各段，适合直接当作品字幕）。
- 退化路径：极少数情况下（上游分段数与音频段数不一致）无法逐段对齐，此时输出整段
  时长按字符均分的字幕，并在控制台打印「分段文本捕获不一致……」提示。
- 发音标注不会污染字幕：字幕里出现的是**原词**，既不会出现 `<|SPECIAL_TOKEN_2|>ZHAO1<…|>`
  这类上游替换产物，也不会残留 `<朝|zhao1>` 写法（例：输入 `长<朝|zhao1>夕` → 字幕
  `长朝夕`）。`subtitle_text=original` 同样会去掉标注记号。

> **升级注意**：`subtitle` 的默认格式从旧版的 `plain` 变为 **`srt`**。若有下游依赖旧
> 的 `>> M:SS.mmm` 文本格式，把 `subtitle_format` 显式设为 `plain` 即可。

### 发音与术语表

两条互补的纠偏手段，都作用于**文本前端**（不改模型权重）：

#### 1）`<文字|读音>` 发音标注（临时纠偏）

写在 `text` 里，就地覆盖某个词的读法：

- 中文用**带声调的拼音**：`<行|XING2>`、`<重|CHONG2>`
- 英文用 **ARPAbet 音素**（可含重音数字）：`<going|G OW1 . IH0 NG>`
- 日文可直接给假名（假名会被识别并直接朗读）

标注在归一化**之后**、分段**之前**处理，因此不受数字读法改写的影响。写错的标注不会
静默失败——`generate` 会在控制台打印问题清单（标签未闭合、缺 `|`、中文拼音没声调、
发音为空等），只提示不中断。

> **IndexTTS Text Preview** 的 `processed_text` / `segments` 会显示上游替换后的
> `<|SPECIAL_TOKEN_1|>…` / `<|SPECIAL_TOKEN_2|>…` 形态——那正是模型真正读的内容，
> 便于核对发音与 token 预算；生成节点输出的 `subtitle` 会自动还原成原词。

#### 2）术语表 `glossary`（可复用，按语言区分）

**放置位置**（模型启动时自动读取，支持热更新）：

```
<ComfyUI>/models/TTS/IndexTTS-2.5/glossary.yaml
```

格式：

```yaml
M.2:
  zh: M 二
  en: M dot two
NVMe: N-V-M-E          # 不区分语言时直接写读法
Stable Diffusion:
  zh: 稳定扩散
```

规则：

- 匹配**大小写不敏感**，按术语长度从长到短替换（`PCIe 5.0` 优先于 `PCIe`）。
- **热更新**：本节点按文件修改时间检测变更，改完 `glossary.yaml` **不用重启 ComfyUI**，
  下一次生成自动生效（控制台打印「术语表已生效：文件 N 条 + 节点 M 条」）。
  上游原生只在模型初始化时读一次，这一层是本节点补的。
- **`glossary_yaml`（节点内联）**：与文件同格式的 YAML，写在节点里，优先级**高于**
  文件（同名词条覆盖，不同名词条叠加）。适合临时试验，不用动模型目录。
- 非法条目（值既不是字符串也不是映射）会被忽略并在控制台提示，不会中断生成。

> **重要前提**：术语替换发生在上游 `normalize()` 内部，因此
> **`text_normalization` 必须为 `true`（默认）**，否则术语表不生效（`<文字|读音>` 标注
> 不受此开关影响，任何时候都生效）。
>
> 另外，上游只在**中文 / 英文**分支里调用术语表，所以 `ja` / `es` / `ar` 下
> `glossary` 不会被应用；这两种语言请直接用 `<文字|读音>` 标注。

#### 建议流程

1. 把文本连参数一起接 **IndexTTS Text Preview** 节点，看 `processed_text`
   （模型实际读什么）和 `segments` / `diagnostics`（怎么分段、有没有超出 token 预算）；
2. 读法不对 → 补 `<文字|读音>` 或加术语表；分段太碎/超长 → 调
   `max_text_tokens_per_segment`；
3. 确认无误再跑 `Index TTS 2.5` 生成，省掉一轮 GPU 时间。

## 内存 / 显存管理

IndexTTS-2.5 整套模型（GPT 3.1 GB + Wav2Vec2-BERT + S2Mel + BigVGAN 等）加载后
常驻 GPU，默认行为是**模型常驻、重复生成快**。若需要把内存/显存腾给 ComfyUI 中的
其它模型（图像 / 视频节点），使用 `auto_free_memory` 开关：

- **`auto_free_memory`（BOOLEAN，默认 `false`）**
  - 勾选后：**每次生成成功**时自动卸载整套模型——清空加载器缓存、显式断开子模型引用、
    `gc.collect()`、`torch.cuda.empty_cache()`（把 PyTorch 缓存分配器预留的显存真正
    归还系统），控制台会打印释放后的 GPU 占用。
  - 代价：下次生成需要重新加载模型（约 10~60 秒，取决于磁盘与设备）。
  - 仅在**生成成功**后释放；推理失败时保留模型缓存以便直接重试。

相关机制（无需操作，自动生效）：

- **进程级共享加载器**：同一工作流中多个 `Index TTS 2.5` 节点复用同一个模型实例，
  不再出现「每个节点一份模型」导致的显存成倍增长；`release()` 也能一次性全部释放。
- **临时文件清理**：参考音频 / 情感参考音频写入的临时 wav 在每次 `generate` 结束
  （无论成功或失败）后自动删除，不再堆积在系统临时目录。

## 模型放置

将 IndexTTS-2.5 权重放到：

```
<ComfyUI>/models/TTS/IndexTTS-2.5/
```

必需文件（官方权重包自带）：

```
config.yaml
gpt.pth
s2mel.pth
codec.pth
wav2vec2bert_stats.pt
feat1.pt
feat2.pt
multilingual_zh_ja_yue_char_del.tiktoken
qwen0.6bemo4-merge/          # 可选，情感文本需要
glossary.yaml                 # 可选，术语表（见「发音与术语表」，支持热更新）
```

下载方式（HuggingFace）：

```bash
hf download IndexTeam/IndexTTS-2.5 --local-dir="<ComfyUI>/models/TTS/IndexTTS-2.5"
```

或 ModelScope：

```bash
modelscope download --model IndexTeam/IndexTTS-2.5 --local_dir "<ComfyUI>/models/TTS/IndexTTS-2.5"
```

## 辅助模型（首次运行自动下载）

三个辅助模型不会出现在权重包内，首次推理时由上游
`ensure_models_available` 自动下载到 `<模型目录>/hf_cache/`：

- `w2v-bert-2.0`（语义编码器）
- `campplus_cn_common.bin`（说话人嵌入）
- `bigvgan`（声码器）

若 HuggingFace 访问慢，建议提前设置镜像：

```bash
export HF_ENDPOINT="https://hf-mirror.com"
```

## 依赖

本节点 vendored 了官方 `indextts` 包源码（`indextts/` 目录），在 ComfyUI 进程内
直接运行，无需额外安装官方仓库。

关键点：官方 `pyproject.toml` 里的 `keras==2.9.0` 是历史遗留依赖，
**IndexTTS-2.5 推理链路并不 import keras**，因此本节点不要求安装 keras，
也不受官方 `Python<3.12` 的限制（可在 Python 3.12 下运行）。

> **transformers 版本**：本节点 vendored 的 `indextts` 源码是 transformers 4.52 的
> fork，已通过 `compatibility_patch.py` 适配到当前环境的 `transformers 4.57.x`
> （`QuantizedCacheConfig`→`QuantizedCache` 等改名）。若环境里的 transformers 版本
> 不同，可能需要重新调整该补丁。

```bash
pip install -r requirements.txt
```

- 日语 TTS 额外需要 `fugashi` + `unidic-lite`（未装时仅日语会报错）。
- `flash-attn` / CUDA kernel 加速默认关闭（避免 torch2.9+cu130 下 ABI 不匹配），
  需要时可在 `model_loader.py` 中开启。

## 节点开发环境

以下为开发 / 验证本节点时使用的环境（ComfyUI 运行于 conda 环境 `cu130`）：

| 项目 | 值 |
| --- | --- |
| 操作系统 | Windows 11（10.0.26200） |
| Python | 3.12.12 |
| PyTorch | 2.9.1+cu130 |
| CUDA（torch 运行时） | 13.0 |
| 显卡驱动 | 610.88（CUDA 13.3） |
| cuDNN | 9.12.0 |
| GPU | NVIDIA GeForce RTX 5090（32 GB） |
| transformers | 4.57.6 |

> `transformers` 版本与本节点的 `compatibility_patch.py` 绑定，升级后若报错需同步调整
> 该补丁（见上文「依赖」一节）。

## 测试

单测不加载真实权重，全部用桩模型（`tests/_bootstrap.py`），覆盖字幕时间轴、流式分段
收集、术语表热更新、文本前端复刻、发音标注校验以及内存/显存释放：

```bash
python -m unittest discover -s tests -t tests
```

其中 `tests/test_upstream_contract.py` 是「上游契约守卫」：用源码断言把本节点依赖的
vendored `indextts` 内部行为（唯一的 `split_text_by_tokens` 调用点、流式 `yield` 形态、
截断告警文案、`gr_progress` 挂钩、glossary 仅在 `normalize` 内生效等）钉住。上游升版
改动这些语句时它会先失败，而不是等线上出现字幕错位或音频莫名变短。

> 若环境里没有 `pytest`，用上面的 `unittest` 命令即可；为避免污染运行环境，建议在
> 独立目录下缓存字节码：`PYTHONPYCACHEPREFIX=/tmp/it25_pyc`。
