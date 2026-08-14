# ComfyUI-IndexTTS-2_5

IndexTTS-2.5 的 ComfyUI 自定义节点。零样本文本转语音，支持中文 / 英文 / 日语 /
西班牙语 / 阿拉伯语，具备情感控制与语速控制。

## 节点

- **Index TTS 2.5**（`IndexTTS2_5Node`）
  - 必填：`text`、`reference_audio`（AUDIO）、`language`、`duration_factor`、`seed`
  - 可选情感控制：`emo_ref_audio` / `emo_vector`（8 维）/ `emo_text`（需 QwenEmotion）
  - 采样参数：`do_sample` / `temperature` / `top_p` / `top_k` / `repetition_penalty` 等
  - 输出：`audio`（AUDIO）、`seed`、`subtitle`

### 工作流预览

![Index TTS 2.5 工作流](preview.png)

参考音频：

<audio controls src="example/ref_audio.mp3"></audio>

生成结果：

<audio controls src="example/result.wav"></audio>

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
