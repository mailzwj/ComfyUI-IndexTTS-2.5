"""
@title: IndexTTS-2.5 for ComfyUI
@description: ComfyUI 接口的 IndexTTS-2.5 零样本文本转语音节点（中/英/日/西/阿）
"""

from .nodes import IndexTTS2_5Node

NODE_CLASS_MAPPINGS = {
    "IndexTTS2_5Node": IndexTTS2_5Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS2_5Node": "Index TTS 2.5",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
