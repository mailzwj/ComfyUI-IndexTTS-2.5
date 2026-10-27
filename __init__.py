"""
@title: IndexTTS-2.5 for ComfyUI
@description: ComfyUI 接口的 IndexTTS-2.5 零样本文本转语音节点（中/英/日/西/阿），含多人对话
"""

from .nodes import IndexTTS2_5Node, EmoSettingsNode, IndexTTSTextPreviewNode
from .nodes_dialogue import (IndexTTSDialogueNode, IndexTTSSpeakerNode,
                             IndexTTSSpeakerPackNode, IndexTTSSettingsNode)

NODE_CLASS_MAPPINGS = {
    "IndexTTS2_5Node": IndexTTS2_5Node,
    "EmoSettingsNode": EmoSettingsNode,
    "IndexTTSTextPreviewNode": IndexTTSTextPreviewNode,
    "IndexTTSSpeakerNode": IndexTTSSpeakerNode,
    "IndexTTSSpeakerPackNode": IndexTTSSpeakerPackNode,
    "IndexTTSSettingsNode": IndexTTSSettingsNode,
    "IndexTTSDialogueNode": IndexTTSDialogueNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS2_5Node": "Index TTS 2.5",
    "EmoSettingsNode": "Emo Settings",
    "IndexTTSTextPreviewNode": "IndexTTS Text Preview",
    "IndexTTSSpeakerNode": "IndexTTS Speaker",
    "IndexTTSSpeakerPackNode": "IndexTTS Speaker Pack",
    "IndexTTSSettingsNode": "IndexTTS Dialogue Settings",
    "IndexTTSDialogueNode": "Index TTS 2.5 Dialogue",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
