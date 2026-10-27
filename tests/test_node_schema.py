"""节点注册表守卫：ComfyUI 只有在真正装载节点时才会暴露的注册期错误，这里离线跑一遍。

覆盖 __init__.py 里的每个节点：INPUT_TYPES 结构、RETURN_TYPES 与 RETURN_NAMES
数量一致、FUNCTION 存在，以及显示名不重复。

运行方式（任选其一）：
    python tests/test_node_schema.py
    pytest tests/test_node_schema.py -v
"""
import importlib.util
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bootstrap import DS, PKG, ROOT  # noqa: E402

DIALOGUE_WORKFLOW = os.path.join(ROOT, "workflow", "index-tts-dialogue.json")
# 只能连线的自定义类型（其余都是面板上的 widget）
LINK_TYPES = {"AUDIO", "SPEAKER", "SPEAKER_PACK", "TTS_OPTIONS"}


def widget_keys(node_class):
    """按 INPUT_TYPES 推导面板 widget 的顺序（示例工作流按这个顺序存值）。"""
    schema = node_class.INPUT_TYPES()
    return [key for group in ("required", "optional")
            for key, spec in schema.get(group, {}).items()
            if not (isinstance(spec[0], str) and spec[0] in LINK_TYPES)]


def load_registry():
    """执行真实的 __init__.py（相对导入落到 _bootstrap 已挂载的 it25 子模块上）。"""
    spec = importlib.util.spec_from_file_location(
        "it25_registry", os.path.join(ROOT, "__init__.py"))
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "it25"
    spec.loader.exec_module(module)
    return module


REGISTRY = load_registry()


class TestNodeRegistry(unittest.TestCase):
    def test_display_names_cover_every_class_and_are_unique(self):
        classes = REGISTRY.NODE_CLASS_MAPPINGS
        names = REGISTRY.NODE_DISPLAY_NAME_MAPPINGS
        self.assertEqual(set(classes), set(names))
        self.assertEqual(len(set(names.values())), len(names))

    def test_input_types_shape(self):
        for class_name, node_class in REGISTRY.NODE_CLASS_MAPPINGS.items():
            with self.subTest(class_name):
                schema = node_class.INPUT_TYPES()
                self.assertIsInstance(schema, dict)
                for group in ("required", "optional"):
                    for key, spec in schema.get(group, {}).items():
                        self.assertIsInstance(spec, (tuple, list), key)
                        self.assertTrue(spec, f"{class_name}.{key} 类型声明为空")
                        # 类型名既可以是一个字符串，也可以是组合下拉的列表
                        choices = spec[0] if isinstance(spec[0], (tuple, list)) else [spec[0]]
                        self.assertTrue(choices, f"{class_name}.{key} 没有类型")
                        for choice in choices:
                            self.assertIsInstance(choice, str, key)
                        if len(spec) > 1:
                            self.assertIsInstance(spec[1], dict, f"{class_name}.{key}")
                        if group == "required" and isinstance(spec[0], list):
                            # 下拉选项里放 None 会让前端渲染出空选项
                            self.assertNotIn(None, spec[0], key)

    def test_returns_and_function(self):
        for class_name, node_class in REGISTRY.NODE_CLASS_MAPPINGS.items():
            with self.subTest(class_name):
                self.assertEqual(len(node_class.RETURN_TYPES),
                                 len(node_class.RETURN_NAMES), class_name)
                self.assertTrue(hasattr(node_class, node_class.FUNCTION),
                                f"{class_name} 没有 {node_class.FUNCTION} 方法")
                self.assertTrue(node_class.CATEGORY, class_name)

    def test_dialogue_outputs_are_four(self):
        dialogue = REGISTRY.NODE_CLASS_MAPPINGS["IndexTTSDialogueNode"]
        self.assertEqual(dialogue.RETURN_TYPES, ("AUDIO", "INT", "STRING", "STRING"))
        self.assertEqual(dialogue.RETURN_NAMES, ("audio", "seed", "subtitle", "report"))

    def test_speaker_pack_chains(self):
        pack = REGISTRY.NODE_CLASS_MAPPINGS["IndexTTSSpeakerPackNode"].INPUT_TYPES()
        self.assertIn("prev_pack", pack["optional"])
        self.assertEqual(pack["optional"]["prev_pack"][0], "SPEAKER_PACK")
        self.assertEqual(REGISTRY.NODE_CLASS_MAPPINGS["IndexTTSSpeakerNode"].RETURN_TYPES,
                         ("SPEAKER",))

    def test_dialogue_custom_slot_types(self):
        dialogue = REGISTRY.NODE_CLASS_MAPPINGS["IndexTTSDialogueNode"].INPUT_TYPES()
        self.assertEqual(dialogue["required"]["speakers"][0], "SPEAKER_PACK")
        self.assertEqual(dialogue["optional"]["options"][0], "TTS_OPTIONS")
        settings = REGISTRY.NODE_CLASS_MAPPINGS["IndexTTSSettingsNode"]
        self.assertEqual(settings.RETURN_TYPES, ("TTS_OPTIONS",))


class TestExampleWorkflow(unittest.TestCase):
    """示例工作流是用户第一接触到的东西，必须和节点定义保持一致。"""

    def setUp(self):
        with open(DIALOGUE_WORKFLOW, encoding="utf-8") as handle:
            self.graph = json.load(handle)
        self.nodes = {node["id"]: node for node in self.graph["nodes"]}

    def test_widget_values_match_node_definitions(self):
        for node in self.graph["nodes"]:
            node_class = REGISTRY.NODE_CLASS_MAPPINGS.get(node["type"])
            if node_class is None:
                continue
            with self.subTest(node["type"]):
                self.assertEqual(len(widget_keys(node_class)),
                                 len(node["widgets_values"]),
                                 "示例里的 widget 数量与 INPUT_TYPES 不一致")

    def test_script_only_uses_registered_speakers(self):
        """旁白翻车回归：示例脚本里的角色必须在示例的 Speaker 节点里都存在。"""
        names = [node["widgets_values"][0]
                 for node in self.graph["nodes"] if node["type"] == "IndexTTSSpeakerNode"]
        dialogue = next(n for n in self.graph["nodes"]
                        if n["type"] == "IndexTTSDialogueNode")
        turns = DS.parse_script(dialogue["widgets_values"][0], known_speakers=names)
        self.assertEqual([t.speaker for t in turns], ["小雪", "阿明", "小雪", "旁白"])

    def test_pack_receives_every_speaker(self):
        pack = next(n for n in self.graph["nodes"]
                    if n["type"] == "IndexTTSSpeakerPackNode")
        wired = [entry["name"] for entry in pack["inputs"]
                 if entry["name"] != "prev_pack" and entry["link"] is not None]
        speakers = [n for n in self.graph["nodes"] if n["type"] == "IndexTTSSpeakerNode"]
        self.assertEqual(wired, [f"s{i}" for i in range(1, len(speakers) + 1)])

    def test_every_link_is_wired_on_both_ends(self):
        for lid, src, sslot, dst, dslot, type_name in self.graph["links"]:
            with self.subTest(lid):
                self.assertIn(lid, self.nodes[src]["outputs"][sslot]["links"])
                target = self.nodes[dst]["inputs"][dslot]
                self.assertEqual(target["link"], lid)
                self.assertEqual(self.nodes[src]["outputs"][sslot]["type"], type_name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
