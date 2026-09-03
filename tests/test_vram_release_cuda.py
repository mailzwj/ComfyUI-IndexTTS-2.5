"""显存回收验证（真实 GPU，秒级完成）。

不加载 3GB+ 的真实权重，而是用与 IndexTTS2 相同属性名的大 CUDA 张量模拟常驻模型，
走 loader.release() 的真实路径（断引用 → gc.collect → cuda.synchronize → empty_cache），
再用 torch.cuda.mem_get_info() 实测「归还给系统的显存」是否达到预期。

运行：
    & "D:\\UserApps\\Conda\\envs\\cu130\\python.exe" tests\\test_vram_release_cuda.py
"""
import os
import sys
import time

NODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NODE_DIR)

import torch  # noqa: E402


class FakeResidentModel:
    """模拟 IndexTTS2 的主要子模型常驻显存（属性名与 _drop_tts_refs 对齐）。"""

    def __init__(self, total_gb=1.5):
        n = int(total_gb * 1024 ** 3 // 4)  # float32 元素数
        self.gpt = torch.zeros(n, device="cuda")                    # 主模型（最大头）
        self.semantic_model = torch.zeros(n // 4, device="cuda")    # Wav2Vec2-BERT
        self.s2mel = torch.zeros(n // 8, device="cuda")             # CFM estimator
        self.bigvgan = torch.zeros(n // 16, device="cuda")          # 声码器
        self.cache_mel = None


def main():
    assert torch.cuda.is_available(), "本测试需要 CUDA"

    from model_loader import IndexTTS2_5Loader

    free0, total = torch.cuda.mem_get_info()
    print(f"[vram] 基线：GPU 总 {total / 2**30:.2f} GB，空闲 {free0 / 2**30:.2f} GB")
    if free0 < 3 * 1024 ** 3:
        print("[vram] FAIL：空闲显存不足 3GB（llama-server 等占用过多），跳过")
        sys.exit(2)

    loader = IndexTTS2_5Loader(device="cuda")
    fake = FakeResidentModel(total_gb=1.5)
    loader._cache[("tts", True, False)] = fake
    torch.cuda.synchronize()

    free1, _ = torch.cuda.mem_get_info()
    alloc_loaded = torch.cuda.memory_allocated() / 2 ** 30
    print(f"[vram] 模拟常驻后：空闲 {free1 / 2**30:.2f} GB，torch 分配 {alloc_loaded:.2f} GB")

    t0 = time.time()
    loader.release()
    release_s = time.time() - t0
    time.sleep(1)

    free2, _ = torch.cuda.mem_get_info()
    alloc_after = torch.cuda.memory_allocated() / 2 ** 30
    delta = (free2 - free1) / 2 ** 30
    print(f"[vram] release 耗时 {release_s:.2f}s")
    print(f"[vram] 释放后：空闲 {free2 / 2**30:.2f} GB，torch 分配 {alloc_after:.2f} GB")
    print(f"[vram] 归还系统显存：{delta:.2f} GB（模拟占用约 {alloc_loaded:.2f} GB）")

    assert loader._cache == {}, "缓存应已清空"
    assert alloc_after < 0.1, f"torch 分配显存应接近 0，实际 {alloc_after:.2f} GB"
    assert delta > 0.8, f"归还显存不足：{delta:.2f} GB"
    print("[vram] PASS：release() 后显存确实归还系统")


if __name__ == "__main__":
    main()
