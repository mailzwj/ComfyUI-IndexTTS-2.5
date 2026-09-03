"""真实模型冒烟测试（CPU 模式）：验证 release() 后内存确实回落。

显存不足时无法在 GPU 上跑完整推理，本脚本只覆盖「加载 → 缓存命中 → 释放」
链路（与生成成功后的释放路径完全一致），并打印各阶段进程 RSS。

运行：
    & "D:\\UserApps\\Conda\\envs\\cosyvoice\\python.exe" tests\\smoke_release_real_model.py
"""
import ctypes
import gc
import os
import sys
import time

NODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NODE_DIR)


def rss_mb() -> float:
    """进程 RSS（MB）。优先 psutil，否则退回系统已用物理内存（粗略）。"""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1048576
    except ImportError:
        class _M(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        m = _M()
        m.dwLength = ctypes.sizeof(_M)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return (m.ullTotalPhys - m.ullAvailPhys) / 1048576


def main():
    from model_loader import IndexTTS2_5Loader

    rss_base = rss_mb()
    print(f"[smoke] 基线 RSS: {rss_base:.0f} MB")

    loader = IndexTTS2_5Loader(device="cpu")
    t0 = time.time()
    tts = loader.get_tts(use_bf16=False, use_qwen_emo=False)
    load_s = time.time() - t0
    print(f"[smoke] 模型加载完成，耗时 {load_s:.1f}s，device={tts.device}")
    rss_loaded = rss_mb()
    print(f"[smoke] 加载后 RSS: {rss_loaded:.0f} MB (+{rss_loaded - rss_base:.0f})")

    # 缓存命中验证
    assert loader.get_tts(use_bf16=False) is tts, "缓存命中失败"
    print("[smoke] 缓存命中 OK")

    # 释放（与节点 auto_free_memory=True 时调用的路径一致）
    t0 = time.time()
    loader.release()
    release_s = time.time() - t0
    gc.collect()
    time.sleep(2)  # 等待 OS 归还页面
    rss_after = rss_mb()
    print(f"[smoke] release 耗时 {release_s:.1f}s")
    print(f"[smoke] 释放后 RSS: {rss_after:.0f} MB（加载前基线见上）")

    delta = rss_loaded - rss_after
    print(f"[smoke] 结论：释放回收约 {delta / 1024:.2f} GB 进程内存")
    if delta < 1 * 1024:  # 至少应回收 1GB 以上才算有效
        print("[smoke] FAIL：回收量不足")
        sys.exit(1)
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
