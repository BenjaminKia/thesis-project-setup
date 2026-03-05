"""
src/utils/check_gpu.py
Verify GPU setup and estimate VRAM usage for each model.

Usage:
    python -m src.utils.check_gpu
"""

import torch
import sys


def check_gpu():
    print("=" * 60)
    print("GPU & Environment Check")
    print("=" * 60)

    print(f"\nPython:  {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n⚠ CUDA not available! Check your PyTorch installation.")
        print("  Install with: conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia")
        return

    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1e9
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total VRAM:  {total_gb:.1f} GB")
        print(f"  Compute cap: {props.major}.{props.minor}")

    # VRAM estimates for your models (with fp16)
    print("\n" + "=" * 60)
    print("VRAM Estimates (fp16 training, per config.yaml)")
    print("=" * 60)

    estimates = [
        ("bert-base-uncased", "110M", "~3.5 GB", "batch=16", "✅ Fits 4060"),
        ("ProsusAI/finbert", "110M", "~3.5 GB", "batch=16", "✅ Fits 4060"),
        ("roberta-base", "125M", "~3.8 GB", "batch=16", "✅ Fits 4060"),
        ("facebook/opt-350m", "350M", "~5.5 GB", "batch=8", "✅ Fits 4060"),
        ("facebook/opt-1.3b", "1.3B", "~12 GB", "batch=4", "❌ Needs Colab"),
    ]

    print(f"\n{'Model':<28} {'Params':<8} {'VRAM':<10} {'Batch':<10} {'Status'}")
    print("-" * 78)
    for model, params, vram, batch, status in estimates:
        print(f"{model:<28} {params:<8} {vram:<10} {batch:<10} {status}")

    print("\nTip: If you hit OOM errors, reduce batch_size and increase")
    print("     gradient_accumulation_steps to maintain effective batch size.")

    # Quick memory test
    print("\n" + "=" * 60)
    print("Quick VRAM allocation test...")
    try:
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.matmul(x, x.T)
        allocated = torch.cuda.memory_allocated() / 1e6
        print(f"  Allocated {allocated:.1f} MB — GPU is working correctly ✅")
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  GPU test failed: {e}")


if __name__ == "__main__":
    check_gpu()
