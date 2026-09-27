#!/usr/bin/env python3
"""
Dynamic GPU Hardware Manager & Multi-GPU Auto-Scaler
Detects single or multi-GPU environments (Tesla T4, A10G, L4, RTX 3090/4090, A100, H100),
calculates optimal layer offload, VRAM sharding (tensor_split), and context window scaling.
"""

import os
import subprocess
import torch

def get_driver_version() -> str:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=2
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "N/A"

def get_hardware_profile() -> dict:
    """
    Returns complete real-time hardware profile, detecting single/multi-GPU,
    VRAM capacities, compute capability, and recommended LLM parameters.
    """
    driver_ver = get_driver_version()
    cuda_ver = getattr(torch.version, "cuda", "N/A")

    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device_count": 0,
            "tier": "CPU Only",
            "primary_device": "CPU (No CUDA Accelerator)",
            "primary_vram_gb": 0.0,
            "total_vram_all_gpus_gb": 0.0,
            "free_vram_all_gpus_gb": 0.0,
            "driver_version": driver_ver,
            "cuda_version": cuda_ver,
            "compute_capability": "0.0",
            "flash_attn_supported": False,
            "recommended_ctx": 1024,
            "recommended_n_gpu_layers": 0,
            "tensor_split": None,
            "devices": []
        }

    device_count = torch.cuda.device_count()
    devices = []
    total_vram_all = 0.0
    free_vram_all = 0.0

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        name = props.name
        cap = f"{props.major}.{props.minor}"
        
        # Query free/total memory for this device
        try:
            free_b, total_b = torch.cuda.mem_get_info(i)
            tot_gb = round(total_b / (1024**3), 2)
            free_gb = round(free_b / (1024**3), 2)
            used_gb = round((total_b - free_b) / (1024**3), 2)
        except Exception:
            tot_gb = round(props.total_memory / (1024**3), 2)
            free_gb = tot_gb
            used_gb = 0.0

        total_vram_all += tot_gb
        free_vram_all += free_gb

        devices.append({
            "index": i,
            "name": name,
            "total_vram_gb": tot_gb,
            "free_vram_gb": free_gb,
            "used_vram_gb": used_gb,
            "compute_capability": cap,
            "major": props.major,
            "minor": props.minor
        })

    primary = devices[0]
    primary_name = primary["name"]
    primary_vram = primary["total_vram_gb"]
    primary_free = primary["free_vram_gb"]
    primary_cap = primary["compute_capability"]
    
    # Check flash attention (Turing 7.5, Ampere 8.0/8.6, Ada 8.9, Hopper 9.0)
    flash_supported = (primary["major"] > 7) or (primary["major"] == 7 and primary["minor"] >= 5)

    # Classify GPU Profile Tier based on primary / aggregate VRAM
    effective_vram = total_vram_all if device_count > 1 else primary_vram
    if effective_vram >= 36.0:
        tier = "Enterprise Ultra-VRAM (A100 / H100 / Multi-GPU)"
        rec_ctx = 8192
    elif effective_vram >= 20.0:
        tier = "Workstation Pro (A10G / L4 / RTX 3090 / RTX 4090)"
        rec_ctx = 4096
    elif effective_vram >= 12.0:
        tier = "Standard Server Acceleration (Tesla T4 / RTX 3080 / RTX 4070)"
        rec_ctx = 2048
    else:
        tier = "Entry Accelerator (< 12 GB VRAM)"
        rec_ctx = 1024

    # Allow environment override for context window
    env_ctx = os.environ.get("LLM_CTX_SIZE")
    if env_ctx:
        try:
            rec_ctx = int(env_ctx)
        except ValueError:
            pass

    # Multi-GPU Tensor Split calculation
    tensor_split = None
    if device_count > 1:
        vrams = [d["total_vram_gb"] for d in devices]
        tot = sum(vrams) if sum(vrams) > 0 else 1.0
        tensor_split = [round(v / tot, 4) for v in vrams]

    return {
        "cuda_available": True,
        "device_count": device_count,
        "tier": tier,
        "primary_device": primary_name,
        "primary_vram_gb": primary_vram,
        "primary_free_vram_gb": primary_free,
        "total_vram_all_gpus_gb": round(total_vram_all, 2),
        "free_vram_all_gpus_gb": round(free_vram_all, 2),
        "driver_version": driver_ver,
        "cuda_version": str(cuda_ver),
        "compute_capability": primary_cap,
        "flash_attn_supported": flash_supported,
        "recommended_ctx": rec_ctx,
        "recommended_n_gpu_layers": -1, # 100% offload
        "tensor_split": tensor_split,
        "devices": devices
    }

def get_llama_init_kwargs(requested_ctx: int = None, model_size_gb: float = 4.5) -> dict:
    """
    Returns optimal kwargs for llama_cpp.Llama initialization based on real-time GPU hardware.
    """
    prof = get_hardware_profile()
    
    # Context window
    ctx = requested_ctx or prof["recommended_ctx"]
    
    # Threading: auto-tune based on available CPU cores
    cpu_cores = os.cpu_count() or 4
    threads = min(8, max(2, cpu_cores // 2))

    kwargs = {
        "n_ctx": ctx,
        "n_threads": threads,
        "verbose": False
    }

    if prof["cuda_available"]:
        kwargs["n_gpu_layers"] = -1 # 100% offload to CUDA
        
        # Multi-GPU support
        if prof["device_count"] > 1 and prof["tensor_split"]:
            kwargs["tensor_split"] = prof["tensor_split"]
            print(f"🚀 Multi-GPU Active ({prof['device_count']} GPUs): Tensor split {prof['tensor_split']}")

        # Flash attention
        if prof["flash_attn_supported"]:
            kwargs["flash_attn"] = True

    else:
        kwargs["n_gpu_layers"] = 0

    return kwargs

def can_run_clm() -> tuple[bool, str]:
    """
    Checks if system hardware can run CLM-8B unquantized without CUDA OOM.
    """
    prof = get_hardware_profile()
    if not prof["cuda_available"]:
        return False, "CUDA tidak tersedia. CLM-8B membutuhkan akselerator GPU."

    free_gb = prof["free_vram_all_gpus_gb"]
    total_gb = prof["total_vram_all_gpus_gb"]
    dev_name = prof["primary_device"]

    # CLM-8B requires ~16 GB free VRAM
    if free_gb >= 15.5:
        return True, f"Hardware GPU memadai: {dev_name} dengan sisa {free_gb} GB bebas dari {total_gb} GB total."
    else:
        return False, (
            f"VRAM Bebas ({free_gb} GB) di bawah ambang batas minimum 16 GB untuk CLM-8B fp16 unquantized. "
            f"Perangkat saat ini: {dev_name} (Total: {total_gb} GB). "
            f"Model ini otomatis aktif penuh pada GPU 24GB - 80GB (seperti A10G, L4, RTX 3090/4090, A100)."
        )

def print_hardware_summary():
    """Prints beautiful formatted terminal banner for GPU detection."""
    p = get_hardware_profile()
    print("=" * 80)
    print("💻 DIAGNOSTIK HARDWARE & DETEKSI AKSELERASI GPU DINAMIS")
    print("=" * 80)
    if not p["cuda_available"]:
        print("  ⚠️  Akselerator CUDA tidak terdeteksi. Berjalan dalam mode CPU.")
        print("=" * 80)
        return

    if p["device_count"] == 1:
        print(f"  • GPU Model            : {p['primary_device']}")
        print(f"  • Kapasitas VRAM       : {p['primary_vram_gb']} GB (Sisa Bebas: {p['primary_free_vram_gb']} GB)")
        print(f"  • Compute Capability   : {p['compute_capability']} (Arsitektur NVIDIA)")
    else:
        print(f"  • Konfigurasi GPU      : MULTI-GPU ({p['device_count']} Perangkat Terdeteksi)")
        for d in p["devices"]:
            print(f"    - GPU {d['index']}: {d['name']} ({d['total_vram_gb']} GB VRAM, {d['free_vram_gb']} GB Bebas)")
        print(f"  • Total VRAM Gabungan  : {p['total_vram_all_gpus_gb']} GB (Bebas: {p['free_vram_all_gpus_gb']} GB)")
        print(f"  • Tensor Split Ratios  : {p['tensor_split']}")

    print(f"  • NVIDIA Driver        : {p['driver_version']}")
    print(f"  • CUDA Runtime         : {p['cuda_version']}")
    print(f"  • Flash Attention      : {'✓ Didukung (Sub-linear KV Memory)' if p['flash_attn_supported'] else 'Standard Attention'}")
    print(f"  • Klasifikasi Tier GPU : {p['tier']}")
    print(f"  • Rekomendasi Context  : {p['recommended_ctx']} tokens")
    print("=" * 80)

if __name__ == "__main__":
    print_hardware_summary()
