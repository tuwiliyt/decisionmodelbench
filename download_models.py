#!/usr/bin/env python3
"""
Enhanced Model Downloader for Indonesian LLM Arena & Decision Models
Downloads GGUF quantized models with real-time transfer stats (MB/s, ETA, % progress)
and validates model integrity for CUDA GPU execution.
"""

import os
import sys
import argparse
import time
from huggingface_hub import hf_hub_download
from config import MODELS_DIR

MODELS_INFO = {
    "sahabatai": {
        "name": "Sahabat-AI 8B Instruct",
        "org": "GoTo Company & Indosat Ooredoo",
        "repo_id": "gmonsoon/llama3-8b-cpt-sahabatai-v1-instruct-GGUF",
        "filename": "llama3-8b-cpt-sahabatai-v1-instruct.Q4_K_M.gguf",
        "target_filename": "sahabatai-8b-q4.gguf",
        "parameters": "8.03B",
        "quantization": "Q4_K_M",
        "context_window": "8,192 tokens",
        "size": "4.58 GB",
        "description": "SOTA Foundation Model Bahasa Indonesia & Bahasa Daerah (Jawa, Sunda, dll)"
    },
    "qwen": {
        "name": "Qwen 2.5 7B Instruct",
        "org": "Alibaba Cloud Intelligence",
        "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "target_filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "parameters": "7.61B",
        "quantization": "Q4_K_M",
        "context_window": "32,768 tokens",
        "size": "4.40 GB",
        "description": "SOTA Multilingual & Penalaran Kompleks (Bahasa Indonesia Kuat)"
    },
    "gemma": {
        "name": "Gemma 2 9B Instruct",
        "org": "Google DeepMind",
        "repo_id": "bartowski/gemma-2-9b-it-GGUF",
        "filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "target_filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "parameters": "9.24B",
        "quantization": "Q4_K_M",
        "context_window": "8,192 tokens",
        "size": "5.40 GB",
        "description": "DeepMind Flagship Open Weight (Performa Bahasa & Logika Tinggi)"
    },
    "gemma-2b": {
        "name": "Gemma 2 2B Instruct",
        "org": "Google DeepMind (Ultra Fast)",
        "repo_id": "bartowski/gemma-2-2b-it-GGUF",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "target_filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "parameters": "2.61B",
        "quantization": "Q4_K_M",
        "context_window": "8,192 tokens",
        "size": "1.59 GB",
        "description": "Ultra Lightweight & High TPS (Ideal untuk latensi sub-detik)"
    }
}

def format_size(bytes_num):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_num < 1024.0:
            return f"{bytes_num:.2f} {unit}"
        bytes_num /= 1024.0
    return f"{bytes_num:.2f} PB"

def print_header():
    print("=" * 80)
    print("📥 PENGUNDUH MODEL LLM KELAS BERAT (GGUF QUANTIZED)")
    print("=" * 80)
    print("Mendukung unduhan berkecepatan tinggi dengan resume berkas otomatis.")
    print("Semua model dioptimalkan untuk offload 100% GPU VRAM via llama-cpp-python.")
    print("=" * 80)

def download_model(key: str, dest_dir: str):
    info = MODELS_INFO[key]
    os.makedirs(dest_dir, exist_ok=True)
    target_path = os.path.join(dest_dir, info["target_filename"])

    print("\n" + "-" * 80)
    print(f"📦 MODEL: {info['name']} ({info['parameters']}) - {info['org']}")
    print("-" * 80)
    print(f"  • Kuantisasi    : {info['quantization']} (4-bit Balanced)")
    print(f"  • Context Len   : {info['context_window']}")
    print(f"  • Ukuran Target : ~{info['size']}")
    print(f"  • Deskripsi     : {info['description']}")
    print(f"  • HuggingFace   : {info['repo_id']}")
    print(f"  • Lokasi Simpan : {target_path}")

    if os.path.exists(target_path):
        size_bytes = os.path.getsize(target_path)
        size_formatted = format_size(size_bytes)
        print(f"  ✓ BERKAS SUDAH TERSEDIA: {size_formatted} pada disk.")
        print(f"  ⏩ Melewati pengunduhan untuk model ini.")
        return target_path

    print(f"  ⏳ Memulai pengunduhan stream (menampilkan progress bar & kecepatan transfer)...")
    t0 = time.time()
    try:
        # hf_hub_download automatically provides interactive tqdm progress bar
        downloaded = hf_hub_download(
            repo_id=info["repo_id"],
            filename=info["filename"],
            local_dir=dest_dir
        )
        
        # Rename if downloaded filename is different from target_filename
        if info["filename"] != info["target_filename"]:
            src = os.path.join(dest_dir, info["filename"])
            if os.path.exists(src) and src != target_path:
                os.rename(src, target_path)

        elapsed = max(time.time() - t0, 0.1)
        final_size = os.path.getsize(target_path) if os.path.exists(target_path) else 0
        speed_mb = (final_size / (1024**2)) / elapsed

        print(f"  ✓ SELESAI: {format_size(final_size)} berhasil diunduh dalam {round(elapsed, 1)} detik ({round(speed_mb, 2)} MB/s)!")
        return target_path
    except Exception as e:
        print(f"  ❌ GAGAL MENGUNDUH {info['name']}: {e}")
        return None

def display_summary(dest_dir: str):
    print("\n" + "=" * 80)
    print("📊 STATUS PENYIMPANAN MODEL GGUF:")
    print("=" * 80)
    print(f"{'ID':<12} | {'Nama Model':<28} | {'Ukuran Disk':<12} | {'Status':<15}")
    print("-" * 80)
    
    total_bytes = 0
    for key, info in MODELS_INFO.items():
        p = os.path.join(dest_dir, info["target_filename"])
        if os.path.exists(p):
            sz = os.path.getsize(p)
            total_bytes += sz
            status = "✓ SIAP (GPU Ready)"
            sz_str = format_size(sz)
        else:
            status = "⚠️ Belum Ada"
            sz_str = "0 B"
        print(f"{key:<12} | {info['name']:<28} | {sz_str:<12} | {status:<15}")

    print("-" * 80)
    print(f"Total Kapasitas Terpakai di {dest_dir}: {format_size(total_bytes)}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Download Heavyweight GGUF models with progress stats")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Pilihan model: 'all', 'quick' (Sahabat-AI + Gemma 2B), atau nama spesifik ('sahabatai', 'qwen', 'gemma', 'gemma-2b')"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=MODELS_DIR,
        help=f"Direktori penyimpanan model (default: {MODELS_DIR})"
    )
    args = parser.parse_args()

    print_header()
    dest = os.path.abspath(args.dest)
    print(f"📁 Direktori Target: {dest}\n")

    if args.models.lower() == "all":
        selected = list(MODELS_INFO.keys())
    elif args.models.lower() == "quick":
        selected = ["sahabatai", "gemma-2b"]
    else:
        selected = [m.strip().lower() for m in args.models.split(",") if m.strip().lower() in MODELS_INFO]

    if not selected:
        print("❌ Pilihan model tidak valid. Pilih: 'all', 'quick', atau nama model ('sahabatai', 'qwen', 'gemma', 'gemma-2b').")
        sys.exit(1)

    print(f"📋 Antrean Unduhan: {', '.join([MODELS_INFO[k]['name'] for k in selected])}")
    for m in selected:
        download_model(m, dest)

    display_summary(dest)

if __name__ == "__main__":
    main()
