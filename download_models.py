#!/usr/bin/env python3
"""
Model Downloader for Indonesian LLM Arena & Decision Models
Downloads GGUF quantized models for local GPU execution on CUDA.
"""

import os
import sys
import argparse
import time
from huggingface_hub import hf_hub_download
from config import MODELS_DIR

MODELS_INFO = {
    "sahabatai": {
        "name": "Sahabat-AI 8B Instruct (GoTo & Indosat)",
        "repo_id": "gmonsoon/llama3-8b-cpt-sahabatai-v1-instruct-GGUF",
        "filename": "llama3-8b-cpt-sahabatai-v1-instruct.Q4_K_M.gguf",
        "target_filename": "sahabatai-8b-q4.gguf",
        "size": "4.6 GB"
    },
    "qwen": {
        "name": "Qwen 2.5 7B Instruct (Alibaba Cloud)",
        "repo_id": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "target_filename": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size": "4.4 GB"
    },
    "gemma": {
        "name": "Gemma 2 9B Instruct (Google DeepMind)",
        "repo_id": "bartowski/gemma-2-9b-it-GGUF",
        "filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "target_filename": "gemma-2-9b-it-Q4_K_M.gguf",
        "size": "5.4 GB"
    },
    "gemma-2b": {
        "name": "Gemma 2 2B Instruct (Google DeepMind Speed)",
        "repo_id": "bartowski/gemma-2-2b-it-GGUF",
        "filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "target_filename": "gemma-2-2b-it-Q4_K_M.gguf",
        "size": "1.6 GB"
    }
}

def download_model(key: str, dest_dir: str):
    info = MODELS_INFO[key]
    os.makedirs(dest_dir, exist_ok=True)
    target_path = os.path.join(dest_dir, info["target_filename"])
    
    if os.path.exists(target_path):
        size_gb = round(os.path.getsize(target_path) / (1024**3), 2)
        print(f"✓ Model '{info['name']}' sudah ada di {target_path} ({size_gb} GB). Melewati unduhan.")
        return target_path

    print(f"\n============================================================")
    print(f"📥 Mengunduh: {info['name']} (~{info['size']})")
    print(f"   Repo: {info['repo_id']}")
    print(f"   Tujuan: {target_path}")
    print(f"============================================================")
    
    t0 = time.time()
    try:
        downloaded = hf_hub_download(
            repo_id=info["repo_id"],
            filename=info["filename"],
            local_dir=dest_dir
        )
        # Rename if filename differs from target_filename
        if info["filename"] != info["target_filename"]:
            src = os.path.join(dest_dir, info["filename"])
            if os.path.exists(src) and src != target_path:
                os.rename(src, target_path)
                
        elapsed = round(time.time() - t0, 1)
        print(f"✓ Berhasil mengunduh {info['name']} dalam {elapsed} detik!")
        return target_path
    except Exception as e:
        print(f"❌ Gagal mengunduh {info['name']}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Download Heavyweight GGUF models for benchmark")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Model yang ingin diunduh: 'all', 'quick' (hanya gemma-2b & sahabatai), atau spesifik seperti 'sahabatai,qwen'"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=MODELS_DIR,
        help=f"Direktori penyimpanan model (default: {MODELS_DIR})"
    )
    args = parser.parse_args()

    dest = os.path.abspath(args.dest)
    print(f"🚀 Memeriksa penyimpanan model di: {dest}")

    if args.models.lower() == "all":
        selected = list(MODELS_INFO.keys())
    elif args.models.lower() == "quick":
        selected = ["gemma-2b", "sahabatai"]
    else:
        selected = [m.strip().lower() for m in args.models.split(",") if m.strip().lower() in MODELS_INFO]

    if not selected:
        print("Pilihan model tidak valid. Pilih dari: all, quick, sahabatai, qwen, gemma, gemma-2b")
        sys.exit(1)

    print(f"Model terpilih: {', '.join(selected)}")
    for m in selected:
        download_model(m, dest)

    print("\n✓ Semua pemeriksaan & unduhan model selesai!")

if __name__ == "__main__":
    main()
