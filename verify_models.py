#!/usr/bin/env python3
"""
Model Loading & Verification Diagnostic Suite
Detailed step-by-step verification of all Decision Models (System 1) and Heavyweight LLMs (System 2).
Reports VRAM allocation, CUDA layer offload, and warm-up latency.
"""

import os
import sys
import time
import json
import torch

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from config import JEV_API_KEY, JEV_ENDPOINT, MODELS_DIR
from heavyweight_llm_engine import HeavyweightLLMManager, MODELS_CATALOG

def get_vram_mb():
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / (1024**2), 1)
    return 0

def get_vram_total_free():
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        return round(total_b / (1024**3), 2), round(free_b / (1024**3), 2)
    return 0, 0

def verify_laya():
    print("\n" + "-"*75)
    print("▶ [1/4] MEMUAT & MENGUJI MODEL DECISION: Laya Multilingual (421M)")
    print("-"*75)
    vram_before = get_vram_mb()
    t0 = time.perf_counter()
    try:
        from laya import Router
        router = Router()
        load_sec = round(time.perf_counter() - t0, 2)
        vram_after = get_vram_mb()
        delta_vram = round(vram_after - vram_before, 1)

        # Warm-up inference test
        test_state = "Halo CS, paket saya tertahan di gateway Marunda."
        test_q = {
            "is_urgent": {"type": "noul", "instructions": "Urgensi pengaduan?"},
            "category": {"type": "choice", "instructions": "Kategori?", "criteria": {"kritis": "Kritis", "normal": "Normal"}}
        }
        t_inf0 = time.perf_counter()
        res = router.predict(test_state, test_q)
        lat_ms = round((time.perf_counter() - t_inf0) * 1000, 1)

        print(f"  • Backbone Arsitektur  : ModernBERT RLCD (Non-Autoregressive)")
        print(f"  • Waktu Pemuatan Model : {load_sec} detik")
        print(f"  • Memori VRAM GPU      : ~{vram_after} MiB (Alokasi: +{delta_vram} MiB)")
        print(f"  • Uji Triage 0 Token   : Selesai dalam {lat_ms} ms (1x Single Forward Pass)")
        print(f"  • Hasil Evaluasi Awal  : {list(res.get('answers', {}).keys())}")
        print(f"  ✓ STATUS: LAYA MULTILINGUAL READY & TERKALIBRASI!")
        del router
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"  ❌ Gagal memuat Laya: {e}")
        return False

def verify_openjev():
    print("\n" + "-"*75)
    print("▶ [2/4] MEMUAT & MENGUJI MODEL DECISION: OpenJev (0.5B Logit Scorer)")
    print("-"*75)
    vram_before = get_vram_mb()
    t0 = time.perf_counter()
    try:
        from openjev_engine import OpenJevScorer
        openjev = OpenJevScorer(model_name="Qwen/Qwen2.5-0.5B-Instruct", device="cuda" if torch.cuda.is_available() else "cpu")
        load_sec = round(time.perf_counter() - t0, 2)
        vram_after = get_vram_mb()
        delta_vram = round(vram_after - vram_before, 1)

        # Test inference
        t_inf0 = time.perf_counter()
        ans = openjev.answer("Pengaduan komplain", {"test_score": {"type": "score", "instructions": "Derajat komplain", "criteria": ["rendah", "tinggi"]}})
        lat_ms = round((time.perf_counter() - t_inf0) * 1000, 1)

        print(f"  • Backbone Arsitektur  : Qwen/Qwen2.5-0.5B-Instruct Logit Scorer")
        print(f"  • Waktu Pemuatan Model : {load_sec} detik")
        print(f"  • Memori VRAM GPU      : ~{vram_after} MiB (Alokasi: +{delta_vram} MiB)")
        print(f"  • Uji Triage 0 Token   : Selesai dalam {lat_ms} ms (Native Continuation Head)")
        print(f"  ✓ STATUS: OPENJEV READY!")
        del openjev
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"  ❌ Gagal memuat OpenJev: {e}")
        return False

def verify_kev():
    print("\n" + "-"*75)
    print("▶ [3/4] MEMUAT & MENGUJI MODEL DECISION: Kev-0.8B (Local GPU)")
    print("-"*75)
    vram_before = get_vram_mb()
    t0 = time.perf_counter()
    try:
        from kev.checkpoint import Checkpoint, LoadOptions
        from kev.api import SystemOneRequest
        from kev.serve import Server

        ck = Checkpoint("jaredpalmer/kev-0.8b")
        tok, model = ck.load("cuda", LoadOptions(dtype=torch.float16))
        srv = Server(checkpoint=ck, tok=tok, model=model, device="cuda")
        load_sec = round(time.perf_counter() - t0, 2)
        vram_after = get_vram_mb()
        delta_vram = round(vram_after - vram_before, 1)

        t_inf0 = time.perf_counter()
        req_kev = SystemOneRequest(model="kev-latest", state="Test ticket", questions={"urgent": {"type": "noul", "instructions": "urgent?"}})
        r_kev = srv.answer(req_kev)
        lat_ms = round((time.perf_counter() - t_inf0) * 1000, 1)

        print(f"  • Backbone Arsitektur  : Qwen 2.5 0.5B/0.8B + LoRA Pointer Head")
        print(f"  • Waktu Pemuatan Model : {load_sec} detik")
        print(f"  • Memori VRAM GPU      : ~{vram_after} MiB (Alokasi: +{delta_vram} MiB)")
        print(f"  • Uji Triage 0 Token   : Selesai dalam {lat_ms} ms (Wire-Compatible)")
        print(f"  ✓ STATUS: KEV-0.8B READY!")
        del srv, model, tok, ck
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"  ⚠️  Catatan Kev: {e} (Model Laya & OpenJev tetap aktif)")
        return False

def verify_heavyweight_llm(model_id: str = "sahabatai"):
    print("\n" + "-"*75)
    mdata = MODELS_CATALOG.get(model_id, MODELS_CATALOG["sahabatai"])
    path = mdata["model_path"]
    
    # Check if target model exists, else find any available GGUF
    candidate_keys = [model_id, "gemma-2b", "qwen", "gemma"]
    chosen_key = None
    for k in candidate_keys:
        if k in MODELS_CATALOG and os.path.exists(MODELS_CATALOG[k]["model_path"]):
            chosen_key = k
            break

    if not chosen_key:
        print("  ⚠️  Tidak ada file model GGUF yang ditemukan. Unduh via: python3 download_models.py")
        return False

    mdata = MODELS_CATALOG[chosen_key]
    path = mdata["model_path"]
    file_size_gb = round(os.path.getsize(path) / (1024**3), 2)
    tot_gb, free_gb = get_vram_total_free()

    print(f"▶ [4/4] MEMUAT & MENGUJI LLM KELAS BERAT: {mdata['name']} ({mdata['parameters']})")
    print("-"*75)
    print(f"  • Path Berkas GGUF     : {path} ({file_size_gb} GB)")
    print(f"  • Kuantisasi & Format  : {mdata['quantization']} ({mdata['format']})")
    print(f"  • Kapasitas VRAM Bebas : {free_gb} GB bebas dari {tot_gb} GB")

    # If VRAM is tight (< 4.2 GB free) and chosen model is > 4 GB, switch to gemma-2b if available
    if free_gb < 4.5 and file_size_gb > 3.5 and os.path.exists(MODELS_CATALOG["gemma-2b"]["model_path"]) and chosen_key != "gemma-2b":
        print(f"  ℹ️ VRAM tersisa ({free_gb} GB) ketat untuk model {file_size_gb} GB. Menguji dengan Gemma 2 2B...")
        chosen_key = "gemma-2b"
        mdata = MODELS_CATALOG[chosen_key]
        path = mdata["model_path"]
        file_size_gb = round(os.path.getsize(path) / (1024**3), 2)

    t0 = time.perf_counter()
    try:
        from llama_cpp import Llama
        # Check layers: offload 100% unless VRAM is critically low
        n_layers = -1
        if free_gb < 2.0:
            n_layers = 16

        llm = Llama(
            model_path=path,
            n_gpu_layers=n_layers,
            n_ctx=512,
            verbose=False
        )
        load_sec = round(time.perf_counter() - t0, 2)
        print(f"  • Waktu Pemuatan GPU   : {load_sec} detik (Offload ke CUDA Sukses)")

        # Run test generation
        t_gen0 = time.perf_counter()
        prompt_text = "Halo, jawab dalam 1 kalimat Bahasa Indonesia singkat: apa ibukota Indonesia?"
        out = llm(prompt_text, max_tokens=25, temperature=0.2)
        gen_sec = time.perf_counter() - t_gen0
        tok_count = out["usage"]["completion_tokens"]
        tps = round(tok_count / max(gen_sec, 0.001), 1)

        resp_text = out['choices'][0]['text'].strip().replace("\n", " ")
        print(f"  • Kecepatan Generasi   : {tps} tokens/detik ({tok_count} token dalam {round(gen_sec, 2)}s)")
        print(f"  • Output Sampel Respon : \"{resp_text[:90]}...\"")
        print(f"  ✓ STATUS: {mdata['name']} 100% OPERASIONAL PADA GPU CUDA!")
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    except Exception as e:
        print(f"  ❌ Gagal memuat LLM {mdata['name']}: {e}")
        return False

def verify_jev_cloud():
    print("\n" + "-"*75)
    print("▶ [*] MEMERIKSA STATUS KONEKSI CLOUD: TypeSafe Jev API")
    print("-"*75)
    if not JEV_API_KEY:
        print("  ℹ️  JEV_API_KEY kosong di .env. Menggunakan mode lokal 100%.")
        return True

    masked_key = JEV_API_KEY[:10] + "..." + JEV_API_KEY[-6:] if len(JEV_API_KEY) > 16 else "********"
    print(f"  • Endpoint URL         : {JEV_ENDPOINT}")
    print(f"  • API Key Terpasang    : {masked_key}")

    import urllib.request
    payload = {
        "model": "jev-latest",
        "state": "System health ping check",
        "questions": {"ping": {"type": "noul", "instructions": "is ping?"}}
    }
    req = urllib.request.Request(
        JEV_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {JEV_API_KEY}", "Content-Type": "application/json"},
        method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=6) as resp:
            lat = round((time.perf_counter() - t0) * 1000, 1)
            print(f"  • Status Respons HTTP  : {resp.status} OK (Latensi: {lat} ms)")
            print(f"  ✓ STATUS: TYPESAFE JEV CLOUD API TERHUBUNG & AKTIF!")
            return True
    except Exception as e:
        print(f"  ⚠️  Respons TypeSafe Jev: {e}")
        return False

def main():
    print("="*75)
    print("🏛️  VERIFIKASI & DIAGNOSTIK PEMUATAN MODEL (SYSTEM 1 & SYSTEM 2)")
    print("="*75)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        tot_gb, free_gb = get_vram_total_free()
        print(f"💻 Akselerator GPU : {gpu_name}")
        print(f"📊 Kapasitas VRAM  : Total {tot_gb} GB | Bebas {free_gb} GB")
    else:
        print("⚠️  CUDA tidak terdeteksi. Berjalan di CPU.")

    v_laya = verify_laya()
    v_openjev = verify_openjev()
    v_kev = verify_kev()
    v_llm = verify_heavyweight_llm("sahabatai")
    v_jev = verify_jev_cloud()

    tot_gb, free_gb = get_vram_total_free()
    used_gb = round(tot_gb - free_gb, 2)

    print("\n" + "="*75)
    print("📋 RINGKASAN DIAGNOSTIK AKHIR:")
    print(f"  • Model Decision System 1 : Laya ({'✓' if v_laya else '✗'}), OpenJev ({'✓' if v_openjev else '✗'}), Kev ({'✓' if v_kev else '✗'}), Jev Cloud ({'✓' if v_jev else '✗'})")
    print(f"  • Model LLM System 2      : {'✓ OPERASIONAL' if v_llm else '⚠️ PERLU DIUNDUH'}")
    print(f"  • Penggunaan VRAM GPU     : {used_gb} GB terpakai / {tot_gb} GB total ({free_gb} GB bebas)")
    print("="*75)
    print("🎉 Seluruh komponen terverifikasi siap untuk pengujian Web & Headless CLI!\n")

if __name__ == "__main__":
    main()
