# 🏛️ PROJECT STATE & CONTEXT PERSISTENCE
# DecisionModelBench: Model Decision (System 1) vs Foundation LLM Indonesia (System 2)

> **Catatan untuk Asisten AI (Context Memory):**  
> Dokumen ini adalah *Single Source of Truth* (SSOT) status, arsitektur, dan konteks lengkap proyek. Setiap kali sesi baru dimulai, baca dokumen ini untuk memulihkan seluruh konteks kerja tanpa kehilangan detail teknis apa pun.

---

## 1. Identitas Proyek & Repository
- **Nama Proyek:** DecisionModelBench
- **Workspace Lokal (Persistent Google Drive):** `/content/drive/MyDrive/AIPROJECT/DecisionModel/`
- **GitHub Repository:** [https://github.com/tuwiliyt/decisionmodelbench](https://github.com/tuwiliyt/decisionmodelbench)
- **Branch:** `main`
- **Tujuan Proyek:** Menguji dan membandingkan secara komprehensif arsitektur **Model Decision Non-Autoregresif (System 1)** melawan **Foundation LLM Autoregresif (System 2)**, khususnya membuktikan keunggulan **Dengan Jev (Two-Tier)** vs **Tanpa Jev (Standalone LLM)** dalam efisiensi latensi, penghematan token, mitigasi halusinasi, dan cost index pada pasar Indonesia.

---

## 2. Lingkungan Hardware & Konfigurasi Server
- **Hardware GPU:** NVIDIA Tesla T4 (15.6 GB VRAM, Compute 7.5, Pasif Server Fan)
- **Driver & CUDA:** Driver 580.82 | CUDA 13.0
- **Python Environment:** Python 3.10+ / 3.13 (`torch.cuda.is_available() == True`)
- **Backend Server:** FastAPI (`app_server.py`) berjalan di port `7860`.
- **Public Tunnel:** Cloudflare Tunnel (`cloudflared tunnel --url http://localhost:7860`).
- **Alokasi VRAM GPU Saat Aktif:**
  - Base System 1 (Laya + OpenJev + Kev): ~4.37 GB VRAM
  - VRAM Bebas untuk LLM Kelas Berat: ~10.5 GB VRAM
  - Hot-swap mechanism: `HeavyweightLLMManager` melakukan unallocation CUDA (`del self.llm; gc.collect(); torch.cuda.empty_cache()`) dengan aman sebelum memuat model 7B/8B/9B lain, menjamin **Zero CUDA OOM**.

---

## 3. Katalog Model Terpasang & Siap Uji

### A. Model Decision (System 1 - Triage Instan 0 Token):
1. **Laya Multilingual (421M):**
   - Arsitektur: ModernBERT RLCD.
   - Latensi: **~50 - 65 ms** di GPU Tesla T4 (Tercepat).
   - Output: 0 tokens, Softmax/Sigmoid matematis.
2. **OpenJev (0.5B):**
   - Arsitektur: Qwen 2.5 Single-Pass Continuation Logit Scorer (`openjev_engine.py`).
   - Latensi: **~180 - 240 ms**.
   - Output: 0 tokens, Normalized Logits.
3. **TypeSafe Jev (Cloud SaaS API):**
   - Endpoint: `https://api.typesafe.ai/v1/systemone`
   - Latensi: **~140 - 180 ms** total network round-trip.
   - Konfigurasi: Dimuat via `.env` (`JEV_API_KEY`).
4. **Kev-0.8B (Local GPU):**
   - Arsitektur: Qwen 2.5 + LoRA Pointer Head (Jared Palmer).
   - Latensi: **~1.1 - 1.4 s**.
5. **CLM-8B (Hardware Guarded):**
   - Arsitektur: Stanford/NVIDIA Contrastive Language Model.
   - Status: Memeriksa VRAM dinamis; menampilkan panduan kuantisasi jika VRAM < 16 GB.

### B. Foundation LLM Kelas Berat (System 2 - Penalaran & Narasi):
Model disimpan di `/root/models/` (atau `./models/` via `resolve_model_path` di `heavyweight_llm_engine.py`):
1. `sahabatai-8b-q4.gguf` (4.6 GB, 8.03B) – GoTo & Indosat Sovereign LLM (~28-32 t/s).
2. `Qwen2.5-7B-Instruct-Q4_K_M.gguf` (4.4 GB, 7.61B) – Alibaba Cloud Multilingual SOTA (~30-35 t/s).
3. `gemma-2-9b-it-Q4_K_M.gguf` (5.4 GB, 9.24B) – Google DeepMind Reasoning (~24-28 t/s).
4. `gemma-2-2b-it-Q4_K_M.gguf` (1.6 GB, 2.61B) – Google DeepMind Ultra-Speed (~50-65 t/s).

---

## 4. Intisari Arsitektur: "Dengan Jev" vs "Tanpa Jev"

| Dimensi | Tanpa Jev (LLM Murni) | Dengan Jev / Decision Model (Two-Tier) |
| :--- | :--- | :--- |
| **Pola Eksekusi** | Single-Tier Autoregresif (100% Kueri ke LLM 8B) | Dual-Tier (Tier 1: Triage 0 Token ➔ Tier 2: LLM On-Demand) |
| **Latensi Klasifikasi** | 3,500 – 6,000 ms per kueri | **50 – 160 ms (50x - 80x lebih cepat)** |
| **Token Klasifikasi** | 150 – 250 tokens per kueri | **0 tokens (Zero Token Waste)** |
| **Kueri Rutin (FAQ)** | Menghabiskan ~200 token & 5 detik GPU | **Fast-Path: Selesai <100ms, LLM dihemat 100%** |
| **Kueri Kritis (Komplain)** | Berisiko halusinasi & drift format | **Triage deterministik terkalibrasi ➔ Eskalasi terarah ke LLM** |
| **Penghematan Biaya** | 0% (Beban komputasi penuh) | **Hemat 80% – 100% kuota token produksi** |

---

## 5. Perkakas & Skrip Utama dalam Repository

1. **`setup.sh` (Skrip Otomatisasi Instalasi Interaktif):**
   - Mendeteksi GPU NVIDIA & CUDA.
   - **Meminta input API Key Jev secara interaktif** dan menyimpannya ke `.env`.
   - Memasang dependensi Python & llama-cpp-python CUDA.
   - Menawarkan opsi unduhan model GGUF (Lengkap / Quick Test / Lewati).
2. **`benchmark_headless.py` (CLI Headless Benchmark):**
   - `python3 benchmark_headless.py --matrix` ➔ Tabel matriks eksekutif lengkap.
   - `python3 benchmark_headless.py --scenario marunda --decision laya --llm sahabatai` ➔ Uji komparasi spesifik.
   - `python3 benchmark_headless.py --all-presets` ➔ Evaluasi 4 skenario batch.
   - `python3 benchmark_headless.py --interactive` ➔ Menu CLI interaktif.
3. **`download_models.py` (Pengunduh Model Cerdas):**
   - Mendukung `--models all`, `--models quick`, dan `--dest <path>`.
   - Otomatis melewati file yang sudah ada di disk.
4. **`app_server.py` (FastAPI Server):**
   - Port 7860.
   - Endpoint: `/api/heavyweight/compare_architectures`, `/api/heavyweight/generate`, `/api/heavyweight/decision`, `/api/heavyweight/two_tier`, `/api/gpu_nvtop`.
5. **`generate_heavyweight_dashboard.py`:**
   - Menghasilkan antarmuka web interaktif `heavyweight_llm_dashboard.html` dengan 5 mode lengkap dan grafik telemetri rolling nvtop 60 detik.

---

## 6. Cara Cepat Melanjutkan Proyek di Sesi Baru
Jika pengguna membuka percakapan baru di workspace ini:
1. Jalankan `git status` dan periksa kelengkapan file.
2. Periksa apakah server backend aktif: `curl -s http://localhost:7860/api/health`
   - Jika belum aktif: `python3 -m uvicorn app_server:app --host 0.0.0.0 --port 7860 &`
3. Pengguna dapat langsung menjalankan pengujian via CLI:
   `python3 benchmark_headless.py --matrix`
   atau mengakses antarmuka web di `/heavyweight`.
