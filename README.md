# 🏛️ DecisionModelBench: Benchmark Model Decision vs Sovereign LLM Indonesia

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CUDA 12/13](https://img.shields.io/badge/CUDA-NVIDIA%20GPU-green.svg)](https://developer.nvidia.com/cuda-zone)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Platform evaluasi komprehensif untuk membandingkan kinerja arsitektur **Model Decision (System 1)** dan **Large Language Model Generatif Kelas Berat (System 2)** pada infrastruktur GPU lokal (NVIDIA Tesla T4/A10G/L4/RTX).

Repository ini membuktikan secara empiris perbedaan arsitektur antara menjalankan **LLM Murni (Tanpa Jev)** melawan **Two-Tier Pipeline (Dengan Jev / Laya / OpenJev / Kev)** dari segi:
- ⚡ **Latensi Triage:** Sub-100 ms vs 3,000–6,000 ms (**hingga 50x–80x lebih cepat**).
- 💸 **Konsumsi Token:** **0 output tokens** pada tahap klasifikasi (Zero Token Waste).
- 💰 **Efisiensi Biaya:** Penghematan **80% hingga 100% token LLM** melalui *smart gating* & *fast-path routing*.
- 🎯 **Konsistensi Keputusan:** 100% deterministik matematis (Softmax/Sigmoid terkalibrasi) tanpa risiko halusinasi format.

---

## 📑 Daftar Isi
1. [Arsitektur Spektrum Model](#-arsitektur-spektrum-model)
2. [Matriks Komparasi Kinerja](#-matriks-komparasi-kinerja)
3. [Panduan Instalasi Lengkap dari Server Kosong](#-panduan-instalasi-lengkap-dari-server-kosong)
4. [Tutorial Uji Perbandingan Headless (CLI Tanpa Tampilan Web)](#-tutorial-uji-perbandingan-headless-cli-tanpa-tampilan-web)
5. [Tutorial Web Dashboard & GPU Monitor (nvtop Style)](#-tutorial-web-dashboard--gpu-monitor-nvtop-style)
6. [Katalog Model & Spesifikasi Hardware](#-katalog-model--spesifikasi-hardware)
7. [Struktur File Repository](#-struktur-file-repository)

---

## 🧠 Arsitektur Spektrum Model

```
                                  [ Masukan Kueri Pelanggan ]
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       ▼                                               ▼
         [ BRANCH A: TANPA JEV (LLM Murni) ]          [ BRANCH B: DENGAN JEV (Two-Tier) ]
                       │                                               │
               Direct 100% Load                                Tier 1: Decision Model
                       │                                      (Laya / OpenJev / Jev / Kev)
             Single-Tier Generatif                                     │
           (Next-Token Autoregression)                      Single Forward Pass (0 Tokens)
                       │                                      Latensi Instan: ~50-160 ms
          Latensi: 3,500 - 6,000 ms                                    │
           Token: 150 - 250 tokens                         [ Evaluasi Smart Gating ]
                       │                                    Kritis? / Butuh Eskalasi?
                       │                                        ├── TIDAK ──► [ Fast-Path Auto-Reply ]
                       │                                        │             • Latensi: <100 ms
                       │                                        │             • 100% Kuota LLM Dihemat
                       │                                        │
                       ▼                                        └── YA ─────► Tier 2: LLM Kelas Berat
         [ Respon Kalimat LLM 8B ]                                            (Sahabat-AI / Qwen / Gemma)
         (Beban Maksimal GPU & Biaya)                                         • Penanganan Empatik Terarah
```

---

## 📊 Matriks Komparasi Kinerja

Dijalankan dan diverifikasi pada GPU **NVIDIA Tesla T4 (15.6 GB VRAM)**:

| Dimensi Kinerja | Tanpa Decision Model (LLM Murni) | Dengan TypeSafe Jev (Cloud) | Dengan Laya Multilingual (421M GPU) | Dengan OpenJev (0.5B GPU) | Dengan Kev-0.8B (Local) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Latensi Keputusan / Triage** | ~3,500 – 6,000 ms | ~140 – 180 ms | **~50 – 75 ms (Tercepat)** | ~180 – 240 ms | ~1,100 – 1,400 ms |
| **Token Output Saat Triage** | 150 – 250 tokens | **0 tokens** | **0 tokens** | **0 tokens** | **0 tokens** |
| **Metode Inferensi** | Next-Token Autoregressive | Single Forward Pass | 1x Forward Pass CUDA | Logit Contrastive Head | Multi-Model Ensemble |
| **Konsistensi Klasifikasi** | Stokastik (Sampling) | 100% Deterministik | 100% Terkalibrasi | 100% Normalisasi Logit | 100% Ensemble Scored |
| **Efisiensi Biaya Token** | 0% Hemat | **Hemat 80–90%** | **Hemat 80–90%** | **Hemat 80–90%** | **Hemat 80–90%** |
| **Overhead VRAM GPU** | 0 MB (LLM Saja) | **0 MB (Offloaded Cloud)** | ~950 MB | ~1,100 MB | ~1,600 MB |
| **Throughput Maksimum** | ~0.2 – 0.3 req/s | ~50+ req/s (Cloud Scaled) | **~15 – 20 req/s** | ~5 – 8 req/s | ~1 req/s |

---

## 🚀 Panduan Instalasi Lengkap dari Server Kosong

Panduan ini ditujukan untuk server Linux baru (Ubuntu 20.04/22.04/24.04, Debian, atau Google Colab GPU Environment).

### 1. Prasyarat Sistem
- **Sistem Operasi:** Linux x86_64
- **GPU:** NVIDIA GPU (minimal 8 GB VRAM untuk mode cepat, 16 GB VRAM untuk suite lengkap)
- **NVIDIA Driver & CUDA Toolkit:** CUDA 12.0+ atau 13.0+
- **Python:** Versi 3.10, 3.11, 3.12, atau 3.13

Periksa kesiapan GPU Anda dengan:
```bash
nvidia-smi
```

### 2. Kloning Repository
```bash
git clone https://github.com/tuwiliyt/decisionmodelbench.git
cd decisionmodelbench
```

### 3. Jalankan Skrip Instalasi Otomatis (`setup.sh`)
Skrip ini akan memeriksa hardware, memasang seluruh dependensi, meminta API Key Jev secara interaktif, dan mengunduh model:

```bash
chmod +x setup.sh
./setup.sh
```

#### 🔑 Alur Input Interaktif & Validasi Online API Key TypeSafe Jev:
Saat menjalankan `./setup.sh`, terminal akan menampilkan prompt interaktif dan melakukan uji koneksi online secara *real-time*:
```text
==============================================================================
🔑 [3/6] KONFIGURASI & VALIDASI API KEY TYPESAFE JEV
==============================================================================
Sistem mengintegrasikan cloud SaaS TypeSafe Jev API (https://typesafe.ai).
Pemberian API Key bersifat opsional:
  • Jika Anda memiliki API Key: masukkan di bawah untuk mengaktifkan TypeSafe Jev Cloud.
  • Jika tidak: tekan [Enter] untuk melewati dan menggunakan 100% model lokal (Laya, OpenJev, Kev).
------------------------------------------------------------------------------
Masukkan TypeSafe Jev API Key: apikey_xxxxxxxxxxxx

⏳ Menguji koneksi langsung ke endpoint cloud TypeSafe Jev...
  • Endpoint URL         : https://api.typesafe.ai/v1/systemone
  • Status Respons HTTP  : 200 OK
  • Latensi Koneksi      : 215.4 ms
  ✓ SUKSES: API KEY VALID & AKTIF TERHUBUNG KE TYPESAFE JEV!
```
- Jika API key valid, skrip otomatis menyimpannya ke `.env` dan mengaktifkan fitur SaaS.
- Jika API key tidak valid (HTTP 401/403), sistem memberikan opsi untuk memasukkan ulang atau langsung melanjutkan dalam mode lokal 100% (**Laya**, **OpenJev**, dan **Kev**).

#### 📥 Deteksi Hardware & Penawaran Model Kelas Berat (Auto-Offering):
Skrip instalasi memeriksa kapasitas VRAM GPU secara otomatis. Jika terdeteksi server Anda memiliki VRAM melimpah (>=20 GB: A10G/L4/RTX 3090/4090/A100 atau Multi-GPU), sistem secara otomatis menawarkan dan merekomendasikan **Paket Enterprise Flagship (+ Qwen 2.5 14B)** untuk pengujian kelas berat lanjutan:
1. **Unduh Paket Enterprise Flagship (+ Qwen 2.5 14B) ~25 GB:** Direkomendasikan untuk GPU VRAM >=20 GB (A10G/L4/RTX 3090/4090/A100).
2. **Unduh Lengkap (~16 GB):** Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B, Gemma 2 2B (Direkomendasikan untuk GPU 12–16 GB: Tesla T4).
3. **Unduh Cepat (~6.2 GB):** Sahabat-AI 8B + Gemma 2 2B (Direkomendasikan untuk GPU hemat VRAM / uji cepat).
4. **Sahabat-AI 8B Saja (~4.6 GB):** Model utama Bahasa Indonesia.
5. **Lewati:** Unduh kapan saja nanti via `python3 download_models.py --models auto`.

#### 🔬 Diagnostik Otomatis Pemuatan Model & Alokasi VRAM (`verify_models.py`):
Di akhir instalasi, skrip otomatis memuat dan menguji setiap model pada CUDA GPU untuk memastikan kesiapan:
- **Laya Multilingual (421M):** Waktu muat 0.04s, VRAM, uji triage 0 token pass.
- **OpenJev (0.5B):** Waktu muat 6.5s, VRAM ~950 MiB, uji normalisasi logit head 270 ms.
- **Kev-0.8B (Local):** Waktu muat 13s, VRAM ~1.4 GB, uji triage wire-compatible.
- **Foundation LLM (CUDA Engine):** 100% GPU layer offload, uji generasi tokens/detik.
- **TypeSafe Jev Cloud API:** Uji ping status HTTP 200 & latensi.

---

### Alternatif: Instalasi Manual (Langkah demi Langkah)

Jika Anda ingin memasang manual tanpa `setup.sh`:

```bash
# 1. Buat virtual environment (opsional namun disarankan)
python3 -m venv venv
source venv/bin/activate

# 2. Pasang dependensi python
pip install --upgrade pip
pip install -r requirements.txt

# 3. Pasang llama-cpp-python dengan akselerasi CUDA & live detail progress
python3 install_llama_cpp.py
# (atau manual: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir -v)

# 4. Pasang Kev dari GitHub
pip install git+https://github.com/jaredpalmer/kev.git

# 5. Salin dan konfigurasi .env
cp .env.example .env
# Edit .env dan masukkan JEV_API_KEY jika ada

# 6. Unduh model GGUF
python3 download_models.py --models quick

# 7. Kompilasi dashboard
python3 generate_dashboard.py
python3 generate_heavyweight_dashboard.py
```

---

## 🖥️ Tutorial Uji Perbandingan Headless (CLI Tanpa Tampilan Web)

Untuk pengujian otomatis, scripting, atau server tanpa GUI/desktop, gunakan perkakas CLI `benchmark_headless.py`.

### 1. Menampilkan Matriks Komparasi Arsitektur
Tampilkan tabel perbandingan seluruh model secara langsung di terminal:
```bash
python3 benchmark_headless.py --matrix
```

### 2. Menjalankan Komparasi Tunggal Berdasarkan Skenario
Jalankan komparasi langsung antara **Tanpa Jev** vs **Dengan Jev/Laya**:
```bash
# Skenario A: Komplain Kritis Marunda (Deteksi Ancaman Viral)
python3 benchmark_headless.py --scenario marunda --decision laya --llm sahabatai

# Skenario B: Tanggap Darurat Fraud Rekening (Deteksi Ancaman Lapor Polisi)
python3 benchmark_headless.py --scenario scam --decision openjev --llm qwen

# Skenario C: Restrukturisasi Nasabah PHK (Fintech OJK)
python3 benchmark_headless.py --scenario phk --decision jev --llm sahabatai

# Skenario D: Kueri Rutin Jam Buka (Membuktikan Fast-Path 100% Hemat LLM)
python3 benchmark_headless.py --scenario faq --decision laya --llm gemma-2b
```

### 3. Menjalankan Semua 4 Skenario Sekaligus (Batch Benchmark)
```bash
python3 benchmark_headless.py --all-presets --decision laya --llm sahabatai
```

### 4. Menu CLI Interaktif
Gunakan menu interaktif berbasis prompt teks di terminal:
```bash
python3 benchmark_headless.py --interactive
```

---

## 🌐 Tutorial Web Dashboard & GPU Monitor (nvtop Style)

### 1. Menjalankan Web Server
Jalankan server FastAPI uvicorn di port 7860:
```bash
python3 -m uvicorn app_server:app --host 0.0.0.0 --port 7860
```

### 2. Akses Antarmuka Dashboard
Buka browser Anda di:
- **Halaman Utama (5 Mode Pengujian & Komparasi Jev):**
  `http://localhost:7860/heavyweight`
- **Halaman Benchmark Kelas Ringan (System 1):**
  `http://localhost:7860/`

### 3. Menjalankan Akses Publik via Cloudflare Tunnel
Jika Anda menggunakan remote server atau Google Colab, buat tunnel publik gratis:
```bash
cloudflared tunnel --url http://localhost:7860
```
Terminal akan memberikan URL publik seperti: `https://xxxx-xxxx.trycloudflare.com/heavyweight`

### Fitur di Web Dashboard:
1. **Mode 1: Generative Chat (Multi-LLM):** Uji bebas empati dan dialek Sahabat-AI, Qwen, dan Gemma.
2. **Mode 2: Multi-LLM Arena:** Komparasi 3 model LLM kelas berat serentak pada prompt yang sama.
3. **Mode 3: Head-to-Head Parallel:** Adu langsung kecepatan inferensi System 1 vs System 2.
4. **Mode 4: Two-Tier Brain Pipeline:** Alur triage otomatis dan eskalasi terpadu.
5. **Mode 5: Komparasi Dengan Jev vs Tanpa Jev:** Tampilan berdampingan (*side-by-side*) yang membuktikan penghematan 80-100% token, mitigasi halusinasi, dan *speedup* 50x.
6. **nvtop Live Monitor:** Pantauan real-time utilisasi CUDA Core, alokasi VRAM, suhu GPU, daya watt, dan proses komputasi aktif.

---

## 📦 Katalog Model & Spesifikasi Hardware

### 1. Model Decision (System 1 - Triage Instan 0 Token)
| Model | Ukuran | Arsitektur | Tipe Eksekusi | Latensi Rata-rata |
| :--- | :--- | :--- | :--- | :--- |
| **Laya Multilingual** | 421M | ModernBERT RLCD | Local GPU CUDA | **~50 - 65 ms** |
| **OpenJev** | 0.5B | Qwen 2.5 Logit Scorer | Local GPU CUDA | **~180 - 240 ms** |
| **TypeSafe Jev** | Proprietary | Decision Architecture | Cloud SaaS API | **~140 - 180 ms** |
| **Kev-0.8B** | 0.8B | Qwen 2.5 LoRA Ensemble | Local GPU CUDA | **~1.1 - 1.4 s** |

### 2. Foundation LLM Kelas Berat (System 2 - Penalaran & Narasi)
| Model | Parameter | Organisasi | Kuantisasi | Kecepatan Inferensi | VRAM Minimum | Rekomendasi Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sahabat-AI 8B Instruct** | 8.03B | GoTo & Indosat | Q4_K_M GGUF | ~28 - 32 t/s | ~4.8 GB | Tesla T4 / RTX 3080/4070 (12–16 GB) |
| **Qwen 2.5 7B Instruct** | 7.61B | Alibaba Cloud | Q4_K_M GGUF | ~30 - 35 t/s | ~4.6 GB | Tesla T4 / RTX 3080/4070 (12–16 GB) |
| **Gemma 2 9B Instruct** | 9.24B | Google DeepMind | Q4_K_M GGUF | ~24 - 28 t/s | ~5.6 GB | Tesla T4 / RTX 3080/4070 (16 GB) |
| **Gemma 2 2B Instruct** | 2.61B | Google DeepMind | Q4_K_M GGUF | ~50 - 65 t/s | ~1.7 GB | Hemat VRAM (GPU 4–8 GB) |
| **Qwen 2.5 14B Instruct** | 14.7B | Alibaba Cloud | Q4_K_M GGUF | ~20 - 26 t/s | ~9.2 GB | **Workstation/Server (24–80 GB: A10G, L4, RTX 3090/4090, A100)** |

### 🚀 Skalabilitas Otomatis Lintas Generasi & Kapasitas GPU:
Sistem ini dilengkapi modul `gpu_manager.py` yang secara dinamis mengenali spesifikasi GPU server tanpa perlu konfigurasi manual:
- **Auto-Detect GPU Tier:** Mengenali otomatis apakah server menggunakan GPU standar (Tesla T4 16GB), Pro Workstation (RTX 3090/4090, A10G, L4 24GB), atau Enterprise Ultra-VRAM (A100 40G/80G, H100).
- **Auto Context Scaling:** Menyesuaikan panjang konteks secara dinamis dari `2,048` tokens pada GPU 16GB, hingga `4,096`–`8,192` tokens pada GPU 24GB, dan `16,384` tokens pada GPU 80GB.
- **Multi-GPU Auto Sharding (Tensor Split):** Jika terdeteksi 2 GPU atau lebih (misal: 2x T4, 2x A100, 4x RTX 4090), sistem otomatis membagi layer model secara proporsional (*proportional tensor-split*) di seluruh GPU yang tersedia.
- **Flash Attention 2.0:** Mengaktifkan optimasi Flash Attention otomatis pada GPU dengan Compute Capability ≥ 7.5 (Turing, Ampere, Ada Lovelace, Hopper), memangkas konsumsi VRAM KV-cache hingga 50% dan meningkatkan kecepatan inferensi.
- **Unlocking CLM-8B:** Pada GPU dengan sisa VRAM bebas ≥ 16 GB, model dual-encoder CLM-8B otomatis terbuka dan dapat dievaluasi secara penuh.

---

## 📁 Struktur File Repository

```text
decisionmodelbench/
├── app_server.py                     # Server FastAPI (Endpoint API, Hot-swap Engine, REST)
├── gpu_manager.py                    # Detektor hardware dinamis, Multi-GPU scaler & auto-tuner
├── benchmark_headless.py             # Perkakas CLI benchmark headless (Rich/Tabulate UI)
├── download_models.py                # Skrip pengunduh otomatis model GGUF dari Hugging Face
├── setup.sh                          # Skrip otomatisasi instalasi & konfigurasi interaktif
├── config.py                         # Modul pembaca konfigurasi & .env
├── .env.example                      # Template file variabel lingkungan
├── requirements.txt                  # Daftar dependensi pustaka Python
├── install_llama_cpp.py              # Installer live detail progress kompilasi llama-cpp-python
├── heavyweight_llm_engine.py         # Engine manajer multi-LLM (CUDA llama-cpp-python)
├── openjev_engine.py                 # Engine native OpenJev logit continuation scorer
├── sahabatai_engine.py               # Engine wrapper Sahabat-AI
├── generate_heavyweight_dashboard.py # Generator antarmuka web dashboard kelas berat (HTML)
├── generate_dashboard.py             # Generator antarmuka web dashboard kelas ringan (HTML)
├── test_jev.py                       # Skrip uji konektivitas TypeSafe Jev API
└── README.md                         # Dokumentasi & panduan teknis lengkap
```

---

## 📄 Lisensi
Proyek ini didistribusikan di bawah lisensi MIT. Silakan gunakan untuk keperluan riset, benchmark perusahaan, dan implementasi arsitektur AI produksi.
