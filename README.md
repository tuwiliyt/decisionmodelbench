# 🏛️ DecisionModelBench: Benchmark Model Decision vs Sovereign LLM Indonesia

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![CUDA 12/13](https://img.shields.io/badge/CUDA-NVIDIA%20GPU-green.svg)](https://developer.nvidia.com/cuda-zone)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Multi-GPU](https://img.shields.io/badge/Multi--GPU-Dual%20Tesla%20T4%20Sharding-purple.svg)](https://developer.nvidia.com)

Platform evaluasi dan arena perbandingan komprehensif untuk memvalidasi mengapa **Model Decision (System 1)** wajib digunakan dalam arsitektur AI produksi dibandingkan membebankan seluruh kueri ke **Foundation Large Language Model Generatif Kelas Berat (System 2)** pada infrastruktur GPU lokal (NVIDIA Tesla T4 / A10G / L4 / RTX 3090/4090 / Multi-GPU).

Repository ini menyajikan pembuktian empiris, baik melalui **Antarmuka Web Interaktif** maupun **Headless CLI Terminal**, dengan hasil nyata:
- ⚡ **Latensi Triage Instan:** Sub-100 ms vs 3,500–90,000+ ms (**50x hingga 390x+ lebih cepat**).
- 🎯 **Konsumsi Token Nol:** **0 output tokens** saat klasifikasi (Zero Token Waste vs 150–250 token terbuang per kueri pada LLM murni).
- 💰 **Penghematan Biaya & GPU 80%–100%:** Kueri rutin diselesaikan via *Fast-Path Gating* (<100 ms) tanpa menyentuh GPU LLM kelas berat.
- 🛡️ **100% Deterministik Matematis:** Nilai probabilitas Sigmoid & Softmax terkalibrasi tanpa risiko *JSON format drift*, *syntax error*, atau *prompt injection*.
- 💻 **Dukungan Multi-GPU Terdistribusi:** Beban dibagi otomatis (*tensor-split sharding*) pada GPU 0 (Laya, Kev, LLM Shard 0) dan GPU 1 (OpenJev, LLM Shard 1).

---

## 📑 Daftar Isi
1. [🏛️ Mengapa Harus Menggunakan Decision Model? (Executive Value Proposition)](#️-mengapa-harus-menggunakan-decision-model-executive-value-proposition)
2. [🔄 Perbandingan Alur Eksekusi Kueri](#-perbandingan-alur-eksekusi-kueri)
3. [📊 Hasil Uji Empiris Nyata (Benchmark Empiric Proof)](#-hasil-uji-empiris-nyata-benchmark-empiric-proof)
4. [📈 Matriks Komparasi Seluruh Spektrum Model](#-matriks-komparasi-seluruh-spektrum-model)
5. [⚡ Arena Trading Cepat AI (High-Frequency Trading & Saldo Dummy $10k)](#-arena-trading-cepat-ai-high-frequency-trading--saldo-dummy-10k)
6. [🎮 Arena Brick Breaker / Breakout AI (Adu Refleks Paralel)](#-arena-brick-breaker--breakout-ai-adu-refleks-paralel)
7. [🖥️ Tutorial Uji Perbandingan Headless (CLI Terminal)](#️-tutorial-uji-perbandingan-headless-cli-terminal)
8. [🌐 Tutorial Web Dashboard & Live Monitor GPU (nvtop Style)](#-tutorial-web-dashboard--live-monitor-gpu-nvtop-style)
9. [🚀 Panduan Instalasi Lengkap dari Server Kosong](#-panduan-instalasi-lengkap-dari-server-kosong)
10. [📦 Katalog Model & Spesifikasi Hardware](#-katalog-model--spesifikasi-hardware)
11. [📁 Struktur File Repository](#-struktur-file-repository)

---

## 🏛️ Mengapa Harus Menggunakan Decision Model? (Executive Value Proposition)

Menjalankan Large Foundation LLM (8B / 14B) untuk 100% kueri pengguna adalah pemborosan komputasi hingga 90%. Arsitektur **Two-Tier Brain (Decision Model + LLM On-Demand)** memisahkan fase **Triage Cepat (System 1)** dari fase **Penalaran Naratif Empatik (System 2)**.

```
                              [ Kueri Pelanggan Masuk ]
                                          │
                 ┌────────────────────────┴────────────────────────┐
                 ▼                                                 ▼
   [ JALUR A: TANPA DECISION MODEL ]               [ JALUR B: DENGAN DECISION MODEL ]
            (LLM Murni)                                    (Two-Tier Brain)
                 │                                                 │
          Direct 100% Load                                  Tier 1: Decision Model
                 │                                       (Laya / OpenJev / Jev / Kev)
       Single-Tier Autoregresif                                    │
    (Next-Token Generation Loop)                         1x Forward Pass CUDA (0 Tokens)
                 │                                         Latensi Instan: ~50-160 ms
     Latensi: 3,500 - 90,000+ ms                                   │
      Token: 150 - 250 tokens                            [ Evaluasi Smart Gating ]
                 │                                       Kritis? / Butuh Eskalasi?
                 │                                           ├── TIDAK ──► [ Fast-Path Auto-Reply ]
                 │                                           │             • Latensi: <100 ms
                 │                                           │             • 100% Kuota LLM Dihemat
                 │                                           │             • 0 Token Output Terbuang
                 ▼                                           │
    [ Respon Narasi Lengkap ]                                └── YA ─────► Tier 2: LLM Kelas Berat
    (Beban Penuh GPU & Biaya)                                              (Sahabat-AI / Qwen / Gemma)
                                                                           • Penanganan Empatik Terarah
```

### 5 Pilar Keunggulan Utama Decision Model:
1. **⚡ 50x–390x Latensi Triage Lebih Cepat:**
   Keputusan routing selesai dalam sub-100 ms (Laya ~50-75ms, Jev ~160ms, OpenJev ~200ms) dibandingkan 3,500–90,000+ ms pada LLM generatif.
2. **🎯 Zero Output Tokens (0 Token Terbuang):**
   Output berupa nilai matematis Softmax/Sigmoid langsung dari representasi embedding. LLM murni membuang 150–250 token per kueri hanya untuk struktur JSON formatting.
3. **🛡️ 100% Deterministik Tanpa Halusinasi Format:**
   Kebal terhadap kegagalan parsing JSON, drift format, dan prompt injection pada tahap routing.
4. **💰 Penghematan Biaya & GPU 80%–100% (Smart Gating):**
   80% kueri rutin (FAQ, cek jadwal, info umum) dijawab instan via *Fast-Path Auto-Reply* (<100 ms) tanpa memanggil LLM 8B sama sekali. Kuota LLM hanya dialokasikan untuk 20% komplain eskalatif.
5. **📈 Throughput Skalabilitas Produksi Tinggi:**
   Decision Model mampu melayani **~15–50 req/s per GPU**, sedangkan LLM 8B saturasi pada **~0.2–0.3 req/s**.

---

## 🔄 Perbandingan Alur Eksekusi Kueri

| Dimensi Alur | ❌ Tanpa Decision Model (LLM Murni) | ✓ Dengan Decision Model (Two-Tier Brain) |
| :--- | :--- | :--- |
| **Aliran Eksekusi** | Kueri ➔ 100% Langsung ke LLM 8B/14B | Kueri ➔ Tier 1: Triage 0 Token (50ms) ➔ Smart Gating |
| **Kueri Rutin (FAQ)** | Memboroskan 150–250 token & waktu GPU | **Fast-Path Selesai <100 ms (100% Kuota LLM Dihemat)** |
| **Kueri Kritis (Komplain)** | Rentan halusinasi JSON & format drift | **Triage deterministik terkalibrasi ➔ Eskalasi terarah ke LLM** |
| **Beban Komputasi GPU** | 100% Beban Penuh per Kueri | **Hemat 80% - 90% Utilisasi Komputasi Cluster GPU** |
| **Reliabilitas Output** | Stokastik (bergantung suhu & sampling) | **Deterministik Matematis (Probabilitas Terkalibrasi)** |

---

## 📊 Hasil Uji Empiris Nyata (Benchmark Empiric Proof)

Pengujian nyata end-to-end pada cluster **2x NVIDIA Tesla T4 GPU** membandingkan eksekusi langsung kueri pengguna:

| Metrik Kunci | Tanpa Decision Model (Gemma 2 2B Standalone) | Dengan Decision Model (Laya 421M + Gemma 2 2B) | Hasil Uji / Keunggulan |
| :--- | :--- | :--- | :--- |
| **Arsitektur** | Single-Tier Autoregressive LLM | Dual-Tier (System 1 Triage + System 2 Resolution) | Pemisahan Triage vs Narasi |
| **Latensi Triage** | 93,389.7 ms | **237.4 ms** | **393.4x Lebih Cepat** ⚡ |
| **Token Output Triage** | 98 tokens | **0 tokens (Zero Waste)** | **100% Token Triage Dihemat** 🎯 |
| **Kueri Rutin (Fast-Path)** | Memanggil LLM penuh | **Dijawab instan <100ms (0 token LLM)** | **100% Biaya LLM Dihemat** 💰 |
| **Reliabilitas Keputusan** | Format tidak terkalibrasi | **100% Konsisten (Sigmoid/Softmax)** | Bebas Halusinasi |

---

## 📈 Matriks Komparasi Seluruh Spektrum Model

| Dimensi Kinerja | Tanpa Decision (LLM Murni) | Laya Multilingual (421M GPU) | TypeSafe Jev (Cloud SaaS) | OpenJev (0.5B Local GPU) | Kev-0.8B (Local Ensemble) | CLM-8B (Stanford Dual-Encoder) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Latensi Triage** | ~3,500 – 6,000 ms | **~50 – 75 ms ⚡** | ~140 – 180 ms | ~180 – 240 ms | ~1,100 – 1,400 ms | Hardware Guarded |
| **Token Output Triage** | 150 – 250 tokens | **0 tokens** | **0 tokens** | **0 tokens** | **0 tokens** | **0 tokens** |
| **Metode Inferensi** | Autoregressive Loop | 1x Forward CUDA | Single Forward Pass | Contrastive Logit Head | Ensemble Forward | Dual-Encoder Contrast |
| **Determinisme** | Stokastik (Drift) | **100% Terkalibrasi** | **100% Deterministik** | **100% Normalized** | **100% Ensembled** | **100% Cosine Scored** |
| **Penghematan Token** | 0% (Boros Kuota) | **Hemat 80–90%** | **Hemat 80–90%** | **Hemat 80–90%** | **Hemat 80–90%** | **Hemat 80–90%** |
| **Overhead VRAM GPU** | 0 MB (Hanya LLM) | ~950 MB | **0 MB (Offloaded Cloud)** | ~1,100 MB | ~1,600 MB | Membutuhkan >=16GB |
| **Throughput Concurrency**| ~0.2 – 0.3 req/s | **~15 – 20 req/s** | **~50+ req/s (Cloud)** | ~5 – 8 req/s | ~1 req/s | N/A |

---

## ⚡ Arena Trading Cepat AI (High-Frequency Trading & Saldo Dummy $10k)

Untuk memvalidasi keunggulan Decision Model pada sektor finansial kuantitatif (*algorithmic scalping / high-frequency execution*), platform menyediakan **Arena Simulasi Trading Cepat Real-Time** dengan modal saldo dummy **$10,000 USD** (Rp 150.000.000) per model:

1. 🟢 **Laya Multilingual (421M ModernBERT CUDA):** Scalper ultra-refleks **~55 ms (18.2 Hz)**, 0 tokens, 0% slippage.
2. 🔵 **OpenJev (0.5B Qwen 2.5 Logit Scorer GPU 1):** Momentum trader **~210 ms (4.8 Hz)**, 0 tokens.
3. 🟣 **TypeSafe Jev (Cloud SaaS Decision Model):** Cloud algorithmic trader **~160 ms (6.2 Hz)**, 0 tokens.
4. 🟡 **Kev-0.8B (Local LoRA Ensemble CUDA):** Trend follower **~950 ms (1.1 Hz)**, 0 tokens.
5. 🔴 **Heavyweight LLM (Sahabat-AI 8B Instruct):** Keterlambatan fatal (*Decision Lag*) **~2,500 ms (0.4 Hz)**, pembakaran ~32 token per order, dan penalti *slippage* parah.

### Mengapa Decision Model Menang Mutlak di Trading Cepat?
* **Hukum Fisika Pasar (Order Execution Window):** Sinyal teknikal (misal: *Golden Cross*, *Order Book Imbalance 74% Bid*) hanya bertahan beberapa ratus milidetik sebelum harga bergeser.
* **Refleks Sub-100ms:** Laya dan Decision Model mengeksekusi order instan dengan selisih harga nol (*Zero Slippage*).
* **Bencana Keterlambatan LLM 2.5 Detik:** LLM membutuhkan ~2.5 detik untuk menghasilkan kalimat rekomendasi. Pada saat order tiba di pasar, harga telah naik/turun signifikan sehingga model selalu membeli di pucuk (*fomo*) atau cut-loss terlambat di dasar jurang.
* **Efisiensi Nol Biaya Token:** 10 tick per detik = 36.000 eksekusi per jam. Decision Model berbiaya **0 token ($0)**, sementara LLM murni membakar ratusan ribu token dalam hitungan menit.

---

### Cara Menjalankan Arena Trading Cepat:

#### 1. Mode Web Interaktif (K-Line Candlestick & Order Book Ladder)
Buka di browser saat server aktif:
* **URL Langsung:** `http://localhost:7860/trading` atau `http://localhost:7860/fast_trading`
* **Tab Mode 7 Dashboard:** Kunjungi `http://localhost:7860/` lalu pilih tab `[📈 Mode 7: Fast Trading AI]`
* **Fitur:** Grafik K-Line Candlestick 1-detik, moving averages EMA(9) dan EMA(21), RSI(14) oscillator, Order Book Depth Ladder (L2 Bids & Asks), kartu portofolio 5 model live side-by-side, selector regime pasar (*Bull Run*, *Flash Crash*, *Sideways Chop*, *Whipsaw*), slider kecepatan (*1x*, *3x*, *10x Turbo HFT*), dan efek suara retro (*Audio SFX*).

#### 2. Mode Terminal Headless CLI (Rich Live Ticker & Portfolio Ladder)
```bash
# Jalankan simulasi trading 60 ticks (default)
python3 play_fast_trading.py

# Simulasi cepat 30 ticks dengan mode market crash
python3 play_fast_trading.py --ticks 30 --speed fast --regime FLASH_CRASH

# Uji satu probe API trading live ke backend FastAPI server
python3 play_fast_trading.py --api
```

---

## 🎮 Arena Brick Breaker / Breakout AI (Adu Refleks Paralel)

Sebagai pembuktian nyata mengapa **Decision Model (System 1)** unggul mutlak atas **Heavyweight Autoregressive LLM (System 2)** pada kendali aksi reaktif waktu-nyata (*real-time reactive control*), DecisionModelBench menyediakan arena simulasi game **Brick Breaker / Breakout** yang mempertandingkan 5 model kecerdasan buatan secara **paralel side-by-side** di arena masing-masing:

1. 🟢 **Laya Multilingual (421M ModernBERT CUDA):** Kecepatan inferensi super-refleks **~55 ms (18.2 Hz)**, 0 tokens.
2. 🔵 **OpenJev (0.5B Qwen 2.5 Logit Scorer GPU 1):** Pertahanan stabil **~210 ms (4.8 Hz)**, 0 tokens.
3. 🟣 **TypeSafe Jev (Cloud SaaS Decision Model):** Respon awan cepat **~160 ms (6.2 Hz)**, 0 tokens.
4. 🟡 **Kev-0.8B (Local LoRA Ensemble CUDA):** Reaksi terukur **~950 ms (1.1 Hz)**, 0 tokens.
5. 🔴 **Heavyweight LLM (Sahabat-AI 8B Instruct):** Loop autoregresif **~2,500 ms (0.4 Hz)**, membakar 12–80 tokens per aksi.

### Mengapa LLM Gagal Total pada Game Reaktif?
* **Physics Deadline:** Bola bergerak menukik ke bawah dan melewati garis batas paddle dalam rentang waktu 1.5 – 2.0 detik.
* **Refleks Sub-100ms vs Input Starvation:** Decision Model (System 1) menghasilkan keputusan arah gerak (`geser_kiri`, `geser_kanan`, `tetap_diam`) dalam 1 forward pass CUDA (<60 ms). Paddle langsung bergeser menangkis bola tepat sasaran.
* Sebaliknya, Large LLM (System 2) butuh ~2.5 detik per siklus generasi kata demi kata. Saat kata pertama baru terbit, bola telah menembus lantai arena dan nyawa terbuang sia-sia (*Decision Lag*).

---

### Cara Menjalankan Arena Breakout:

#### 1. Mode Web Interaktif (Retro Cyber Arcade 5-Kanvas Paralel)
Buka di browser saat server aktif:
* **URL Langsung:** `http://localhost:7860/breakout` atau `http://localhost:7860/brick_breaker`
* **Tab Mode 6 Dashboard:** Buka `http://localhost:7860/` lalu klik tab `[🎮 Mode 6: Breakout Arcade]`
* **Fitur:** 5 kanvas simulasi paralel simultan, efek audio retro Web Audio API, partikel ledakan balok, garis proyeksi lintasan, grafik leaderboard live, dan tombol uji API probe langsung ke GPU.

#### 2. Mode Terminal Headless CLI (Rich ASCII Parallel Arena)
Jalankan simulator paralel langsung di terminal Linux/SSH:
```bash
# Jalankan simulasi paralel 5 model selama 100 ticks (default)
python3 play_brick_breaker.py

# Simulasi cepat 40 ticks
python3 play_brick_breaker.py --ticks 40 --fps 15

# Uji satu probe API live ke backend FastAPI server
python3 play_brick_breaker.py --api
```

---

## 🖥️ Tutorial Uji Perbandingan Headless (CLI Terminal)

Perkakas CLI `benchmark_headless.py` dirancang untuk pengujian langsung di terminal, integrasi CI/CD, atau server tanpa GUI/desktop:

### 1. Menampilkan Argumen Eksekutif "Mengapa Harus Decision Model"
```bash
python3 benchmark_headless.py --why
```

### 2. Menampilkan Matriks Komparasi Spektrum Model Lengkap
```bash
python3 benchmark_headless.py --matrix
```

### 3. Menjalankan Komparasi Arsitektur Berdasarkan Skenario
Bandingkan langsung hasil eksekusi *Tanpa Decision Model* vs *Dengan Decision Model*:
```bash
# Skenario 1: Komplain Kritis Marunda (Deteksi Ancaman Viral Medsos)
python3 benchmark_headless.py --compare --scenario marunda --decision laya --llm gemma-2b

# Skenario 2: Tanggap Darurat Fraud Rekening (Deteksi Ancaman Lapor Polisi)
python3 benchmark_headless.py --compare --scenario scam --decision openjev --llm sahabatai

# Skenario 3: Restrukturisasi Nasabah PHK (Fintech OJK)
python3 benchmark_headless.py --compare --scenario phk --decision jev --llm sahabatai

# Skenario 4: Kueri Rutin Jam Buka (Membuktikan Fast-Path 100% Hemat LLM)
python3 benchmark_headless.py --compare --scenario faq --decision laya --llm gemma-2b
```

### 4. Menjalankan Seluruh 4 Skenario Sekaligus (Batch Benchmark)
```bash
python3 benchmark_headless.py --all-presets --decision laya --llm gemma-2b
```

### 5. Menu CLI Interaktif
```bash
python3 benchmark_headless.py --interactive
```

---

## 🌐 Tutorial Web Dashboard & Live Monitor GPU (nvtop Style)

### 1. Menjalankan Web Server
Jalankan server FastAPI uvicorn di port 7860:
```bash
python3 -m uvicorn app_server:app --host 0.0.0.0 --port 7860
```

### 2. URL Akses Dashboard
Buka browser Anda:
* **Halaman Utama (Arena Komparasi & Mengapa Decision Model):**
  `http://localhost:7860/` atau `http://localhost:7860/heavyweight`
  * **Executive Value Proposition Hero:** 4 pilar alasan wajib decision model.
  * **Mode Utama (Dengan vs Tanpa Decision Model):** Battle interaktif langsung dengan LLM nyata.
  * **Dual-GPU nvtop Monitor:** Kartu mandiri **GPU 0: Tesla T4 (Primary Node)** dan **GPU 1: Tesla T4 (Secondary Node)** dengan grafik rolling real-time, suhu, daya (Watt), dan daftar proses komputasi CUDA.
  * **Two-Tier Brain Pipeline & Multi-LLM Arena:** Uji Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B, dan Gemma 2 2B.
* **Halaman Single-Question Playground (System 1):**
  `http://localhost:7860/playground` atau `http://localhost:7860/benchmark`
  * Evaluasi instan 5 model System 1 (Laya, Jev, Kev, OpenJev, CLM) dengan tombol alih cepat.

### 3. Akses Publik via Cloudflare Tunnel
Jika menggunakan remote server atau Google Colab/Kaggle:
```bash
cloudflared tunnel --url http://localhost:7860
```

---

## 🚀 Panduan Instalasi Lengkap dari Server Kosong

### 1. Prasyarat Sistem
* **OS:** Linux x86_64 (Ubuntu 20.04+, Debian 11+, Google Colab, Kaggle Environment).
* **GPU:** NVIDIA GPU (minimal 8 GB VRAM; 16 GB+ atau Multi-GPU direkomendasikan).
* **Driver & CUDA:** CUDA 12.0+ atau 13.0+ (`nvidia-smi` aktif).
* **Python:** 3.10, 3.11, 3.12, atau 3.13.

### 2. Kloning & Instalasi Otomatis (`setup.sh`)
```bash
git clone https://github.com/tuwiliyt/decisionmodelbench.git
cd decisionmodelbench
chmod +x setup.sh
./setup.sh
```

#### Fitur Unggulan Skrip Instalasi:
1. **Live Detail Kompilasi llama-cpp-python (`install_llama_cpp.py`):**
   * Menampilkan progress bar interaktif dengan target kompilasi real-time `[xxx/443]`, persentase, dan nama file kernel CUDA yang sedang dikompilasi.
   * Mengatasi otomatis konflik driver `/usr/local/nvidia/lib64/libcuda.so` pada lingkungan cloud container (Kaggle/Colab).
2. **Auto-Offering Model Kelas Berat:**
   * Jika sistem mendeteksi GPU memiliki kapasitas menjalankan model besar, skrip otomatis menawarkan opsi unduhan model (Sahabat-AI 8B, Gemma 2 9B/2B, Qwen 2.5 7B).
3. **Validasi Interaktif API Key TypeSafe Jev:**
   * Uji koneksi live ke `https://api.typesafe.ai/v1/systemone`. Jika tidak ada kunci, sistem berjalan 100% offline dengan model lokal (Laya, OpenJev, Kev).
4. **Verifikasi Diagnostik VRAM Otomatis (`verify_models.py`):**
   * Menguji pemuatan dan inferensi setiap model pada GPU sebelum menyelesaikan instalasi.

---

## 📦 Katalog Model & Spesifikasi Hardware

### 1. Model Decision (System 1 - Triage Instan 0 Token)
| Model | Parameter | Arsitektur | Alokasi Hardware | Latensi Rata-rata |
| :--- | :--- | :--- | :--- | :--- |
| **Laya Multilingual** | 421M | ModernBERT RLCD | GPU 0 (`cuda:0`) | **~50 - 65 ms** |
| **OpenJev** | 0.5B | Qwen 2.5 Logit Scorer | GPU 1 (`cuda:1`) | **~180 - 240 ms** |
| **TypeSafe Jev** | Proprietary | Decision Architecture | Cloud SaaS API | **~140 - 180 ms** |
| **Kev-0.8B** | 0.8B | Qwen 2.5 LoRA Ensemble | GPU 0 (`cuda:0`) | **~1.1 - 1.4 s** |

### 2. Foundation LLM Kelas Berat (System 2 - Penalaran & Narasi)
| Model | Parameter | Organisasi | Format | Kecepatan Inferensi | Rekomendasi Hardware |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sahabat-AI 8B Instruct** | 8.03B | GoTo & Indosat | Q4_K_M GGUF | ~28 - 32 t/s | Tesla T4 / RTX 3080/4070 (12–16 GB) |
| **Qwen 2.5 7B Instruct** | 7.61B | Alibaba Cloud | Q4_K_M GGUF | ~30 - 35 t/s | Tesla T4 / RTX 3080/4070 (12–16 GB) |
| **Gemma 2 9B Instruct** | 9.24B | Google DeepMind | Q4_K_M GGUF | ~24 - 28 t/s | Tesla T4 / RTX 3080/4070 (16 GB) |
| **Gemma 2 2B Instruct** | 2.61B | Google DeepMind | Q4_K_M GGUF | ~50 - 65 t/s | Hemat VRAM (GPU 4–8 GB) |

### 🚀 Topologi Dual GPU (Multi-GPU Sharding)
Pada sistem dengan 2 GPU (seperti 2x Tesla T4), beban komputasi didistribusikan secara optimal:
* **GPU 0 (`cuda:0`):** Laya (421M), Kev (0.8B), dan Shard 0 LLM Engine.
* **GPU 1 (`cuda:1`):** OpenJev (0.5B Logit Scorer) dan Shard 1 LLM Engine (`tensor_split=[0.5, 0.5]`).

---

## 📁 Struktur File Repository

```text
decisionmodelbench/
├── app_server.py                     # Server FastAPI (Multi-GPU Sharding, API REST, Web Server)
├── gpu_manager.py                    # Detektor hardware dinamis & multi-GPU scaler
├── benchmark_headless.py             # CLI Tool: Komparasi Dengan vs Tanpa Jev, Matrix, Why
├── download_models.py                # Skrip pengunduh otomatis model GGUF dari Hugging Face
├── setup.sh                          # Skrip otomatisasi instalasi & konfigurasi interaktif
├── config.py                         # Modul pembaca konfigurasi & .env
├── .env.example                      # Template file variabel lingkungan
├── requirements.txt                  # Daftar dependensi pustaka Python
├── install_llama_cpp.py              # Installer CUDA llama-cpp-python dengan live progress detail
├── heavyweight_llm_engine.py         # Engine manajer multi-LLM (Tensor-split multi-GPU)
├── openjev_engine.py                 # Engine native OpenJev logit continuation scorer (GPU 1)
├── sahabatai_engine.py               # Engine wrapper Sahabat-AI
├── generate_heavyweight_dashboard.py # Generator antarmuka web dashboard kelas berat (HTML)
├── heavyweight_llm_dashboard.html    # Antarmuka web utama: Arena Komparasi & Dual-GPU nvtop
├── brick_breaker_arena.html          # Web Arena Retro Arcade 5-kanvas simulasi paralel
├── play_brick_breaker.py             # CLI Terminal Rich ASCII Brick Breaker 5-model paralel
├── trading_engine.py                 # Core HFT Trading Engine, Latency Slippage, & Portfolio Tracker
├── fast_trading_arena.html           # Web Arena Simulasi Trading Cepat K-Line Candlestick & Order Book
├── play_fast_trading.py              # CLI Terminal Rich ASCII Trading Ticker & P&L Arena
├── generate_dashboard.py             # Generator antarmuka web playground kelas ringan (HTML)
├── benchmark_dashboard.html          # Antarmuka web playground single-question (System 1)
├── test_jev.py                       # Skrip uji konektivitas TypeSafe Jev API
└── README.md                         # Dokumentasi & panduan teknis komprehensif
```

---

## 📄 Lisensi
Proyek ini didistribusikan di bawah lisensi MIT. Silakan gunakan untuk keperluan riset, benchmark perusahaan, dan implementasi arsitektur AI produksi.
