# Project Context & State: AI Decision Models (System One)

Dokumen ini adalah ringkasan konteks percakapan dan status proyek pengujian *Probabilistic Non-Autoregressive Decision Models*.

---

## 1. Lingkungan Hardware & Dependensi

* **Lokasi Workspace:** `/content/drive/MyDrive/AIPROJECT/DecisionModel`
* **GPU Aktif:** NVIDIA Tesla T4 (15.6 GB VRAM) | CUDA Driver 580.82 / CUDA 13.0
* **Python Environment:** Python 3.13.15, PyTorch dengan dukungan CUDA (`torch.cuda.is_available() == True`)
* **Libraries Terpasang:**
  * `laya` (v0.3.20) – Engine System One non-autoregresif lokal (~421M parameter)
  * `kev` (v0.1.0) – Model keputusan lokal berbasis Qwen3.5 + LoRA pointer head (100% wire-compatible dengan TypeSafe Jev)
  * `peft`, `torchao` (v0.18.0), `transformers` (v5.16.1), `datasets`
* **API Key TypeSafe Jev (Terverifikasi Aktif):**
  ```text
  apikey_22539801ad3ed10f4c298ead877d832613c8_cf1f0eaa3f565712acd771c6dc036e0afcfab5e5d5fc8cd56c650d48fce4b226
  ```

---

## 2. Hasil Benchmark Perbandingan (Jev vs Laya vs Kev)

Pengujian dilakukan menggunakan script [`benchmark_comparison.py`](file:///content/drive/MyDrive/AIPROJECT/DecisionModel/benchmark_comparison.py) dengan 3 skenario nyata:

| Skenario | Model | Deployment | Latensi | Ringkasan Output Keputusan |
| :--- | :--- | :--- | :--- | :--- |
| **Kasus 1: Billing & Churn (EN)** | **Jev (TypeSafe)** | Cloud API | ~197 ms | `is_cancellation=0.99`, `category=pricing (100%)`, `churn_risk=2.91` |
| | **Laya (421M)** | Local GPU | ~65 - 94 ms | `is_cancellation=0.94`, `category=pricing (98%)`, `churn_risk=1.04` |
| | **Kev-0.8B** | Local GPU | ~268 ms | `is_cancellation=0.94`, `category=pricing (55%)`, `churn_risk=1.62` |
| **Kasus 2: Incident Triage (EN)** | **Jev (TypeSafe)** | Cloud API | ~165 ms | `is_urgent=0.90`, `fault_domain=database (100%)`, `severity=2.99` |
| | **Laya (421M)** | Local GPU | **94.0 ms** | `is_urgent=0.08`, `fault_domain=database (94%)`, `severity=2.16` |
| | **Kev-0.8B** | Local GPU | ~1.8 s | `is_urgent=0.70`, `fault_domain=database (97%)`, `severity=2.23` |
| **Kasus 3: Tiket Komplain (ID)** | **Jev (TypeSafe)** | Cloud API | ~162 ms | `is_cancellation=0.68`, `category=pricing (100%)`, `urgency=2.10` |
| | **Laya (421M)** | Local GPU | **78.3 ms** | `is_cancellation=0.55`, `category=pricing (70%)`, `urgency=1.36` |
| | **Kev-0.8B** | Local GPU | ~1.3 s | `is_cancellation=0.83`, `category=pricing (61%)`, `urgency=1.80` |

---

## 3. Analisis & Karakteristik Model

1. **Jev (TypeSafe AI Cloud API):**
   * **Akurasi & Kalibrasi:** Sangat tajam dan percaya diri tinggi (sering mencapai confidence 99-100% pada kategori yang jelas).
   * **Latensi:** ~69 ms upstream server, ~160-200 ms total network round-trip.
   * **Kelebihan:** Nol penggunaan resource lokal (VRAM/RAM), zero setup maintenance.

2. **Laya (Local GPU - ModernBERT RLCD):**
   * **Kecepatan:** **Tercepat secara lokal (<80-95 ms di Tesla T4)** setelah model di-cache.
   * **Resource:** Sangat hemat VRAM (~800 MB).
   * **Multilingual:** Mampu memproses input Bahasa Indonesia secara native.
   * **Output Token:** 0 tokens generated (murni non-autoregresif).

3. **Kev-0.8B (Local GPU - Qwen3.5 LoRA):**
   * **Wire-Compatibility:** 100% kompatibel dengan schema API TypeSafe (`SystemOneRequest`, `to_record`, `to_answers`).
   * **Resource:** Menggunakan ~2 GB VRAM.
   * **Keunggulan:** Pemahaman logika teks yang dalam (berbasis backbone LLM Qwen3.5) dan mendukung fine-tuning sendiri.

4. **OpenJev (Local GPU - Qwen2.5 Logit Scorer):**
   * **Arsitektur:** Menggunakan teknik *single-pass continuation logit scoring* (mirip `daseinlabs/open-jev`) pada model terbuka tanpa decoding token.
   * **Latensi:** ~220 ms di GPU Tesla T4.
   * **Resource:** ~1 GB VRAM.

5. **CLM-8B (Stanford & NVIDIA Contrastive Language Model):**
   * **Arsitektur:** Dual-encoder (State Head + Action Head) di atas backbone Qwen3-8B.
   * **Hardware Guard:** Sistem live playground mendeteksi kapasitas VRAM secara dinamis. Jika VRAM < 16 GB, sistem menampilkan peringatan hardware dan panduan kuantisasi (AWQ/GPTQ) untuk mencegah *CUDA Out of Memory*.


6. **Sahabat-AI 8B (Sovereign Foundation LLM Indonesia - GoTo & Indosat):**
   * **Arsitektur:** Continuous Pre-Training di atas Llama-3-8B dengan korpus bahasa Indonesia, budaya lokal, hukum, bisnis, dan dialek daerah.
   * **Deployment:** Dijalankan secara lokal di GPU Tesla T4 via format GGUF Q4_K_M dengan offload CUDA penuh (`n_gpu_layers=-1`).
   * **Kecepatan Inferensi:** **~28 - 32 tokens/detik** pada Tesla T4.
   * **Penggunaan Resource:** ~4.8 GB VRAM (~9.7 GB total sistem aktif bersama System 1).
   * **Peran Operasional:** Berfungsi sebagai **System 2 Generative & Empathy Engine** dalam arsitektur hibrida Two-Tier Brain.

---

## 4. File yang Tersedia di Workspace & Server Live

* **Public Cloudflare Tunnel URLs:**
  * Dashboard Utama (Kelas Ringan / Decision Models): `https://closed-similarly-offers-approved.trycloudflare.com/`
  * Dashboard Khusus Kelas Berat (Sahabat-AI 8B & Two-Tier Brain): `https://closed-similarly-offers-approved.trycloudflare.com/heavyweight`
* [`heavyweight_llm_dashboard.html`](file:///content/drive/MyDrive/AIPROJECT/DecisionModel/heavyweight_llm_dashboard.html): Halaman web interaktif pengujian kelas berat (Mode Chat Generatif, Head-to-Head System 1 vs 2, dan Simulasi Two-Tier Pipeline).
* [`sahabatai_engine.py`](file:///content/drive/MyDrive/AIPROJECT/DecisionModel/sahabatai_engine.py): Modul inferensi terintegrasi Sahabat-AI 8B (chat, decision mode, two-tier pipeline).
* [`app_server.py`](file:///content/drive/MyDrive/AIPROJECT/DecisionModel/app_server.py): FastAPI backend daemon yang melayani semua model (Laya, Kev, Jev, OpenJev, CLM, Sahabat-AI 8B) di port `7860`.
* [`benchmark_dashboard.html`](file:///content/drive/MyDrive/AIPROJECT/DecisionModel/benchmark_dashboard.html): Dashboard komparasi 5 decision models dengan link lintas halaman.
* [`openjev_engine.py`](file:///content/drive/MyDrive/AIPROJECT/DecisionModel/openjev_engine.py): Engine inferensi OpenJev berbasis logit scoring Qwen2.5.
* [`indonesia_benchmark_suite.py`](file:///content/drive/MyDrive/AIPROJECT/DecisionModel/indonesia_benchmark_suite.py): Script pengujian 8 skenario nyata di Indonesia.
* [`PROJECT_STATE.md`](file:///content/drive/MyDrive/AIPROJECT/DecisionModel/PROJECT_STATE.md): Catatan status proyek dan riwayat benchmark.

---

## 5. Ringkasan Temuan Pengujian Kelas Berat (System 1 vs System 2)

| Parameter | System 1: Decision Model (Laya/OpenJev) | System 2: Generative LLM (Sahabat-AI 8B) |
| :--- | :--- | :--- |
| **Waktu Respon (Latensi)** | **~60 - 220 ms** | **~3,700 - 5,800 ms** (15-80x lebih lama) |
| **Output Token** | **0 tokens** (murni probabilitas logit) | **~100 - 250 tokens** (autoregresif) |
| **Biaya Komputasi / Token** | $0 biaya token generatif | Membutuhkan kuota token kontinu |
| **Konsistensi Format** | 100% deterministik, skema kaku | Butuh JSON parser & rentan variasi teks |
| **Kemampuan Teks/Empati** | Tidak ada generasi teks (hanya klasifikasi) | **Sangat fasih, empatik, memahami dialek lokal** |
| **Rekomendasi Terbaik** | **Triage 100% chat masuk (Filter Lapis 1)** | **Hanya menangani eskalasi/kasus krisis (Lapis 2)** |

