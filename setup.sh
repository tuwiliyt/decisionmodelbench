#!/usr/bin/env bash
# ==============================================================================
# Decision Model & Heavyweight LLM Benchmark - Automated Setup Script
# Repository: https://github.com/tuwiliyt/decisionmodelbench
# ==============================================================================
set -e

# Terminal ANSI Color Definitions
BOLD="\033[1m"
GREEN="\033[0;32m"
CYAN="\033[0;36m"
YELLOW="\033[1;33m"
RED="\033[0;31m"
MAGENTA="\033[0;35m"
NC="\033[0m" # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BOLD}${CYAN}==============================================================================${NC}"
echo -e "${BOLD}${CYAN}🏛️  INSTALASI & SETUP BENCHMARK DECISION MODEL & FOUNDATION LLM INDONESIA${NC}"
echo -e "${BOLD}${CYAN}==============================================================================${NC}"
echo -e "Repositori : ${MAGENTA}https://github.com/tuwiliyt/decisionmodelbench${NC}"
echo -e "Target OS  : Ubuntu 20.04/22.04/24.04 / Debian / Google Colab"
echo -e "Hardware   : GPU NVIDIA Tesla T4 / A10G / L4 / A100 / RTX Series (CUDA)"
echo -e "${BOLD}${CYAN}==============================================================================${NC}\n"

# ------------------------------------------------------------------------------
# 1. Hardware & GPU Detection
# ------------------------------------------------------------------------------
echo -e "${BOLD}${YELLOW}🔍 [1/6] Memeriksa Perangkat Keras & Akselerator GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    echo -e "  • Perangkat GPU   : ${GREEN}$GPU_NAME${NC}"
    echo -e "  • Total VRAM      : ${GREEN}$GPU_VRAM${NC}"
    echo -e "  • Driver NVIDIA   : ${GREEN}$GPU_DRIVER${NC}"
    echo -e "  ${GREEN}✓ GPU Terdeteksi & Siap untuk Akselerasi CUDA 100% Layer Offload.${NC}"
else
    echo -e "  ${YELLOW}⚠️  nvidia-smi tidak ditemukan. Model akan berjalan di CPU (kecepatan terbatas).${NC}"
fi

# ------------------------------------------------------------------------------
# 2. Python Environment Check
# ------------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${YELLOW}🐍 [2/6] Memeriksa Lingkungan Runtime Python...${NC}"
if command -v python3 &> /dev/null; then
    PY_VER=$(python3 --version)
    echo -e "  • Versi Python    : ${GREEN}$PY_VER${NC}"
    echo -e "  ${GREEN}✓ Lingkungan Python valid.${NC}"
else
    echo -e "  ${RED}❌ Python 3 tidak ditemukan. Harap pasang Python 3.10 atau lebih baru.${NC}"
    exit 1
fi

# ------------------------------------------------------------------------------
# 3. Interactive Jev API Key Configuration & Live Online Validation
# ------------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${CYAN}==============================================================================${NC}"
echo -e "${BOLD}${CYAN}🔑 [3/6] KONFIGURASI & VALIDASI API KEY TYPESAFE JEV${NC}"
echo -e "${BOLD}${CYAN}==============================================================================${NC}"
echo -e "Sistem mengintegrasikan cloud SaaS TypeSafe Jev API (https://typesafe.ai)."
echo -e "Pemberian API Key bersifat opsional:"
echo -e "  • Jika Anda memiliki API Key: masukkan di bawah untuk mengaktifkan TypeSafe Jev Cloud."
echo -e "  • Jika tidak: tekan ${BOLD}[Enter]${NC} untuk melewati dan menggunakan 100% model lokal (Laya, OpenJev, Kev)."
echo -e "------------------------------------------------------------------------------"

CONFIGURED_KEY=""
while true; do
    read -r -p "Masukkan TypeSafe Jev API Key: " INPUT_JEV_KEY

    if [ -z "$INPUT_JEV_KEY" ]; then
        echo -e "\n${YELLOW}ℹ️  Jev API Key dilewati. Seluruh model Decision lokal (Laya, OpenJev, Kev) tetap aktif 100%.${NC}"
        CONFIGURED_KEY=""
        break
    else
        echo -e "\n⏳ Menguji koneksi langsung ke endpoint cloud TypeSafe Jev..."
        # Run live connectivity verification script via python
        PING_RESULT=$(python3 -c "
import urllib.request, json, time, sys
url = 'https://api.typesafe.ai/v1/systemone'
headers = {'Authorization': 'Bearer $INPUT_JEV_KEY', 'Content-Type': 'application/json'}
payload = json.dumps({'model': 'jev-latest', 'state': 'Setup ping validation', 'questions': {'p': {'type': 'noul', 'instructions': 'ping'}}}).encode('utf-8')
t0 = time.perf_counter()
try:
    req = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=8) as r:
        lat = round((time.perf_counter() - t0) * 1000, 1)
        print(f'SUCCESS:{r.status}:{lat}')
except urllib.error.HTTPError as e:
    print(f'HTTP_ERROR:{e.code}')
except Exception as ex:
    print(f'ERROR:{ex}')
")

        if [[ "$PING_RESULT" == SUCCESS:* ]]; then
            HTTP_STATUS=$(echo "$PING_RESULT" | cut -d':' -f2)
            LATENCY_MS=$(echo "$PING_RESULT" | cut -d':' -f3)
            echo -e "  • Endpoint URL         : https://api.typesafe.ai/v1/systemone"
            echo -e "  • Status Respons HTTP  : ${GREEN}${HTTP_STATUS} OK${NC}"
            echo -e "  • Latensi Koneksi      : ${GREEN}${LATENCY_MS} ms${NC}"
            echo -e "  ${GREEN}✓ SUKSES: API KEY VALID & AKTIF TERHUBUNG KE TYPESAFE JEV!${NC}"
            CONFIGURED_KEY="$INPUT_JEV_KEY"
            break
        elif [[ "$PING_RESULT" == HTTP_ERROR:* ]]; then
            HTTP_CODE=$(echo "$PING_RESULT" | cut -d':' -f2)
            echo -e "  ${RED}❌ GAGAL AUTENTIKASI (HTTP $HTTP_CODE: Invalid API Key / Unauthorized).${NC}"
            echo -e "  Pilihan:"
            echo -e "    1) Coba masukkan ulang API Key"
            echo -e "    2) Lewati dan lanjutkan dengan mode lokal (Laya, OpenJev, Kev)"
            read -r -p "  Pilihan Anda [1/2, default 1]: " RETRY_CHOICE
            if [ "$RETRY_CHOICE" == "2" ]; then
                CONFIGURED_KEY=""
                echo -e "  ${YELLOW}ℹ️  Melanjutkan instalasi dalam mode lokal 100%.${NC}"
                break
            fi
        else
            echo -e "  ${YELLOW}⚠️  Koneksi timeout/tidak terjangkau: $PING_RESULT${NC}"
            echo -e "  Menyimpan API Key ke file .env untuk percobaan lanjutan."
            CONFIGURED_KEY="$INPUT_JEV_KEY"
            break
        fi
    fi
done

# Persist to .env
cat <<EOF > .env
# Auto-generated by setup.sh
JEV_API_KEY="$CONFIGURED_KEY"
JEV_ENDPOINT="https://api.typesafe.ai/v1/systemone"
MODELS_DIR="/root/models"
PORT=7860
EOF
echo -e "${GREEN}✓ Konfigurasi tersimpan di berkas .env!${NC}"

# ------------------------------------------------------------------------------
# 4. Core Python Dependencies Installation
# ------------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${YELLOW}📦 [4/6] Memasang Dependensi Python (pip requirements)...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Install Kev framework if not already installed
if ! python3 -c "import kev" &> /dev/null; then
    echo -e "Mengunduh & memasang pustaka Kev dari GitHub..."
    pip install git+https://github.com/jaredpalmer/kev.git || echo "⚠️ Pemasangan kev via git gagal, lewati."
else
    echo -e "  ${GREEN}✓ Pustaka Kev (jaredpalmer/kev) sudah terpasang.${NC}"
fi

# Check llama-cpp-python CUDA acceleration
if ! python3 -c "import llama_cpp" &> /dev/null; then
    echo -e "Memasang llama-cpp-python dengan akselerasi CUDA GPU..."
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir || pip install llama-cpp-python
else
    echo -e "  ${GREEN}✓ llama-cpp-python (CUDA Engine) sudah terpasang.${NC}"
fi

# ------------------------------------------------------------------------------
# 5. Heavyweight LLM Models Download
# ------------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${CYAN}==============================================================================${NC}"
echo -e "${BOLD}${CYAN}📥 [5/6] PENGUNDUHAN MODEL LLM KELAS BERAT (GGUF 4-BIT QUANTIZED)${NC}"
echo -e "${BOLD}${CYAN}==============================================================================${NC}"
echo -e "Pilih paket pengunduhan model LLM System 2:"
echo -e "  1) Unduh Lengkap (~16 GB: Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B, Gemma 2 2B)"
echo -e "  2) Unduh Cepat (~6.2 GB: Sahabat-AI 8B + Gemma 2 2B) [Rekomendasi Uji Cepat]"
echo -e "  3) Unduh Sahabat-AI 8B Saja (~4.6 GB)"
echo -e "  4) Lewati sekarang (Unduh nanti dengan: python3 download_models.py)"
echo -e "------------------------------------------------------------------------------"
read -r -p "Pilihan Anda [1/2/3/4, default: 2]: " DOWNLOAD_CHOICE
DOWNLOAD_CHOICE=${DOWNLOAD_CHOICE:-2}

case "$DOWNLOAD_CHOICE" in
    1)
        echo -e "${CYAN}Mengunduh seluruh 4 model kelas berat...${NC}"
        python3 download_models.py --models all
        ;;
    2)
        echo -e "${CYAN}Mengunduh paket cepat: Sahabat-AI 8B + Gemma 2 2B...${NC}"
        python3 download_models.py --models quick
        ;;
    3)
        echo -e "${CYAN}Mengunduh Sahabat-AI 8B Instruct...${NC}"
        python3 download_models.py --models sahabatai
        ;;
    *)
        echo -e "${YELLOW}Unduhan model dilewati. Anda dapat mengunduh kapan saja dengan:${NC}"
        echo -e "  $ python3 download_models.py"
        ;;
esac

# ------------------------------------------------------------------------------
# 6. Model Verification & GPU Diagnostic Suite
# ------------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${CYAN}==============================================================================${NC}"
echo -e "${BOLD}${CYAN}🔬 [6/6] DIAGNOSTIK & VERIFIKASI PEMUATAN MODEL (SYSTEM 1 & SYSTEM 2)${NC}"
echo -e "${BOLD}${CYAN}==============================================================================${NC}"
echo -e "Menjalankan pemuatan model secara berurutan untuk memeriksa VRAM, offload layer, dan latensi..."
python3 verify_models.py

# ------------------------------------------------------------------------------
# Generate Dashboards
# ------------------------------------------------------------------------------
echo -e "\n🎨 Mengompilasi Antarmuka Web Dashboard..."
python3 generate_dashboard.py
python3 generate_heavyweight_dashboard.py

# ------------------------------------------------------------------------------
# Completion Summary
# ------------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}==============================================================================${NC}"
echo -e "${BOLD}${GREEN}🎉 INSTALASI SELESAI & SISTEM 100% SIAP DIGUNAKAN!${NC}"
echo -e "${BOLD}${GREEN}==============================================================================${NC}"
echo -e ""
echo -e "${BOLD}PILIHAN CARA MENJALANKAN BENCHMARK:${NC}"
echo -e ""
echo -e "${CYAN}1. MODE HEADLESS / TERMINAL CLI (Tanpa Perlu Browser):${NC}"
echo -e "   $ ${BOLD}python3 benchmark_headless.py --compare --preset marunda${NC}   (Komparasi Head-to-Head)"
echo -e "   $ ${BOLD}python3 benchmark_headless.py --all-presets${NC}                (Uji 4 skenario industri)"
echo -e "   $ ${BOLD}python3 benchmark_headless.py --matrix${NC}                     (Tabel matriks eksekutif)"
echo -e "   $ ${BOLD}python3 benchmark_headless.py --interactive${NC}                (Menu terminal interaktif)"
echo -e ""
echo -e "${CYAN}2. MODE WEB DASHBOARD (Antarmuka Visual + Grafik nvtop GPU):${NC}"
echo -e "   $ ${BOLD}python3 -m uvicorn app_server:app --host 0.0.0.0 --port 7860${NC}"
echo -e "   Buka browser di: ${BOLD}http://localhost:7860/heavyweight${NC}"
echo -e ""
echo -e "${CYAN}3. MODE PUBLIK (Cloudflare Tunnel Publik Otomatis):${NC}"
echo -e "   $ ${BOLD}cloudflared tunnel --url http://localhost:7860${NC}"
echo -e "${BOLD}${GREEN}==============================================================================${NC}\n"
