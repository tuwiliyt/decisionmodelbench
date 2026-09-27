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
python3 gpu_manager.py

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

# Check llama-cpp-python installation & compilation with live detailed progress
if ! python3 -c "import llama_cpp" &> /dev/null; then
    python3 install_llama_cpp.py
else
    echo -e "  ${GREEN}✓ llama-cpp-python (CUDA Engine) sudah terpasang.${NC}"
fi

# ------------------------------------------------------------------------------
# 5. Heavyweight LLM Models Download with Dynamic Hardware Auto-Offering
# ------------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${CYAN}==============================================================================${NC}"
echo -e "${BOLD}${CYAN}📥 [5/6] DETEKSI HARDWARE & PENAWARAN MODEL LLM KELAS BERAT (GGUF)${NC}"
echo -e "${BOLD}${CYAN}==============================================================================${NC}"

HW_INFO=$(python3 -c "
try:
    from gpu_manager import get_hardware_profile
    hw = get_hardware_profile()
    vram = float(hw.get('total_vram_all_gpus_gb', 0.0))
    dev = hw.get('primary_device', 'CPU')
    tier = hw.get('tier', 'Unknown')
    count = int(hw.get('device_count', 0))
    print(f'{vram}|{dev}|{tier}|{count}')
except Exception:
    print('0.0|CPU|CPU|0')
")

VRAM_TOTAL=$(echo "$HW_INFO" | cut -d'|' -f1)
PRIMARY_DEV=$(echo "$HW_INFO" | cut -d'|' -f2)
GPU_TIER=$(echo "$HW_INFO" | cut -d'|' -f3)
GPU_COUNT=$(echo "$HW_INFO" | cut -d'|' -f4)

CAN_RUN_14B=$(python3 -c "print(1 if float('$VRAM_TOTAL') >= 20.0 else 0)")
CAN_RUN_FULL=$(python3 -c "print(1 if float('$VRAM_TOTAL') >= 12.0 else 0)")

if [ "$CAN_RUN_14B" -eq 1 ]; then
    DEFAULT_CHOICE="4"
    echo -e "${BOLD}${GREEN}🚀 KAPASITAS HARDWARE BESAR TERDETEKSI: ${VRAM_TOTAL} GB VRAM (${GPU_COUNT}x GPU)!${NC}"
    echo -e "  • Perangkat Primer   : ${GREEN}${PRIMARY_DEV}${NC}"
    echo -e "  • Klasifikasi Tier   : ${GREEN}${GPU_TIER}${NC}"
    echo -e "  • Total VRAM Gabungan: ${GREEN}${VRAM_TOTAL} GB${NC}"
    echo -e ""
    echo -e "${BOLD}${MAGENTA}🔥 PENAWARAN MODEL BESAR (FLAGSHIP HEAVYWEIGHT):${NC}"
    echo -e "  Sistem mendeteksi hardware Anda ${BOLD}${GREEN}SANGAT MAMPU${NC} menjalankan model LLM besar:"
    echo -e "    ⭐ ${BOLD}Qwen 2.5 14B Instruct${NC} (14.7B Parameter, 32K context, ~9.0 GB VRAM)"
    echo -e "    ⭐ ${BOLD}Gemma 2 9B Instruct${NC} (9.24B Parameter, ~5.4 GB VRAM)"
    echo -e "    ⭐ ${BOLD}Sahabat-AI 8B Instruct${NC} (8.03B Parameter, ~4.58 GB VRAM)"
    echo -e "  Model 14B sangat direkomendasikan untuk pengujian kelas berat & penalaran kompleks."
    echo -e "  ${YELLOW}Sistem otomatis merekomendasikan: Pilihan 4 (Paket Enterprise Flagship).${NC}"
elif [ "$CAN_RUN_FULL" -eq 1 ]; then
    DEFAULT_CHOICE="1"
    echo -e "${BOLD}${GREEN}💡 GPU SERVER TERDETEKSI: ${VRAM_TOTAL} GB VRAM (${PRIMARY_DEV})!${NC}"
    echo -e "  • Klasifikasi Tier   : ${GREEN}${GPU_TIER}${NC}"
    echo -e ""
    echo -e "${BOLD}${CYAN}🎯 PENAWARAN MODEL KELAS BERAT:${NC}"
    echo -e "  GPU Anda mampu menjalankan model kelas berat 8B/9B:"
    echo -e "    ⭐ ${BOLD}Gemma 2 9B Instruct${NC} (~5.4 GB VRAM)"
    echo -e "    ⭐ ${BOLD}Sahabat-AI 8B Instruct${NC} (~4.58 GB VRAM)"
    echo -e "    ⭐ ${BOLD}Qwen 2.5 7B Instruct${NC} (~4.40 GB VRAM)"
    echo -e "  ${YELLOW}Sistem otomatis merekomendasikan: Pilihan 1 (Unduh Lengkap 4 Model).${NC}"
else
    DEFAULT_CHOICE="2"
    echo -e "${YELLOW}ℹ️  Kapasitas VRAM terbatas (${VRAM_TOTAL} GB VRAM / ${PRIMARY_DEV}).${NC}"
    echo -e "  Direkomendasikan paket 'Unduh Cepat' (Sahabat-AI 8B + Gemma 2 2B) untuk menghemat VRAM."
fi

echo -e "------------------------------------------------------------------------------"
echo -e "Pilih paket pengunduhan model LLM System 2:"
if [ "$CAN_RUN_14B" -eq 1 ]; then
    echo -e "  1) Unduh Lengkap (~16 GB: Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B, Gemma 2 2B)"
    echo -e "  2) Unduh Cepat (~6.2 GB: Sahabat-AI 8B + Gemma 2 2B)"
    echo -e "  3) Unduh Sahabat-AI 8B Saja (~4.6 GB)"
    echo -e "  ${BOLD}${GREEN}4) Unduh Paket Enterprise Flagship (+ Qwen 2.5 14B) ~25 GB  [⭐ DIREKOMENDASIKAN UNTUK GPU ANDA]${NC}"
    echo -e "  5) Lewati sekarang (Unduh nanti dengan: python3 download_models.py)"
elif [ "$CAN_RUN_FULL" -eq 1 ]; then
    echo -e "  ${BOLD}${GREEN}1) Unduh Lengkap (~16 GB: Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B, Gemma 2 2B)  [⭐ DIREKOMENDASIKAN UNTUK GPU ANDA]${NC}"
    echo -e "  2) Unduh Cepat (~6.2 GB: Sahabat-AI 8B + Gemma 2 2B)"
    echo -e "  3) Unduh Sahabat-AI 8B Saja (~4.6 GB)"
    echo -e "  4) Unduh Paket Enterprise Flagship (+ Qwen 2.5 14B) ~25 GB (Membutuhkan >=20 GB VRAM)"
    echo -e "  5) Lewati sekarang (Unduh nanti dengan: python3 download_models.py)"
else
    echo -e "  1) Unduh Lengkap (~16 GB: Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B, Gemma 2 2B)"
    echo -e "  ${BOLD}${GREEN}2) Unduh Cepat (~6.2 GB: Sahabat-AI 8B + Gemma 2 2B)  [⭐ DIREKOMENDASIKAN UNTUK KAPASITAS ANDA]${NC}"
    echo -e "  3) Unduh Sahabat-AI 8B Saja (~4.6 GB)"
    echo -e "  4) Unduh Paket Enterprise Flagship (+ Qwen 2.5 14B) ~25 GB"
    echo -e "  5) Lewati sekarang (Unduh nanti dengan: python3 download_models.py)"
fi
echo -e "------------------------------------------------------------------------------"
read -r -p "Pilihan Anda [1/2/3/4/5, default: $DEFAULT_CHOICE]: " DOWNLOAD_CHOICE
DOWNLOAD_CHOICE=${DOWNLOAD_CHOICE:-$DEFAULT_CHOICE}

case "$DOWNLOAD_CHOICE" in
    1)
        echo -e "${CYAN}Mengunduh seluruh 4 model kelas berat standar...${NC}"
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
    4)
        echo -e "${CYAN}Mengunduh paket Enterprise Flagship (+ Qwen 14B)...${NC}"
        python3 download_models.py --models enterprise
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
