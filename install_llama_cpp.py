#!/usr/bin/env python3
"""
install_llama_cpp.py: Detailed Live Progress Installer for llama-cpp-python
Provides real-time compilation progress feedback with:
- GPU & CUDA detection
- Parallel CPU core compilation (CMAKE_BUILD_PARALLEL_LEVEL)
- Real-time CMake configuration stream
- Live C++ and CUDA kernel compilation percentages [ 12%], [ 45%], etc.
- Dynamic elapsed time counter
- Automated fallback to standard CPU build if CUDA build fails
"""

import sys
import os
import time
import shutil
import re
import subprocess
import argparse

# Terminal ANSI Color Definitions
BOLD = "\033[1m"
GREEN = "\033[0;32m"
CYAN = "\033[0;36m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
MAGENTA = "\033[0;35m"
DIM = "\033[2m"
NC = "\033[0m"

def detect_cuda():
    """Detect whether CUDA GPU is available for GGML compilation."""
    # 1. Try torch
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, f"NVIDIA CUDA ({device_name})"
    except Exception:
        pass

    # 2. Try nvcc
    if shutil.which("nvcc") is not None:
        try:
            out = subprocess.check_output(["nvcc", "--version"], text=True)
            m = re.search(r"release (\d+\.\d+)", out)
            ver = m.group(1) if m else "toolkit"
            return True, f"NVIDIA CUDA Toolkit (nvcc {ver})"
        except Exception:
            pass

    # 3. Try nvidia-smi
    if shutil.which("nvidia-smi") is not None:
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
            if out:
                return True, f"NVIDIA GPU ({out.splitlines()[0]})"
        except Exception:
            pass

    return False, "CPU Only (OpenMP Acceleration)"

def ensure_cuda_symlinks():
    """Ensure libcuda.so and kitware cmake are properly linked on Linux/Kaggle environments."""
    try:
        if not os.path.exists("/usr/lib/x86_64-linux-gnu/libcuda.so"):
            candidates = [
                "/usr/local/nvidia/lib64/libcuda.so",
                "/usr/local/nvidia/lib64/libcuda.so.1",
                "/usr/local/cuda/compat/libcuda.so"
            ]
            for c in candidates:
                if os.path.exists(c):
                    subprocess.run(["ln", "-sf", c, "/usr/lib/x86_64-linux-gnu/libcuda.so"], check=False)
                    subprocess.run(["ldconfig"], check=False)
                    break
        real_cmake = "/usr/local/lib/python3.12/dist-packages/cmake/data/bin/cmake"
        if os.path.exists(real_cmake) and not os.path.islink("/usr/local/bin/cmake"):
            subprocess.run(["ln", "-sf", real_cmake, "/usr/local/bin/cmake"], check=False)
    except Exception:
        pass

def format_duration(seconds):
    mins = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{mins:02d}:{secs:02d}"

def execute_build(enable_cuda=True, force=False):
    ensure_cuda_symlinks()
    cpu_cores = os.cpu_count() or 4
    
    env = os.environ.copy()
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(cpu_cores)
    
    cuda_arch = "75"
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            cuda_arch = f"{major}{minor}"
    except Exception:
        pass

    if enable_cuda:
        env["CMAKE_ARGS"] = f"-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"
        mode_desc = f"NVIDIA CUDA GPU (-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES={cuda_arch})"
    else:
        env["CMAKE_ARGS"] = "-DGGML_BLAS=off"
        mode_desc = "Standard CPU / Native OpenMP"

    cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python", "--no-cache-dir", "-v"]
    if force:
        cmd.append("--force-reinstall")

    print(f"\n{BOLD}{CYAN}=============================================================================={NC}")
    print(f"{BOLD}{CYAN}⚙️  KOMPILASI & INSTALASI LLAMA-CPP-PYTHON DENGAN LIVE DETAIL PROGRESS{NC}")
    print(f"{BOLD}{CYAN}=============================================================================={NC}")
    print(f"  • Mode Akselerasi    : {GREEN}{mode_desc}{NC}")
    print(f"  • Parallel Build     : {GREEN}{cpu_cores} CPU Threads{NC} (CMAKE_BUILD_PARALLEL_LEVEL={cpu_cores})")
    print(f"  • Interpreter Python : {DIM}{sys.executable}{NC}")
    print(f"{YELLOW}⏳ Memulai build native C++/CUDA GGML... Seluruh detail progress ditampilkan live:{NC}\n")

    t_start = time.time()
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    captured_logs = []
    pct_pattern = re.compile(r"\[\s*(\d+)%\]")
    ninja_pattern = re.compile(r"\[\s*(\d+)/(\d+)\]")

    for raw_line in iter(proc.stdout.readline, ''):
        line = raw_line.strip()
        if not line:
            continue
        captured_logs.append(line)
        elapsed = format_duration(time.time() - t_start)

        # Detect progress percentage (Make or Ninja format)
        prog_label = None
        ninja_match = ninja_pattern.search(line)
        if ninja_match:
            curr = int(ninja_match.group(1))
            total = int(ninja_match.group(2))
            pct = int((curr / max(total, 1)) * 100)
            prog_label = f"[{curr}/{total}] [{pct:3d}%]"
        else:
            pct_match = pct_pattern.search(line)
            if pct_match:
                pct = int(pct_match.group(1))
                prog_label = f"[{pct:3d}%]"

        if prog_label:
            # Identify compilation object
            if "Building CUDA object" in line or ".cu" in line:
                target = line.split("Building CUDA object")[-1].strip() if "Building CUDA object" in line else line.split()[-1]
                target_short = os.path.basename(target)
                print(f"  {CYAN}[{elapsed}] {prog_label}{NC} {YELLOW}⚡ Mengompilasi Kernel CUDA:{NC} {target_short}")
            elif "Building CXX object" in line or "Building C object" in line or ".cpp" in line or ".c" in line:
                target = line.split("object")[-1].strip() if "object" in line else line.split()[-1]
                target_short = os.path.basename(target)
                print(f"  {CYAN}[{elapsed}] {prog_label}{NC} {GREEN}🔨 Mengompilasi C/C++:{NC} {target_short}")
            elif "Linking" in line or ".so" in line:
                target_short = os.path.basename(line.split()[-1])
                print(f"  {CYAN}[{elapsed}] {prog_label}{NC} {MAGENTA}🔗 Linking Shared Library:{NC} {target_short}")
            else:
                print(f"  {CYAN}[{elapsed}] {prog_label}{NC} {line[:75]}")
            sys.stdout.flush()
            continue

        # CMake Configuration steps
        if line.startswith("-- "):
            cmake_msg = line[3:].strip()
            if any(k in cmake_msg.lower() for k in ["compiler", "cuda", "found", "ggml", "version", "architecture", "gpu"]):
                print(f"  {DIM}[{elapsed}] [CMake]{NC} {cmake_msg}")
                sys.stdout.flush()
            continue

        # Wheel packaging steps
        if "Building wheel" in line or "Created wheel" in line:
            print(f"  {CYAN}[{elapsed}] [100%]{NC} {BOLD}{GREEN}📦 Memaketkan wheel binary Python...{NC}")
            sys.stdout.flush()
            continue

        # Pip collection
        if "Downloading" in line or "Collecting" in line:
            print(f"  {DIM}[{elapsed}] [Pip]{NC} {line}")
            sys.stdout.flush()
            continue

        # Success message from pip
        if "Successfully installed" in line:
            print(f"  {BOLD}{GREEN}✓ {line}{NC}")
            sys.stdout.flush()
            continue

        # Compilation errors
        if "error:" in line.lower() and not "ignoring" in line.lower():
            print(f"  {RED}[ERROR]{NC} {line}")
            sys.stdout.flush()

    proc.wait()
    total_time = format_duration(time.time() - t_start)

    if proc.returncode == 0:
        print(f"\n{BOLD}{GREEN}=============================================================================={NC}")
        print(f"{BOLD}{GREEN}🎉 BERHASIL: llama-cpp-python selesai dikompilasi & dipasang dalam {total_time}!{NC}")
        print(f"{BOLD}{GREEN}=============================================================================={NC}")
        
        # Verify import
        try:
            import llama_cpp
            ver = getattr(llama_cpp, "__version__", "unknown")
            print(f"  • Verifikasi Import  : {GREEN}✓ Berhasil (llama_cpp v{ver}){NC}")
            print(f"  • Status Runtime     : {GREEN}✓ Siap untuk inferensi model LLM GGUF{NC}\n")
            return True
        except Exception as e:
            print(f"  • Verifikasi Import  : {YELLOW}⚠️ Import error: {e}{NC}\n")
            return False
    else:
        print(f"\n{BOLD}{RED}=============================================================================={NC}")
        print(f"{BOLD}{RED}❌ GAGAL: Kompilasi llama-cpp-python berhenti dengan kode {proc.returncode} ({total_time}){NC}")
        print(f"{BOLD}{RED}=============================================================================={NC}")
        print(f"{YELLOW}Cuplikan 10 baris terakhir log kompilasi:{NC}")
        for err_line in captured_logs[-10:]:
            print(f"  {DIM}{err_line}{NC}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Detailed Live Progress Installer for llama-cpp-python")
    parser.add_argument("--force", action="store_true", help="Force reinstall/recompile llama-cpp-python")
    parser.add_argument("--cpu-only", action="store_true", help="Compile with CPU only, without CUDA")
    args = parser.parse_args()

    # Check if already installed
    if not args.force:
        try:
            import llama_cpp
            ver = getattr(llama_cpp, "__version__", "unknown")
            print(f"\n{GREEN}✓ llama-cpp-python (v{ver}) sudah terpasang dan siap digunakan.{NC}")
            print(f"{DIM}Gunakan 'python3 install_llama_cpp.py --force' jika ingin mengompilasi ulang.{NC}\n")
            return 0
        except ImportError:
            pass

    has_cuda, cuda_desc = detect_cuda()
    enable_cuda = has_cuda and not args.cpu_only

    print(f"\n🔍 Deteksi Akselerator: {GREEN}{cuda_desc}{NC}")

    # Step 1: Attempt build
    success = execute_build(enable_cuda=enable_cuda, force=args.force)

    # Step 2: Fallback to CPU if CUDA failed
    if not success and enable_cuda:
        print(f"\n{YELLOW}⚠️  Kompilasi dengan akselerasi CUDA gagal.{NC}")
        print(f"{CYAN}🔄 Mencoba kompilasi ulang dengan backend CPU standar...{NC}")
        success = execute_build(enable_cuda=False, force=args.force)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
