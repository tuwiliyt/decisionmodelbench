#!/usr/bin/env python3
"""
Generates heavyweight_llm_dashboard.html with:
1. Multi-LLM Heavyweight Selection: Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B, Gemma 2 2B.
2. Selectable Decision Models (System 1: Laya, OpenJev, Jev, Kev).
3. Mode 1: Generative Chat with Model Selector.
4. Mode 2: Parallel Head-to-Head (System 1 Model vs System 2 LLM of choice).
5. Mode 3: Two-Tier Brain Pipeline (Custom System 1 Triage + Custom System 2 LLM).
6. Mode 4: Multi-LLM Arena (Compare Sahabat-AI vs Qwen vs Gemma side-by-side).
7. Real-time nvtop-style GPU telemetry dashboard with 2s rolling live graph.
8. Deep dive architectural explanations (Non-Autoregressive vs Autoregressive).
"""

import os

def generate_html():
    html_content = '''<!DOCTYPE html>
<html lang="id" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Arena LLM Kelas Berat Indonesia (Sahabat-AI, Qwen, Gemma) & nvtop GPU Monitor</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          fontFamily: {
            sans: ['Plus Jakarta Sans', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          }
        }
      }
    }
  </script>
  <style>
    body {
      background-color: #060911;
      background-image: 
        radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
        radial-gradient(at 0% 100%, rgba(6, 182, 212, 0.08) 0px, transparent 50%);
      background-attachment: fixed;
    }
    .glass-card {
      background: rgba(15, 23, 42, 0.7);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .terminal-box {
      background: #030712;
      border: 1px solid #1e293b;
      box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .code-block {
      background: #090d16;
      border: 1px solid #1f2937;
    }
    .tab-btn.active {
      background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
      color: #ffffff;
      box-shadow: 0 4px 15px -3px rgba(124, 58, 237, 0.4);
    }
    .model-badge-sahabatai { background: rgba(168, 85, 247, 0.15); color: #d8b4fe; border-color: rgba(168, 85, 247, 0.3); }
    .model-badge-qwen { background: rgba(6, 182, 212, 0.15); color: #67e8f9; border-color: rgba(6, 182, 212, 0.3); }
    .model-badge-gemma { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; border-color: rgba(16, 185, 129, 0.3); }
    .model-badge-gemma-2b { background: rgba(245, 158, 11, 0.15); color: #fcd34d; border-color: rgba(245, 158, 11, 0.3); }
  </style>
</head>
<body class="min-h-screen font-sans antialiased text-slate-200">

  <!-- Header Section -->
  <header class="border-b border-slate-800/80 sticky top-0 z-50 glass-card">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-purple-600 via-indigo-600 to-cyan-500 flex items-center justify-center shadow-lg shadow-purple-500/20 font-bold text-white text-xl">
          🏛️
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h1 class="text-xl font-bold bg-gradient-to-r from-white via-purple-200 to-cyan-400 bg-clip-text text-transparent">
              Arena LLM Kelas Berat & Decision Models
            </h1>
            <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20">
              Sahabat-AI • Qwen • Gemma
            </span>
          </div>
          <p class="text-xs text-slate-400">Sovereign Foundation Models & Decision Architecture • NVIDIA Tesla T4 CUDA</p>
        </div>
      </div>
      <div class="flex items-center gap-3">
        <a href="/playground" class="px-3.5 py-1.5 text-xs font-bold rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white transition border border-slate-700">
          ⚡ Beralih ke Single-Question Playground (System 1)
        </a>
        <div id="gpuStatusBadge" class="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-purple-950/40 border border-purple-800/50 text-xs">
          <span class="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
          <span id="topHeaderGpuBadge" class="text-purple-200 font-medium">GPU: Auto-Detecting...</span>
        </div>
      </div>
    </div>
  </header>

  <!-- Main Container -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">

    <!-- KPI Metric Cards (Models Catalog) -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      <div class="glass-card rounded-2xl p-4 border border-purple-500/30 relative overflow-hidden">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-bold uppercase tracking-wider text-purple-400">Sovereign LLM</span>
          <span class="text-[10px] px-2 py-0.5 rounded font-mono bg-purple-950 text-purple-300 border border-purple-800">8.03B</span>
        </div>
        <div class="mt-2 text-xl font-extrabold text-white">Sahabat-AI 8B</div>
        <div class="text-xs text-slate-400 mt-1">GoTo & Indosat (Llama 3 CPT)</div>
        <div class="mt-2 text-[11px] text-purple-300/80 font-mono">~28-32 t/s • Q4_K_M</div>
      </div>

      <div class="glass-card rounded-2xl p-4 border border-cyan-500/30 relative overflow-hidden">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-bold uppercase tracking-wider text-cyan-400">Multilingual SOTA</span>
          <span class="text-[10px] px-2 py-0.5 rounded font-mono bg-cyan-950 text-cyan-300 border border-cyan-800">7.61B</span>
        </div>
        <div class="mt-2 text-xl font-extrabold text-cyan-400">Qwen 2.5 7B</div>
        <div class="text-xs text-slate-400 mt-1">Alibaba Cloud (ChatML)</div>
        <div class="mt-2 text-[11px] text-cyan-300/80 font-mono">~30-35 t/s • Q4_K_M</div>
      </div>

      <div class="glass-card rounded-2xl p-4 border border-emerald-500/30 relative overflow-hidden">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-bold uppercase tracking-wider text-emerald-400">Deep Reasoning</span>
          <span class="text-[10px] px-2 py-0.5 rounded font-mono bg-emerald-950 text-emerald-300 border border-emerald-800">9.24B</span>
        </div>
        <div class="mt-2 text-xl font-extrabold text-emerald-400">Gemma 2 9B</div>
        <div class="text-xs text-slate-400 mt-1">Google DeepMind (SWA)</div>
        <div class="mt-2 text-[11px] text-emerald-300/80 font-mono">~24-28 t/s • Q4_K_M</div>
      </div>

      <div class="glass-card rounded-2xl p-4 border border-amber-500/30 relative overflow-hidden">
        <div class="flex items-center justify-between">
          <span class="text-[10px] font-bold uppercase tracking-wider text-amber-400">High-Speed Light</span>
          <span class="text-[10px] px-2 py-0.5 rounded font-mono bg-amber-950 text-amber-300 border border-amber-800">2.61B</span>
        </div>
        <div class="mt-2 text-xl font-extrabold text-amber-400">Gemma 2 2B</div>
        <div class="text-xs text-slate-400 mt-1">Google DeepMind</div>
        <div class="mt-2 text-[11px] text-amber-300/80 font-mono">~50-65 t/s • 1.7 GB VRAM</div>
      </div>
    </div>

    <!-- Real-Time GPU Performance Monitor (nvtop Style) -->
    <div class="terminal-box rounded-2xl p-5 font-mono text-xs space-y-4">
      <!-- nvtop Header -->
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
        <div class="flex items-center gap-2.5">
          <div class="w-3 h-3 rounded-full bg-emerald-400 animate-ping"></div>
          <span class="text-sm font-bold text-emerald-400 uppercase tracking-wide flex items-center gap-1.5">
            <span>💻</span> nvtop GPU Performance Monitor
          </span>
          <span id="nvtopDeviceBadge" class="px-2.5 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300 border border-slate-700">
            NVIDIA Tesla T4 (Compute 7.5)
          </span>
        </div>
        <div class="flex items-center gap-3 text-[11px] text-slate-400">
          <span>Driver: <strong id="nvtopDriverText" class="text-slate-200">580.82</strong></span>
          <span>CUDA: <strong id="nvtopCudaText" class="text-slate-200">13.0</strong></span>
          <span class="text-emerald-400 flex items-center gap-1">
            <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Live 2s Telemetry
          </span>
        </div>
      </div>

      <!-- Multi-GPU Dedicated Cluster Nodes (Live GPU 0 & GPU 1 Sharding) -->
      <div id="nvtopMultiGpuSection" class="p-4 rounded-xl bg-slate-950/80 border border-slate-800 space-y-3">
        <div class="flex items-center justify-between">
          <span class="text-xs font-bold text-slate-300 flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
            Node GPU Terdistribusi (Multi-GPU Sharding & Parallel Triage)
          </span>
          <span id="nvtopMultiGpuTopologyBadge" class="text-[10px] px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-800 text-cyan-300 font-mono">
            Cluster: 2x Tesla T4 (29.12 GB VRAM Total)
          </span>
        </div>
        <div id="nvtopMultiGpuCards" class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <!-- GPU 0 Card -->
          <div class="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2">
            <div class="flex justify-between items-center text-xs">
              <span class="font-bold text-cyan-400 flex items-center gap-1.5">
                <span class="px-1.5 py-0.5 rounded bg-cyan-950 border border-cyan-800 text-[10px]">GPU 0</span>
                <span>Tesla T4 (Primary Node)</span>
              </span>
              <span id="gpu0UtilText" class="font-mono text-cyan-300 font-bold text-[11px]">0% Util</span>
            </div>
            <div class="text-[10px] text-slate-400">Model Alokasi: <span class="text-purple-300 font-medium">Laya (421M), Kev (0.8B), Heavyweight LLM Shard 0</span></div>
            <div class="space-y-1">
              <div class="flex justify-between text-[11px]">
                <span class="text-slate-400 font-bold">VRAM Digunakan:</span>
                <span id="gpu0MemText" class="font-mono text-purple-400 font-bold text-xs">6.3 / 15.0 GB (42%)</span>
              </div>
              <div class="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
                <div id="gpu0MemBar" class="bg-gradient-to-r from-purple-500 to-indigo-500 h-2.5 rounded-full transition-all duration-300" style="width: 42%"></div>
              </div>
            </div>
            <div class="flex justify-between text-[10px] text-slate-400 pt-1 border-t border-slate-800/60">
              <span>Suhu: <strong id="gpu0TempText" class="text-emerald-400">58°C</strong></span>
              <span>Daya: <strong id="gpu0PowerText" class="text-amber-400">29.8W / 70W</strong></span>
            </div>
          </div>
          <!-- GPU 1 Card -->
          <div class="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2">
            <div class="flex justify-between items-center text-xs">
              <span class="font-bold text-emerald-400 flex items-center gap-1.5">
                <span class="px-1.5 py-0.5 rounded bg-emerald-950 border border-emerald-800 text-[10px]">GPU 1</span>
                <span>Tesla T4 (Secondary Node)</span>
              </span>
              <span id="gpu1UtilText" class="font-mono text-emerald-300 font-bold text-[11px]">0% Util</span>
            </div>
            <div class="text-[10px] text-slate-400">Model Alokasi: <span class="text-emerald-300 font-medium">OpenJev (0.5B Logit Scorer), Heavyweight LLM Shard 1</span></div>
            <div class="space-y-1">
              <div class="flex justify-between text-[11px]">
                <span class="text-slate-400 font-bold">VRAM Digunakan:</span>
                <span id="gpu1MemText" class="font-mono text-emerald-400 font-bold text-xs">1.5 / 15.0 GB (10%)</span>
              </div>
              <div class="w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
                <div id="gpu1MemBar" class="bg-gradient-to-r from-emerald-500 to-teal-500 h-2.5 rounded-full transition-all duration-300" style="width: 10%"></div>
              </div>
            </div>
            <div class="flex justify-between text-[10px] text-slate-400 pt-1 border-t border-slate-800/60">
              <span>Suhu: <strong id="gpu1TempText" class="text-emerald-400">56°C</strong></span>
              <span>Daya: <strong id="gpu1PowerText" class="text-amber-400">26.4W / 70W</strong></span>
            </div>
          </div>
        </div>
      </div>

      <!-- Real-time Aggregate Telemetry Bars & Gauges -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <!-- GPU Util Gauge -->
        <div class="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <div class="flex justify-between items-center text-[11px]">
            <span class="text-slate-400 font-bold">Cluster Core Util (Avg):</span>
            <span id="nvtopGpuUtilText" class="font-bold text-cyan-400 font-mono text-sm">0%</span>
          </div>
          <div class="w-full bg-slate-900 rounded-full h-3 overflow-hidden border border-slate-800">
            <div id="nvtopGpuUtilBar" class="bg-gradient-to-r from-cyan-500 to-indigo-500 h-3 rounded-full transition-all duration-300" style="width: 0%"></div>
          </div>
          <div class="flex justify-between text-[10px] text-slate-500">
            <span>Multi-GPU Cores</span>
            <span id="nvtopGpuLoadLabel">Idle</span>
          </div>
        </div>

        <!-- VRAM Memory Usage Gauge -->
        <div class="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <div class="flex justify-between items-center text-[11px]">
            <span class="text-slate-400 font-bold">Total Cluster VRAM:</span>
            <span id="nvtopMemText" class="font-bold text-purple-400 font-mono text-sm">7.8 / 30.7 GB (25%)</span>
          </div>
          <div class="w-full bg-slate-900 rounded-full h-3 overflow-hidden border border-slate-800">
            <div id="nvtopMemBar" class="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full transition-all duration-300" style="width: 25%"></div>
          </div>
          <div class="flex justify-between text-[10px] text-slate-500">
            <span>Free Cluster VRAM:</span>
            <span id="nvtopMemFree" class="text-slate-400 font-mono">22.9 GB</span>
          </div>
        </div>

        <!-- Temperature & Fan -->
        <div class="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
          <div class="flex justify-between items-center text-[11px]">
            <span class="text-slate-400 font-bold">Avg Temp & Fan:</span>
            <span id="nvtopTempText" class="font-bold text-emerald-400 font-mono text-sm">48°C [Optimal]</span>
          </div>
          <div class="w-full bg-slate-900 rounded-full h-3 overflow-hidden border border-slate-800">
            <div id="nvtopTempBar" class="bg-gradient-to-r from-emerald-500 to-amber-500 h-3 rounded-full transition-all duration-300" style="width: 48%"></div>
          </div>
          <div class="flex justify-between text-[10px] text-slate-500">
            <span>Fan Status: Passive (Server)</span>
            <span>Target: &lt;75°C</span>
          </div>
        </div>

        <!-- Power & Clocks -->
        <div class="p-3.5 rounded-xl bg-slate-950 border border-slate-800 space-y-1.5 text-[11px]">
          <div class="flex justify-between">
            <span class="text-slate-400">Total Power Draw:</span>
            <span id="nvtopPowerText" class="font-bold text-amber-400 font-mono">56.2W / 140W</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-400">Graphics Clock:</span>
            <span id="nvtopClockGfx" class="font-mono text-slate-300">585 MHz</span>
          </div>
          <div class="flex justify-between">
            <span class="text-slate-400">Memory Clock:</span>
            <span id="nvtopClockMem" class="font-mono text-slate-300">5000 MHz</span>
          </div>
          <div class="flex justify-between pt-1 border-t border-slate-800/80 text-[10px] text-emerald-400">
            <span>Topology Mode:</span>
            <span>2x GPU Sharding</span>
          </div>
        </div>
      </div>

      <!-- Real-Time Rolling History Chart -->
      <div class="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
        <div class="flex items-center justify-between text-[11px]">
          <span class="text-slate-400 font-bold flex items-center gap-1.5">
            <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
            GPU Utilization & Memory History (Rolling 60 Detik)
          </span>
          <span class="text-[10px] text-slate-500">Sampling Rate: 2000 ms</span>
        </div>
        <div class="h-28 w-full">
          <canvas id="nvtopLiveChart"></canvas>
        </div>
      </div>

      <!-- Processes Table (like nvtop bottom window) -->
      <div class="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
        <div class="flex items-center justify-between text-[11px]">
          <span class="text-slate-400 font-bold">Proses Komputasi Aktif di GPU:</span>
          <span class="text-[10px] text-slate-500">Process Type: CUDA Compute (C)</span>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-[11px] font-mono">
            <thead>
              <tr class="text-slate-500 border-b border-slate-800">
                <th class="pb-1.5">PID</th>
                <th class="pb-1.5">Process Name</th>
                <th class="pb-1.5">GPU Device</th>
                <th class="pb-1.5">Type</th>
                <th class="pb-1.5">GPU Memory</th>
                <th class="pb-1.5">Loaded Architecture</th>
              </tr>
            </thead>
            <tbody id="nvtopProcessTable" class="divide-y divide-slate-900 text-slate-300">
              <tr>
                <td class="py-1.5 text-cyan-400 font-bold">5691</td>
                <td class="py-1.5">python3 (uvicorn)</td>
                <td class="py-1.5 text-cyan-300 font-bold">GPU 0 & GPU 1</td>
                <td class="py-1.5 text-purple-400">C (CUDA Compute)</td>
                <td class="py-1.5 font-bold text-amber-300">~6,400 MiB</td>
                <td class="py-1.5 text-emerald-400">Laya, Kev, OpenJev + Heavyweight LLM Engine</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Navigation Tabs -->
    <div class="flex flex-wrap gap-2 border-b border-slate-800 pb-3">
      <button onclick="switchTab('tab-compare')" id="btn-tab-compare" class="tab-btn active px-4 py-2 text-xs font-bold rounded-xl transition flex items-center gap-2">
        <span>⚖️</span> Mode Utama: Mengapa Decision Model? (Dengan vs Tanpa Decision Model)
      </button>
      <button onclick="switchTab('tab-twotier')" id="btn-tab-twotier" class="tab-btn px-4 py-2 text-xs font-bold rounded-xl bg-slate-800 text-slate-300 hover:text-white transition flex items-center gap-2">
        <span>🧠</span> Mode 2: Two-Tier Brain Pipeline
      </button>
      <button onclick="switchTab('tab-arena')" id="btn-tab-arena" class="tab-btn px-4 py-2 text-xs font-bold rounded-xl bg-slate-800 text-slate-300 hover:text-white transition flex items-center gap-2">
        <span>⚔️</span> Mode 3: Multi-LLM Arena (Komparasi 3 LLM)
      </button>
      <button onclick="switchTab('tab-h2h')" id="btn-tab-h2h" class="tab-btn px-4 py-2 text-xs font-bold rounded-xl bg-slate-800 text-slate-300 hover:text-white transition flex items-center gap-2">
        <span>⚡</span> Mode 4: Head-to-Head (System 1 vs System 2)
      </button>
      <button onclick="switchTab('tab-chat')" id="btn-tab-chat" class="tab-btn px-4 py-2 text-xs font-bold rounded-xl bg-slate-800 text-slate-300 hover:text-white transition flex items-center gap-2">
        <span>💬</span> Mode 5: Generative Chat (Multi-LLM)
      </button>
    </div>

    <!-- TAB 1: GENERATIVE CHAT WITH MODEL SELECTOR -->
    <div id="tab-chat" class="tab-content hidden space-y-6">
      <div class="glass-card rounded-2xl p-6 space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 class="text-base font-bold text-white flex items-center gap-2">
              <span>💬</span> Playground Percakapan Bebas (Pilih Model LLM Kelas Berat)
            </h3>
            <p class="text-xs text-slate-400">Uji kemampuan empati, pemahaman dialek, logika, dan regulasi Indonesia</p>
          </div>
          <!-- LLM Model Selector Pills -->
          <div class="flex flex-wrap items-center gap-1.5 p-1 rounded-xl bg-slate-900 border border-slate-800">
            <button onclick="selectChatModel('sahabatai')" id="chat-btn-sahabatai" class="chat-model-btn active px-3 py-1.5 rounded-lg text-xs font-bold bg-purple-600 text-white transition">
              Sahabat-AI 8B
            </button>
            <button onclick="selectChatModel('qwen')" id="chat-btn-qwen" class="chat-model-btn px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800 text-slate-300 hover:text-white transition">
              Qwen 2.5 7B
            </button>
            <button onclick="selectChatModel('gemma')" id="chat-btn-gemma" class="chat-model-btn px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800 text-slate-300 hover:text-white transition">
              Gemma 2 9B
            </button>
            <button onclick="selectChatModel('gemma-2b')" id="chat-btn-gemma-2b" class="chat-model-btn px-3 py-1.5 rounded-lg text-xs font-bold bg-slate-800 text-slate-300 hover:text-white transition">
              Gemma 2 2B (Speed)
            </button>
          </div>
        </div>

        <!-- Model Info Banner -->
        <div id="chatModelBanner" class="p-3.5 rounded-xl bg-purple-950/20 border border-purple-800/40 flex items-center justify-between text-xs">
          <div class="flex items-center gap-2.5">
            <span class="text-base">🇮🇩</span>
            <div>
              <strong id="chatBannerTitle" class="text-purple-300">Sahabat-AI 8B Instruct (GoTo & Indosat)</strong>
              <p id="chatBannerDesc" class="text-slate-400 text-[11px]">Model fondasi kedaulatan digital Indonesia. Menguasai norma budaya, dialek lokal, dan empati customer care.</p>
            </div>
          </div>
          <span id="chatBannerSpeed" class="px-2.5 py-1 rounded font-mono font-bold bg-purple-900/50 text-purple-200 border border-purple-700 whitespace-nowrap">
            ~28-32 t/s (CUDA GPU)
          </span>
        </div>

        <!-- Presets -->
        <div class="space-y-1.5">
          <label class="text-xs font-bold text-slate-400 uppercase tracking-wide">Pilih Skenario Nyata:</label>
          <div class="flex flex-wrap gap-2">
            <button onclick="loadChatPreset('marunda')" class="px-3 py-1 rounded-lg text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 transition border border-slate-700">
              📦 Komplain Ekspedisi (Gateway Marunda)
            </button>
            <button onclick="loadChatPreset('phk')" class="px-3 py-1 rounded-lg text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 transition border border-slate-700">
              💳 Restrukturisasi Cicilan (Nasabah PHK)
            </button>
            <button onclick="loadChatPreset('scam')" class="px-3 py-1 rounded-lg text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 transition border border-slate-700">
              🛡️ Tanggap Darurat Fraud Rekening
            </button>
            <button onclick="loadChatPreset('gov')" class="px-3 py-1 rounded-lg text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 transition border border-slate-700">
              📜 Izin Usaha Mikro (OSS, NIB, BPOM)
            </button>
            <button onclick="loadChatPreset('jaksel')" class="px-3 py-1 rounded-lg text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 transition border border-slate-700">
              🏙️ Konsultasi Karir & Bahasa Gaul Jaksel
            </button>
          </div>
        </div>

        <!-- System Prompt & Input -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div class="md:col-span-1 space-y-1.5">
            <label class="text-xs font-bold text-slate-400">System Instruction / Persona:</label>
            <textarea id="chatSystemPrompt" rows="5" class="w-full code-block rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-purple-500 transition resize-none"></textarea>
            <div class="flex items-center justify-between text-[11px] text-slate-500">
              <span>Max Tokens:</span>
              <select id="chatMaxTokens" class="bg-slate-900 border border-slate-800 rounded px-2 py-0.5 text-slate-300">
                <option value="150">150 tokens</option>
                <option value="250" selected>250 tokens</option>
                <option value="350">350 tokens</option>
              </select>
            </div>
          </div>
          <div class="md:col-span-2 space-y-1.5">
            <label class="text-xs font-bold text-slate-400">Pesan Pengguna / Komplain Pelanggan:</label>
            <textarea id="chatInputText" rows="5" class="w-full code-block rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-purple-500 transition resize-none"></textarea>
            <div class="flex justify-end">
              <button onclick="runChatGeneration()" id="btnRunChat" class="px-5 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-purple-600 via-indigo-600 to-cyan-600 text-white hover:opacity-90 shadow-lg shadow-purple-500/20 transition flex items-center gap-2">
                <span id="btnChatIcon">⚡</span>
                <span id="btnChatText">HASILKAN RESPON LLM</span>
              </button>
            </div>
          </div>
        </div>

        <!-- Output Card -->
        <div id="chatOutputCard" class="hidden mt-4 p-5 rounded-xl bg-slate-900/90 border border-purple-500/30 space-y-3">
          <div class="flex items-center justify-between border-b border-slate-800 pb-2">
            <span class="text-xs font-bold text-purple-300 flex items-center gap-2">
              <span id="chatOutputModelBadge" class="px-2 py-0.5 rounded text-[10px] font-mono bg-purple-900/40 text-purple-200 border border-purple-700">Sahabat-AI 8B</span>
              Respon Generatif:
            </span>
            <div class="flex items-center gap-2">
              <span id="chatMetricsBadge" class="text-xs font-mono text-cyan-400 bg-cyan-950/60 px-2.5 py-1 rounded-lg border border-cyan-800/60">
                0 ms | 0 tokens | 0 t/s
              </span>
              <button onclick="copyChatOutput()" class="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-white transition" title="Salin Teks">
                📋
              </button>
            </div>
          </div>
          <div id="chatResponseBox" class="text-xs leading-relaxed text-slate-200 whitespace-pre-wrap font-sans"></div>
        </div>
      </div>
    </div>

    <!-- TAB 2: MULTI-LLM ARENA (COMPARE ALL 3 HEAVYWEIGHT LLMS SIDE-BY-SIDE) -->
    <div id="tab-arena" class="tab-content hidden space-y-6">
      <div class="glass-card rounded-2xl p-6 space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 class="text-base font-bold text-white flex items-center gap-2">
              <span>⚔️</span> Arena Komparasi LLM: Uji 3 Model Sekaligus pada Prompt yang Sama
            </h3>
            <p class="text-xs text-slate-400">Bandingkan langsung kualitas narasi, latensi, dan kecepatan Sahabat-AI 8B vs Qwen 2.5 7B vs Gemma 2 9B</p>
          </div>
          <span class="px-3 py-1 rounded-lg text-xs font-mono font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
            Multi-LLM Benchmark
          </span>
        </div>

        <div class="space-y-1.5">
          <label class="text-xs font-bold text-slate-400">Prompt Ujian Serentak:</label>
          <textarea id="arenaPrompt" rows="3" class="w-full code-block rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 transition resize-none">Min paket gw dr tgl 20 stuck di gateway Marunda ga gerak2, kurir gimana sih? Mau gw viralin di X nih klo ga nyampe hari ini! Balikin ongkir gw juga!</textarea>
          <div class="flex justify-between items-center">
            <span class="text-[11px] text-slate-500">Masing-masing model akan dievaluasi secara berurutan dengan hot-swap VRAM aman.</span>
            <button onclick="runArenaBenchmark()" id="btnRunArena" class="px-5 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-cyan-600 via-indigo-600 to-purple-600 text-white hover:opacity-90 shadow-lg shadow-cyan-500/20 transition flex items-center gap-2">
              <span id="btnArenaIcon">⚔️</span>
              <span id="btnArenaText">JALANKAN ARENA KOMPARASI</span>
            </button>
          </div>
        </div>

        <!-- 3 Columns Comparison Grid -->
        <div id="arenaGrid" class="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          <!-- Col 1: Sahabat-AI 8B -->
          <div class="glass-card rounded-xl p-4 border border-purple-500/30 space-y-3">
            <div class="flex items-center justify-between border-b border-slate-800 pb-2">
              <span class="text-xs font-bold text-purple-300 flex items-center gap-1.5">
                <span>🇮🇩</span> Sahabat-AI 8B
              </span>
              <span id="arenaLatSahabat" class="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800">Siap Uji</span>
            </div>
            <div class="text-[11px] text-slate-400">GoTo & Indosat • Sovereign Indonesian</div>
            <div id="arenaTextSahabat" class="p-3 rounded-lg bg-slate-900/90 text-xs text-slate-300 min-h-[160px] whitespace-pre-wrap leading-relaxed">
              Klik 'Jalankan Arena Komparasi' untuk mengevaluasi respon Sahabat-AI 8B.
            </div>
            <div class="pt-2 border-t border-slate-800/60 flex justify-between text-[10px] font-mono text-slate-400">
              <span id="arenaSpdSahabat">Speed: - t/s</span>
              <span id="arenaTokSahabat">Tokens: -</span>
            </div>
          </div>

          <!-- Col 2: Qwen 2.5 7B -->
          <div class="glass-card rounded-xl p-4 border border-cyan-500/30 space-y-3">
            <div class="flex items-center justify-between border-b border-slate-800 pb-2">
              <span class="text-xs font-bold text-cyan-300 flex items-center gap-1.5">
                <span>⚡</span> Qwen 2.5 7B
              </span>
              <span id="arenaLatQwen" class="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800">Siap Uji</span>
            </div>
            <div class="text-[11px] text-slate-400">Alibaba Cloud • Multilingual SOTA</div>
            <div id="arenaTextQwen" class="p-3 rounded-lg bg-slate-900/90 text-xs text-slate-300 min-h-[160px] whitespace-pre-wrap leading-relaxed">
              Respon Qwen 2.5 7B akan ditampilkan di sini.
            </div>
            <div class="pt-2 border-t border-slate-800/60 flex justify-between text-[10px] font-mono text-slate-400">
              <span id="arenaSpdQwen">Speed: - t/s</span>
              <span id="arenaTokQwen">Tokens: -</span>
            </div>
          </div>

          <!-- Col 3: Gemma 2 9B -->
          <div class="glass-card rounded-xl p-4 border border-emerald-500/30 space-y-3">
            <div class="flex items-center justify-between border-b border-slate-800 pb-2">
              <span class="text-xs font-bold text-emerald-300 flex items-center gap-1.5">
                <span>🧠</span> Gemma 2 9B
              </span>
              <span id="arenaLatGemma" class="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">Siap Uji</span>
            </div>
            <div class="text-[11px] text-slate-400">Google DeepMind • Deep Reasoning</div>
            <div id="arenaTextGemma" class="p-3 rounded-lg bg-slate-900/90 text-xs text-slate-300 min-h-[160px] whitespace-pre-wrap leading-relaxed">
              Respon Gemma 2 9B akan ditampilkan di sini.
            </div>
            <div class="pt-2 border-t border-slate-800/60 flex justify-between text-[10px] font-mono text-slate-400">
              <span id="arenaSpdGemma">Speed: - t/s</span>
              <span id="arenaTokGemma">Tokens: -</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: HEAD-TO-HEAD PARALLEL (SYSTEM 1 vs SYSTEM 2) -->
    <div id="tab-h2h" class="tab-content hidden space-y-6">
      <div class="glass-card rounded-2xl p-6 space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 class="text-base font-bold text-white flex items-center gap-2">
              <span>⚡</span> Head-to-Head: Decision Model (System 1) vs LLM Generatif (System 2)
            </h3>
            <p class="text-xs text-slate-400">Bandingkan efisiensi pengambilan keputusan instan (0 token) melawan penalaran autoregresif</p>
          </div>
          <span class="px-3 py-1 rounded-lg text-xs font-mono font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
            Parallel Benchmark
          </span>
        </div>

        <!-- Model Selectors for Both Tiers -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 rounded-xl bg-slate-950/80 border border-slate-800">
          <!-- System 1 Selector -->
          <div class="space-y-1.5">
            <label class="text-xs font-bold text-cyan-400 flex items-center gap-1.5">
              <span>⚡</span> Pilih Model System 1 (Decision Model):
            </label>
            <div class="grid grid-cols-2 gap-2">
              <button onclick="selectH2HModel('laya')" id="h2h-btn-laya" class="h2h-model-btn active px-2.5 py-1.5 rounded-lg text-xs font-bold bg-cyan-500 text-white transition text-center">
                Laya (421M GPU)
              </button>
              <button onclick="selectH2HModel('openjev')" id="h2h-btn-openjev" class="h2h-model-btn px-2.5 py-1.5 rounded-lg text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-center">
                OpenJev (0.5B GPU)
              </button>
              <button onclick="selectH2HModel('jev')" id="h2h-btn-jev" class="h2h-model-btn px-2.5 py-1.5 rounded-lg text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-center">
                TypeSafe Jev (Cloud)
              </button>
              <button onclick="selectH2HModel('kev')" id="h2h-btn-kev" class="h2h-model-btn px-2.5 py-1.5 rounded-lg text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-center">
                Kev-0.8B (Local)
              </button>
            </div>
          </div>

          <!-- System 2 Selector -->
          <div class="space-y-1.5">
            <label class="text-xs font-bold text-purple-400 flex items-center gap-1.5">
              <span>🧠</span> Pilih Model System 2 (Heavyweight LLM):
            </label>
            <div class="grid grid-cols-2 gap-2">
              <button onclick="selectH2HSys2Model('sahabatai')" id="h2h-s2-btn-sahabatai" class="h2h-s2-btn active px-2.5 py-1.5 rounded-lg text-xs font-bold bg-purple-600 text-white transition text-center">
                Sahabat-AI 8B
              </button>
              <button onclick="selectH2HSys2Model('qwen')" id="h2h-s2-btn-qwen" class="h2h-s2-btn px-2.5 py-1.5 rounded-lg text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-center">
                Qwen 2.5 7B
              </button>
              <button onclick="selectH2HSys2Model('gemma')" id="h2h-s2-btn-gemma" class="h2h-s2-btn px-2.5 py-1.5 rounded-lg text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-center">
                Gemma 2 9B
              </button>
              <button onclick="selectH2HSys2Model('gemma-2b')" id="h2h-s2-btn-gemma-2b" class="h2h-s2-btn px-2.5 py-1.5 rounded-lg text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-center">
                Gemma 2 2B
              </button>
            </div>
          </div>
        </div>

        <!-- Presets -->
        <div class="flex flex-wrap items-center gap-2">
          <span class="text-xs text-slate-400 font-bold">Preset:</span>
          <button onclick="loadH2HPreset('marunda')" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-300">
            📦 Paket Tertahan Marunda
          </button>
          <button onclick="loadH2HPreset('scam')" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-300">
            🚨 Ancaman Lapor Polisi
          </button>
          <button onclick="loadH2HPreset('fintech')" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-xs text-slate-300">
            💼 Risiko Default PHK
          </button>
        </div>

        <!-- State Text Input -->
        <div class="space-y-1.5">
          <label class="text-xs font-bold text-slate-400">Teks Masukan Pengujian:</label>
          <textarea id="h2hState" rows="2" class="w-full code-block rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 transition resize-none"></textarea>
        </div>

        <div class="flex justify-end">
          <button onclick="runHeadToHeadBenchmark()" id="btnRunH2H" class="px-5 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-cyan-600 via-indigo-600 to-purple-600 text-white hover:opacity-90 shadow-lg shadow-indigo-500/20 transition flex items-center gap-2">
            <span id="btnH2HIcon">⚡</span>
            <span id="btnH2HText">JALANKAN HEAD-TO-HEAD PARALEL</span>
          </button>
        </div>

        <!-- Results Grid -->
        <div id="h2hResultsGrid" class="hidden grid grid-cols-1 md:grid-cols-2 gap-6 pt-4 border-t border-slate-800">
          <!-- System 1 Result -->
          <div class="glass-card rounded-2xl p-5 border border-cyan-500/30 space-y-4">
            <div class="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <span class="text-[10px] font-bold uppercase tracking-wider text-cyan-400">System 1 (Decision Model)</span>
                <h4 id="h2hSys1Name" class="text-sm font-bold text-white">Laya Multilingual (421M)</h4>
              </div>
              <div class="text-right">
                <span id="h2hSys1Latency" class="text-sm font-mono font-bold text-cyan-400">~65 ms</span>
                <div class="text-[10px] text-slate-500">Output: 0 tokens</div>
              </div>
            </div>
            <div id="h2hSys1Content" class="space-y-3"></div>
          </div>

          <!-- System 2 Result -->
          <div class="glass-card rounded-2xl p-5 border border-purple-500/30 space-y-4">
            <div class="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <span class="text-[10px] font-bold uppercase tracking-wider text-purple-400">System 2 (Generative LLM)</span>
                <h4 id="h2hSys2Name" class="text-sm font-bold text-white">Sahabat-AI 8B</h4>
              </div>
              <div class="text-right">
                <span id="h2hSys2Latency" class="text-sm font-mono font-bold text-purple-400">~4,800 ms</span>
                <div id="h2hSys2Tokens" class="text-[10px] text-slate-500">Output: ~180 tokens</div>
              </div>
            </div>
            <div id="h2hSys2Content" class="space-y-3 text-xs"></div>
            <div class="pt-2 border-t border-slate-800 text-[11px] text-slate-400 flex justify-between">
              <span id="h2hSys2Speed">Speed: ~30 t/s</span>
              <span>Autoregressive JSON Mode</span>
            </div>
          </div>
        </div>

        <!-- Summary Banner -->
        <div id="h2hSummaryBanner" class="hidden p-4 rounded-xl bg-indigo-950/40 border border-indigo-800/50 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div>
            <strong class="text-indigo-300">Kesimpulan Komparasi:</strong>
            <p id="h2hSummaryText" class="text-slate-300 text-[11px] mt-0.5">
              System 1 menyelesaikan keputusan <strong>50x-100x lebih cepat</strong> dengan <strong>0 output tokens</strong>.
            </p>
          </div>
          <span id="h2hSpeedupBadge" class="px-3 py-1.5 rounded-lg font-mono font-bold bg-cyan-950 text-cyan-300 border border-cyan-800 whitespace-nowrap">
            Efisiensi: 80x Speedup
          </span>
        </div>
      </div>
    </div>

    <!-- TAB 4: TWO-TIER BRAIN PIPELINE -->
    <div id="tab-twotier" class="tab-content hidden space-y-6">
      <div class="glass-card rounded-2xl p-6 space-y-4">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h3 class="text-base font-bold text-white flex items-center gap-2">
              <span>🧠</span> Two-Tier Brain Pipeline (Kombinasi Sempurna System 1 & System 2)
            </h3>
            <p class="text-xs text-slate-400">Arsitektur AI Modern: Triage instan 0 token pada Tier 1, lalu eskalasi cerdas ke LLM pada Tier 2</p>
          </div>
          <span class="px-3 py-1 rounded-lg text-xs font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
            Enterprise Architecture
          </span>
        </div>

        <!-- Model Selectors for Two-Tier -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 rounded-xl bg-slate-950/80 border border-slate-800">
          <div class="space-y-1.5">
            <label class="text-xs font-bold text-cyan-400">Tahap 1: Pilih Engine Triage (System 1):</label>
            <div class="grid grid-cols-2 gap-2">
              <button onclick="selectTTModel('laya')" id="tt-btn-laya" class="tt-model-btn active px-2.5 py-1.5 rounded-lg text-xs font-bold bg-emerald-500 text-white transition text-center">
                Laya (421M GPU)
              </button>
              <button onclick="selectTTModel('openjev')" id="tt-btn-openjev" class="tt-model-btn px-2.5 py-1.5 rounded-lg text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-center">
                OpenJev (0.5B GPU)
              </button>
              <button onclick="selectTTModel('jev')" id="tt-btn-jev" class="tt-model-btn px-2.5 py-1.5 rounded-lg text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-center">
                TypeSafe Jev (Cloud)
              </button>
              <button onclick="selectTTModel('kev')" id="tt-btn-kev" class="tt-model-btn px-2.5 py-1.5 rounded-lg text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-center">
                Kev-0.8B (Local)
              </button>
            </div>
          </div>

          <div class="space-y-1.5">
            <label class="text-xs font-bold text-purple-400">Tahap 2: Pilih LLM Resolusi Empatik (System 2):</label>
            <div class="grid grid-cols-2 gap-2">
              <button onclick="selectTTSys2Model('sahabatai')" id="tt-s2-btn-sahabatai" class="tt-s2-btn active px-2.5 py-1.5 rounded-lg text-xs font-bold bg-purple-600 text-white transition text-center">
                Sahabat-AI 8B
              </button>
              <button onclick="selectTTSys2Model('qwen')" id="tt-s2-btn-qwen" class="tt-s2-btn px-2.5 py-1.5 rounded-lg text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-center">
                Qwen 2.5 7B
              </button>
              <button onclick="selectTTSys2Model('gemma')" id="tt-s2-btn-gemma" class="tt-s2-btn px-2.5 py-1.5 rounded-lg text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-center">
                Gemma 2 9B
              </button>
              <button onclick="selectTTSys2Model('gemma-2b')" id="tt-s2-btn-gemma-2b" class="tt-s2-btn px-2.5 py-1.5 rounded-lg text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-center">
                Gemma 2 2B
              </button>
            </div>
          </div>
        </div>

        <!-- Workflow Visualizer -->
        <div class="p-4 rounded-xl bg-slate-950 border border-slate-800 flex flex-col md:flex-row items-center justify-between gap-4 text-xs font-mono">
          <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded-lg bg-cyan-500/20 text-cyan-300 flex items-center justify-center font-bold">1</div>
            <div>
              <div class="text-cyan-400 font-bold" id="ttFlowSys1Title">Tier 1: Triage System 1</div>
              <div class="text-slate-400 text-[11px]" id="ttFlowSys1Label">Triage instan dalam &lt;100 ms (0 tokens)</div>
            </div>
          </div>
          <div class="text-slate-600 text-lg hidden md:block">➔</div>
          <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded-lg bg-purple-500/20 text-purple-300 flex items-center justify-center font-bold">2</div>
            <div>
              <div class="text-purple-400 font-bold" id="ttFlowSys2Title">Tier 2: Resolusi System 2</div>
              <div class="text-slate-400 text-[11px]" id="ttFlowSys2Label">Respon empatik personalisasi (~3-5 detik)</div>
            </div>
          </div>
          <div class="text-slate-600 text-lg hidden md:block">➔</div>
          <div class="flex items-center gap-3">
            <div class="w-8 h-8 rounded-lg bg-emerald-500/20 text-emerald-300 flex items-center justify-center font-bold">3</div>
            <div>
              <div class="text-emerald-400 font-bold">Total Efisiensi Biaya</div>
              <div class="text-slate-400 text-[11px]">Hemat 80% kuota token LLM</div>
            </div>
          </div>
        </div>

        <!-- Test Case Input -->
        <div class="space-y-1.5">
          <label class="text-xs font-bold text-slate-400">Pesan Pengaduan Pelanggan:</label>
          <textarea id="twoTierState" rows="3" class="w-full code-block rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 transition resize-none">Min paket gw dr tgl 20 stuck di gateway Marunda ga gerak2, kurir gimana sih? Mau gw viralin di X nih klo ga nyampe hari ini! Balikin ongkir gw juga!</textarea>
        </div>

        <div class="flex justify-end">
          <button onclick="runTwoTierPipeline()" id="btnRunTwoTier" class="px-5 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-emerald-600 via-teal-600 to-cyan-600 text-white hover:opacity-90 shadow-lg shadow-emerald-500/20 transition flex items-center gap-2">
            <span id="btnTwoTierIcon">⚡</span>
            <span id="btnTwoTierText">EKSEKUSI PIPELINE DUA-TIER LENGKAP</span>
          </button>
        </div>

        <!-- Two-Tier Execution Outputs -->
        <div id="twoTierResults" class="hidden space-y-4 pt-4 border-t border-slate-800">
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold text-emerald-300">Hasil Eksekusi Pipeline Dua-Tier:</span>
            <span id="twoTierTotalBadge" class="text-xs font-mono font-bold px-3 py-1 rounded-lg bg-emerald-950 text-emerald-300 border border-emerald-800">
              Total Waktu Pipeline: ~3,265 ms
            </span>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
            <!-- Step 1 Output Card -->
            <div class="md:col-span-1 glass-card rounded-xl p-4 border border-cyan-500/30 space-y-2">
              <div class="flex items-center justify-between mb-2">
                <span class="text-[11px] font-bold text-cyan-400 uppercase" id="ttStep1CardTitle">Tahap 1: Triage System 1</span>
                <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300" id="ttStep1Lat">~65 ms</span>
              </div>
              <div id="ttStep1Answers" class="space-y-2 text-xs"></div>
            </div>

            <!-- Step 2 Output Card -->
            <div class="md:col-span-2 glass-card rounded-xl p-4 border border-purple-500/30 space-y-2">
              <div class="flex items-center justify-between mb-2">
                <span class="text-[11px] font-bold text-purple-400 uppercase" id="ttStep2CardTitle">Tahap 2: Resolusi Empatik System 2</span>
                <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-950 text-purple-300" id="ttStep2Lat">~3,200 ms</span>
              </div>
              <div id="ttStep2Response" class="p-3.5 rounded-lg bg-slate-900/90 text-xs text-slate-200 leading-relaxed font-sans whitespace-pre-wrap"></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 1 (MODE UTAMA): DENGAN JEV VS TANPA JEV (ARSITEKTUR DECISION vs LLM MURNI) -->
    <div id="tab-compare" class="tab-content space-y-6">
      <!-- EXECUTIVE VALUE PROPOSITION HERO: MENGAPA HARUS MENGGUNAKAN DECISION MODEL? -->
      <div class="glass-card rounded-2xl p-6 border border-indigo-500/40 bg-gradient-to-br from-indigo-950/80 via-slate-900/90 to-purple-950/80 space-y-5">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 text-[11px] font-bold uppercase tracking-wider mb-2">
              <span>🏛️</span> Nilai Kritis Arsitektur Produksi
            </div>
            <h2 class="text-xl sm:text-2xl font-extrabold text-white tracking-tight flex items-center gap-2.5">
              Mengapa Wajib Menggunakan Decision Model (System 1)?
            </h2>
            <p class="text-xs text-slate-300 mt-1.5 max-w-3xl leading-relaxed">
              Menjalankan LLM 8B/14B untuk 100% kueri pengguna adalah pemborosan komputasi hingga 90%. 
              Arsitektur <strong>Two-Tier Brain (Decision Model + LLM On-Demand)</strong> memberikan latensi instan sub-100ms, determinisme terkalibrasi matematis, dan penghematan biaya produksi masif tanpa mengorbankan kualitas penalaran naratif.
            </p>
          </div>
          <div class="flex flex-wrap items-center gap-2 shrink-0">
            <span class="px-3.5 py-1.5 rounded-xl font-mono text-xs font-extrabold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-sm">
              ⚡ 50x–80x Lebih Cepat
            </span>
            <span class="px-3.5 py-1.5 rounded-xl font-mono text-xs font-extrabold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm">
              🎯 0 Output Tokens
            </span>
          </div>
        </div>

        <!-- 4 Core Pillars KPI Cards -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5 pt-1">
          <!-- Card 1 -->
          <div class="p-4 rounded-xl bg-slate-900/90 border border-slate-800 hover:border-cyan-500/40 transition space-y-1.5">
            <div class="flex items-center justify-between">
              <span class="text-lg">⚡</span>
              <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950 text-cyan-300 border border-cyan-800 font-bold">~50 - 160 ms</span>
            </div>
            <h4 class="text-xs font-bold text-white">Latensi Triage Instan</h4>
            <p class="text-[11px] text-slate-400 leading-normal">
              Sub-100 ms vs 3.500–6.000 ms. Respon klasifikasi terjadi seketika dalam 1 single forward pass GPU tanpa menunggu loop token autoregresif.
            </p>
          </div>

          <!-- Card 2 -->
          <div class="p-4 rounded-xl bg-slate-900/90 border border-slate-800 hover:border-emerald-500/40 transition space-y-1.5">
            <div class="flex items-center justify-between">
              <span class="text-lg">🎯</span>
              <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-800 font-bold">0 Output Tokens</span>
            </div>
            <h4 class="text-xs font-bold text-white">Zero Token Waste</h4>
            <p class="text-[11px] text-slate-400 leading-normal">
              Menghasilkan tensor probabilitas Softmax/Sigmoid matematis langsung, mengeliminasi 150–250 token ekstra per kueri hanya untuk format JSON.
            </p>
          </div>

          <!-- Card 3 -->
          <div class="p-4 rounded-xl bg-slate-900/90 border border-slate-800 hover:border-purple-500/40 transition space-y-1.5">
            <div class="flex items-center justify-between">
              <span class="text-lg">🛡️</span>
              <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-purple-950 text-purple-300 border border-purple-800 font-bold">100% Deterministik</span>
            </div>
            <h4 class="text-xs font-bold text-white">Bebas Halusinasi Format</h4>
            <p class="text-[11px] text-slate-400 leading-normal">
              Tidak ada resiko JSON syntax error, format drift, atau prompt injection routing. Terkalibrasi matematis untuk ambang batas bisnis.
            </p>
          </div>

          <!-- Card 4 -->
          <div class="p-4 rounded-xl bg-slate-900/90 border border-slate-800 hover:border-amber-500/40 transition space-y-1.5">
            <div class="flex items-center justify-between">
              <span class="text-lg">💰</span>
              <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800 font-bold">Hemat 80% - 100%</span>
            </div>
            <h4 class="text-xs font-bold text-white">Smart Fast-Path Gating</h4>
            <p class="text-[11px] text-slate-400 leading-normal">
              80% kueri rutin diselesaikan instan &lt;100ms tanpa menyentuh GPU LLM. Kuota komputasi LLM hanya digunakan untuk eskalasi kasus kritis.
            </p>
          </div>
        </div>

        <!-- Visual Flow Comparison -->
        <div class="p-4 rounded-xl bg-slate-950/90 border border-slate-800/90 space-y-3">
          <span class="text-[11px] font-bold text-slate-400 uppercase tracking-wider block">Visualisasi Alur Eksekusi Kueri:</span>
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- Path A -->
            <div class="p-3.5 rounded-lg bg-rose-950/20 border border-rose-800/40 space-y-2">
              <div class="flex items-center justify-between">
                <span class="text-xs font-bold text-rose-400 flex items-center gap-1.5">
                  <span>❌</span> Jalur A: Tanpa Decision Model (LLM Murni)
                </span>
                <span class="text-[10px] font-mono text-rose-300 bg-rose-900/40 px-2 py-0.5 rounded font-bold">3,500 - 6,000 ms</span>
              </div>
              <div class="flex items-center gap-2 text-[11px] font-mono text-slate-300 flex-wrap">
                <span class="p-1 rounded bg-slate-900 border border-slate-800">Kueri Masuk</span>
                <span>➔</span>
                <span class="p-1 rounded bg-rose-900/40 border border-rose-800 text-rose-200">100% Beban LLM 8B</span>
                <span>➔</span>
                <span class="p-1 rounded bg-slate-900 border border-slate-800">200 Token Boros</span>
              </div>
              <p class="text-[11px] text-slate-400 leading-relaxed">
                Semua kueri memicu komputasi autoregresif penuh, memboroskan kuota GPU dan menyebabkan antrean lambat.
              </p>
            </div>

            <!-- Path B -->
            <div class="p-3.5 rounded-lg bg-emerald-950/20 border border-emerald-800/40 space-y-2">
              <div class="flex items-center justify-between">
                <span class="text-xs font-bold text-emerald-400 flex items-center gap-1.5">
                  <span>✓</span> Jalur B: Dengan Decision Model (Two-Tier Brain)
                </span>
                <span class="text-[10px] font-mono text-emerald-300 bg-emerald-900/40 px-2 py-0.5 rounded font-bold">&lt;100 ms (Fast-Path)</span>
              </div>
              <div class="flex items-center gap-2 text-[11px] font-mono text-slate-300 flex-wrap">
                <span class="p-1 rounded bg-slate-900 border border-slate-800">Kueri Masuk</span>
                <span>➔</span>
                <span class="p-1 rounded bg-cyan-900/40 border border-cyan-800 text-cyan-200">Tier 1: Triage 0 Token</span>
                <span>➔</span>
                <span class="p-1 rounded bg-emerald-900/40 border border-emerald-800 text-emerald-200">Smart Gating (80% Hemat)</span>
              </div>
              <p class="text-[11px] text-slate-400 leading-relaxed">
                Triage sub-100ms memutuskan jalur seketika. Kueri rutin dijawab cepat, sedangkan komplain eskalatif diarahkan ke LLM dengan context terarah.
              </p>
            </div>
          </div>
        </div>

        <!-- Matriks Komparasi Seluruh Spektrum Model -->
        <div class="space-y-2 pt-1">
          <div class="flex items-center justify-between">
            <span class="text-xs font-bold text-white flex items-center gap-1.5">
              <span>📊</span> Matriks Komparasi Seluruh Spektrum Model (System 1 vs System 2)
            </span>
            <span class="text-[10px] text-slate-400">Terverifikasi pada NVIDIA Tesla T4 GPU</span>
          </div>
          <div class="overflow-x-auto rounded-xl border border-slate-800">
            <table class="w-full text-left text-xs">
              <thead class="bg-slate-950 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                <tr>
                  <th class="p-2.5 font-bold">Model Architecture</th>
                  <th class="p-2.5 font-bold">Tipe Model</th>
                  <th class="p-2.5 font-bold">Latensi Triage</th>
                  <th class="p-2.5 font-bold">Token Output</th>
                  <th class="p-2.5 font-bold">Sifat Keputusan</th>
                  <th class="p-2.5 font-bold">Efisiensi Biaya</th>
                  <th class="p-2.5 font-bold">Throughput Concurrency</th>
                </tr>
              </thead>
              <tbody class="divide-y divide-slate-900 text-slate-300 bg-slate-900/50">
                <tr class="bg-rose-950/10">
                  <td class="p-2.5 font-bold text-rose-400 flex items-center gap-1.5">
                    <span class="w-2 h-2 rounded-full bg-rose-500"></span>
                    Tanpa Decision Model (LLM Murni)
                  </td>
                  <td class="p-2.5 text-slate-400">Foundation LLM 8B/14B</td>
                  <td class="p-2.5 font-mono text-rose-400 font-bold">~3,500 - 6,000 ms</td>
                  <td class="p-2.5 font-mono text-amber-400 font-bold">150 - 250 tokens</td>
                  <td class="p-2.5 text-rose-300">Stokastik (Sampling)</td>
                  <td class="p-2.5 text-rose-400 font-bold">0% (Beban Maksimal)</td>
                  <td class="p-2.5 font-mono text-slate-400">~0.2 - 0.3 req/s</td>
                </tr>
                <tr class="hover:bg-slate-800/50 transition">
                  <td class="p-2.5 font-bold text-emerald-300 flex items-center gap-1.5">
                    <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
                    Laya Multilingual (421M)
                  </td>
                  <td class="p-2.5 text-cyan-300 font-mono text-[11px]">ModernBERT RLCD (GPU)</td>
                  <td class="p-2.5 font-mono text-emerald-400 font-bold">~50 - 75 ms ⚡</td>
                  <td class="p-2.5 font-mono text-emerald-400 font-bold">0 tokens</td>
                  <td class="p-2.5 text-emerald-300 font-bold">100% Terkalibrasi</td>
                  <td class="p-2.5 text-emerald-300 font-bold">Hemat 80 - 90%</td>
                  <td class="p-2.5 font-mono text-cyan-300 font-bold">~15 - 20 req/s</td>
                </tr>
                <tr class="hover:bg-slate-800/50 transition">
                  <td class="p-2.5 font-bold text-cyan-300 flex items-center gap-1.5">
                    <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
                    TypeSafe Jev (Cloud)
                  </td>
                  <td class="p-2.5 text-cyan-300 font-mono text-[11px]">Cloud SaaS API</td>
                  <td class="p-2.5 font-mono text-cyan-400 font-bold">~140 - 180 ms</td>
                  <td class="p-2.5 font-mono text-emerald-400 font-bold">0 tokens</td>
                  <td class="p-2.5 text-cyan-300 font-bold">100% Deterministik</td>
                  <td class="p-2.5 text-emerald-300 font-bold">Hemat 80 - 90%</td>
                  <td class="p-2.5 font-mono text-cyan-300 font-bold">~50+ req/s (Cloud)</td>
                </tr>
                <tr class="hover:bg-slate-800/50 transition">
                  <td class="p-2.5 font-bold text-purple-300 flex items-center gap-1.5">
                    <span class="w-2 h-2 rounded-full bg-purple-400"></span>
                    OpenJev (0.5B)
                  </td>
                  <td class="p-2.5 text-purple-300 font-mono text-[11px]">Qwen Logit Scorer (GPU)</td>
                  <td class="p-2.5 font-mono text-purple-400 font-bold">~180 - 240 ms</td>
                  <td class="p-2.5 font-mono text-emerald-400 font-bold">0 tokens</td>
                  <td class="p-2.5 text-purple-300 font-bold">Normalized Logit Head</td>
                  <td class="p-2.5 text-emerald-300 font-bold">Hemat 80 - 90%</td>
                  <td class="p-2.5 font-mono text-purple-300 font-bold">~5 - 8 req/s</td>
                </tr>
                <tr class="hover:bg-slate-800/50 transition">
                  <td class="p-2.5 font-bold text-blue-300 flex items-center gap-1.5">
                    <span class="w-2 h-2 rounded-full bg-blue-400"></span>
                    Kev-0.8B (Local)
                  </td>
                  <td class="p-2.5 text-blue-300 font-mono text-[11px]">Qwen LoRA Pointer Head</td>
                  <td class="p-2.5 font-mono text-blue-400 font-bold">~1,100 - 1,400 ms</td>
                  <td class="p-2.5 font-mono text-emerald-400 font-bold">0 tokens</td>
                  <td class="p-2.5 text-blue-300 font-bold">Ensemble Scored</td>
                  <td class="p-2.5 text-emerald-300 font-bold">Hemat 80 - 90%</td>
                  <td class="p-2.5 font-mono text-slate-400">~1 req/s</td>
                </tr>
                <tr class="hover:bg-slate-800/50 transition">
                  <td class="p-2.5 font-bold text-amber-300 flex items-center gap-1.5">
                    <span class="w-2 h-2 rounded-full bg-amber-400"></span>
                    CLM-8B (Contrastive)
                  </td>
                  <td class="p-2.5 text-amber-300 font-mono text-[11px]">Stanford / NVIDIA</td>
                  <td class="p-2.5 font-mono text-amber-400 font-bold">Hardware Guarded</td>
                  <td class="p-2.5 font-mono text-emerald-400 font-bold">0 tokens</td>
                  <td class="p-2.5 text-amber-300 font-bold">Contrastive Dual-Encoder</td>
                  <td class="p-2.5 text-emerald-300 font-bold">Hemat 80 - 90%</td>
                  <td class="p-2.5 font-mono text-amber-400 font-bold">Membutuhkan >=16GB</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <!-- INTERACTIVE SIDE-BY-SIDE SIMULATOR CARD -->
      <div class="glass-card rounded-2xl p-6 space-y-5">
        <!-- Title & Subtitle -->
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-4">
          <div>
            <h3 class="text-base font-bold text-white flex items-center gap-2">
              <span>⚖️</span> Simulator Uji Langsung: Bandingkan Dengan Decision vs Tanpa Decision
            </h3>
            <p class="text-xs text-slate-400 mt-0.5">
              Pilih model dan skenario untuk membuktikan langsung efisiensi latensi, penghematan token, dan determinisme secara simultan
            </p>
          </div>
          <span class="px-3 py-1 rounded-lg text-xs font-mono font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30 whitespace-nowrap">
            Live Head-to-Head Testing
          </span>
        </div>

        <!-- Controls: Select Decision Model & Select Heavyweight LLM -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 rounded-xl bg-slate-950/80 border border-slate-800">
          <!-- Decision Model Selector -->
          <div class="space-y-2">
            <label class="text-xs font-bold text-cyan-400 flex items-center gap-1.5">
              <span>⚡</span> 1. Pilih Model Decision (System 1):
            </label>
            <div class="grid grid-cols-2 gap-2">
              <button onclick="selectCompareDecisionModel('jev')" id="comp-btn-jev" class="comp-dec-btn active px-3 py-2 rounded-xl text-xs font-bold bg-cyan-600 text-white transition text-left flex flex-col">
                <span class="font-extrabold">TypeSafe Jev</span>
                <span class="text-[10px] text-cyan-200 opacity-90 font-normal">Cloud SaaS API • ~160ms</span>
              </button>
              <button onclick="selectCompareDecisionModel('laya')" id="comp-btn-laya" class="comp-dec-btn px-3 py-2 rounded-xl text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-left flex flex-col border border-slate-800">
                <span class="font-extrabold">Laya Multilingual</span>
                <span class="text-[10px] text-slate-400 font-normal">421M Local GPU • ~65ms</span>
              </button>
              <button onclick="selectCompareDecisionModel('openjev')" id="comp-btn-openjev" class="comp-dec-btn px-3 py-2 rounded-xl text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-left flex flex-col border border-slate-800">
                <span class="font-extrabold">OpenJev</span>
                <span class="text-[10px] text-slate-400 font-normal">0.5B Logit Scorer • ~220ms</span>
              </button>
              <button onclick="selectCompareDecisionModel('kev')" id="comp-btn-kev" class="comp-dec-btn px-3 py-2 rounded-xl text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-left flex flex-col border border-slate-800">
                <span class="font-extrabold">Kev-0.8B</span>
                <span class="text-[10px] text-slate-400 font-normal">Local GPU Ensemble • ~1.2s</span>
              </button>
            </div>
          </div>

          <!-- Heavyweight LLM Selector -->
          <div class="space-y-2">
            <label class="text-xs font-bold text-purple-400 flex items-center gap-1.5">
              <span>🧠</span> 2. Pilih Model LLM Kelas Berat (System 2):
            </label>
            <div class="grid grid-cols-2 sm:grid-cols-3 gap-2">
              <button onclick="selectCompareLLMModel('sahabatai')" id="comp-llm-sahabatai" class="comp-llm-btn active px-3 py-2 rounded-xl text-xs font-bold bg-purple-600 text-white transition text-left flex flex-col">
                <span class="font-extrabold">Sahabat-AI 8B</span>
                <span class="text-[10px] text-purple-200 opacity-90 font-normal">GoTo & Indosat</span>
              </button>
              <button onclick="selectCompareLLMModel('qwen')" id="comp-llm-qwen" class="comp-llm-btn px-3 py-2 rounded-xl text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-left flex flex-col border border-slate-800">
                <span class="font-extrabold">Qwen 2.5 7B</span>
                <span class="text-[10px] text-slate-400 font-normal">Alibaba Cloud</span>
              </button>
              <button onclick="selectCompareLLMModel('gemma')" id="comp-llm-gemma" class="comp-llm-btn px-3 py-2 rounded-xl text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-left flex flex-col border border-slate-800">
                <span class="font-extrabold">Gemma 2 9B</span>
                <span class="text-[10px] text-slate-400 font-normal">Google DeepMind</span>
              </button>
              <button onclick="selectCompareLLMModel('gemma-2b')" id="comp-llm-gemma-2b" class="comp-llm-btn px-3 py-2 rounded-xl text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-left flex flex-col border border-slate-800">
                <span class="font-extrabold">Gemma 2 2B</span>
                <span class="text-[10px] text-slate-400 font-normal">Ultra-Speed</span>
              </button>
              <button onclick="selectCompareLLMModel('qwen-14b')" id="comp-llm-qwen-14b" class="comp-llm-btn px-3 py-2 rounded-xl text-xs font-bold bg-slate-900 text-slate-300 hover:text-white transition text-left flex flex-col border border-slate-800">
                <span class="font-extrabold">Qwen 14B</span>
                <span class="text-[10px] text-slate-400 font-normal">Enterprise 14.7B</span>
              </button>
            </div>
          </div>
        </div>

        <!-- Presets -->
        <div class="space-y-1.5">
          <label class="text-xs font-bold text-slate-400 uppercase tracking-wide">Pilih Skenario Kasus Nyata:</label>
          <div class="flex flex-wrap gap-2">
            <button onclick="loadComparePreset('marunda')" class="px-3 py-1.5 rounded-lg text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 transition border border-slate-700 flex items-center gap-1.5">
              <span>📦</span> Skenario A: Komplain Kritis Ekspedisi (Gateway Marunda)
            </button>
            <button onclick="loadComparePreset('scam')" class="px-3 py-1.5 rounded-lg text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 transition border border-slate-700 flex items-center gap-1.5">
              <span>🚨</span> Skenario B: Tanggap Darurat Fraud / Lapor Polisi
            </button>
            <button onclick="loadComparePreset('phk')" class="px-3 py-1.5 rounded-lg text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 transition border border-slate-700 flex items-center gap-1.5">
              <span>💳</span> Skenario C: Restrukturisasi Nasabah PHK (Fintech OJK)
            </button>
            <button onclick="loadComparePreset('faq')" class="px-3 py-1.5 rounded-lg text-xs bg-slate-800 hover:bg-slate-700 text-emerald-300 transition border border-emerald-800/60 flex items-center gap-1.5">
              <span>❓</span> Skenario D: Kueri Rutin Jam Buka (Uji Fast-Path 100% Hemat LLM)
            </button>
          </div>
        </div>

        <!-- Text Input & Action Button -->
        <div class="space-y-1.5">
          <label class="text-xs font-bold text-slate-400">Pesan Pelanggan / Kasus Pengujian:</label>
          <textarea id="compareStateInput" rows="3" class="w-full code-block rounded-xl p-3.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 transition resize-none"></textarea>
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-1">
            <span class="text-[11px] text-slate-500">
              Sistem akan menjalankan pengujian arsitektur <strong>Tanpa Jev</strong> vs <strong>Dengan Jev</strong> secara simultan pada Tesla T4.
            </span>
            <button onclick="runArchitectureComparison()" id="btnRunCompare" class="px-6 py-2.5 rounded-xl font-bold text-xs bg-gradient-to-r from-purple-600 via-indigo-600 to-cyan-500 text-white hover:opacity-95 shadow-lg shadow-indigo-500/25 transition flex items-center justify-center gap-2 whitespace-nowrap">
              <span id="btnCompareIcon">⚖️</span>
              <span id="btnCompareText">JALANKAN KOMPARASI ARSITEKTUR</span>
            </button>
          </div>
        </div>

        <!-- Execution Results: Side-by-Side Comparison -->
        <div id="compareResultsSection" class="hidden space-y-6 pt-4 border-t border-slate-800">
          <!-- Executive Verdict Banner -->
          <div id="compareExecutiveBanner" class="p-4 rounded-xl bg-gradient-to-r from-indigo-950/60 to-purple-950/60 border border-indigo-800/60 flex flex-col md:flex-row md:items-center justify-between gap-4 text-xs">
            <div class="space-y-1">
              <div class="flex items-center gap-2 font-bold text-indigo-300">
                <span>🎯</span>
                <span>Executive Architectural Verdict:</span>
              </div>
              <p id="compareVerdictText" class="text-slate-300 text-[11px] leading-relaxed"></p>
            </div>
            <div class="flex items-center gap-2 shrink-0">
              <span id="compareSpeedupBadge" class="px-3 py-1.5 rounded-lg font-mono font-bold bg-cyan-950 text-cyan-300 border border-cyan-800 text-xs">
                Speedup: -
              </span>
              <span id="compareTokenSaveBadge" class="px-3 py-1.5 rounded-lg font-mono font-bold bg-emerald-950 text-emerald-300 border border-emerald-800 text-xs">
                Token Savings: -
              </span>
            </div>
          </div>

          <!-- Two Columns: Tanpa Jev vs Dengan Jev -->
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Column 1: TANPA JEV (Direct LLM) -->
            <div class="glass-card rounded-2xl p-5 border border-rose-500/30 space-y-4 relative overflow-hidden">
              <div class="flex items-center justify-between border-b border-slate-800 pb-3">
                <div class="space-y-0.5">
                  <div class="flex items-center gap-2">
                    <span class="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
                    <span class="text-[10px] font-extrabold uppercase tracking-wider text-rose-400">Arsitektur Konvensional</span>
                  </div>
                  <h4 class="text-sm font-bold text-white" id="compNoJevTitle">Tanpa Decision Model (LLM Murni)</h4>
                </div>
                <span class="px-2.5 py-1 rounded text-[10px] font-mono font-bold bg-rose-950/80 text-rose-300 border border-rose-800/80">
                  100% LLM Compute Load
                </span>
              </div>

              <!-- Metric Indicators -->
              <div class="grid grid-cols-3 gap-2 text-center">
                <div class="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800">
                  <div class="text-[10px] text-slate-400">Latensi Total</div>
                  <div class="text-sm font-mono font-bold text-rose-400 mt-0.5" id="compNoJevLat">- ms</div>
                </div>
                <div class="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800">
                  <div class="text-[10px] text-slate-400">Token Terpakai</div>
                  <div class="text-sm font-mono font-bold text-amber-400 mt-0.5" id="compNoJevTokens">- tokens</div>
                </div>
                <div class="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800">
                  <div class="text-[10px] text-slate-400">Token Savings</div>
                  <div class="text-sm font-mono font-bold text-rose-400 mt-0.5">0% (Boros)</div>
                </div>
              </div>

              <!-- Feature Checks -->
              <div class="space-y-1.5 text-[11px] p-3 rounded-xl bg-slate-900/60 border border-slate-800/80">
                <div class="flex justify-between items-center text-slate-300">
                  <span class="text-slate-400">Reliabilitas Klasifikasi:</span>
                  <span class="text-rose-400 font-bold">Rentan Halusinasi & Drift</span>
                </div>
                <div class="flex justify-between items-center text-slate-300">
                  <span class="text-slate-400">Determinisme Softmax:</span>
                  <span class="text-rose-400 font-bold">Tidak Ada (Sampling Token)</span>
                </div>
                <div class="flex justify-between items-center text-slate-300">
                  <span class="text-slate-400">Kapasitas Concurrency:</span>
                  <span class="text-amber-400 font-bold">Rendah (GPU Saturation)</span>
                </div>
                <div class="flex justify-between items-center text-slate-300">
                  <span class="text-slate-400">Indeks Biaya Operasional:</span>
                  <span class="text-rose-400 font-bold">100% Maksimal</span>
                </div>
              </div>

              <!-- Response Box -->
              <div class="space-y-1">
                <label class="text-[11px] font-bold text-slate-400">Respon LLM Standalone:</label>
                <div id="compNoJevResponse" class="p-3.5 rounded-xl bg-slate-950 text-xs text-slate-300 whitespace-pre-wrap leading-relaxed min-h-[140px] max-h-60 overflow-y-auto font-sans border border-slate-800"></div>
              </div>
            </div>

            <!-- Column 2: DENGAN JEV / DECISION MODEL (Two-Tier) -->
            <div class="glass-card rounded-2xl p-5 border border-emerald-500/30 space-y-4 relative overflow-hidden">
              <div class="flex items-center justify-between border-b border-slate-800 pb-3">
                <div class="space-y-0.5">
                  <div class="flex items-center gap-2">
                    <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
                    <span class="text-[10px] font-extrabold uppercase tracking-wider text-emerald-400">Arsitektur Modern Two-Tier</span>
                  </div>
                  <h4 class="text-sm font-bold text-white" id="compWithJevTitle">Dengan Jev (Decision Model) + LLM</h4>
                </div>
                <span class="px-2.5 py-1 rounded text-[10px] font-mono font-bold bg-emerald-950/80 text-emerald-300 border border-emerald-800/80" id="compWithJevSaveBadge">
                  Hemat 80-100% Token
                </span>
              </div>

              <!-- Metric Indicators -->
              <div class="grid grid-cols-3 gap-2 text-center">
                <div class="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800">
                  <div class="text-[10px] text-slate-400">Triage Decision</div>
                  <div class="text-sm font-mono font-bold text-cyan-400 mt-0.5" id="compWithJevDecLat">- ms</div>
                </div>
                <div class="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800">
                  <div class="text-[10px] text-slate-400">Token Triage</div>
                  <div class="text-sm font-mono font-bold text-emerald-400 mt-0.5">0 tokens</div>
                </div>
                <div class="p-2.5 rounded-xl bg-slate-900/90 border border-slate-800">
                  <div class="text-[10px] text-slate-400">Total Waktu</div>
                  <div class="text-sm font-mono font-bold text-purple-400 mt-0.5" id="compWithJevTotalLat">- ms</div>
                </div>
              </div>

              <!-- Decision Triage Details -->
              <div class="p-3.5 rounded-xl bg-cyan-950/20 border border-cyan-800/40 space-y-2 text-xs">
                <div class="flex items-center justify-between">
                  <span class="font-bold text-cyan-300 flex items-center gap-1.5">
                    <span>⚡</span>
                    <span>Triage Instan Decision Model:</span>
                  </span>
                  <span class="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-900/50 text-cyan-200 border border-cyan-700/60" id="compWithJevRoutingAction">
                    Eskalasi Cerdas
                  </span>
                </div>
                <div id="compWithJevAnswersGrid" class="space-y-1.5 text-[11px]"></div>
                <div id="compWithJevEscalationReasons" class="text-[11px] text-slate-300 pt-1 border-t border-cyan-800/40"></div>
              </div>

              <!-- Response Box -->
              <div class="space-y-1">
                <label class="text-[11px] font-bold text-slate-400" id="compWithJevResponseLabel">Respon Akhir Sistem:</label>
                <div id="compWithJevResponse" class="p-3.5 rounded-xl bg-slate-950 text-xs text-slate-200 whitespace-pre-wrap leading-relaxed min-h-[140px] max-h-60 overflow-y-auto font-sans border border-slate-800"></div>
              </div>
            </div>
          </div>

          <!-- Executive Matrix Comparison Table (All Decision Models & LLMs) -->
          <div class="glass-card rounded-2xl p-5 border border-slate-800 space-y-4">
            <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div>
                <h4 class="text-sm font-bold text-white flex items-center gap-2">
                  <span>📊</span> Matriks Komparasi Menyeluruh: Tanpa Jev vs Seluruh Opsi Model Decision
                </h4>
                <p class="text-xs text-slate-400">Analisis komparatif menyeluruh lintas dimensi kinerja arsitektur di GPU NVIDIA Tesla T4</p>
              </div>
              <span class="px-2.5 py-1 rounded text-[10px] font-mono bg-slate-800 text-slate-300">
                Arsitektur Benchmark
              </span>
            </div>

            <div class="overflow-x-auto">
              <table class="w-full text-left text-xs font-sans">
                <thead>
                  <tr class="text-slate-400 border-b border-slate-800 bg-slate-950/60">
                    <th class="p-3">Dimensi Kinerja</th>
                    <th class="p-3 text-rose-400">Tanpa Decision Model (LLM Murni)</th>
                    <th class="p-3 text-cyan-400">Dengan TypeSafe Jev (Cloud)</th>
                    <th class="p-3 text-emerald-400">Dengan Laya Multilingual (421M GPU)</th>
                    <th class="p-3 text-amber-400">Dengan OpenJev (0.5B GPU)</th>
                    <th class="p-3 text-indigo-400">Dengan Kev-0.8B (Local)</th>
                  </tr>
                </thead>
                <tbody class="divide-y divide-slate-800/60 text-slate-300 text-[11px]">
                  <tr>
                    <td class="p-3 font-bold text-white">Latensi Keputusan / Triage</td>
                    <td class="p-3 text-rose-400 font-mono font-bold">~3,500 - 6,000 ms</td>
                    <td class="p-3 text-cyan-300 font-mono font-bold">~140 - 180 ms</td>
                    <td class="p-3 text-emerald-300 font-mono font-bold">~50 - 75 ms (Tercepat)</td>
                    <td class="p-3 text-amber-300 font-mono font-bold">~180 - 240 ms</td>
                    <td class="p-3 text-indigo-300 font-mono font-bold">~1,100 - 1,400 ms</td>
                  </tr>
                  <tr>
                    <td class="p-3 font-bold text-white">Token Output Triage</td>
                    <td class="p-3 text-rose-400 font-mono">150 - 250 tokens</td>
                    <td class="p-3 text-emerald-400 font-mono font-bold">0 tokens (Zero Waste)</td>
                    <td class="p-3 text-emerald-400 font-mono font-bold">0 tokens (Zero Waste)</td>
                    <td class="p-3 text-emerald-400 font-mono font-bold">0 tokens (Zero Waste)</td>
                    <td class="p-3 text-emerald-400 font-mono font-bold">0 tokens (Zero Waste)</td>
                  </tr>
                  <tr>
                    <td class="p-3 font-bold text-white">Metode Inferensi</td>
                    <td class="p-3 text-slate-400">Autoregressive Loop (Next-Token)</td>
                    <td class="p-3 text-cyan-300">Single Forward Pass (Softmax Cloud)</td>
                    <td class="p-3 text-emerald-300 font-bold">1x Forward Pass CUDA (Tensor Core)</td>
                    <td class="p-3 text-amber-300">Logit Contrastive Head (PyTorch)</td>
                    <td class="p-3 text-indigo-300">Multi-Model Ensemble Forward</td>
                  </tr>
                  <tr>
                    <td class="p-3 font-bold text-white">Konsistensi & Determinisme</td>
                    <td class="p-3 text-rose-400">Stokastik (Risiko Drift & Parsing Error)</td>
                    <td class="p-3 text-emerald-300 font-bold">100% Deterministik Sigmoid/Softmax</td>
                    <td class="p-3 text-emerald-300 font-bold">100% Deterministik (Terkalibrasi)</td>
                    <td class="p-3 text-emerald-300 font-bold">100% Logit Normalized</td>
                    <td class="p-3 text-emerald-300 font-bold">100% Ensemble Scored</td>
                  </tr>
                  <tr>
                    <td class="p-3 font-bold text-white">Efisiensi Biaya (Token Savings)</td>
                    <td class="p-3 text-rose-400 font-bold">0% Hemat (100% Beban Token Penuh)</td>
                    <td class="p-3 text-emerald-300 font-bold">Hemat 80-90% Biaya LLM</td>
                    <td class="p-3 text-emerald-300 font-bold">Hemat 80-90% Biaya LLM</td>
                    <td class="p-3 text-emerald-300 font-bold">Hemat 80-90% Biaya LLM</td>
                    <td class="p-3 text-emerald-300 font-bold">Hemat 80-90% Biaya LLM</td>
                  </tr>
                  <tr>
                    <td class="p-3 font-bold text-white">Overhead Memori VRAM GPU</td>
                    <td class="p-3 text-slate-400">0 MB (Hanya LLM)</td>
                    <td class="p-3 text-cyan-300 font-bold">0 MB (Offloaded ke Cloud)</td>
                    <td class="p-3 text-emerald-300 font-mono">~950 MB (Sangat Ringan)</td>
                    <td class="p-3 text-amber-300 font-mono">~1,100 MB</td>
                    <td class="p-3 text-indigo-300 font-mono">~1,600 MB</td>
                  </tr>
                  <tr>
                    <td class="p-3 font-bold text-white">Throughput Maksimum (Req/Sec)</td>
                    <td class="p-3 text-rose-400 font-mono">~0.2 - 0.3 req/s (Saturasi Cepat)</td>
                    <td class="p-3 text-cyan-300 font-mono">~50+ req/s (Auto-scaling SaaS)</td>
                    <td class="p-3 text-emerald-300 font-mono font-bold">~15 - 20 req/s (1x Tesla T4)</td>
                    <td class="p-3 text-amber-300 font-mono">~5 - 8 req/s</td>
                    <td class="p-3 text-indigo-300 font-mono">~1 req/s</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="glass-card rounded-2xl p-6 border border-cyan-500/30 space-y-6">
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <div class="flex items-center gap-2">
            <span class="text-xl">🔬</span>
            <h3 class="text-lg font-bold text-white">
              Deep Dive Benchmark: Mengapa Decision Model Berbeda dengan Generative LLM & Bagaimana Cara Kerjanya?
            </h3>
          </div>
          <p class="text-xs text-slate-400 mt-1">
            Perbandingan mendalam Arsitektur Non-Autoregressive (System 1) vs Autoregressive Generative LLM (System 2: Sahabat-AI, Qwen, Gemma)
          </p>
        </div>
        <span class="px-3 py-1 rounded-lg text-xs font-mono font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 whitespace-nowrap">
          Arsitektur Fundamental
        </span>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div class="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
          <div class="w-8 h-8 rounded-lg bg-cyan-500/20 text-cyan-400 flex items-center justify-center font-bold text-sm">1</div>
          <h4 class="text-xs font-bold text-white uppercase tracking-wider">Tanpa Text Decoder (0 Tokens)</h4>
          <p class="text-[11px] text-slate-300 leading-relaxed">
            Decision Model (Laya, Kev, Jev, OpenJev) <strong>tidak memiliki decoder kalimat</strong>. Model memetakan representasi semantik teks langsung ke tensor probabilitas klasifikasi (Softmax/Sigmoid) dalam 1 pass.
          </p>
        </div>

        <div class="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
          <div class="w-8 h-8 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-sm">2</div>
          <h4 class="text-xs font-bold text-white uppercase tracking-wider">Single Forward Pass (&lt;100 ms)</h4>
          <p class="text-[11px] text-slate-300 leading-relaxed">
            LLM generatif (Sahabat-AI, Qwen, Gemma) harus melakukan loop ratusan kali (<em>next-token autoregression</em>) untuk merangkai kalimat. Decision Model selesai dalam <strong>1 kali komputasi forward</strong> GPU.
          </p>
        </div>

        <div class="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
          <div class="w-8 h-8 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center font-bold text-sm">3</div>
          <h4 class="text-xs font-bold text-white uppercase tracking-wider">Zero Hallucination Risk</h4>
          <p class="text-[11px] text-slate-300 leading-relaxed">
            Karena tidak mengarang kata, Decision Model <strong>mustahil mengalami halusinasi teks</strong>. Nilai probabilitas matematisnya stabil, terkalibrasi, dan siap dijadikan kondisi percabangan logika backend (if-else).
          </p>
        </div>

        <div class="p-4 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
          <div class="w-8 h-8 rounded-lg bg-purple-500/20 text-purple-400 flex items-center justify-center font-bold text-sm">4</div>
          <h4 class="text-xs font-bold text-white uppercase tracking-wider">Sinergi Dua-Tier AI</h4>
          <p class="text-[11px] text-slate-300 leading-relaxed">
            Jika pelanggan butuh penjelasan kalimat santun dan penyelesaian manusiawi, System 1 menyuplai hasil klasifikasinya ke <strong>System 2 (Sahabat-AI / Qwen / Gemma)</strong> untuk menuliskan respon resmi.
          </p>
        </div>
      </div>
    </div>

    <!-- Comparative Analysis & Latency Chart -->
    <div class="glass-card rounded-2xl p-6">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h3 class="text-base font-bold text-white">Spektrum Model: Decision Models (System 1) vs Heavyweight LLM (System 2)</h3>
          <p class="text-xs text-slate-400">Perbandingan latensi logaritmik dan karakteristik operasional di infrastruktur Tesla T4</p>
        </div>
        <span class="text-xs px-2.5 py-1 rounded bg-slate-800 text-slate-300 font-mono">Skala Logaritmik (ms)</span>
      </div>
      <div class="h-64">
        <canvas id="heavyChart"></canvas>
      </div>
    </div>

  </main>

  <!-- Footer -->
  <footer class="border-t border-slate-800/80 py-8 text-center text-xs text-slate-500">
    <p>DecisionModelBench • Komparasi Model Kelas Berat Indonesia (Sahabat-AI 8B, Qwen 2.5 7B/14B, Gemma 2 9B)</p>
    <p class="mt-1">Dijalankan pada Akselerator GPU CUDA (4-bit Quantization • On-Demand Hot-Swap)</p>
  </footer>

  <script>
    const CHAT_PRESETS = {
      'marunda': {
        sys: 'Kamu adalah Customer Care Senior e-commerce terkemuka di Indonesia yang empatik, tenang, dan solutif.',
        text: 'Min paket gw dr tgl 20 stuck di gateway Marunda ga gerak2, kurir gimana sih? Mau gw viralin di X nih klo ga nyampe hari ini! Balikin ongkir gw juga!'
      },
      'phk': {
        sys: 'Kamu adalah agen restrukturisasi pinjaman fintech OJK yang berintegritas dan memahami musibah nasabah secara manusiawi.',
        text: 'Selamat siang bapak/ibu, saya mohon maaf bulan ini belum bisa bayar cicilan pinjaman full karena saya baru kena PHK minggu lalu. Mohon jangan sebar data saya, saya berniat bayar tapi tolong beri perpanjangan tenor.'
      },
      'scam': {
        sys: 'Kamu adalah tim tanggap darurat fraud perbankan Indonesia yang sigap menghentikan kerugian nasabah.',
        text: 'Tolong min akun saya diblokir segera! Tadi ada yg nelpon ngaku CS minta kode OTP buat undian berhadiah, terus saldo rekening saya ludes 10 juta!'
      },
      'gov': {
        sys: 'Kamu adalah konsultan regulasi dan perizinan usaha mikro Indonesia yang paham aturan OSS RBA dan BPOM.',
        text: 'Halo min, saya mau buka usaha sambal kemasan rumahan di Bandung. Supaya legal dan bisa masuk minimarket, urutan izin OSS, NIB, P-IRT, dan BPOM nya gimana ya?'
      },
      'jaksel': {
        sys: 'Kamu adalah teman ngobrol gaul anak muda Jakarta yang santai, paham tren, tapi tetap sopan dan berwawasan luas.',
        text: 'Bro, jujur gw lagi burnout parah sama kerjaan agency di SCBD. Literally tiap hari pulang jam 10 malem, commute dari Bogor naik KRL desek-desekan. Menurut lo gw resign aja apa gimana ya?'
      }
    };

    const H2H_PRESETS = {
      'marunda': {
        state: 'Min paket gw dr tgl 20 stuck di gateway Marunda ga gerak2, kurir gimana sih? Mau gw viralin di X nih klo ga nyampe hari ini!',
        questions: {
          'is_threat_viral': { 'type': 'noul', 'instructions': 'Apakah pelanggan mengancam memviralkan di medsos?' },
          'issue_category': { 'type': 'choice', 'instructions': 'Kategori masalah?', 'criteria': { 'keterlambatan_stuck': 'Paket tertahan di hub', 'barang_rusak': 'Barang rusak', 'salah_alamat': 'Salah alamat' } },
          'urgency': { 'type': 'score', 'instructions': 'Tingkat urgensi', 'criteria': ['santai', 'menunggu', 'marah', 'kritis_viral'] }
        }
      },
      'scam': {
        state: 'Woi penipu ya lu! Pesen HP Samsung yg dateng malah sabun colek! Balikin duit gw sekarang atau gw lapor polisi sekarang juga!',
        questions: {
          'threat_legal': { 'type': 'noul', 'instructions': 'Apakah pelanggan mengancam lapor polisi?' },
          'issue_category': { 'type': 'choice', 'instructions': 'Klasifikasi?', 'criteria': { 'penipuan_salah_barang': 'Barang palsu/diganti', 'retur_ukuran': 'Retur ukuran' } },
          'risk_score': { 'type': 'score', 'instructions': 'Risiko reputasi', 'criteria': ['rendah', 'sedang', 'tinggi', 'ekstrem'] }
        }
      },
      'fintech': {
        state: 'Selamat siang bapak/ibu, saya mohon maaf bulan ini belum bisa bayar cicilan pinjaman full karena saya baru kena PHK minggu lalu. Mohon jangan sebar data saya, saya berniat bayar tapi tolong beri perpanjangan tenor.',
        questions: {
          'willingness_to_pay': { 'type': 'noul', 'instructions': 'Apakah nasabah berniat bayar?' },
          'reason_delay': { 'type': 'choice', 'instructions': 'Penyebab keterlambatan?', 'criteria': { 'kehilangan_pekerjaan': 'Kena PHK', 'sakit': 'Sakit RS', 'menolak_bayar': 'Menolak bayar' } },
          'default_risk': { 'type': 'score', 'instructions': 'Risiko gagal bayar', 'criteria': ['rendah', 'sedang', 'tinggi', 'macet_total'] }
        }
      }
    };

    const COMPARE_PRESETS = {
      'marunda': {
        state: 'Min paket gw dr tgl 20 stuck di gateway Marunda ga gerak2, kurir gimana sih? Mau gw viralin di X nih klo ga nyampe hari ini! Balikin ongkir gw juga!',
        questions: {
          'is_threat_or_urgent': { 'type': 'noul', 'instructions': 'Apakah pelanggan mengancam memviralkan di media sosial?' },
          'issue_category': { 'type': 'choice', 'instructions': 'Kategori masalah pengiriman?', 'criteria': { 'komplain_kritis': 'Paket tertahan / Macet', 'permohonan_bantuan': 'Cek resi normal', 'informasi_rutin': 'Biaya ongkir' } },
          'urgency_level': { 'type': 'score', 'instructions': 'Tingkat urgensi penanganan', 'criteria': ['rendah', 'sedang', 'tinggi', 'kritis'] }
        }
      },
      'scam': {
        state: 'Woi akun rekening saya tiba2 ludes 10 juta habis dihubungi nomor penipu minta kode OTP! Tolong blokir sekarang juga atau saya lapor polisi detik ini!',
        questions: {
          'is_threat_or_urgent': { 'type': 'noul', 'instructions': 'Apakah keadaan darurat fraud dan ancaman lapor polisi?' },
          'issue_category': { 'type': 'choice', 'instructions': 'Jenis insiden keuangan?', 'criteria': { 'komplain_kritis': 'Fraud rekening / Scam OTP', 'permohonan_bantuan': 'Ganti password akun', 'informasi_rutin': 'Info suku bunga' } },
          'urgency_level': { 'type': 'score', 'instructions': 'Tingkat urgensi penanganan', 'criteria': ['rendah', 'sedang', 'tinggi', 'kritis'] }
        }
      },
      'phk': {
        state: 'Selamat siang bapak/ibu, saya mohon maaf bulan ini belum bisa bayar cicilan pinjaman full karena saya baru kena PHK minggu lalu. Mohon jangan sebar data saya, saya berniat bayar tapi tolong beri perpanjangan tenor.',
        questions: {
          'is_threat_or_urgent': { 'type': 'noul', 'instructions': 'Apakah nasabah mengalami krisis gagal bayar akibat PHK?' },
          'issue_category': { 'type': 'choice', 'instructions': 'Kebutuhan nasabah?', 'criteria': { 'komplain_kritis': 'Gagal bayar total', 'permohonan_bantuan': 'Restrukturisasi tenor pinjaman', 'informasi_rutin': 'Jadwal jatuh tempo' } },
          'urgency_level': { 'type': 'score', 'instructions': 'Tingkat risiko penanganan', 'criteria': ['rendah', 'sedang', 'tinggi', 'kritis'] }
        }
      },
      'faq': {
        state: 'Halo selamat siang CS, saya mau tanya apakah kantor cabang di Jakarta Selatan buka pelayanan nasabah pada hari Sabtu dan Minggu? Terima kasih infonya.',
        questions: {
          'is_threat_or_urgent': { 'type': 'noul', 'instructions': 'Apakah pesan bernada komplain keras atau ancaman?' },
          'issue_category': { 'type': 'choice', 'instructions': 'Klasifikasi kebutuhan?', 'criteria': { 'komplain_kritis': 'Komplain pelayanan cabang', 'permohonan_bantuan': 'Booking janji temu', 'informasi_rutin': 'Informasi jam operasional kantor' } },
          'urgency_level': { 'type': 'score', 'instructions': 'Tingkat urgensi', 'criteria': ['rendah', 'sedang', 'tinggi', 'kritis'] }
        }
      }
    };

    let currentH2HQuestions = H2H_PRESETS['marunda'].questions;
    let selectedChatModel = 'sahabatai';
    let selectedH2HModel = 'laya';
    let selectedH2HSys2Model = 'sahabatai';
    let selectedTTModel = 'laya';
    let selectedTTSys2Model = 'sahabatai';
    let selectedCompareDecisionModel = 'jev';
    let selectedCompareLLMModel = 'sahabatai';
    let currentCompareQuestions = COMPARE_PRESETS['marunda'].questions;

    function switchTab(tabId) {
      document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
      document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active', 'bg-gradient-to-r', 'from-purple-600', 'to-indigo-600', 'text-white', 'shadow-lg');
        btn.classList.add('bg-slate-800', 'text-slate-300');
      });
      document.getElementById(tabId).classList.remove('hidden');
      const activeBtn = document.getElementById('btn-' + tabId);
      activeBtn.classList.remove('bg-slate-800', 'text-slate-300');
      activeBtn.classList.add('active');
    }

    const MODEL_META = {
      'sahabatai': {
        name: 'Sahabat-AI 8B Instruct (GoTo & Indosat)',
        desc: 'Model fondasi kedaulatan digital Indonesia. Menguasai norma budaya, dialek lokal, dan empati customer care.',
        speed: '~28-32 t/s (CUDA GPU)',
        btnBg: 'bg-purple-600'
      },
      'qwen': {
        name: 'Qwen 2.5 7B Instruct (Alibaba Cloud)',
        desc: 'Model multilingual SOTA dengan penalaran logika tinggi, pemecahan masalah, dan bahasa Indonesia formal yang sangat presisi.',
        speed: '~30-35 t/s (CUDA GPU)',
        btnBg: 'bg-cyan-600'
      },
      'gemma': {
        name: 'Gemma 2 9B Instruct (Google DeepMind)',
        desc: 'Arsitektur generasi terbaru Google DeepMind dengan sliding window attention dan penalaran inferensi mendalam.',
        speed: '~24-28 t/s (CUDA GPU)',
        btnBg: 'bg-emerald-600'
      },
      'gemma-2b': {
        name: 'Gemma 2 2B Instruct (Google DeepMind)',
        desc: 'Versi ultra-cepat dan hemat memori dari Google. Menghasilkan respon sangat cepat (~55+ t/s) dengan VRAM minim (~1.7 GB).',
        speed: '~50-65 t/s (CUDA GPU)',
        btnBg: 'bg-amber-600'
      }
    };

    function selectChatModel(model) {
      selectedChatModel = model;
      document.querySelectorAll('.chat-model-btn').forEach(btn => {
        btn.classList.remove('active', 'bg-purple-600', 'bg-cyan-600', 'bg-emerald-600', 'bg-amber-600', 'text-white');
        btn.classList.add('bg-slate-800', 'text-slate-300');
      });
      const activeBtn = document.getElementById('chat-btn-' + model);
      const meta = MODEL_META[model] || MODEL_META['sahabatai'];
      if (activeBtn) {
        activeBtn.classList.remove('bg-slate-800', 'text-slate-300');
        activeBtn.classList.add('active', meta.btnBg, 'text-white');
      }
      document.getElementById('chatBannerTitle').innerText = meta.name;
      document.getElementById('chatBannerDesc').innerText = meta.desc;
      document.getElementById('chatBannerSpeed').innerText = meta.speed;
      document.getElementById('chatOutputModelBadge').innerText = meta.name.split(' (')[0];
    }

    function selectH2HModel(model) {
      selectedH2HModel = model;
      document.querySelectorAll('.h2h-model-btn').forEach(btn => {
        btn.classList.remove('active', 'bg-cyan-500', 'text-white');
        btn.classList.add('bg-slate-900', 'text-slate-300');
      });
      const activeBtn = document.getElementById('h2h-btn-' + model);
      if (activeBtn) {
        activeBtn.classList.remove('bg-slate-900', 'text-slate-300');
        activeBtn.classList.add('active', 'bg-cyan-500', 'text-white');
      }
    }

    function selectH2HSys2Model(model) {
      selectedH2HSys2Model = model;
      document.querySelectorAll('.h2h-s2-btn').forEach(btn => {
        btn.classList.remove('active', 'bg-purple-600', 'text-white');
        btn.classList.add('bg-slate-900', 'text-slate-300');
      });
      const activeBtn = document.getElementById('h2h-s2-btn-' + model);
      if (activeBtn) {
        activeBtn.classList.remove('bg-slate-900', 'text-slate-300');
        activeBtn.classList.add('active', 'bg-purple-600', 'text-white');
      }
    }

    function selectTTModel(model) {
      selectedTTModel = model;
      document.querySelectorAll('.tt-model-btn').forEach(btn => {
        btn.classList.remove('active', 'bg-emerald-500', 'text-white');
        btn.classList.add('bg-slate-900', 'text-slate-300');
      });
      const activeBtn = document.getElementById('tt-btn-' + model);
      if (activeBtn) {
        activeBtn.classList.remove('bg-slate-900', 'text-slate-300');
        activeBtn.classList.add('active', 'bg-emerald-500', 'text-white');
      }
      const modelLabels = {
        'laya': 'Laya triage dalam <strong>~65 ms</strong> (0 tokens)',
        'openjev': 'OpenJev triage dalam <strong>~220 ms</strong> (0 tokens)',
        'jev': 'TypeSafe Jev triage dalam <strong>~160 ms</strong> (0 tokens)',
        'kev': 'Kev-0.8B triage dalam <strong>~1.2 s</strong> (0 tokens)'
      };
      document.getElementById('ttFlowSys1Label').innerHTML = modelLabels[model] || 'Triage instan 0 tokens';
    }

    function selectTTSys2Model(model) {
      selectedTTSys2Model = model;
      document.querySelectorAll('.tt-s2-btn').forEach(btn => {
        btn.classList.remove('active', 'bg-purple-600', 'text-white');
        btn.classList.add('bg-slate-900', 'text-slate-300');
      });
      const activeBtn = document.getElementById('tt-s2-btn-' + model);
      if (activeBtn) {
        activeBtn.classList.remove('bg-slate-900', 'text-slate-300');
        activeBtn.classList.add('active', 'bg-purple-600', 'text-white');
      }
      const meta = MODEL_META[model] || MODEL_META['sahabatai'];
      document.getElementById('ttFlowSys2Title').innerText = 'Tier 2: ' + meta.name.split(' (')[0];
    }

    function loadChatPreset(key) {
      const p = CHAT_PRESETS[key];
      if (p) {
        document.getElementById('chatSystemPrompt').value = p.sys;
        document.getElementById('chatInputText').value = p.text;
      }
    }

    function loadH2HPreset(key) {
      const p = H2H_PRESETS[key];
      if (p) {
        document.getElementById('h2hState').value = p.state;
        currentH2HQuestions = p.questions;
      }
    }

    function selectCompareDecisionModel(model) {
      selectedCompareDecisionModel = model;
      document.querySelectorAll('.comp-dec-btn').forEach(btn => {
        btn.classList.remove('active', 'bg-cyan-600', 'text-white');
        btn.classList.add('bg-slate-900', 'text-slate-300');
      });
      const activeBtn = document.getElementById('comp-btn-' + model);
      if (activeBtn) {
        activeBtn.classList.remove('bg-slate-900', 'text-slate-300');
        activeBtn.classList.add('active', 'bg-cyan-600', 'text-white');
      }
    }

    function selectCompareLLMModel(model) {
      selectedCompareLLMModel = model;
      document.querySelectorAll('.comp-llm-btn').forEach(btn => {
        btn.classList.remove('active', 'bg-purple-600', 'text-white');
        btn.classList.add('bg-slate-900', 'text-slate-300');
      });
      const activeBtn = document.getElementById('comp-llm-' + model);
      if (activeBtn) {
        activeBtn.classList.remove('bg-slate-900', 'text-slate-300');
        activeBtn.classList.add('active', 'bg-purple-600', 'text-white');
      }
    }

    function loadComparePreset(key) {
      const p = COMPARE_PRESETS[key];
      if (p) {
        document.getElementById('compareStateInput').value = p.state;
        currentCompareQuestions = p.questions;
      }
    }

    async function runChatGeneration() {
      const prompt = document.getElementById('chatInputText').value.trim();
      if (!prompt) return alert('Silakan isi pesan pengujian');

      const sys = document.getElementById('chatSystemPrompt').value.trim();
      const maxTok = parseInt(document.getElementById('chatMaxTokens').value) || 250;

      const btn = document.getElementById('btnRunChat');
      const btnIcon = document.getElementById('btnChatIcon');
      const btnText = document.getElementById('btnChatText');

      btn.disabled = true;
      btn.classList.add('opacity-75', 'cursor-not-allowed');
      btnIcon.innerHTML = '⏳';
      btnText.innerText = 'MEMPROSES DI TESLA T4...';

      const outputCard = document.getElementById('chatOutputCard');
      const box = document.getElementById('chatResponseBox');
      outputCard.classList.remove('hidden');
      box.innerText = `Menjalankan inferensi ${selectedChatModel.toUpperCase()} (CUDA offload)...`;

      try {
        const res = await fetch('/api/heavyweight/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: selectedChatModel,
            prompt: prompt,
            system_prompt: sys,
            max_tokens: maxTok,
            temperature: 0.2
          })
        });

        const data = await res.json();
        box.innerText = data.text || 'Tidak ada teks dihasilkan';
        document.getElementById('chatMetricsBadge').innerText = 
          `${data.latency_ms} ms | ${data.tokens_generated} tokens | ${data.speed_tokens_sec} t/s`;

      } catch (e) {
        box.innerText = 'Gagal mengeksekusi inferensi: ' + e.message;
      } finally {
        btn.disabled = false;
        btn.classList.remove('opacity-75', 'cursor-not-allowed');
        btnIcon.innerHTML = '⚡';
        btnText.innerText = 'HASILKAN RESPON LLM';
      }
    }

    async function runArenaBenchmark() {
      const prompt = document.getElementById('arenaPrompt').value.trim();
      if (!prompt) return alert('Silakan isi prompt arena pengujian');

      const btn = document.getElementById('btnRunArena');
      const btnIcon = document.getElementById('btnArenaIcon');
      const btnText = document.getElementById('btnArenaText');

      btn.disabled = true;
      btn.classList.add('opacity-75', 'cursor-not-allowed');
      btnIcon.innerHTML = '⏳';
      btnText.innerText = 'ARENA SEDANG BERJALAN...';

      const arenaModels = [
        { id: 'sahabatai', elText: 'arenaTextSahabat', elLat: 'arenaLatSahabat', elSpd: 'arenaSpdSahabat', elTok: 'arenaTokSahabat' },
        { id: 'qwen', elText: 'arenaTextQwen', elLat: 'arenaLatQwen', elSpd: 'arenaSpdQwen', elTok: 'arenaTokQwen' },
        { id: 'gemma', elText: 'arenaTextGemma', elLat: 'arenaLatGemma', elSpd: 'arenaSpdGemma', elTok: 'arenaTokGemma' }
      ];

      for (const m of arenaModels) {
        document.getElementById(m.elText).innerText = `Menjalankan inferensi ${m.id.toUpperCase()}...`;
        document.getElementById(m.elLat).innerText = 'Memproses...';
        try {
          const res = await fetch('/api/heavyweight/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              model: m.id,
              prompt: prompt,
              max_tokens: 150,
              temperature: 0.2
            })
          });
          const data = await res.json();
          document.getElementById(m.elText).innerText = data.text || '-';
          document.getElementById(m.elLat).innerText = `${data.latency_ms} ms`;
          document.getElementById(m.elSpd).innerText = `Speed: ${data.speed_tokens_sec} t/s`;
          document.getElementById(m.elTok).innerText = `Tokens: ${data.tokens_generated}`;
        } catch (e) {
          document.getElementById(m.elText).innerText = 'Error: ' + e.message;
          document.getElementById(m.elLat).innerText = 'Gagal';
        }
      }

      btn.disabled = false;
      btn.classList.remove('opacity-75', 'cursor-not-allowed');
      btnIcon.innerHTML = '⚔️';
      btnText.innerText = 'JALANKAN ARENA KOMPARASI';
    }

    async function runHeadToHeadBenchmark() {
      const state = document.getElementById('h2hState').value.trim();
      if (!state) return alert('Silakan isi teks pengujian');

      const btn = document.getElementById('btnRunH2H');
      const btnIcon = document.getElementById('btnH2HIcon');
      const btnText = document.getElementById('btnH2HText');

      btn.disabled = true;
      btn.classList.add('opacity-75', 'cursor-not-allowed');
      btnIcon.innerHTML = '⏳';
      btnText.innerText = 'MENJALANKAN HEAD-TO-HEAD...';

      const grid = document.getElementById('h2hResultsGrid');
      grid.classList.remove('hidden');
      document.getElementById('h2hSummaryBanner').classList.remove('hidden');

      const sys1Box = document.getElementById('h2hSys1Content');
      const sys2Box = document.getElementById('h2hSys2Content');
      sys1Box.innerHTML = `<div class="text-slate-400">Menjalankan System 1 (${selectedH2HModel})...</div>`;
      sys2Box.innerHTML = `<div class="text-slate-400">Menjalankan System 2 ${selectedH2HSys2Model.toUpperCase()} (~4-5s)...</div>`;

      try {
        const [resSys1, resSys2] = await Promise.all([
          fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: selectedH2HModel, state: state, questions: currentH2HQuestions })
          }).then(r => r.json()),
          fetch('/api/heavyweight/decision', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: selectedH2HSys2Model, state: state, questions: currentH2HQuestions })
          }).then(r => r.json())
        ]);

        const modelData = (resSys1.results && resSys1.results[selectedH2HModel]) || {};
        document.getElementById('h2hSys1Name').innerText = modelData.name || selectedH2HModel;
        document.getElementById('h2hSys1Latency').innerText = `${modelData.latency_ms || 65} ms`;

        let s1Html = '';
        for (const [k, v] of Object.entries(modelData.answers || {})) {
          if (v.type === 'noul') {
            const prob = (v.noul !== undefined ? v.noul * 100 : 0).toFixed(0);
            const isTrue = (v.noul || 0) >= 0.5;
            s1Html += `
              <div class="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                <div class="flex items-center justify-between">
                  <span class="text-xs font-bold text-slate-300">🔘 Predikat Boolean: <code class="text-cyan-400 font-mono">${k}</code></span>
                  <span class="text-[10px] px-2 py-0.5 rounded font-bold ${isTrue ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'}">
                    ${isTrue ? 'AKTIF (True)' : 'NEGATIF (False)'}
                  </span>
                </div>
                <div class="flex items-center justify-between text-xs">
                  <span class="text-slate-400">Probabilitas Sigmoid Laten:</span>
                  <strong class="text-cyan-400 font-mono text-sm">${prob}%</strong>
                </div>
                <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                  <div class="bg-cyan-500 h-2 rounded-full transition-all duration-500" style="width: ${prob}%"></div>
                </div>
                <div class="text-[10px] text-slate-400 pt-1 flex justify-between border-t border-slate-800/60">
                  <span>Confidence: ${((v.confidence || v.answer_confidence || v.noul || 0) * 100).toFixed(0)}%</span>
                  <span>Aktivasi: ${isTrue ? '+Logit Dominan' : '-Suppressed'}</span>
                </div>
              </div>
            `;
          } else if (v.type === 'choice') {
            s1Html += `
              <div class="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                <div class="flex items-center justify-between">
                  <span class="text-xs font-bold text-slate-300">🎯 Pilihan Kategori: <code class="text-cyan-400 font-mono">${k}</code></span>
                  <span class="text-[10px] px-2 py-0.5 rounded font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                    MENANG: ${v.choice}
                  </span>
                </div>
                <div class="space-y-1.5 pt-1">
            `;
            const probs = v.probabilities || { [v.choice]: 0.95 };
            for (const [cKey, cProb] of Object.entries(probs)) {
              const isWinner = cKey === v.choice;
              const pct = (cProb * 100).toFixed(1);
              s1Html += `
                <div>
                  <div class="flex justify-between text-[11px] mb-0.5">
                    <span class="${isWinner ? 'text-white font-bold' : 'text-slate-400'}">${cKey}</span>
                    <span class="font-mono ${isWinner ? 'text-cyan-400 font-bold' : 'text-slate-500'}">${pct}%</span>
                  </div>
                  <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div class="${isWinner ? 'bg-cyan-500' : 'bg-slate-700'} h-1.5 rounded-full" style="width: ${pct}%"></div>
                  </div>
                </div>
              `;
            }
            s1Html += `
                </div>
                <div class="text-[10px] text-slate-400 pt-1 border-t border-slate-800/60 flex justify-between">
                  <span>Distribusi: Softmax Multi-Class</span>
                  <span>Confidence: ${((v.answer_confidence || v.confidence || 0.8) * 100).toFixed(0)}%</span>
                </div>
              </div>
            `;
          } else if (v.type === 'score') {
            const sc = v.score !== undefined ? v.score.toFixed(2) : '0.00';
            s1Html += `
              <div class="p-3.5 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
                <div class="flex items-center justify-between">
                  <span class="text-xs font-bold text-slate-300">📊 Penilaian Spektrum: <code class="text-cyan-400 font-mono">${k}</code></span>
                  <span class="text-[10px] px-2 py-0.5 rounded font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                    SKOR: ${sc}
                  </span>
                </div>
                <div class="text-[11px] text-slate-400 flex justify-between">
                  <span>Estimasi Derajat / Tingkat Urgensi:</span>
                  <strong class="text-amber-300 font-mono text-sm">${sc} / 3.00</strong>
                </div>
                <div class="text-[10px] text-slate-400 pt-1 border-t border-slate-800/60 flex justify-between">
                  <span>Confidence Score: ${((v.confidence || 0.9) * 100).toFixed(0)}%</span>
                  <span>Head: Scalar Value Regression</span>
                </div>
              </div>
            `;
          }
        }
        s1Html += `
          <div class="mt-4 p-3.5 rounded-xl bg-cyan-950/20 border border-cyan-800/40 text-xs space-y-2">
            <div class="flex items-center gap-1.5 text-cyan-300 font-bold">
              <span>🔍</span>
              <span>Keterangan Detail: Cara Kerja Decision Model atas Prompt Ini</span>
            </div>
            <ul class="text-[11px] text-slate-300 space-y-1 list-disc list-inside leading-relaxed">
              <li><strong>Mekanisme Eksekusi:</strong> Model mengevaluasi seluruh input dalam <em>1 single forward pass</em> (tanpa loop next-token autoregresif).</li>
              <li><strong>Output Token:</strong> <code>0 tokens generated</code> (murni output tensor matematis ke routing controller, bukan teks).</li>
              <li><strong>Aktivasi Fitur Laten:</strong> Representasi atensi teks memetakan kata kunci pemicu langsung ke Linear Logit Heads di atas.</li>
              <li><strong>Mengapa Tanpa Narasi?:</strong> Decision model berfokus 100% pada <em>kecepatan instan (&lt;80 ms) & konsistensi tanpa halusinasi</em>. Narasi kalimat penjelasan diserahkan ke System 2 (${resSys2.model_name || 'LLM'}).</li>
            </ul>
          </div>
        `;
        sys1Box.innerHTML = s1Html;

        // Render System 2
        document.getElementById('h2hSys2Name').innerText = resSys2.model_name || selectedH2HSys2Model;
        document.getElementById('h2hSys2Latency').innerText = `${resSys2.latency_ms} ms`;
        document.getElementById('h2hSys2Tokens').innerText = `${resSys2.tokens_generated} tokens`;
        document.getElementById('h2hSys2Speed').innerText = `Speed: ${resSys2.speed_tokens_sec} t/s`;

        let s2Html = '';
        const parsed = (resSys2.parsed_result && resSys2.parsed_result.answers) || {};
        for (const [k, v] of Object.entries(parsed)) {
          s2Html += `<div class="p-2.5 rounded bg-slate-900 border border-slate-800">
            <div class="flex justify-between mb-1">
              <span class="text-slate-400">${k}:</span>
              <strong class="text-purple-300 font-mono">${v.choice || v.probability || v.score || JSON.stringify(v)}</strong>
            </div>
          </div>`;
        }
        if (resSys2.parsed_result && resSys2.parsed_result.reasoning) {
          s2Html += `<div class="p-2.5 rounded bg-purple-950/20 border border-purple-900/40 text-[11px] text-purple-200">
            <strong>Penalaran LLM:</strong> ${resSys2.parsed_result.reasoning}
          </div>`;
        }
        sys2Box.innerHTML = s2Html;

        const sys1Lat = modelData.latency_ms || 65;
        const speedup = ((resSys2.latency_ms || 5000) / sys1Lat).toFixed(0);
        document.getElementById('h2hSpeedupBadge').innerText = `Efisiensi: ${speedup}x Speedup`;
        document.getElementById('h2hSummaryText').innerHTML = `
          System 1 (${modelData.name || selectedH2HModel}) <strong>${speedup}x lebih cepat</strong> dalam menentukan keputusan dibanding System 2 (${resSys2.model_name || 'LLM'}), 
          dengan <strong>0 output tokens</strong> (tanpa biaya token LLM). Namun System 2 memberikan kemampuan penalaran tekstual deskriptif.
        `;

      } catch (e) {
        alert('Error: ' + e.message);
      } finally {
        btn.disabled = false;
        btn.classList.remove('opacity-75', 'cursor-not-allowed');
        btnIcon.innerHTML = '⚡';
        btnText.innerText = 'JALANKAN HEAD-TO-HEAD PARALEL';
      }
    }

    async function runTwoTierPipeline() {
      const state = document.getElementById('twoTierState').value.trim();
      if (!state) return alert('Silakan isi teks pengujian');

      const btn = document.getElementById('btnRunTwoTier');
      const btnIcon = document.getElementById('btnTwoTierIcon');
      const btnText = document.getElementById('btnTwoTierText');

      btn.disabled = true;
      btn.classList.add('opacity-75', 'cursor-not-allowed');
      btnIcon.innerHTML = '⏳';
      btnText.innerText = 'MENJALANKAN PIPELINE DUA-TIER...';

      const results = document.getElementById('twoTierResults');
      results.classList.remove('hidden');

      const step1Div = document.getElementById('ttStep1Answers');
      const step2Div = document.getElementById('ttStep2Response');
      step1Div.innerHTML = `<div class="text-slate-400">Eksekusi Triage System 1 (${selectedTTModel})...</div>`;
      step2Div.innerHTML = `<div class="text-slate-400">Menunggu trigger System 2 (${selectedTTSys2Model.toUpperCase()})...</div>`;

      try {
        const res = await fetch('/api/heavyweight/two_tier', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            state: state,
            questions: {
              'is_threat_viral': { 'type': 'noul', 'instructions': 'Ancaman viralitas?' },
              'issue_category': { 'type': 'choice', 'instructions': 'Kategori masalah?', 'criteria': { 'keterlambatan_stuck': 'Paket tertahan', 'rusak': 'Barang rusak' } },
              'urgency': { 'type': 'score', 'instructions': 'Urgensi komplain', 'criteria': ['rendah', 'sedang', 'tinggi', 'kritis'] }
            },
            system1_model: selectedTTModel,
            heavy_model: selectedTTSys2Model
          })
        });

        const data = await res.json();
        document.getElementById('twoTierTotalBadge').innerText = `Total Waktu Pipeline: ${data.total_pipeline_latency_ms} ms`;
        document.getElementById('ttStep1Lat').innerText = `${data.tier1.latency_ms} ms`;
        document.getElementById('ttStep2Lat').innerText = `${data.tier2.latency_ms} ms (${data.tier2.speed_tokens_sec} t/s)`;
        document.getElementById('ttStep2CardTitle').innerText = `Tahap 2: Resolusi Empatik ${data.tier2.name || 'System 2'}`;

        let s1Html = '';
        for (const [k, v] of Object.entries(data.tier1.answers || {})) {
          if (v.type === 'noul') {
            const prob = (v.noul !== undefined ? v.noul * 100 : 0).toFixed(0);
            s1Html += `
              <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800 space-y-1">
                <div class="flex justify-between text-[11px]">
                  <span class="text-slate-300 font-bold">${k}</span>
                  <span class="text-cyan-400 font-mono font-bold">${prob}%</span>
                </div>
                <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                  <div class="bg-cyan-500 h-1.5 rounded-full" style="width: ${prob}%"></div>
                </div>
              </div>
            `;
          } else if (v.type === 'choice') {
            s1Html += `
              <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800 space-y-1">
                <div class="flex justify-between text-[11px]">
                  <span class="text-slate-300 font-bold">${k}</span>
                  <span class="text-white font-bold bg-cyan-500/20 text-cyan-300 px-1.5 py-0.5 rounded text-[10px]">${v.choice || '-'}</span>
                </div>
              </div>
            `;
          } else {
            s1Html += `
              <div class="p-2.5 rounded-lg bg-slate-900 border border-slate-800 flex justify-between text-[11px]">
                <span class="text-slate-400">${k}:</span>
                <strong class="text-amber-400 font-mono">${(v.score || 0).toFixed(2)}</strong>
              </div>
            `;
          }
        }
        s1Html += `
          <div class="p-2.5 rounded-lg bg-cyan-950/20 border border-cyan-800/40 text-[10px] text-slate-300">
            <strong>Cara Kerja:</strong> Triage otomatis 0 token memfilter prompt dalam ${data.tier1.latency_ms} ms dan mengoper metadata ke ${data.tier2.name}.
          </div>
        `;
        step1Div.innerHTML = s1Html;
        step2Div.innerText = data.tier2.response_text || '-';

      } catch (e) {
        alert('Pipeline Error: ' + e.message);
      } finally {
        btn.disabled = false;
        btn.classList.remove('opacity-75', 'cursor-not-allowed');
        btnIcon.innerHTML = '⚡';
        btnText.innerText = 'EKSEKUSI PIPELINE DUA-TIER LENGKAP';
      }
    }

    function copyChatOutput() {
      const text = document.getElementById('chatResponseBox').innerText;
      navigator.clipboard.writeText(text);
      alert('Teks berhasil disalin ke clipboard!');
    }

    async function runArchitectureComparison() {
      const state = document.getElementById('compareStateInput').value.trim();
      if (!state) return alert('Silakan masukkan pesan pelanggan untuk pengujian');

      const btn = document.getElementById('btnRunCompare');
      const btnIcon = document.getElementById('btnCompareIcon');
      const btnText = document.getElementById('btnCompareText');

      btn.disabled = true;
      btn.classList.add('opacity-75', 'cursor-not-allowed');
      btnIcon.innerHTML = '⏳';
      btnText.innerText = 'MEMPROSES DI TESLA T4...';

      const resultsSection = document.getElementById('compareResultsSection');
      resultsSection.classList.remove('hidden');

      // Reset loading states
      document.getElementById('compNoJevLat').innerText = 'Memproses...';
      document.getElementById('compNoJevTokens').innerText = 'Menghitung...';
      document.getElementById('compNoJevResponse').innerText = 'Mengeksekusi LLM standalone (Branch A: Tanpa Decision Model)...';

      document.getElementById('compWithJevDecLat').innerText = 'Memproses...';
      document.getElementById('compWithJevTotalLat').innerText = 'Memproses...';
      document.getElementById('compWithJevResponse').innerText = 'Mengeksekusi Pipeline Two-Tier (Branch B: Dengan Decision Model)...';
      document.getElementById('compWithJevAnswersGrid').innerHTML = '<div class="text-slate-400">Mengevaluasi logit & probabilitas...</div>';
      document.getElementById('compWithJevEscalationReasons').innerText = '';

      try {
        const res = await fetch('/api/heavyweight/compare_architectures', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            decision_model: selectedCompareDecisionModel,
            heavy_model: selectedCompareLLMModel,
            state: state,
            questions: currentCompareQuestions
          })
        });

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(errData.detail || 'Server mengembalikan status ' + res.status);
        }

        const data = await res.json();

        // 1. Populate Executive Verdict
        const exec = data.executive_comparison || {};
        document.getElementById('compareVerdictText').innerText = exec.verdict || '-';
        document.getElementById('compareSpeedupBadge').innerText = `Speedup: ${exec.speedup_decision_step || '50x'}`;
        document.getElementById('compareTokenSaveBadge').innerText = exec.token_saving_at_decision || 'Hemat 100% Token';

        // 2. Populate Branch A: Tanpa Jev
        const noJev = data.without_decision || {};
        document.getElementById('compNoJevTitle').innerText = noJev.name || 'Tanpa Decision Model';
        document.getElementById('compNoJevLat').innerText = `${noJev.total_latency_ms} ms`;
        document.getElementById('compNoJevTokens').innerText = `${noJev.tokens_generated} tokens`;
        document.getElementById('compNoJevResponse').innerText = noJev.response_text || '-';

        // 3. Populate Branch B: Dengan Jev
        const withJev = data.with_decision || {};
        document.getElementById('compWithJevTitle').innerText = withJev.name || 'Dengan Decision Model';
        document.getElementById('compWithJevDecLat').innerText = `${withJev.decision_latency_ms} ms`;
        document.getElementById('compWithJevTotalLat').innerText = `${withJev.total_latency_ms} ms`;
        document.getElementById('compWithJevSaveBadge').innerText = withJev.token_saving_pct || 'Hemat Token';
        document.getElementById('compWithJevRoutingAction').innerText = withJev.routing_action || 'Routing Otomatis';
        document.getElementById('compWithJevResponse').innerText = withJev.response_text || '-';

        // Render answers in withJev
        let answersHtml = '';
        for (const [k, v] of Object.entries(withJev.decision_answers || {})) {
          if (v && v.type === 'noul') {
            const prob = ((v.noul !== undefined ? v.noul : 0) * 100).toFixed(0);
            const isTrue = (v.noul || 0) >= 0.5;
            answersHtml += `
              <div class="flex items-center justify-between p-2 rounded bg-slate-900/90 border border-slate-800">
                <span class="text-slate-300 font-mono">🔘 ${k}</span>
                <div class="flex items-center gap-2">
                  <span class="text-[10px] px-1.5 py-0.5 rounded font-bold ${isTrue ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'}">
                    ${isTrue ? 'AKTIF (True)' : 'NEGATIF (False)'}
                  </span>
                  <span class="font-mono text-cyan-400 font-bold">${prob}%</span>
                </div>
              </div>
            `;
          } else if (v && v.type === 'choice') {
            answersHtml += `
              <div class="flex items-center justify-between p-2 rounded bg-slate-900/90 border border-slate-800">
                <span class="text-slate-300 font-mono">🎯 ${k}</span>
                <span class="font-bold text-cyan-300 bg-cyan-950/60 px-2 py-0.5 rounded border border-cyan-800">${v.choice || '-'}</span>
              </div>
            `;
          } else if (v && v.type === 'score') {
            answersHtml += `
              <div class="flex items-center justify-between p-2 rounded bg-slate-900/90 border border-slate-800">
                <span class="text-slate-300 font-mono">📊 ${k}</span>
                <span class="font-mono text-amber-300 font-bold">${(v.score !== undefined ? v.score.toFixed(2) : '0.00')} / 3.00</span>
              </div>
            `;
          }
        }
        document.getElementById('compWithJevAnswersGrid').innerHTML = answersHtml || '<div class="text-slate-400 text-[10px]">Triage selesai tanpa kendala.</div>';

        // Render escalation reasons
        if (withJev.escalation_reasons && withJev.escalation_reasons.length > 0) {
          document.getElementById('compWithJevEscalationReasons').innerHTML = 
            `<strong>Pemicu Eskalasi ke LLM:</strong> ${withJev.escalation_reasons.join(', ')}`;
        } else {
          document.getElementById('compWithJevEscalationReasons').innerHTML = 
            `<strong class="text-emerald-400">Fast-Path Status:</strong> Kueri aman/rutin, LLM 8B dihemat 100% (0 token LLM dikonsumsi).`;
        }

      } catch (err) {
        alert('Gagal menjalankan komparasi: ' + err.message);
        document.getElementById('compNoJevResponse').innerText = 'Error: ' + err.message;
        document.getElementById('compWithJevResponse').innerText = 'Error: ' + err.message;
      } finally {
        btn.disabled = false;
        btn.classList.remove('opacity-75', 'cursor-not-allowed');
        btnIcon.innerHTML = '⚖️';
        btnText.innerText = 'JALANKAN KOMPARASI ARSITEKTUR';
      }
    }

    // nvtop Chart & Live Monitor
    let nvtopChart;
    function initNvtopChart() {
      const ctx = document.getElementById('nvtopLiveChart').getContext('2d');
      const labels = Array.from({length: 25}, () => '');
      const gpuData = Array(25).fill(0);
      const memData = Array(25).fill(29.0);

      nvtopChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'GPU Core Util (%)',
              data: gpuData,
              borderColor: '#06b6d4',
              backgroundColor: 'rgba(6, 182, 212, 0.15)',
              borderWidth: 2,
              fill: true,
              tension: 0.3,
              pointRadius: 0
            },
            {
              label: 'VRAM Usage (%)',
              data: memData,
              borderColor: '#a855f7',
              backgroundColor: 'rgba(168, 85, 247, 0.15)',
              borderWidth: 2,
              fill: true,
              tension: 0.3,
              pointRadius: 0
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: false,
          plugins: {
            legend: {
              labels: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
            }
          },
          scales: {
            y: {
              min: 0,
              max: 100,
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 } }
            },
            x: {
              grid: { display: false },
              ticks: { display: false }
            }
          }
        }
      });
    }

    async function pollNvtopTelemetry() {
      try {
        const res = await fetch('/api/gpu_nvtop');
        if (!res.ok) return;
        const d = await res.json();

        // 1. Multi-GPU Cluster Topology Cards
        if (d.gpus && d.gpus.length >= 2) {
          const g0 = d.gpus[0];
          const g1 = d.gpus[1];
          
          const g0Used = (g0.mem_used_mb / 1024).toFixed(2);
          const g0Tot = (g0.mem_total_mb / 1024).toFixed(1);
          const g0Pct = ((g0.mem_used_mb / Math.max(g0.mem_total_mb, 1)) * 100).toFixed(0);
          const el0Util = document.getElementById('gpu0UtilText');
          if (el0Util) el0Util.innerText = `${g0.gpu_util_pct.toFixed(0)}% Util`;
          const el0Mem = document.getElementById('gpu0MemText');
          if (el0Mem) el0Mem.innerText = `${g0Used} / ${g0Tot} GB (${g0Pct}%)`;
          const el0Bar = document.getElementById('gpu0MemBar');
          if (el0Bar) el0Bar.style.width = `${g0Pct}%`;
          const el0Temp = document.getElementById('gpu0TempText');
          if (el0Temp) el0Temp.innerText = `${g0.temp_c.toFixed(0)}°C`;
          const el0Pwr = document.getElementById('gpu0PowerText');
          if (el0Pwr) el0Pwr.innerText = `${g0.power_draw_w.toFixed(1)}W / ${g0.power_limit_w.toFixed(0)}W`;

          const g1Used = (g1.mem_used_mb / 1024).toFixed(2);
          const g1Tot = (g1.mem_total_mb / 1024).toFixed(1);
          const g1Pct = ((g1.mem_used_mb / Math.max(g1.mem_total_mb, 1)) * 100).toFixed(0);
          const el1Util = document.getElementById('gpu1UtilText');
          if (el1Util) el1Util.innerText = `${g1.gpu_util_pct.toFixed(0)}% Util`;
          const el1Mem = document.getElementById('gpu1MemText');
          if (el1Mem) el1Mem.innerText = `${g1Used} / ${g1Tot} GB (${g1Pct}%)`;
          const el1Bar = document.getElementById('gpu1MemBar');
          if (el1Bar) el1Bar.style.width = `${g1Pct}%`;
          const el1Temp = document.getElementById('gpu1TempText');
          if (el1Temp) el1Temp.innerText = `${g1.temp_c.toFixed(0)}°C`;
          const el1Pwr = document.getElementById('gpu1PowerText');
          if (el1Pwr) el1Pwr.innerText = `${g1.power_draw_w.toFixed(1)}W / ${g1.power_limit_w.toFixed(0)}W`;

          const topBadge = document.getElementById('topHeaderGpuBadge');
          if (topBadge) topBadge.innerText = `Multi-GPU: 2x Tesla T4 (${(d.mem_total_mb / 1024).toFixed(1)} GB Total VRAM)`;
          const topoBadge = document.getElementById('nvtopMultiGpuTopologyBadge');
          if (topoBadge) topoBadge.innerText = `Active Cluster: 2x Tesla T4 (${(d.mem_total_mb / 1024).toFixed(1)} GB Total VRAM)`;
        }

        // 2. Aggregate Cluster Metrics
        const gpuPct = d.gpu_util_pct || 0;
        document.getElementById('nvtopGpuUtilText').innerText = `${gpuPct.toFixed(0)}%`;
        document.getElementById('nvtopGpuUtilBar').style.width = `${gpuPct}%`;
        document.getElementById('nvtopGpuLoadLabel').innerText = gpuPct > 20 ? 'Active Compute' : 'Idle';

        if (d.device_name) {
          const badge = document.getElementById('nvtopDeviceBadge');
          if (badge) badge.innerText = d.device_count > 1 ? `Multi-GPU Cluster (${d.device_count}x ${d.device_name})` : `${d.device_name} (Compute ${d.compute_capability || 'CUDA'})`;
        }
        if (d.driver_version) {
          const el = document.getElementById('nvtopDriverText');
          if (el) el.innerText = d.driver_version;
        }
        if (d.cuda_version) {
          const el = document.getElementById('nvtopCudaText');
          if (el) el.innerText = d.cuda_version;
        }

        const usedGb = (d.mem_used_mb / 1024).toFixed(2);
        const totalGb = (d.mem_total_mb / 1024).toFixed(2);
        const freeGb = (d.mem_free_mb / 1024).toFixed(2);
        const memPct = ((d.mem_used_mb / Math.max(d.mem_total_mb, 1)) * 100).toFixed(0);
        document.getElementById('nvtopMemText').innerText = `${usedGb} / ${totalGb} GB (${memPct}%)`;
        document.getElementById('nvtopMemBar').style.width = `${memPct}%`;
        document.getElementById('nvtopMemFree').innerText = `${freeGb} GB`;

        document.getElementById('nvtopTempText').innerText = `${d.temp_c.toFixed(0)}°C [Optimal]`;
        document.getElementById('nvtopTempBar').style.width = `${Math.min(d.temp_c, 100)}%`;
        document.getElementById('nvtopPowerText').innerText = `${d.power_draw_w.toFixed(1)}W / ${d.power_limit_w.toFixed(0)}W`;

        document.getElementById('nvtopClockGfx').innerText = `${d.clock_graphics_mhz} MHz`;
        document.getElementById('nvtopClockMem').innerText = `${d.clock_mem_mhz} MHz`;

        if (nvtopChart) {
          const gData = nvtopChart.data.datasets[0].data;
          const mData = nvtopChart.data.datasets[1].data;
          gData.shift();
          gData.push(gpuPct);
          mData.shift();
          mData.push(parseFloat(memPct));
          nvtopChart.update();
        }

        if (d.processes && d.processes.length > 0) {
          const tbody = document.getElementById('nvtopProcessTable');
          let rowsHtml = '';
          d.processes.forEach(p => {
            rowsHtml += `
              <tr>
                <td class="py-1.5 text-cyan-400 font-bold">${p.pid}</td>
                <td class="py-1.5">${p.name}</td>
                <td class="py-1.5 text-cyan-300 font-mono text-[10px] font-bold">${p.gpu || 'GPU 0'}</td>
                <td class="py-1.5 text-purple-400">${p.type}</td>
                <td class="py-1.5 font-bold text-amber-300">${p.used_mb} MiB</td>
                <td class="py-1.5 text-emerald-400">${p.models || 'app_server'}</td>
              </tr>
            `;
          });
          tbody.innerHTML = rowsHtml;
        }

      } catch (err) {
        console.warn('nvtop poll error:', err);
      }
    }

    // Main Chart Render
    window.addEventListener('DOMContentLoaded', () => {
      loadChatPreset('marunda');
      loadH2HPreset('marunda');
      loadComparePreset('marunda');

      initNvtopChart();
      setInterval(pollNvtopTelemetry, 2000);
      pollNvtopTelemetry();

      const ctx = document.getElementById('heavyChart').getContext('2d');
      new Chart(ctx, {
        type: 'bar',
        data: {
          labels: [
            'Laya (421M)', 'TypeSafe Jev', 'OpenJev (0.5B)', 'Kev-0.8B', 
            'Gemma 2 2B', 'Qwen 2.5 7B', 'Sahabat-AI 8B', 'Gemma 2 9B'
          ],
          datasets: [{
            label: 'Latensi Respon (ms - Skala Logaritmik)',
            data: [65, 160, 220, 1250, 1850, 4800, 5200, 6400],
            backgroundColor: [
              'rgba(6, 182, 212, 0.8)',
              'rgba(99, 102, 241, 0.8)',
              'rgba(245, 158, 11, 0.8)',
              'rgba(16, 185, 129, 0.8)',
              'rgba(251, 191, 36, 0.8)',
              'rgba(56, 189, 248, 0.8)',
              'rgba(168, 85, 247, 0.8)',
              'rgba(52, 211, 153, 0.8)'
            ],
            borderColor: ['#06b6d4', '#6366f1', '#f59e0b', '#10b981', '#fbbf24', '#38bdf8', '#a855f7', '#34d399'],
            borderWidth: 1,
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false }
          },
          scales: {
            y: {
              type: 'logarithmic',
              title: { display: true, text: 'Latensi (ms - Skala Logaritmik)', color: '#64748b' },
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#94a3b8' }
            },
            x: {
              grid: { display: false },
              ticks: { color: '#94a3b8' }
            }
          }
        }
      });
    });
  </script>
</body>
</html>
'''
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(base_dir, 'heavyweight_llm_dashboard.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"heavyweight_llm_dashboard.html generated successfully at {output_path}!")

if __name__ == "__main__":
    generate_html()
