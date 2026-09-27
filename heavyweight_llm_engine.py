"""
Unified Heavyweight LLM Engine for Indonesian LLM Arena & Decision Model Integration
Supported Models:
1. Sahabat-AI 8B Instruct (GoTo Company & Indosat Ooredoo Hutchison - Llama 3 CPT)
2. Qwen 2.5 7B Instruct (Alibaba Cloud - Top Multilingual & Indonesian)
3. Gemma 2 9B Instruct (Google DeepMind - Deep Reasoning & Multilingual)
4. Gemma 2 2B Instruct (Google DeepMind - Ultra-Fast Lightweight Heavyweight)
Runtime: llama-cpp-python with CUDA GPU Offload on NVIDIA Tesla T4
"""

import time
import json
import re
import gc
import os
import torch
from llama_cpp import Llama

def resolve_model_path(filename: str) -> str:
    possible_dirs = [
        os.environ.get("MODELS_DIR", ""),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
        "/root/models",
        os.path.expanduser("~/models"),
        "./models"
    ]
    for d in possible_dirs:
        if d and os.path.exists(os.path.join(d, filename)):
            return os.path.join(d, filename)
    base = os.environ.get("MODELS_DIR", "/root/models")
    return os.path.join(base, filename)

MODELS_CATALOG = {
    "sahabatai": {
        "id": "sahabatai",
        "name": "Sahabat-AI 8B Instruct",
        "org": "GoTo & Indosat",
        "parameters": "8.03B",
        "quantization": "Q4_K_M GGUF",
        "model_path": resolve_model_path("sahabatai-8b-q4.gguf"),
        "badge_color": "purple",
        "description": "Model fondasi kedaulatan digital Indonesia hasil kolaborasi GoTo & Indosat. Menguasai norma budaya lokal, dialek daerah, dan empati customer care.",
        "typical_speed_tps": "28 - 32 t/s",
        "vram_mb": 4800,
        "format": "llama3",
        "stop": ["<|eot_id|>", "<|end_of_text|>"]
    },
    "qwen": {
        "id": "qwen",
        "name": "Qwen 2.5 7B Instruct",
        "org": "Alibaba Cloud",
        "parameters": "7.61B",
        "quantization": "Q4_K_M GGUF",
        "model_path": resolve_model_path("Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
        "badge_color": "cyan",
        "description": "Model multilingual kelas dunia dengan penalaran logika, pemecahan masalah, coding, dan bahasa Indonesia formal yang sangat tajam dan presisi.",
        "typical_speed_tps": "30 - 35 t/s",
        "vram_mb": 4600,
        "format": "chatml",
        "stop": ["<|im_end|>", "<|endoftext|>"]
    },
    "gemma": {
        "id": "gemma",
        "name": "Gemma 2 9B Instruct",
        "org": "Google DeepMind",
        "parameters": "9.24B",
        "quantization": "Q4_K_M GGUF",
        "model_path": resolve_model_path("gemma-2-9b-it-Q4_K_M.gguf"),
        "badge_color": "emerald",
        "description": "Arsitektur generasi terbaru Google DeepMind dengan sliding window attention, penalaran inferensi mendalam, dan pemahaman multilingual tinggi.",
        "typical_speed_tps": "24 - 28 t/s",
        "vram_mb": 5600,
        "format": "gemma",
        "stop": ["<end_of_turn>"]
    },
    "gemma-2b": {
        "id": "gemma-2b",
        "name": "Gemma 2 2B Instruct",
        "org": "Google DeepMind",
        "parameters": "2.61B",
        "quantization": "Q4_K_M GGUF",
        "model_path": resolve_model_path("gemma-2-2b-it-Q4_K_M.gguf"),
        "badge_color": "amber",
        "description": "Versi ultra-cepat dan hemat memori dari Google. Mampu menghasilkan 50+ token/detik dengan konsumsi VRAM sangat rendah (~1.7 GB).",
        "typical_speed_tps": "50 - 65 t/s",
        "vram_mb": 1700,
        "format": "gemma",
        "stop": ["<end_of_turn>"]
    },
    "qwen-14b": {
        "id": "qwen-14b",
        "name": "Qwen 2.5 14B Instruct",
        "org": "Alibaba Cloud",
        "parameters": "14.7B",
        "quantization": "Q4_K_M GGUF",
        "model_path": resolve_model_path("Qwen2.5-14B-Instruct-Q4_K_M.gguf"),
        "badge_color": "blue",
        "description": "Flagship 14.7B Heavyweight untuk GPU VRAM besar (16GB-80GB: A10G, L4, RTX 3090/4090, A100). Penalaran tingkat tinggi dan konteks panjang Bahasa Indonesia.",
        "typical_speed_tps": "20 - 26 t/s",
        "vram_mb": 9200,
        "format": "chatml",
        "stop": ["<|im_end|>", "<|endoftext|>"]
    }
}

class HeavyweightLLMManager:
    def __init__(self, default_model: str = "sahabatai", n_ctx: int = None):
        self.catalog = MODELS_CATALOG
        from gpu_manager import get_hardware_profile
        hw = get_hardware_profile()
        self.n_ctx = n_ctx or hw["recommended_ctx"]
        self.active_model_id = None
        self.llm = None
        self.default_model = default_model
        
    def get_catalog(self) -> dict:
        result = {}
        for mid, mdata in self.catalog.items():
            item = dict(mdata)
            item["exists_on_disk"] = os.path.exists(item["model_path"])
            item["is_active"] = (self.active_model_id == mid)
            result[mid] = item
        return result

    def get_or_load_model(self, model_id: str = None) -> tuple[Llama, dict]:
        mid = (model_id or self.default_model).lower()
        if mid not in self.catalog:
            mid = "sahabatai"
            
        mdata = self.catalog[mid]
        path = mdata["model_path"]
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found on disk.")
            
        if self.active_model_id == mid and self.llm is not None:
            return self.llm, mdata
            
        # Hot-swap: unload previous model from CUDA memory
        if self.llm is not None:
            print(f"Hot-swap: Unloading {self.active_model_id} from VRAM...")
            del self.llm
            self.llm = None
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(0.3)
            
        from gpu_manager import get_llama_init_kwargs, get_hardware_profile
        hw = get_hardware_profile()
        init_kwargs = get_llama_init_kwargs(requested_ctx=self.n_ctx, model_size_gb=mdata.get("vram_mb", 4800)/1024)

        print(f"Loading {mdata['name']} ({mdata['parameters']}) on {hw['primary_device']} ({hw['tier']})...")
        t0 = time.perf_counter()
        self.llm = Llama(
            model_path=path,
            **init_kwargs
        )
        load_sec = round(time.perf_counter() - t0, 2)
        self.active_model_id = mid
        print(f"✓ {mdata['name']} ready in {load_sec}s with {hw['primary_device']} GPU acceleration (ctx={self.n_ctx}).")
        return self.llm, mdata

    def format_prompt(self, model_id: str, user_prompt: str, system_prompt: str = None) -> str:
        mid = (model_id or "sahabatai").lower()
        mdata = self.catalog.get(mid, self.catalog["sahabatai"])
        fmt = mdata["format"]

        if fmt == "llama3":
            sys = system_prompt or (
                "Kamu adalah Sahabat-AI, model bahasa kecerdasan buatan terdepan untuk Indonesia "
                "yang dikembangkan untuk memahami konteks budaya, hukum, bisnis, dan dialek lokal Indonesia secara mendalam. "
                "Berikan jawaban yang cerdas, solutif, empatik, dan berintegritas tinggi."
            )
            return (
                f"<|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        elif fmt == "chatml":
            sys = system_prompt or (
                "Kamu adalah Qwen, asisten AI canggih berbahasa Indonesia yang sangat cerdas, akurat, analitis, dan solutif."
            )
            return (
                f"<|im_start|>system\n{sys}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        elif fmt == "gemma":
            if system_prompt:
                combined = f"Instruksi Sistem: {system_prompt}\n\nPesan Pengguna:\n{user_prompt}"
            else:
                combined = user_prompt
            return (
                f"<start_of_turn>user\n{combined}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
        else:
            return user_prompt

    def generate(self, model_id: str = "sahabatai", prompt: str = "", system_prompt: str = None, max_tokens: int = 250, temperature: float = 0.2) -> dict:
        llm, mdata = self.get_or_load_model(model_id)
        formatted_prompt = self.format_prompt(model_id, prompt, system_prompt)
        
        t0 = time.perf_counter()
        output = llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=mdata["stop"]
        )
        duration = time.perf_counter() - t0
        text = output["choices"][0]["text"].strip()
        tokens = output["usage"]["completion_tokens"]
        speed_tps = round(tokens / max(duration, 0.001), 2)
        
        return {
            "model_id": mdata["id"],
            "model_name": mdata["name"],
            "model_org": mdata["org"],
            "parameters": mdata["parameters"],
            "badge_color": mdata["badge_color"],
            "text": text,
            "latency_ms": round(duration * 1000, 1),
            "tokens_generated": tokens,
            "speed_tokens_sec": speed_tps,
            "temperature": temperature
        }

    def decision(self, model_id: str = "sahabatai", state: str = "", questions: dict = None, max_tokens: int = 300) -> dict:
        """
        Prompt the selected LLM to produce structured JSON answers for comparison against System 1.
        """
        q_clean = questions or {}
        prompt = (
            f"Analisis input teks pelanggan berikut ini dan hasilkan KEPUTUSAN TERSTRUKTUR "
            f"hanya dalam format JSON valid tanpa teks pengantar atau penutup apapun.\n\n"
            f"Teks Pelanggan:\n\"{state}\"\n\n"
            f"Daftar Pertanyaan Keputusan:\n"
            f"{json.dumps(q_clean, indent=2, ensure_ascii=False)}\n\n"
            f"Ketentuan Format JSON:\n"
            f"- Untuk tipe 'noul': berikan float probabilitas antara 0.0 sampai 1.0 (contoh: {{\"noul\": 0.95}})\n"
            f"- Untuk tipe 'choice': pilih satu key dari kriteria pilihan yang paling tepat (contoh: {{\"choice\": \"keterlambatan_stuck\", \"confidence\": 0.90}})\n"
            f"- Untuk tipe 'score': berikan nilai float estimasi (contoh: {{\"score\": 2.85, \"confidence\": 0.92}})\n\n"
            f"Format Output Wajib:\n"
            f"{{\n"
            f"  \"answers\": {{\n"
            f"    \"<question_key>\": {{ ... }},\n"
            f"    \"<question_key>\": {{ ... }}\n"
            f"  }},\n"
            f"  \"reasoning\": \"Penjelasan singkat 1-2 kalimat\"\n"
            f"}}"
        )
        sys_prompt = "Kamu adalah decision engine extractor bahasa Indonesia yang sangat teliti dan selalu membalas dalam format JSON valid murni."
        t0 = time.perf_counter()
        res = self.generate(model_id=model_id, prompt=prompt, system_prompt=sys_prompt, max_tokens=max_tokens, temperature=0.1)
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        raw_text = res["text"]
        parsed_json = None
        json_valid = False

        try:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                parsed_json = json.loads(match.group(0))
                json_valid = True
            else:
                parsed_json = json.loads(raw_text)
                json_valid = True
        except Exception as e:
            parsed_json = {"error": f"JSON parse error: {str(e)}", "raw": raw_text}
            json_valid = False

        return {
            "model_id": res["model_id"],
            "model_name": res["model_name"],
            "latency_ms": duration_ms,
            "tokens_generated": res["tokens_generated"],
            "speed_tokens_sec": res["speed_tokens_sec"],
            "json_valid": json_valid,
            "parsed_result": parsed_json,
            "raw_output": raw_text
        }

    def two_tier_pipeline(self, heavy_model_id: str, state: str, questions: dict, tier1_func, tier1_name: str = "System 1 (Decision Model)"):
        """
        Executes Two-Tier Brain:
        Tier 1 (System 1 - Laya/Kev/Jev/OpenJev): <100ms ultra-fast triage
        Tier 2 (System 2 - Selected LLM: Sahabat-AI / Qwen / Gemma): Empathic generation
        """
        t0 = time.perf_counter()
        tier1_result = tier1_func(state, questions)
        tier1_duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        prompt = (
            f"Pelanggan mengirimkan pesan komplain berikut:\n"
            f"\"{state}\"\n\n"
            f"Hasil Triage Otomatis {tier1_name} (Kecepatan: {tier1_duration_ms} ms):\n"
            f"{json.dumps(tier1_result, indent=2, ensure_ascii=False)}\n\n"
            f"Sebagai agen Customer Care senior berintegritas tinggi di Indonesia, "
            f"buatkan respon balasan resmi yang sangat santun, profesional, solutif, "
            f"dan mampu meredakan emosi pelanggan tanpa menjanjikan hal yang melanggar SOP."
        )

        t1 = time.perf_counter()
        tier2_res = self.generate(model_id=heavy_model_id, prompt=prompt, max_tokens=300, temperature=0.3)
        tier2_duration_ms = round((time.perf_counter() - t1) * 1000, 1)
        total_pipeline_ms = round(tier1_duration_ms + tier2_duration_ms, 1)

        return {
            "tier1": {
                "name": tier1_name,
                "latency_ms": tier1_duration_ms,
                "answers": tier1_result
            },
            "tier2": {
                "model_id": tier2_res["model_id"],
                "name": f"System 2 ({tier2_res['model_name']})",
                "latency_ms": tier2_duration_ms,
                "tokens_generated": tier2_res["tokens_generated"],
                "speed_tokens_sec": tier2_res["speed_tokens_sec"],
                "response_text": tier2_res["text"]
            },
            "total_pipeline_latency_ms": total_pipeline_ms
        }
