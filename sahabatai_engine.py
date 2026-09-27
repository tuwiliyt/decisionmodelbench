"""
Sahabat-AI 8B Engine for Heavyweight LLM & Two-Tier Brain Testing
Model: GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct (Q4_K_M GGUF)
Runtime: llama-cpp-python with full CUDA GPU offload on NVIDIA Tesla T4
"""

import time
import json
import re
import torch
from llama_cpp import Llama

DEFAULT_SYSTEM_PROMPT = (
    "Kamu adalah Sahabat-AI, model bahasa kecerdasan buatan terdepan untuk Indonesia "
    "yang dikembangkan untuk memahami konteks budaya, hukum, bisnis, dan dialek lokal Indonesia secara mendalam. "
    "Berikan jawaban yang cerdas, solutif, empatik, dan berintegritas tinggi."
)

class SahabatAIEngine:
    def __init__(self, model_path: str = "/root/models/sahabatai-8b-q4.gguf", n_ctx: int = 2048, n_gpu_layers: int = -1):
        print(f"Loading Sahabat-AI 8B (Q4 GGUF, full CUDA GPU offload) from {model_path}...")
        t0 = time.perf_counter()
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers, # -1 offloads all layers to CUDA GPU
            n_ctx=n_ctx,
            verbose=False
        )
        self.load_duration_sec = round(time.perf_counter() - t0, 2)
        print(f"✓ Sahabat-AI 8B loaded in {self.load_duration_sec}s with {n_gpu_layers} layers offloaded.")

    def format_llama3_prompt(self, user_prompt: str, system_prompt: str = None) -> str:
        sys = system_prompt or DEFAULT_SYSTEM_PROMPT
        return (
            f"<|start_header_id|>system<|end_header_id|>\n"
            f"{sys}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{user_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
        )

    def generate(self, prompt: str, system_prompt: str = None, max_tokens: int = 256, temperature: float = 0.2):
        formatted_prompt = self.format_llama3_prompt(prompt, system_prompt)
        t0 = time.perf_counter()
        output = self.llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        duration = time.perf_counter() - t0
        text = output["choices"][0]["text"].strip()
        tokens = output["usage"]["completion_tokens"]
        speed_tps = round(tokens / duration, 2) if duration > 0 else 0

        return {
            "model": "Sahabat-AI 8B (GoTo & Indosat)",
            "text": text,
            "latency_ms": round(duration * 1000, 1),
            "tokens_generated": tokens,
            "speed_tokens_sec": speed_tps,
            "temperature": temperature
        }

    def decision(self, state: str, questions: dict, max_tokens: int = 300):
        """
        Prompt Sahabat-AI to generate structured JSON decision outputs (Autoregressive System 2),
        to compare directly against System 1 (Laya, Jev, Kev, OpenJev).
        """
        prompt = (
            f"Analisis input teks pelanggan berikut ini dan hasilkan KEPUTUSAN TERSTRUKTUR "
            f"hanya dalam format JSON valid tanpa teks pengantar atau penutup apapun.\n\n"
            f"Teks Pelanggan:\n\"{state}\"\n\n"
            f"Daftar Keputusan yang harus dijawab:\n"
            f"{json.dumps(questions, indent=2, ensure_ascii=False)}\n\n"
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
        res = self.generate(prompt, system_prompt=sys_prompt, max_tokens=max_tokens, temperature=0.1)
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        raw_text = res["text"]
        parsed_json = None
        json_valid = False

        # Attempt to parse json
        try:
            # find JSON block if wrapped in markdown
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
            "model": "Sahabat-AI 8B (Autoregressive JSON Mode)",
            "latency_ms": duration_ms,
            "tokens_generated": res["tokens_generated"],
            "speed_tokens_sec": res["speed_tokens_sec"],
            "json_valid": json_valid,
            "parsed_result": parsed_json,
            "raw_output": raw_text
        }

    def two_tier_pipeline(self, state: str, questions: dict, tier1_func, tier1_name: str = "System 1 (Decision Model)"):
        """
        Executes Two-Tier Brain:
        Tier 1 (System 1 - Laya/Kev/Jev/OpenJev): <100ms ultra-fast triage
        Tier 2 (System 2 - Sahabat-AI 8B): Empathic generation if escalation triggered
        """
        t0 = time.perf_counter()
        tier1_result = tier1_func(state, questions)
        tier1_duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Build prompt for Sahabat-AI with Tier 1 metadata
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
        tier2_res = self.generate(prompt, max_tokens=300, temperature=0.3)
        tier2_duration_ms = round((time.perf_counter() - t1) * 1000, 1)
        total_pipeline_ms = round(tier1_duration_ms + tier2_duration_ms, 1)

        return {
            "tier1": {
                "name": tier1_name,
                "latency_ms": tier1_duration_ms,
                "answers": tier1_result
            },
            "tier2": {
                "name": "System 2 (Sahabat-AI 8B)",
                "latency_ms": tier2_duration_ms,
                "tokens_generated": tier2_res["tokens_generated"],
                "speed_tokens_sec": tier2_res["speed_tokens_sec"],
                "response_text": tier2_res["text"]
            },
            "total_pipeline_latency_ms": total_pipeline_ms
        }

