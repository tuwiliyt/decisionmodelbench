"""
Unified AI Decision Models Benchmark: Jev (Cloud) vs Laya (Local GPU) vs Kev (Local GPU)
Workspace: /content/drive/MyDrive/AIPROJECT/DecisionModel
"""

import time
import json
import urllib.request
import urllib.error
import torch

from laya import Router
from kev.checkpoint import Checkpoint, LoadOptions
from kev.api import SystemOneRequest
from kev.serve import Server

from config import JEV_API_KEY, JEV_ENDPOINT

def query_jev(state: str, questions: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {JEV_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "jev-latest",
        "state": state,
        "questions": questions
    }
    req = urllib.request.Request(
        JEV_ENDPOINT,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        latency = (time.perf_counter() - t0) * 1000
        res = json.loads(resp.read().decode("utf-8"))
        res["client_latency_ms"] = round(latency, 2)
        return res

def run_benchmarks():
    print("=" * 70)
    print("🤖 AI DECISION MODELS BENCHMARK (System One / Non-Autoregressive)")
    print("=" * 70)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Hardware: {gpu_name} ({vram:.1f} GB VRAM) | CUDA Active")
    else:
        print("Hardware: CPU Only")
    print("=" * 70)

    # 1. Initialize Local Models
    print("\n[1/3] Loading Laya (ModernBERT-based, ~421M params)...")
    laya_router = Router()

    print("[2/3] Loading Kev-0.8B (Qwen3.5-0.8B + LoRA Pointer Head)...")
    ck = Checkpoint("jaredpalmer/kev-0.8b")
    tok, model = ck.load("cuda", LoadOptions(dtype=torch.float16))
    kev_srv = Server(checkpoint=ck, tok=tok, model=model, device="cuda")

    print("[3/3] Cloud API ready (TypeSafe Jev)...")

    # Define Test Cases
    test_cases = [
        {
            "name": "Case 1: Customer Churn & Billing Triage (English)",
            "state": "The user wants to cancel subscription due to high billing charges and demands an immediate refund.",
            "questions": {
                "is_cancellation": {
                    "type": "noul",
                    "instructions": "Is the user requesting a cancellation?"
                },
                "category": {
                    "type": "choice",
                    "instructions": "What is the primary department category?",
                    "criteria": {
                        "pricing": "Issues regarding cost, invoices, or billing",
                        "technical": "Issues regarding bugs, outages, or system failures",
                        "general": "General questions or feedback"
                    }
                },
                "churn_risk": {
                    "type": "score",
                    "instructions": "Rate the churn risk level",
                    "criteria": ["low", "medium", "high", "critical"]
                }
            }
        },
        {
            "name": "Case 2: Technical Incident Triage (English)",
            "state": "Database connection pool exhausted after deployment v2.4. Production API returning 500 errors across all nodes.",
            "questions": {
                "is_urgent": {
                    "type": "noul",
                    "instructions": "Does this require emergency P0 on-call escalation?"
                },
                "fault_domain": {
                    "type": "choice",
                    "instructions": "Which subsystem is failing?",
                    "criteria": {
                        "database": "Database, connections, storage",
                        "frontend": "UI, browser rendering, client app",
                        "billing": "Payment gateway, invoicing"
                    }
                },
                "severity": {
                    "type": "score",
                    "instructions": "Rate incident severity",
                    "criteria": ["low", "medium", "high", "critical"]
                }
            }
        },
        {
            "name": "Case 3: Multilingual Support Ticket (Bahasa Indonesia)",
            "state": "Halo admin, akun saya terdebit dua kali untuk langganan bulan ini padahal saya sudah minta berhenti berlangganan minggu lalu. Mohon pengembalian dana segera.",
            "questions": {
                "is_cancellation": {
                    "type": "noul",
                    "instructions": "Apakah pengguna meminta pembatalan langganan?"
                },
                "category": {
                    "type": "choice",
                    "instructions": "Kategori permasalahan tiket?",
                    "criteria": {
                        "pricing": "Masalah biaya, tagihan ganda, refund",
                        "technical": "Masalah bug atau kendala sistem",
                        "other": "Pertanyaan umum lainnya"
                    }
                },
                "urgency": {
                    "type": "score",
                    "instructions": "Tingkat urgensi komplain",
                    "criteria": ["rendah", "sedang", "tinggi", "kritis"]
                }
            }
        }
    ]

    for tc in test_cases:
        print("\n" + "#" * 70)
        print(f"📋 {tc['name']}")
        print(f"State: \"{tc['state']}\"")
        print("#" * 70)

        # A. Jev (Cloud API)
        try:
            res_jev = query_jev(tc["state"], tc["questions"])
            jev_lat = res_jev["client_latency_ms"]
            jev_ans = res_jev.get("answers", {})
        except Exception as e:
            res_jev = None
            jev_lat = -1
            jev_ans = {"error": str(e)}

        # B. Laya (Local GPU)
        t0 = time.perf_counter()
        res_laya = laya_router.predict(tc["state"], tc["questions"])
        laya_lat = (time.perf_counter() - t0) * 1000
        laya_ans = res_laya.get("answers", {})

        # C. Kev (Local GPU)
        req_kev = SystemOneRequest(model="kev-latest", state=tc["state"], questions=tc["questions"])
        t0 = time.perf_counter()
        res_kev = kev_srv.answer(req_kev)
        kev_lat = (time.perf_counter() - t0) * 1000
        kev_ans = res_kev.get("answers", {})

        # Display Summary Table
        print(f"\n{'Model':<15} | {'Deployment':<12} | {'Latency':<10} | {'Key Decision Output'}")
        print("-" * 75)

        # Helper to format answer summary
        def summarize(ans):
            items = []
            for k, v in ans.items():
                if v.get("type") == "noul":
                    items.append(f"{k}={v.get('noul', 0):.2f}")
                elif v.get("type") == "choice":
                    p = v.get("probabilities", {}).get(v.get("choice"), 0)
                    items.append(f"{k}={v.get('choice')}({p*100:.0f}%)")
                elif v.get("type") == "score":
                    items.append(f"{k}={v.get('score', 0):.2f}")
            return ", ".join(items)

        print(f"{'Jev (TypeSafe)':<15} | {'Cloud API':<12} | {jev_lat:<7.1f} ms | {summarize(jev_ans)}")
        print(f"{'Laya (421M)':<15} | {'Local GPU':<12} | {laya_lat:<7.1f} ms | {summarize(laya_ans)}")
        print(f"{'Kev-0.8B':<15} | {'Local GPU':<12} | {kev_lat:<7.1f} ms | {summarize(kev_ans)}")

    print("\n" + "=" * 70)
    print("✅ Benchmark Selesai!")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmarks()
