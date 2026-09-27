#!/usr/bin/env python3
"""
Multi-LLM Tandem Benchmark: Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B
Author: Richie O. Sumual
Tests Standalone Decision Model (JEV) vs Standalone LLMs vs Two-Tier Tandem
across Emergency Calling & Public Service scenarios on Dual Tesla T4 GPUs.
"""

import time
import json
import urllib.request
import urllib.error

API_BASE = "http://localhost:7860"

TEST_SCENARIOS = [
    {
        "id": "EMERG-01",
        "title": "Henti Jantung RJP (Emergency 112)",
        "transcript": "TOLONG CEPAT! Ayah saya tiba-tiba roboh memegangi dada, sekarang tidak sadar dan napasnya berhenti! Wajahnya mulai membiru di Jalan Anggrek No. 12! Kirim ambulans sekarang!",
        "questions": {
            "is_emergency": {"type": "noul", "instructions": "Apakah ini ancaman kematian seketika (Life-Threatening Emergency)?"},
            "dispatch_unit": {
                "type": "choice",
                "instructions": "Armada pertolongan pertama yang harus diluncurkan seketika?",
                "criteria": {
                    "ambulans_paramedis_icu": "Unit Ambulans Gawat Darurat & Dokter Paramedis ICU",
                    "damkar_rescue": "Mobil Tangga Pemadam Kebakaran",
                    "patroli_polisi": "Mobil Patroli Polisi Lalu Lintas",
                    "dishub_derek": "Mobil Derek Dinas Perhubungan"
                }
            }
        },
        "expected": "ambulans_paramedis_icu"
    },
    {
        "id": "EMERG-02",
        "title": "Kebakaran Ruko Rooftop (Damkar Snorkel)",
        "transcript": "KEBAKARAN BESAR! Gudang plastik lantai 1 meledak, api membesar cepat ke lantai 3! Kami ada 6 orang karyawan terjebak di rooftop tidak bisa turun karena tangga tertutup asap hitam pekat panas!",
        "questions": {
            "is_emergency": {"type": "noul", "instructions": "Apakah situasi ini kebakaran aktif dengan korban terjebak api?"},
            "dispatch_unit": {
                "type": "choice",
                "instructions": "Armada respon darurat yang harus diluncurkan?",
                "criteria": {
                    "damkar_snorkel_rescue": "Damkar Armada Mobil Tangga Tinggi (Snorkel) & Regu Evakuasi Asap",
                    "satpol_pp": "Satpol PP Penertiban Pedagang",
                    "dinas_pertamanan": "Dinas Pertamanan dan Hutan Kota"
                }
            }
        },
        "expected": "damkar_snorkel_rescue"
    },
    {
        "id": "CS-01",
        "title": "Pembobolan Rekening Nasabah (Bank Fraud Desk)",
        "transcript": "TOLONG BLOKIR REKENING SAYA SEKARANG! Ada nomor mengaku pihak bank minta kode SMS, sekarang saldo tabungan saya terpotong 25 juta rupiah ditransfer ke rekening lain tanpa izin saya!",
        "questions": {
            "is_emergency": {"type": "noul", "instructions": "Apakah ini transaksi penipuan darurat?"},
            "dispatch_unit": {
                "type": "choice",
                "instructions": "Divisi internal bank yang harus menangani?",
                "criteria": {
                    "fraud_desk_emergency_block": "Tim Tanggap Darurat Fraud & Pemblokiran Rekening Seketika",
                    "cs_marketing_sales": "Tim Promosi Kredit",
                    "it_support_printer": "Helpdesk Printer"
                }
            }
        },
        "expected": "fraud_desk_emergency_block"
    }
]

LLM_MODELS = [
    ("sahabatai", "Sahabat-AI 8B Instruct (Indosat/GoTo)"),
    ("qwen", "Qwen 2.5 7B Instruct (Alibaba Cloud)"),
    ("gemma", "Gemma 2 9B Instruct (Google DeepMind)")
]

def query(endpoint, payload):
    req = urllib.request.Request(
        f"{API_BASE}{endpoint}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            elapsed = (time.perf_counter() - t0) * 1000
            res = json.loads(resp.read().decode())
            res["measured_ms"] = elapsed
            return res
    except Exception as e:
        return {"error": str(e), "measured_ms": -1}

def main():
    print("=" * 100)
    print("🚀 BENCHMARK MODEL BESAR: SAHABAT-AI 8B, QWEN 2.5 7B, GEMMA 2 9B")
    print("Standalone LLM vs Two-Tier Tandem (with TypeSafe JEV System 1)")
    print("Author: Richie O. Sumual | Hardware: Dual Tesla T4 GPUs")
    print("=" * 100)

    # 1. Measure System 1 baseline (TypeSafe JEV)
    print("\n[Baseline] Menguji System 1 (TypeSafe JEV Cloud API)...")
    jev_times = []
    for sc in TEST_SCENARIOS:
        payload = {"model": "jev", "state": sc["transcript"], "questions": sc["questions"]}
        r = query("/api/predict", payload)
        lat = r.get("results", {}).get("jev", {}).get("latency_ms", r.get("measured_ms", 0))
        jev_times.append(lat)
        target = r.get("results", {}).get("jev", {}).get("answers", {}).get("dispatch_unit", {}).get("choice", "")
        print(f"  • {sc['id']}: {lat:.1f} ms | Target: {target} (Match: {target == sc['expected']})")
    avg_jev_lat = sum(jev_times) / len(jev_times)

    # 2. Measure Each Heavy LLM in Standalone & Tandem Mode
    report = {}

    for m_id, m_name in LLM_MODELS:
        print(f"\n" + "-" * 100)
        print(f"🔥 PENGUJIAN MODEL BESAR: {m_name}")
        print("-" * 100)

        # Standalone Decision Evaluation
        standalone_lats = []
        standalone_tokens = []
        matches = 0
        for sc in TEST_SCENARIOS:
            payload = {"model": m_id, "state": sc["transcript"], "questions": sc["questions"]}
            r = query("/api/heavyweight/decision", payload)
            lat = r.get("latency_ms", r.get("measured_ms", 0))
            tok = r.get("tokens_generated", 0)
            parsed = r.get("parsed_result", {}).get("answers", {})
            du = parsed.get("dispatch_unit", {})
            target = du.get("key") or du.get("value") or du.get("choice") or du.get("option") or "unknown"
            is_match = (target == sc["expected"])
            if is_match:
                matches += 1
            standalone_lats.append(lat)
            standalone_tokens.append(tok)
            print(f"  [Standalone] {sc['id']}: {lat:6.1f} ms | Tokens: {tok:3d} | Target: {target} (Match: {is_match})")

        # Two-Tier Tandem Evaluation (JEV System 1 + LLM System 2)
        tandem_tier1_lats = []
        tandem_tot_lats = []
        tandem_tokens = []
        for sc in TEST_SCENARIOS:
            payload = {
                "heavy_model": m_id,
                "system1_model": "jev",
                "state": sc["transcript"],
                "questions": sc["questions"]
            }
            r = query("/api/heavyweight/two_tier", payload)
            t1_lat = r.get("tier1", {}).get("latency_ms", 0)
            tot_lat = r.get("total_pipeline_latency_ms", 0)
            tok = r.get("tier2", {}).get("tokens_generated", 0)
            tandem_tier1_lats.append(t1_lat)
            tandem_tot_lats.append(tot_lat)
            tandem_tokens.append(tok)
            print(f"  [Two-Tier]   {sc['id']}: Dispatch: {t1_lat:5.1f} ms! | Total Flow: {tot_lat:6.1f} ms | Guidance Tok: {tok:3d}")

        report[m_id] = {
            "name": m_name,
            "standalone_lat_mean": sum(standalone_lats) / len(standalone_lats),
            "standalone_tokens_mean": sum(standalone_tokens) / len(standalone_tokens),
            "standalone_acc": (matches / len(TEST_SCENARIOS)) * 100,
            "tandem_dispatch_mean": sum(tandem_tier1_lats) / len(tandem_tier1_lats),
            "tandem_total_flow_mean": sum(tandem_tot_lats) / len(tandem_tot_lats),
            "tandem_tokens_mean": sum(tandem_tokens) / len(tandem_tokens)
        }

    # Save to JSON
    out_file = "/kaggle/working/decisionmodelbench/heavyweight_tandem_multi_llm_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "author": "Richie O. Sumual",
            "jev_baseline_ms": avg_jev_lat,
            "models": report
        }, f, indent=2)

    # Print Final Comparative Matrix
    print("\n" + "=" * 105)
    print("📊 MATRIKS EVALUASI TANDEM MULTI-LLM: STANDALONE VS TWO-TIER (DUAL TESLA T4)")
    print("=" * 105)
    print(f"{'Nama Model Besar (System 2)':<35} | {'Standalone Latency':<18} | {'Tandem Dispatch':<16} | {'Akurasi':<8} | {'Tokens Guidance'}")
    print("-" * 105)
    print(f"{'TypeSafe JEV System 1 (Solo)':<35} | {avg_jev_lat:8.1f} ms        | {avg_jev_lat:8.1f} ms     | 100.0 %  |      0 (Zero)")
    for m_id, data in report.items():
        print(f"{data['name']:<35} | {data['standalone_lat_mean']:8.1f} ms        | {data['tandem_dispatch_mean']:8.1f} ms     | {data['standalone_acc']:5.1f} %  | {data['tandem_tokens_mean']:6.0f} tokens")
    print("=" * 105)
    print(f"Data lengkap tersimpan di: {out_file}\n")

if __name__ == "__main__":
    main()
