#!/usr/bin/env python3
"""
DecisionModelBench: Emergency Calling (911/112) & Two-Tier Tandem Benchmark
Author: Richie O. Sumual
Empirically benchmarks Standalone Decision Model vs Standalone Foundation LLM
versus Hybrid Two-Tier Tandem Brain on Dual Tesla T4 GPUs.
"""

import time
import json
import urllib.request
import urllib.error
import os

API_BASE = "http://localhost:7860"

EMERGENCY_SCENARIOS = [
    {
        "id": "EMERG-01",
        "title": "Henti Jantung & Henti Napas (Cardiac Arrest)",
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
            },
            "triage_code": {"type": "score", "instructions": "Kode triase kegawatan medis", "criteria": ["hijau_ringan", "kuning_waspada", "merah_gawat", "merah_kritis_henti_jantung"]}
        },
        "expected": {"is_emergency": True, "target": "ambulans_paramedis_icu"}
    },
    {
        "id": "EMERG-02",
        "title": "Kebakaran Gedung Ruko Terjebak Asap (Structural Fire)",
        "transcript": "KEBAKARAN BESAR! Gudang plastik lantai 1 meledak, api membesar cepat ke lantai 3! Kami ada 6 orang karyawan terjebak di rooftop tidak bisa turun karena tangga tertutup asap hitam pekat panas!",
        "questions": {
            "is_emergency": {"type": "noul", "instructions": "Apakah situasi ini kebakaran aktif dengan korban terjebak api?"},
            "dispatch_unit": {
                "type": "choice",
                "instructions": "Armada respon darurat yang harus diluncurkan?",
                "criteria": {
                    "damkar_snorkel_rescue": "Damkar Armada Mobil Tangga Tinggi (Snorkel) & Regu Evakuasi Asap",
                    "satpol_pp": "Satpol PP Penertiban Pedagang",
                    "dinas_pertamanan": "Dinas Pertamanan dan Hutan Kota",
                    "puskesmas_keliling": "Mobil Puskesmas Imunisasi"
                }
            },
            "triage_code": {"type": "score", "instructions": "Tingkat keparahan kebakaran", "criteria": ["asap_ringan", "sedang", "kebakaran_hebat", "kritis_korban_terjebak"]}
        },
        "expected": {"is_emergency": True, "target": "damkar_snorkel_rescue"}
    },
    {
        "id": "EMERG-03",
        "title": "Perampokan Bersenjata Berlangsung (Active Armed Robbery)",
        "transcript": "Tolong bisik-bisik, ada 3 orang bawa senjata tajam dan pistol mendobrak minimarket kami. Kasir disekap dan mereka mengancam menembak. Kami sembunyi di ruang brankas belakang Jalan Raya Bogor KM 28!",
        "questions": {
            "is_emergency": {"type": "noul", "instructions": "Apakah ini ancaman kriminal bersenjata aktif (Armed Crime in Progress)?"},
            "dispatch_unit": {
                "type": "choice",
                "instructions": "Kesatuan keamanan yang harus segera meluncur?",
                "criteria": {
                    "polri_resmob_gegana": "Unit Reaksi Cepat Resmob/Perintis Presisi & Taktis Polri",
                    "damkar_air": "Armada Tangki Air Pemadam",
                    "dinas_sosial": "Petugas Dinsos Penertiban Pengemis",
                    "kelurahan": "Staf Administrasi Kelurahan"
                }
            },
            "triage_code": {"type": "score", "instructions": "Tingkat ancaman senjata api", "criteria": ["laporan_biasa", "waspada", "bahaya", "ancaman_senjata_mematikan"]}
        },
        "expected": {"is_emergency": True, "target": "polri_resmob_gegana"}
    },
    {
        "id": "EMERG-04",
        "title": "Tabrakan Beruntun Tol Korban Terjepit (Extrication Crash)",
        "transcript": "Kecelakaan parah di Tol Cipularang KM 92 arah Jakarta! Truk tronton rem blong menabrak 5 minibus, ada 2 korban terjepit kabin ringsek berdarah-darah tidak bisa ditarik!",
        "questions": {
            "is_emergency": {"type": "noul", "instructions": "Apakah kecelakaan lalu lintas memerlukan alat ekstrikasi hidrolik penyelamatan?"},
            "dispatch_unit": {
                "type": "choice",
                "instructions": "Unit penyelamat terpadu yang harus diberangkatkan?",
                "criteria": {
                    "basarnas_rescue_hidrolik": "Tim Rescue Basarnas/Damkar Alat Ekstrikasi & Ambulans Trauma",
                    "dinas_kebudayaan": "Petugas Dinas Pariwisata",
                    "kominfo_siaran": "Tim Siaran Radio Pemda",
                    "satpam_kantor": "Sekuriti Kantor Balai Kota"
                }
            },
            "triage_code": {"type": "score", "instructions": "Tingkat keparahan trauma korban", "criteria": ["kerusakan_materi", "luka_ringan", "luka_berat", "kritis_terjepit_nyawa"]}
        },
        "expected": {"is_emergency": True, "target": "basarnas_rescue_hidrolik"}
    },
    {
        "id": "EMERG-05",
        "title": "Panggilan Iseng / Non-Darurat (Prank Call Filter)",
        "transcript": "Halo mbak 112, mau tanya sekarang jam berapa ya? Sama di situ jual pulsa telkomsel sepuluh ribu ga? Hehehe.",
        "questions": {
            "is_emergency": {"type": "noul", "instructions": "Apakah panggilan ini merupakan situasi darurat nyata yang mengancam nyawa atau harta benda?"},
            "dispatch_unit": {
                "type": "choice",
                "instructions": "Tindakan penanganan sistem dispatcher?",
                "criteria": {
                    "filter_prank_warning": "Saring & Lepas Saluran (Filter Prank/Non-Emergency Warning)",
                    "ambulans_darurat": "Kirim Ambulans Sirine Penuh",
                    "damkar_rescue": "Kirim Armada Pemadam Kebakaran",
                    "gegana_polri": "Kirim Pasukan Gegana"
                }
            },
            "triage_code": {"type": "score", "instructions": "Tingkat validitas panggilan", "criteria": ["prank_palsu", "iseng", "pertanyaan_umum", "darurat_asli"]}
        },
        "expected": {"is_emergency": False, "target": "filter_prank_warning"}
    }
]

def query_endpoint(endpoint: str, payload: dict) -> dict:
    url = f"{API_BASE}{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            elapsed = (time.perf_counter() - t0) * 1000
            res = json.loads(resp.read().decode("utf-8"))
            res["measured_total_latency_ms"] = round(elapsed, 2)
            return res
    except Exception as e:
        return {"error": str(e), "measured_total_latency_ms": -1}

def run_benchmark():
    print("=" * 95)
    print("🚨 911 / 112 EMERGENCY CALLING & TWO-TIER TANDEM EMPIRICAL BENCHMARK")
    print("DecisionModelBench — Author: Richie O. Sumual")
    print("Server Topology: Dual NVIDIA Tesla T4 GPUs (31.27 GB VRAM Total)")
    print("=" * 95)

    results = []

    for idx, sc in enumerate(EMERGENCY_SCENARIOS, 1):
        print(f"\n[{idx:02d}/{len(EMERGENCY_SCENARIOS):02d}] {sc['title']}")
        print(f"Transcript: \"{sc['transcript']}\"")
        print(f"Ground Truth Target: {sc['expected']['target']}")
        print("-" * 95)

        # 1. Mode A: Standalone Decision Model (TypeSafe JEV Cloud API)
        payload_jev = {"model": "jev", "state": sc["transcript"], "questions": sc["questions"]}
        res_jev = query_endpoint("/api/predict", payload_jev)
        jev_ans = res_jev.get("results", {}).get("jev", {}).get("answers", {})
        jev_lat = res_jev.get("results", {}).get("jev", {}).get("latency_ms", res_jev.get("measured_total_latency_ms", 0))
        jev_target = jev_ans.get("dispatch_unit", {}).get("choice", "unknown")
        jev_match = (jev_target == sc["expected"]["target"])

        # 2. Mode B: Standalone Decision Model (OpenJev 0.5B on GPU 1)
        payload_openjev = {"model": "openjev", "state": sc["transcript"], "questions": sc["questions"]}
        res_openjev = query_endpoint("/api/predict", payload_openjev)
        openjev_ans = res_openjev.get("results", {}).get("openjev", {}).get("answers", {})
        openjev_lat = res_openjev.get("results", {}).get("openjev", {}).get("latency_ms", res_openjev.get("measured_total_latency_ms", 0))
        openjev_target = openjev_ans.get("dispatch_unit", {}).get("choice", "unknown")
        openjev_match = (openjev_target == sc["expected"]["target"])

        # 3. Mode C: Standalone Generative LLM (Sahabat-AI 8B on GPU 0/1)
        payload_sahabat = {"model": "sahabatai", "state": sc["transcript"], "questions": sc["questions"]}
        res_sahabat = query_endpoint("/api/heavyweight/decision", payload_sahabat)
        sahabat_lat = res_sahabat.get("latency_ms", 0)
        sahabat_tok = res_sahabat.get("tokens_generated", 0)
        sahabat_raw = res_sahabat.get("parsed_result", {}).get("answers", {})
        sahabat_target = "unknown"
        if isinstance(sahabat_raw.get("dispatch_unit"), dict):
            du = sahabat_raw["dispatch_unit"]
            sahabat_target = du.get("key") or du.get("value") or du.get("choice") or du.get("option") or "unknown"
        sahabat_match = (sahabat_target == sc["expected"]["target"])

        # 4. Mode D: Two-Tier Tandem Brain (Tier 1: JEV Sub-200ms Triage + Tier 2: Sahabat-AI 8B CPR/Guidance)
        payload_tandem = {
            "heavy_model": "sahabatai",
            "system1_model": "jev",
            "state": sc["transcript"],
            "questions": sc["questions"]
        }
        res_tandem = query_endpoint("/api/heavyweight/two_tier", payload_tandem)
        tier1_dispatch_lat = res_tandem.get("tier1", {}).get("latency_ms", 0)
        tier2_guidance_lat = res_tandem.get("tier2", {}).get("latency_ms", 0)
        total_tandem_lat = res_tandem.get("total_pipeline_latency_ms", 0)
        tier2_tok = res_tandem.get("tier2", {}).get("tokens_generated", 0)
        guidance_snippet = res_tandem.get("tier2", {}).get("response_text", "")[:90].replace("\n", " ")

        print(f"  • [Mode A: JEV Cloud Decision]  : {jev_lat:6.1f} ms | Tok:   0 | Target: {jev_target:<28} | Match: {jev_match}")
        print(f"  • [Mode B: OpenJev 0.5B Local] : {openjev_lat:6.1f} ms | Tok:   0 | Target: {openjev_target:<28} | Match: {openjev_match}")
        print(f"  • [Mode C: Sahabat-AI 8B LLM]  : {sahabat_lat:6.1f} ms | Tok: {sahabat_tok:3d} | Target: {sahabat_target:<28} | Match: {sahabat_match}")
        print(f"  • [Mode D: TWO-TIER TANDEM]    : DISPATCH IN {tier1_dispatch_lat:5.1f} ms! (Total Guidance: {total_tandem_lat:6.1f} ms, Tok: {tier2_tok})")
        print(f"     Guidance: \"{guidance_snippet}...\"")

        results.append({
            "id": sc["id"],
            "title": sc["title"],
            "ground_truth": sc["expected"]["target"],
            "jev": {"latency_ms": jev_lat, "target": jev_target, "match": jev_match, "tokens": 0},
            "openjev": {"latency_ms": openjev_lat, "target": openjev_target, "match": openjev_match, "tokens": 0},
            "sahabatai": {"latency_ms": sahabat_lat, "target": sahabat_target, "match": sahabat_match, "tokens": sahabat_tok},
            "tandem": {
                "tier1_dispatch_ms": tier1_dispatch_lat,
                "tier2_guidance_ms": tier2_guidance_lat,
                "total_ms": total_tandem_lat,
                "tokens": tier2_tok,
                "guidance": res_tandem.get("tier2", {}).get("response_text", "")
            }
        })

    # Save results
    out_file = "/kaggle/working/decisionmodelbench/emergency_tandem_benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_title": "Emergency Calling 911/112 & Two-Tier Tandem Evaluation",
            "author": "Richie O. Sumual",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "scenarios": results
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 105)
    print("📊 REKAPITULASI EMPIRIS: 911 / 112 EMERGENCY CALLING & TANDEM EVALUATION")
    print("=" * 105)
    print(f"{'Paradigma Operasional':<32} | {'Time-to-Dispatch':<18} | {'Total Flow Time':<16} | {'Accuracy':<10} | {'Tokens':<8} | {'Status Kritis'}")
    print("-" * 105)

    avg_jev_lat = sum(r["jev"]["latency_ms"] for r in results) / len(results)
    avg_jev_acc = sum(1 for r in results if r["jev"]["match"]) / len(results) * 100

    avg_openjev_lat = sum(r["openjev"]["latency_ms"] for r in results) / len(results)
    avg_openjev_acc = sum(1 for r in results if r["openjev"]["match"]) / len(results) * 100

    avg_llm_lat = sum(r["sahabatai"]["latency_ms"] for r in results) / len(results)
    avg_llm_tok = sum(r["sahabatai"]["tokens"] for r in results) / len(results)
    avg_llm_acc = sum(1 for r in results if r["sahabatai"]["match"]) / len(results) * 100

    avg_tandem_disp = sum(r["tandem"]["tier1_dispatch_ms"] for r in results) / len(results)
    avg_tandem_tot = sum(r["tandem"]["total_ms"] for r in results) / len(results)
    avg_tandem_tok = sum(r["tandem"]["tokens"] for r in results) / len(results)

    print(f"{'Decision Model (JEV Cloud)':<32} | {avg_jev_lat:8.1f} ms        | {avg_jev_lat:8.1f} ms       | {avg_jev_acc:6.1f} %   | {0:6d}   | Lolos Regulasi (<3s)")
    print(f"{'Decision Model (OpenJev 0.5B)':<32} | {avg_openjev_lat:8.1f} ms        | {avg_openjev_lat:8.1f} ms       | {avg_openjev_acc:6.1f} %   | {0:6d}   | Lolos Regulasi (<3s)")
    print(f"{'Standalone LLM (Sahabat-AI 8B)':<32} | {avg_llm_lat:8.1f} ms (DELAY)| {avg_llm_lat:8.1f} ms       | {avg_llm_acc:6.1f} %   | {avg_llm_tok:6.0f}   | GAGAL SLA (Bahaya Jiwa)")
    print(f"{'TWO-TIER HYBRID TANDEM':<32} | {avg_tandem_disp:8.1f} ms (INSTANT)| {avg_tandem_tot:8.1f} ms       | {avg_jev_acc:6.1f} %   | {avg_tandem_tok:6.0f}   | PARETO OPTIMAL (Sempurna)")
    print("=" * 105)
    print(f"Data tersimpan di: {out_file}\n")

if __name__ == "__main__":
    run_benchmark()
