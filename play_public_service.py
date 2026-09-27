#!/usr/bin/env python3
"""
Interactive CLI for Public Service & Enterprise Customer Support Triage Arena
DecisionModelBench — Author: Richie O. Sumual
Allows testing real-life citizen grievances and customer service emergencies
across System 1 (Decision Models) and System 2 (Generative LLMs).
"""

import sys
import time
import json
import urllib.request

API_BASE = "http://localhost:7860"

SAMPLE_CASES = [
    {
        "id": "1",
        "title": "Banjir Bandang & Tanggul Jebol (Darurat 112)",
        "text": "DARURAT! Tanggul kali Ciliwung jebol di RW 07, air naik 2 meter sangat cepat. Banyak lansia dan balita terjebak di atap rumah, tolong perahu karet evakuasi sekarang!",
        "questions": {
            "is_emergency": {"type": "noul", "instructions": "Apakah situasi ini merupakan keadaan darurat bencana yang mengancam nyawa?"},
            "routing_target": {
                "type": "choice",
                "instructions": "Instansi yang harus segera dimobilisasi?",
                "criteria": {
                    "bpbd_basarnas_damkar": "Tim Tanggap Darurat Bencana, Basarnas, dan Pemadam Kebakaran",
                    "disdukcapil": "Dinas Kependudukan dan Catatan Sipil",
                    "dinas_pariwisata": "Dinas Kebudayaan dan Pariwisata",
                    "dinas_pendidikan": "Dinas Pendidikan"
                }
            },
            "urgency": {"type": "score", "instructions": "Tingkat urgensi eskalasi", "criteria": ["rendah", "sedang", "tinggi", "kritis_darurat"]}
        }
    },
    {
        "id": "2",
        "title": "Pungutan Liar Pembuatan e-KTP (SP4N LAPOR)",
        "text": "Saya mau lapor pungli di kantor kelurahan Sukamaju. Oknum staf minta uang pelicin 150 ribu per orang kalau mau e-KTP jadi cepat, kalau tidak bayar dibilang blangko kosong berbulan-bulan.",
        "questions": {
            "is_malpractice": {"type": "noul", "instructions": "Apakah laporan ini mengandung unsur pungli, korupsi, atau malpraktik birokrasi?"},
            "routing_target": {
                "type": "choice",
                "instructions": "Unit pengawas yang berwenang menindaklanjuti?",
                "criteria": {
                    "inspektorat_ombudsman": "Inspektorat Daerah, Tim Saber Pungli, dan Ombudsman RI",
                    "dinas_kebersihan": "Dinas Lingkungan Hidup dan Kebersihan",
                    "puskesmas": "Puskesmas dan Layanan Kesehatan",
                    "dishub": "Dinas Perhubungan"
                }
            },
            "urgency": {"type": "score", "instructions": "Tingkat prioritas investigasi", "criteria": ["informasi", "rendah", "prioritas_tinggi", "darurat"]}
        }
    },
    {
        "id": "3",
        "title": "Pembobolan Rekening Nasabah via Social Engineering (Fraud Desk)",
        "text": "TOLONG BLOKIR REKENING SAYA SEKARANG! Ada nomor mengaku pihak bank minta kode SMS, sekarang saldo tabungan saya terpotong 25 juta rupiah ditransfer ke rekening lain tanpa izin saya!",
        "questions": {
            "is_unauthorized_fraud": {"type": "noul", "instructions": "Apakah laporan ini mengindikasikan transaksi penipuan / unauthorized account takeover?"},
            "routing_target": {
                "type": "choice",
                "instructions": "Antrean layanan darurat internal perbankan?",
                "criteria": {
                    "fraud_desk_emergency_block": "Tim Tanggap Darurat Fraud & Pemblokiran Kartu/Rekening Seketika",
                    "cs_marketing_sales": "Tim Promosi Kredit dan Asuransi",
                    "cs_general_faq": "Layanan Tanya Jawab Umum Saldo",
                    "it_support_internal": "Helpdesk Printer Kantor"
                }
            },
            "urgency": {"type": "score", "instructions": "Kecepatan waktu respons yang diwajibkan", "criteria": ["24_jam", "4_jam", "1_jam", "seketika_di_bawah_60_detik"]}
        }
    },
    {
        "id": "4",
        "title": "Barang Palsu Salah Kirim & Ancaman Viral (Dispute E-Commerce)",
        "text": "Saya order iPhone 15 Pro Max 20 juta tapi yang datang sabun colek dalam kardus basah! Penjual ga respon dan kurir cuci tangan. Kalau ga ada refund hari ini saya viralin di Twitter/TikTok dan laporkan pasal penipuan!",
        "questions": {
            "is_viral_threat": {"type": "noul", "instructions": "Apakah konsumen mengancam eskalasi reputasi viral di media sosial atau jalur hukum?"},
            "routing_target": {
                "type": "choice",
                "instructions": "Tim eskalasi prioritas yang harus menangani?",
                "criteria": {
                    "priority_dispute_retur": "Tim Sengketa Transaksi Prioritas Tinggi & Investigasi Penjual",
                    "general_courier_faq": "FAQ Ongkos Kirim Biasa",
                    "seller_onboarding": "Tim Pendaftaran Toko Baru",
                    "warehouse_inventory": "Staf Gudang Pengepakan"
                }
            },
            "urgency": {"type": "score", "instructions": "Tingkat risiko reputasi perusahaan", "criteria": ["rendah", "sedang", "tinggi", "krisis_viral"]}
        }
    }
]

MODELS_LIST = [
    ("jev", "TypeSafe JEV System One (Cloud SaaS API)", "/api/predict"),
    ("openjev", "OpenJev 0.5B Logit Scorer (Local GPU 1)", "/api/predict"),
    ("kev", "Kev-0.8B LoRA Pointer (Local GPU 0)", "/api/predict"),
    ("laya", "Laya Multilingual 421M (Local GPU 0)", "/api/predict"),
    ("sahabatai", "Sahabat-AI 8B Instruct (Generative LLM)", "/api/heavyweight/decision")
]

def run_triage(case_data):
    print("\n" + "=" * 95)
    print(f"🏛️ SKENARIO TRISE: {case_data['title']}")
    print("=" * 95)
    print(f"Konteks Laporan: \"{case_data['text']}\"\n")
    print(f"{'Nama Model':<38} | {'Latensi':<10} | {'Tokens':<8} | {'Klasifikasi Target / Routing':<32}")
    print("-" * 95)

    for m_key, m_name, m_endpoint in MODELS_LIST:
        payload = {
            "model": m_key,
            "state": case_data["text"],
            "questions": case_data["questions"]
        }
        req = urllib.request.Request(
            f"{API_BASE}{m_endpoint}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                elapsed = (time.perf_counter() - t0) * 1000
                res = json.loads(resp.read().decode())
                
                if m_endpoint == "/api/predict":
                    ans = res.get("results", {}).get(m_key, {}).get("answers", {})
                    lat = res.get("results", {}).get(m_key, {}).get("latency_ms", elapsed)
                    tok = 0
                    target = "unknown"
                    for qk, qv in ans.items():
                        if isinstance(qv, dict) and qv.get("type") == "choice":
                            target = qv.get("choice", "unknown")
                else:
                    lat = res.get("latency_ms", elapsed)
                    tok = res.get("tokens_generated", 0)
                    parsed = res.get("parsed_result", {}).get("answers", {})
                    target = "unknown"
                    for qk, qv in parsed.items():
                        if isinstance(qv, dict) and qv.get("type") == "choice":
                            target = qv.get("key") or qv.get("value") or qv.get("choice") or "unknown"

                print(f"{m_name:<38} | {lat:7.1f} ms | {tok:6d}   | {target:<32}")
        except Exception as e:
            print(f"{m_name:<38} | ERROR      | {0:6d}   | {str(e)[:30]}")

    print("=" * 95)

def main():
    print("=========================================================================================")
    print("🏛️ ARENA SIMULASI TRIASE LAYANAN PUBLIK & CUSTOMER SERVICE KORPORAT")
    print("DecisionModelBench — Author: Richie O. Sumual")
    print("=========================================================================================")
    print("Pilih Skenario Pengujian:")
    for sc in SAMPLE_CASES:
        print(f" [{sc['id']}] {sc['title']}")
    print(" [A] Jalankan Semua Skenario Secara Sekuensial")
    print(" [Q] Keluar")
    print("-----------------------------------------------------------------------------------------")

    if len(sys.argv) > 1:
        choice = sys.argv[1].upper()
    else:
        choice = input("Masukkan Pilihan (1-4 / A / Q): ").strip().upper()

    if choice == "Q":
        return
    elif choice == "A":
        for sc in SAMPLE_CASES:
            run_triage(sc)
            time.sleep(1)
    else:
        selected = next((s for s in SAMPLE_CASES if s["id"] == choice), SAMPLE_CASES[0])
        run_triage(selected)

if __name__ == "__main__":
    main()
