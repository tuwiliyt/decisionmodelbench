#!/usr/bin/env python3
"""
build_github_pages.py
Generates the complete GitHub Pages portal and Google Scholar-optimized landing pages
for DecisionModelBench research papers by Richie O. Sumual (PANITA GORONTALO).
"""

import os
import shutil
import pypdf

BASE_URL = "https://tuwiliyt.github.io/decisionmodelbench"
AUTHOR = "Richie O. Sumual"
AFFILIATION = "PANITA GORONTALO & Advanced Agentic AI & Distributed Systems Research, Gorontalo, Indonesia"
EMAIL = "richie@panita.web.id"
PUB_DATE = "2026/09/27"
PUB_YEAR = "2026"

PAPERS_INFO = [
    {
        "id": "decisionmodelbench-en",
        "title": "The Autoregression Fallacy in Time-Critical Cyber-Physical Systems and Algorithmic Trading: An Empirical Benchmark of Non-Autoregressive Decision Models versus Foundation Large Language Models",
        "lang": "en",
        "lang_label": "English",
        "report_id": "PANITA-RR-2026-02-EN",
        "journal": "PANITA GORONTALO Independent Research Reports",
        "pdf_filename": "IEEE_Paper_DecisionModelBench_EN.pdf",
        "tex_filename": "IEEE_Paper_DecisionModelBench_EN.tex",
        "md_filename": "IEEE_Paper_DecisionModelBench_EN.md",
        "pages": 7,
        "badge": "Cyber-Physical Systems & Algorithmic Trading",
        "abstract": "The prevailing tendency to deploy autoregressive Large Language Models (LLMs) across arbitrary decision boundaries has introduced severe architectural pathologies into time-critical cyber-physical systems (CPS) and sub-second algorithmic trading. Autoregressive decoders suffer from an intrinsic generation lag: sequential next-token synthesis enforces an O(N) temporal execution delay and memory bandwidth saturation, inducing catastrophic state-decision drift where physical or market conditions evolve faster than policy resolution. In this paper, we present DecisionModelBench, a multi-domain empirical benchmark evaluating non-autoregressive decision models against foundation generative LLMs (Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B, and Gemma 2 2B) across distributed dual NVIDIA Tesla T4 GPU topologies and managed cloud APIs. We formulate the Continuous State Drift integral, Latency-Induced Order Book Slippage, and Negative Alpha Decay. Across three adversarial environments—tactical air defense (C-RAM/Iron Dome multi-threat discrimination under 20-missile magazine constraints and 3.5s reload windows), sub-second order book execution ($10,000 portfolio), and continuous dynamic arcade interception—non-autoregressive decision models (Laya 421M at 55 ms, TypeSafe JEV at 160 ms, OpenJev 0.5B at 210 ms) achieve superior performance with zero token waste. Conversely, autoregressive LLMs (2,200–2,800 ms latency) precipitate structural failure: total city collapse in air defense, -$19.41 P&L versus +$40.90 (Laya) in trading, and 100% loss-of-control in arcade kinematics. We rigorously define the microstructural boundaries separating ultra-high-frequency hardware (FPGA/C++ in microsecond regimes) from sub-second decision models, demonstrating that remote cloud decision APIs outperform local 8B LLMs due to sequential memory bus bottlenecks. Finally, we formalize a Decoupled Two-Tier Cognitive Architecture uniting sub-100ms deterministic reflex triage (System 1) with out-of-band asynchronous strategic deliberation (System 2).",
        "keywords": "Cyber-Physical Systems; Non-Autoregressive Models; Foundation Large Language Models; Algorithmic Trading; Air Defense Simulation; Real-Time Control; Latency Slippage; Alpha Decay; Two-Tier Architecture",
        "bibtex": """@article{sumual2026autoregression,
  title={The Autoregression Fallacy in Time-Critical Cyber-Physical Systems and Algorithmic Trading: An Empirical Benchmark of Non-Autoregressive Decision Models versus Foundation Large Language Models},
  author={Sumual, Richie O.},
  journal={PANITA GORONTALO Independent Research Reports},
  volume={2026},
  number={PANITA-RR-2026-02-EN},
  pages={1--7},
  year={2026},
  month={September},
  institution={PANITA GORONTALO},
  address={Gorontalo, Indonesia},
  url={https://tuwiliyt.github.io/decisionmodelbench/papers/decisionmodelbench-en.html}
}"""
    },
    {
        "id": "decisionmodelbench-id",
        "title": "Kekeliruan Autoregresi dalam Sistem Siber-Fisik Kritis-Waktu dan Perdagangan Algoritmik: Tolok Ukur Empiris Model Keputusan Non-Autoregresif Melawan Model Bahasa Besar Fondasional",
        "lang": "id",
        "lang_label": "Bahasa Indonesia",
        "report_id": "PANITA-RR-2026-02-ID",
        "journal": "Laporan Riset Mandiri PANITA GORONTALO",
        "pdf_filename": "IEEE_Paper_DecisionModelBench_ID.pdf",
        "tex_filename": "IEEE_Paper_DecisionModelBench_ID.tex",
        "md_filename": "IEEE_Paper_DecisionModelBench_ID.md",
        "pages": 7,
        "badge": "Sistem Siber-Fisik & Perdagangan Algoritmik",
        "abstract": "Kecenderungan mutakhir untuk menerapkan Model Bahasa Besar (Large Language Models atau LLM) berbasis autoregresif pada seluruh domain komputasi keputusan telah memicu patologi arsitektural yang parah dalam sistem siber-fisik (cyber-physical systems/CPS) kritis-waktu dan perdagangan algoritmik sub-detik. Dekoder autoregresif memiliki keterlambatan inferensi intrinsik: pembentukan token demi token secara berurutan memaksakan latensi eksekusi temporal berorde O(N) dan saturasi lebar pita memori (memory bandwidth saturation), yang menimbulkan State-Decision Drift (pergeseran status-keputusan) katastropik di mana kondisi fisik atau pasar bergerak lebih cepat daripada siklus resolusi kebijakan kendali. Artikel ini menyajikan DecisionModelBench, sebuah kerangka pengujian empiris multi-domain yang mengevaluasi model-model keputusan non-autoregresif melawan LLM fondasional generatif (Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B, dan Gemma 2 2B) pada klaster komputasi terdistribusi dual NVIDIA Tesla T4 GPU dan infrastruktur cloud API. Kami merumuskan secara analitis integral Pergeseran Status Kontinu, formula Pergeseran Harga Akibat Latensi, serta Pembusukan Alfa Eksponensial. Melalui tiga lingkungan uji berlawanan—pertahanan udara taktis (klasifikasi multi-ancaman C-RAM/Iron Dome dengan kendala magazin baterai 20-rudal dan jeda reload 3,5 detik), eksekusi buku pesanan sub-detik (portofolio tiruan $10.000), dan penangkisan proyektil dinamik kontinu (vektor kecepatan bola Brick Breaker)—model keputusan non-autoregresif (Laya 421M dengan latensi 55 ms, TypeSafe JEV 160 ms, OpenJev 0.5B 210 ms) secara konsisten mempertahankan integritas operasional dengan pemborosan nol token. Sebaliknya, LLM autoregresif (latensi 2.200–2.800 ms) berujung pada keruntuhan struktural: kehancuran kota total pada pertahanan udara, P&L negatif -$19,41 berbanding +$40,90 (Laya) pada perdagangan, dan kegagalan kendali 100% pada kinematika arkade. Kami secara ilmiah mendefinisikan batas mikrostruktur pasar dan merumuskan Arsitektur Kognitif Dua-Tingkat yang menyatukan penapisan refleks deterministik sub-100ms (Sistem 1) dengan penalaran strategis asinkron di luar jalur kritis (Sistem 2).",
        "keywords": "Sistem Siber-Fisik; Model Non-Autoregresif; Model Bahasa Besar Fondasional; Perdagangan Algoritmik; Pertahanan Udara; Kendali Waktu-Nyata; Pergeseran Latensi; Pembusukan Alfa; Arsitektur Dua-Tingkat",
        "bibtex": """@article{sumual2026kekeliruan,
  title={Kekeliruan Autoregresi dalam Sistem Siber-Fisik Kritis-Waktu dan Perdagangan Algoritmik: Tolok Ukur Empiris Model Keputusan Non-Autoregresif Melawan Model Bahasa Besar Fondasional},
  author={Sumual, Richie O.},
  journal={Laporan Riset Mandiri PANITA GORONTALO},
  volume={2026},
  number={PANITA-RR-2026-02-ID},
  pages={1--7},
  year={2026},
  month={September},
  institution={PANITA GORONTALO},
  address={Gorontalo, Indonesia},
  url={https://tuwiliyt.github.io/decisionmodelbench/papers/decisionmodelbench-id.html}
}"""
    },
    {
        "id": "publicservice-cs-en",
        "title": "Deterministic Intent Gating, Sovereign Emergency Calling, and Two-Tier Civic Triage: An Empirical Evaluation of Non-Autoregressive Decision Models versus Foundation Large Language Models in Municipal 911/112 Operations and High-Throughput Public Administration",
        "lang": "en",
        "lang_label": "English",
        "report_id": "PANITA-RR-2026-03-EN",
        "journal": "PANITA GORONTALO Independent Research Reports",
        "pdf_filename": "IEEE_Paper_PublicService_CS_EN.pdf",
        "tex_filename": "IEEE_Paper_PublicService_CS_EN.tex",
        "md_filename": "IEEE_Paper_PublicService_CS_EN.md",
        "pages": 15,
        "badge": "Emergency Calling 911/112 & Multi-LLM Tandem",
        "abstract": "The wholesale integration of autoregressive Large Language Models (LLMs) into municipal emergency hotlines (e.g., 911 / Indonesia 112), public administration dispatch channels, and enterprise customer support has introduced severe operational vulnerabilities: queue divergence, catastrophic service-level agreement (SLA) breaches, non-deterministic schema mutation, and privacy exposure of personally identifiable information (PII). In life-safety emergency dispatch, sequential next-token synthesis incurs multi-second latency (>4,500 ms to >12,500 ms), severely compromising the 3-to-5 minute clinical 'golden period' of out-of-hospital cardiac arrest and violating statutory dispatch timing regulations (<3 s). In this paper, we present an empirical evaluation conducted on DecisionModelBench, benchmarking non-autoregressive decision models against flagship foundation generative LLMs—including the Indonesian sovereign model Sahabat-AI 8B Instruct, Alibaba Cloud's Qwen 2.5 7B Instruct, and Google DeepMind's Gemma 2 9B Instruct deployed on dual NVIDIA Tesla T4 GPUs—across fifteen comprehensive scenarios: ten high-throughput civic and enterprise support cases and five critical life-safety 911/112 emergency calling scenarios. We formalize Erlang-C queue explosion dynamics, statutory SLA breach probability integrals, clinical survival probability decay functions, and schema entropy failure laws. Across civic triage, non-autoregressive decision models (TypeSafe JEV System One at 197.6 ms, OpenJev 0.5B at 540.6 ms, Kev-0.8B at 648.5 ms, and dedicated Laya 421M at 35.8–63.2 ms) achieve 100% schema compliance, up to 100% routing accuracy, zero token waste, and 100% SLA compliance (<500 ms) at $0.05 per 100,000 queries. In critical emergency calling, Standalone LLMs exhibit dangerous dispatch latencies: 3,340.0 ms for Sahabat-AI 8B, 10,024.4 ms for Qwen 2.5 7B, and 12,681.7 ms for Gemma 2 9B, with Qwen exhibiting severe syntax fragility (33.3% standalone accuracy due to markdown fence wrapping). Conversely, our Two-Tier Hybrid Tandem Pipeline decouples physical CAD mobilization (<220 ms, up to 64.2x acceleration, 100% deterministic accuracy, 0 tokens) from asynchronous high-fidelity conversational guidance (207–294 tokens), establishing the definitive paradigm for life-safety emergency operations.",
        "keywords": "Public Administration AI; Sovereign AI; Emergency Calling 911/112; Two-Tier Hybrid Brain Pipeline; Intent Gating; Non-Autoregressive Decision Models; Large Language Models; Erlang-C Queueing; Golden Period Resuscitation; Data Sovereignty; Total Cost of Ownership",
        "bibtex": """@article{sumual2026deterministic,
  title={Deterministic Intent Gating, Sovereign Emergency Calling, and Two-Tier Civic Triage: An Empirical Evaluation of Non-Autoregressive Decision Models versus Foundation Large Language Models in Municipal 911/112 Operations and High-Throughput Public Administration},
  author={Sumual, Richie O.},
  journal={PANITA GORONTALO Independent Research Reports},
  volume={2026},
  number={PANITA-RR-2026-03-EN},
  pages={1--15},
  year={2026},
  month={September},
  institution={PANITA GORONTALO},
  address={Gorontalo, Indonesia},
  url={https://tuwiliyt.github.io/decisionmodelbench/papers/publicservice-cs-en.html}
}"""
    },
    {
        "id": "publicservice-cs-id",
        "title": "Penapisan Niat Deterministik, Panggilan Darurat Berdaulat 112/911, dan Triase Sipil Dua-Tingkat: Evaluasi Empiris Model Keputusan Non-Autoregresif Melawan Model Bahasa Besar Fondasional pada Operasi Penyelamatan Nyawa dan Administrasi Publik Bervolume Tinggi",
        "lang": "id",
        "lang_label": "Bahasa Indonesia",
        "report_id": "PANITA-RR-2026-03-ID",
        "journal": "Laporan Riset Mandiri PANITA GORONTALO",
        "pdf_filename": "IEEE_Paper_PublicService_CS_ID.pdf",
        "tex_filename": "IEEE_Paper_PublicService_CS_ID.tex",
        "md_filename": "IEEE_Paper_PublicService_CS_ID.md",
        "pages": 13,
        "badge": "Panggilan Darurat 112/911 & Tandem Multi-LLM",
        "abstract": "Integrasi masif Model Bahasa Besar (Large Language Models/LLM) autoregresif ke dalam kanal tanggap darurat terpadu pemerintah daerah (Panggilan Darurat 112 Indonesia / 911), kanal disposisi administrasi publik nasional (SP4N-LAPOR!), serta pusat layanan pelanggan korporat telah menyingkap kerentanan operasional yang kritis: penumpukan antrean ekstrem, kegagalan pemenuhan kesepakatan tingkat layanan (Service Level Agreement/SLA), mutasi sintaksis skema data JSON yang non-deterministik, serta risiko kebocoran data pribadi warga (Personally Identifiable Information/PII). Pada triase darurat penyelamatan nyawa, sintesis teks sekuensial token-demi-token menimbulkan latensi multi-detik (>3.300 ms hingga >12.600 ms) yang memangkas secara fatal batas klinis 'periode emas' (golden period) 3 hingga 5 menit pada kasus henti jantung di luar rumah sakit serta melanggar regulasi batas waktu pengiriman armada (<3 detik). Dalam makalah ini, kami menyajikan evaluasi empiris menggunakan tolok ukur DecisionModelBench, yang menguji model keputusan non-autoregresif melawan jajaran LLM generatif fondasional unggulan dunia—meliputi model kedaulatan Indonesia Sahabat-AI 8B Instruct, Qwen 2.5 7B Instruct dari Alibaba Cloud, dan Gemma 2 9B Instruct dari Google DeepMind pada peladen akselerator ganda NVIDIA Tesla T4—melintasi lima belas skenario komprehensif: sepuluh skenario sipil dan korporat serta lima skenario kritis panggilan darurat 112/911. Kami memformulasikan dinamika ledakan antrean Erlang-C, integral probabilitas pelanggaran batas waktu SLA, fungsi peluruhan probabilitas kelangsungan hidup klinis pasien henti jantung, serta hukum entropi kegagalan skema JSON. Secara empiris, model keputusan non-autoregresif mencapai 100% kepatuhan skema, akurasi perutean semantik hingga 100%, serta nol emisi token dengan biaya $0,05 per 100.000 transaksi. Sebaliknya, LLM Mandiri menunjukkan latensi pengiriman armada yang berbahaya: 3.340,0 ms untuk Sahabat-AI 8B, 10.024,4 ms untuk Qwen 2.5 7B, dan 12.681,7 ms untuk Gemma 2 9B. Menjawab krisis ini, kami mengevaluasi Pipeline Tandem Hibrida Dua-Tingkat di mana Tingkat 1 (Model Keputusan JEV) mengeksekusi disipasi sinyal pengiriman armada penyelamat secara instan dalam 178,5 ms hingga 217,7 ms (<220 ms, akselerasi hingga 64,2x, akurasi 100%, 0 token), sementara Tingkat 2 secara paralel/asinkron mengasimilasi metadata triase guna menyintesis panduan verbal resusitasi jantung paru (RJP) yang mendalam (207–294 token).",
        "keywords": "Kecerdasan Buatan Sektor Publik; Kedaulatan AI; Panggilan Darurat 112/911; Pipeline Tandem Hibrida Dua-Tingkat; Penapisan Niat; Model Keputusan Non-Autoregresif; Model Bahasa Besar; Antrean Erlang-C; Resusitasi Periode Emas; Pelindungan Data Pribadi; Total Biaya Kepemilikan",
        "bibtex": """@article{sumual2026penapisan,
  title={Penapisan Niat Deterministik, Panggilan Darurat Berdaulat 112/911, dan Triase Sipil Dua-Tingkat: Evaluasi Empiris Model Keputusan Non-Autoregresif Melawan Model Bahasa Besar Fondasional pada Operasi Penyelamatan Nyawa dan Administrasi Publik Bervolume Tinggi},
  author={Sumual, Richie O.},
  journal={Laporan Riset Mandiri PANITA GORONTALO},
  volume={2026},
  number={PANITA-RR-2026-03-ID},
  pages={1--13},
  year={2026},
  month={September},
  institution={PANITA GORONTALO},
  address={Gorontalo, Indonesia},
  url={https://tuwiliyt.github.io/decisionmodelbench/papers/publicservice-cs-id.html}
}"""
    }
]

def render_scholar_meta_tags(paper):
    pdf_url = f"{BASE_URL}/papers/{paper['pdf_filename']}"
    landing_url = f"{BASE_URL}/papers/{paper['id']}.html"
    return f"""
    <!-- Google Scholar & Highwire Press Bibliographic Meta Tags -->
    <meta name="citation_title" content="{paper['title']}">
    <meta name="citation_author" content="Sumual, Richie O.">
    <meta name="citation_author" content="Richie O. Sumual">
    <meta name="citation_author_institution" content="PANITA GORONTALO">
    <meta name="citation_author_email" content="{EMAIL}">
    <meta name="citation_publication_date" content="{PUB_DATE}">
    <meta name="citation_date" content="{PUB_DATE}">
    <meta name="citation_online_date" content="{PUB_DATE}">
    <meta name="citation_year" content="{PUB_YEAR}">
    <meta name="citation_journal_title" content="{paper['journal']}">
    <meta name="citation_technical_report_number" content="{paper['report_id']}">
    <meta name="citation_pdf_url" content="{pdf_url}">
    <meta name="citation_language" content="{paper['lang']}">
    <meta name="citation_keywords" content="{paper['keywords']}">
    <meta name="citation_abstract" content="{paper['abstract']}">

    <!-- Dublin Core Metadata for Academic Repositories -->
    <meta name="DC.title" content="{paper['title']}">
    <meta name="DC.creator" content="Sumual, Richie O.">
    <meta name="DC.contributor" content="PANITA GORONTALO">
    <meta name="DC.date" content="2026-09-27">
    <meta name="DC.type" content="Text">
    <meta name="DC.format" content="application/pdf">
    <meta name="DC.language" content="{paper['lang']}">
    <meta name="DC.publisher" content="PANITA GORONTALO">
    <meta name="DC.identifier" content="{pdf_url}">
    <meta name="DC.description" content="{paper['abstract']}">

    <!-- Schema.org ScholarlyArticle JSON-LD -->
    <script type="application/ld+json">
    {{
      "@context": "https://schema.org",
      "@type": "ScholarlyArticle",
      "headline": "{paper['title']}",
      "name": "{paper['title']}",
      "author": {{
        "@type": "Person",
        "name": "Richie O. Sumual",
        "email": "{EMAIL}",
        "affiliation": {{
          "@type": "Organization",
          "name": "PANITA GORONTALO",
          "address": {{
            "@type": "PostalAddress",
            "addressLocality": "Gorontalo",
            "addressCountry": "ID"
          }}
        }}
      }},
      "datePublished": "2026-09-27",
      "description": "{paper['abstract']}",
      "url": "{landing_url}",
      "encoding": {{
        "@type": "MediaObject",
        "contentUrl": "{pdf_url}",
        "encodingFormat": "application/pdf"
      }}
    }}
    </script>
"""

def generate_paper_landing_page(paper):
    pdf_url = f"{BASE_URL}/papers/{paper['pdf_filename']}"
    meta_tags = render_scholar_meta_tags(paper)
    
    html = f"""<!DOCTYPE html>
<html lang="{paper['lang']}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{paper['title']} | Richie O. Sumual</title>
    {meta_tags}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0f1d;
            --bg-secondary: #121a2f;
            --bg-card: #18233e;
            --border-color: rgba(99, 140, 255, 0.2);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.35);
            --accent-green: #10b981;
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: var(--font-sans);
            line-height: 1.65;
            padding: 0;
            margin: 0;
        }}
        .navbar {{
            background: rgba(18, 26, 47, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            padding: 16px 24px;
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .nav-brand {{
            font-weight: 700;
            font-size: 1.1rem;
            color: #fff;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .nav-brand span {{
            color: var(--accent);
        }}
        .nav-links {{
            display: flex;
            gap: 18px;
        }}
        .nav-links a {{
            color: var(--text-muted);
            text-decoration: none;
            font-size: 0.9rem;
            font-weight: 500;
            transition: color 0.2s;
        }}
        .nav-links a:hover {{
            color: #fff;
        }}
        .container {{
            max-width: 960px;
            margin: 40px auto;
            padding: 0 24px;
        }}
        .badge-bar {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }}
        .badge-blue {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
        .badge-green {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-purple {{ background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }}
        h1.paper-title {{
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1.3;
            margin-bottom: 24px;
            color: #ffffff;
            letter-spacing: -0.02em;
        }}
        .author-box {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 32px;
        }}
        .author-name {{
            font-size: 1.25rem;
            font-weight: 700;
            color: #fff;
        }}
        .author-affiliation {{
            color: #93c5fd;
            font-weight: 600;
            margin: 4px 0;
            font-size: 0.95rem;
        }}
        .author-meta {{
            color: var(--text-muted);
            font-size: 0.88rem;
        }}
        .author-meta a {{
            color: var(--accent);
            text-decoration: none;
        }}
        .cta-box {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            margin-bottom: 40px;
        }}
        .btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            padding: 14px 28px;
            border-radius: 8px;
            font-size: 0.95rem;
            font-weight: 600;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .btn-primary {{
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: #fff;
            box-shadow: 0 4px 14px var(--accent-glow);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .btn-primary:hover {{
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px var(--accent-glow);
        }}
        .btn-secondary {{
            background: var(--bg-card);
            color: #e2e8f0;
            border: 1px solid var(--border-color);
        }}
        .btn-secondary:hover {{
            background: #202d4f;
            color: #fff;
            transform: translateY(-2px);
        }}
        .section-title {{
            font-size: 1.25rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 8px;
        }}
        .abstract-text {{
            font-size: 1.05rem;
            line-height: 1.8;
            color: #cbd5e1;
            text-align: justify;
            margin-bottom: 32px;
            background: rgba(18, 26, 47, 0.5);
            padding: 24px;
            border-radius: 8px;
            border-left: 4px solid var(--accent);
        }}
        .keywords-box {{
            margin-bottom: 36px;
        }}
        .keyword-tag {{
            display: inline-block;
            background: rgba(255, 255, 255, 0.06);
            color: #cbd5e1;
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 0.85rem;
            margin: 4px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .bibtex-box {{
            background: #0d1322;
            border: 1px solid rgba(99, 140, 255, 0.25);
            border-radius: 8px;
            padding: 20px;
            position: relative;
            margin-bottom: 40px;
        }}
        pre.bibtex-code {{
            font-family: var(--font-mono);
            font-size: 0.85rem;
            color: #93c5fd;
            white-space: pre-wrap;
            word-break: break-all;
            margin: 0;
        }}
        .copy-btn {{
            position: absolute;
            top: 14px;
            right: 14px;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            color: #e2e8f0;
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .copy-btn:hover {{
            background: var(--accent);
            color: #fff;
        }}
        .footer {{
            border-top: 1px solid var(--border-color);
            padding: 32px 24px;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 60px;
        }}
    </style>
</head>
<body>
    <nav class="navbar">
        <a href="{BASE_URL}/" class="nav-brand">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2.2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
            DecisionModel<span>Bench</span>
        </a>
        <div class="nav-links">
            <a href="{BASE_URL}/">Home & Papers</a>
            <a href="{BASE_URL}/papers/{paper['pdf_filename']}">Direct PDF</a>
            <a href="https://github.com/tuwiliyt/decisionmodelbench" target="_blank">GitHub Repo</a>
        </div>
    </nav>

    <main class="container">
        <div class="badge-bar">
            <span class="badge badge-blue">{paper['badge']}</span>
            <span class="badge badge-green">RISET MANDIRI</span>
            <span class="badge badge-purple">{paper['lang_label']}</span>
            <span class="badge badge-blue">IEEE Standard ({paper['pages']} Pages)</span>
        </div>

        <h1 class="paper-title">{paper['title']}</h1>

        <div class="author-box">
            <div class="author-name">{AUTHOR}</div>
            <div class="author-affiliation">PANITA GORONTALO</div>
            <div class="author-meta">
                Advanced Agentic AI & Distributed Systems Research &bull; Gorontalo, Indonesia<br>
                Official Email: <a href="mailto:{EMAIL}">{EMAIL}</a> &bull; Report ID: <code>{paper['report_id']}</code> &bull; Date: September 2026
            </div>
        </div>

        <div class="cta-box">
            <a href="{paper['pdf_filename']}" class="btn btn-primary" download>
                <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
                Download Full Paper (PDF)
            </a>
            <a href="{BASE_URL}/papers/{paper['pdf_filename']}" class="btn btn-secondary" target="_blank">
                <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                View PDF in Browser
            </a>
            <a href="{BASE_URL}/" class="btn btn-secondary">
                <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
                Back to All Papers
            </a>
        </div>

        <h2 class="section-title">Abstract</h2>
        <div class="abstract-text">
            {paper['abstract']}
        </div>

        <h2 class="section-title">Index Terms & Keywords</h2>
        <div class="keywords-box">
            {''.join([f'<span class="keyword-tag">{k.strip()}</span>' for k in paper['keywords'].split(';')])}
        </div>

        <h2 class="section-title">BibTeX Citation</h2>
        <div class="bibtex-box">
            <button class="copy-btn" onclick="copyBibtex()">Copy BibTeX</button>
            <pre class="bibtex-code" id="bibtexText">{paper['bibtex']}</pre>
        </div>

        <h2 class="section-title">Archival & Academic Metadata</h2>
        <div style="background: var(--bg-card); padding: 18px 24px; border-radius: 8px; border: 1px solid var(--border-color); font-size: 0.9rem;">
            <p><strong>Full Title:</strong> {paper['title']}</p>
            <p><strong>Primary Author:</strong> {AUTHOR}</p>
            <p><strong>Affiliation:</strong> PANITA GORONTALO (Gorontalo, Indonesia)</p>
            <p><strong>Publication Series:</strong> {paper['journal']}</p>
            <p><strong>Permanent Document Link:</strong> <a href="{pdf_url}" style="color: var(--accent);">{pdf_url}</a></p>
            <p><strong>Google Scholar Metadata Compatibility:</strong> Highwire Press, Dublin Core, Schema.org ScholarlyArticle verified.</p>
        </div>
    </main>

    <footer class="footer">
        <p>&copy; 2026 Richie O. Sumual &bull; PANITA GORONTALO &bull; Advanced Agentic AI & Distributed Systems Research</p>
        <p style="margin-top: 6px; font-size: 0.8rem; color: #64748b;">Gorontalo, Indonesia &bull; Contact: {EMAIL}</p>
    </footer>

    <script>
        function copyBibtex() {{
            const text = document.getElementById('bibtexText').innerText;
            navigator.clipboard.writeText(text).then(() => {{
                const btn = document.querySelector('.copy-btn');
                btn.innerText = 'Copied!';
                setTimeout(() => btn.innerText = 'Copy BibTeX', 2000);
            }});
        }}
    </script>
</body>
</html>
"""
    return html


def generate_homepage():
    cards_html = ""
    for p in PAPERS_INFO:
        cards_html += f"""
        <div class="paper-card">
            <div class="card-badges">
                <span class="badge badge-blue">{p['badge']}</span>
                <span class="badge badge-green">RISET MANDIRI</span>
                <span class="badge badge-purple">{p['lang_label']}</span>
                <span class="badge badge-blue">{p['pages']} Pages</span>
            </div>
            <h3 class="card-title">
                <a href="papers/{p['id']}.html">{p['title']}</a>
            </h3>
            <div class="card-author">
                <strong>{AUTHOR}</strong> &bull; <span style="color: #60a5fa;">PANITA GORONTALO</span> (Gorontalo, Indonesia)
            </div>
            <p class="card-abstract">
                {p['abstract'][:280]}...
            </p>
            <div class="card-actions">
                <a href="papers/{p['id']}.html" class="btn btn-secondary btn-sm">
                    View Paper & Metadata
                </a>
                <a href="papers/{p['pdf_filename']}" class="btn btn-primary btn-sm" download>
                    <svg width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
                    Download PDF
                </a>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DecisionModelBench &bull; Research Publications &bull; Richie O. Sumual (PANITA GORONTALO)</title>
    <meta name="description" content="Official Research Publications Portal for DecisionModelBench by Richie O. Sumual, PANITA GORONTALO. Empirical benchmarks comparing Non-Autoregressive Decision Models against Foundation Large Language Models.">
    <meta name="author" content="Richie O. Sumual">
    <meta name="keywords" content="Decision Models, LLM Benchmark, Richie O. Sumual, PANITA GORONTALO, Sahabat-AI, Qwen 2.5, Gemma 2, Emergency 112, Algorithmic Trading, Air Defense">

    <!-- Schema.org Portal Metadata -->
    <script type="application/ld+json">
    {{
      "@context": "https://schema.org",
      "@type": "CollectionPage",
      "name": "DecisionModelBench Research Publications",
      "author": {{
        "@type": "Person",
        "name": "Richie O. Sumual",
        "email": "{EMAIL}",
        "affiliation": {{
          "@type": "Organization",
          "name": "PANITA GORONTALO"
        }}
      }},
      "description": "Empirical benchmarks comparing Non-Autoregressive Decision Models against Foundation Large Language Models across Cyber-Physical Systems, Algorithmic Trading, Emergency Calling, and Public Administration."
    }}
    </script>

    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0f1d;
            --bg-secondary: #121a2f;
            --bg-card: #18233e;
            --border-color: rgba(99, 140, 255, 0.2);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.35);
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: var(--font-sans);
            line-height: 1.6;
        }}
        .navbar {{
            background: rgba(18, 26, 47, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-color);
            padding: 18px 32px;
            position: sticky;
            top: 0;
            z-index: 100;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .nav-brand {{
            font-weight: 800;
            font-size: 1.2rem;
            color: #fff;
            text-decoration: none;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .nav-brand span {{ color: var(--accent); }}
        .nav-links {{ display: flex; gap: 20px; }}
        .nav-links a {{
            color: var(--text-muted);
            text-decoration: none;
            font-size: 0.95rem;
            font-weight: 500;
            transition: color 0.2s;
        }}
        .nav-links a:hover {{ color: #fff; }}
        .hero {{
            padding: 70px 24px 50px;
            text-align: center;
            background: radial-gradient(circle at 50% 20%, rgba(59, 130, 246, 0.15) 0%, transparent 60%);
            border-bottom: 1px solid var(--border-color);
        }}
        .hero-inst {{
            display: inline-block;
            background: rgba(59, 130, 246, 0.12);
            color: #93c5fd;
            border: 1px solid rgba(59, 130, 246, 0.35);
            padding: 6px 16px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            margin-bottom: 16px;
            text-transform: uppercase;
        }}
        .hero-title {{
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #ffffff;
            margin-bottom: 16px;
            line-height: 1.2;
        }}
        .hero-desc {{
            font-size: 1.2rem;
            color: var(--text-muted);
            max-width: 800px;
            margin: 0 auto 28px;
            line-height: 1.7;
        }}
        .author-badge {{
            display: inline-flex;
            align-items: center;
            gap: 12px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            padding: 10px 22px;
            border-radius: 9999px;
            font-size: 0.95rem;
        }}
        .author-badge strong {{ color: #fff; }}
        .container {{
            max-width: 1100px;
            margin: 50px auto;
            padding: 0 24px;
        }}
        .section-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-bottom: 28px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 14px;
        }}
        .section-title {{
            font-size: 1.6rem;
            font-weight: 700;
            color: #fff;
        }}
        .paper-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 24px;
            margin-bottom: 60px;
        }}
        @media (max-width: 600px) {{
            .paper-grid {{ grid-template-columns: 1fr; }}
            .hero-title {{ font-size: 2rem; }}
        }}
        .paper-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 26px;
            display: flex;
            flex-direction: column;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .paper-card:hover {{
            transform: translateY(-3px);
            border-color: rgba(99, 140, 255, 0.45);
        }}
        .card-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 14px;
        }}
        .badge {{
            display: inline-flex;
            align-items: center;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }}
        .badge-blue {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
        .badge-green {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
        .badge-purple {{ background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }}
        .card-title {{
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1.4;
            margin-bottom: 12px;
        }}
        .card-title a {{
            color: #fff;
            text-decoration: none;
            transition: color 0.2s;
        }}
        .card-title a:hover {{ color: var(--accent); }}
        .card-author {{
            font-size: 0.88rem;
            color: var(--text-muted);
            margin-bottom: 14px;
        }}
        .card-abstract {{
            font-size: 0.92rem;
            color: #cbd5e1;
            line-height: 1.6;
            margin-bottom: 22px;
            flex-grow: 1;
        }}
        .card-actions {{
            display: flex;
            gap: 12px;
            margin-top: auto;
        }}
        .btn {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 10px 18px;
            border-radius: 7px;
            font-size: 0.88rem;
            font-weight: 600;
            text-decoration: none;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .btn-sm {{ padding: 8px 14px; font-size: 0.84rem; }}
        .btn-primary {{
            background: linear-gradient(135deg, #2563eb, #1d4ed8);
            color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .btn-primary:hover {{ background: #2563eb; transform: translateY(-1px); }}
        .btn-secondary {{
            background: var(--bg-card);
            color: #e2e8f0;
            border: 1px solid var(--border-color);
        }}
        .btn-secondary:hover {{ background: #223055; color: #fff; transform: translateY(-1px); }}
        .sim-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 18px;
            margin-bottom: 60px;
        }}
        .sim-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 20px;
            text-decoration: none;
            color: inherit;
            display: block;
            transition: all 0.2s;
        }}
        .sim-card:hover {{
            border-color: var(--accent);
            transform: translateY(-2px);
        }}
        .sim-title {{ font-size: 1.05rem; font-weight: 700; color: #fff; margin-bottom: 6px; }}
        .sim-desc {{ font-size: 0.85rem; color: var(--text-muted); }}
        .footer {{
            border-top: 1px solid var(--border-color);
            padding: 36px 24px;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.88rem;
        }}
    </style>
</head>
<body>
    <nav class="navbar">
        <a href="{BASE_URL}/" class="nav-brand">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2.2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
            DecisionModel<span>Bench</span>
        </a>
        <div class="nav-links">
            <a href="#papers">Papers</a>
            <a href="#arenas">Interactive Arenas</a>
            <a href="https://github.com/tuwiliyt/decisionmodelbench" target="_blank">GitHub</a>
        </div>
    </nav>

    <header class="hero">
        <div class="hero-inst">PANITA GORONTALO &bull; RISET MANDIRI</div>
        <h1 class="hero-title">DecisionModelBench</h1>
        <p class="hero-desc">
            Empirical benchmarks evaluating Non-Autoregressive Decision Models against Foundation Large Language Models (Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B) across Time-Critical Cyber-Physical Defense, Algorithmic Trading, and Sovereign Emergency 911/112 Triage.
        </p>
        <div class="author-badge">
            <span>Author: <strong>{AUTHOR}</strong></span>
            <span>&bull;</span>
            <span>Affiliation: <strong style="color: #60a5fa;">PANITA GORONTALO</strong></span>
            <span>&bull;</span>
            <span>Gorontalo, Indonesia</span>
            <span>&bull;</span>
            <span><a href="mailto:{EMAIL}" style="color: var(--accent); text-decoration: none;">{EMAIL}</a></span>
        </div>
    </header>

    <main class="container">
        <div class="section-header" id="papers">
            <div>
                <h2 class="section-title">Research Publications</h2>
                <p style="color: var(--text-muted); font-size: 0.95rem;">Peer-review quality IEEE Transactions format manuscripts with complete empirical telemetry data.</p>
            </div>
        </div>

        <div class="paper-grid">
            {cards_html}
        </div>

        <div class="section-header" id="arenas">
            <div>
                <h2 class="section-title">Interactive Simulators & Telemetry Dashboards</h2>
                <p style="color: var(--text-muted); font-size: 0.95rem;">Explore live browser-based physics engines and multi-LLM benchmark evaluation suites.</p>
            </div>
        </div>

        <div class="sim-grid">
            <a href="air_defense_arena.html" class="sim-card">
                <div class="sim-title">Tactical Air Defense Arena</div>
                <div class="sim-desc">Iron Dome/C-RAM radar simulation testing sub-100ms kinetic missile interception vs LLM lag.</div>
            </a>
            <a href="fast_trading_arena.html" class="sim-card">
                <div class="sim-title">Sub-Second Trading Arena</div>
                <div class="sim-desc">L2 orderbook matching & slippage simulation across $10,000 algorithmic portfolio.</div>
            </a>
            <a href="brick_breaker_arena.html" class="sim-card">
                <div class="sim-title">Arcade Trajectory Interception</div>
                <div class="sim-desc">Ball-paddle continuous physics testing state drift under 55ms vs 2,500ms inferencing.</div>
            </a>
            <a href="heavyweight_llm_dashboard.html" class="sim-card">
                <div class="sim-title">Heavyweight Multi-LLM Dashboard</div>
                <div class="sim-desc">Comparative multi-GPU telemetry for Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B, and JEV.</div>
            </a>
        </div>
    </main>

    <footer class="footer">
        <p>&copy; 2026 <strong>Richie O. Sumual</strong> &bull; <strong>PANITA GORONTALO</strong> &bull; Advanced Agentic AI & Distributed Systems Research</p>
        <p style="margin-top: 8px; font-size: 0.82rem; color: #64748b;">
            Gorontalo, Indonesia &bull; Pos-el / Email: <a href="mailto:{EMAIL}" style="color: var(--accent);">{EMAIL}</a> &bull;
            Indexed for Google Scholar, Semantic Scholar, and Open Academic Research.
        </p>
    </footer>
</body>
</html>
"""
    return html


def generate_sitemap():
    urls = [
        f"{BASE_URL}/",
        f"{BASE_URL}/air_defense_arena.html",
        f"{BASE_URL}/fast_trading_arena.html",
        f"{BASE_URL}/brick_breaker_arena.html",
        f"{BASE_URL}/heavyweight_llm_dashboard.html",
    ]
    for p in PAPERS_INFO:
        urls.append(f"{BASE_URL}/papers/{p['id']}.html")
        urls.append(f"{BASE_URL}/papers/{p['pdf_filename']}")
    
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
"""
    for u in urls:
        xml += f"""  <url>
    <loc>{u}</loc>
    <lastmod>2026-09-27</lastmod>
    <changefreq>monthly</changefreq>
    <priority>{"1.0" if u.endswith('/') or '.pdf' in u else "0.8"}</priority>
  </url>
"""
    xml += "</urlset>\n"
    return xml


def generate_robots():
    return f"""User-agent: *
Allow: /

User-agent: Googlebot
Allow: /

User-agent: Googlebot-Scholar
Allow: /

User-agent: Bingbot
Allow: /

User-agent: Slurp
Allow: /

Sitemap: {BASE_URL}/sitemap.xml
"""


def main():
    root_dir = "/kaggle/working/decisionmodelbench"
    papers_dir = os.path.join(root_dir, "papers")
    docs_dir = os.path.join(root_dir, "docs")
    docs_papers_dir = os.path.join(docs_dir, "papers")
    
    os.makedirs(papers_dir, exist_ok=True)
    os.makedirs(docs_papers_dir, exist_ok=True)

    print("Generating Google Scholar Landing Pages...")
    for p in PAPERS_INFO:
        html = generate_paper_landing_page(p)
        # Write to papers/
        p_path = os.path.join(papers_dir, f"{p['id']}.html")
        with open(p_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  -> Generated {p_path}")
        
        # Also copy to docs/papers/
        p_docs_path = os.path.join(docs_papers_dir, f"{p['id']}.html")
        with open(p_docs_path, "w", encoding="utf-8") as f:
            f.write(html)

        # Copy PDF to docs/papers/ if it exists
        src_pdf = os.path.join(papers_dir, p['pdf_filename'])
        dst_pdf = os.path.join(docs_papers_dir, p['pdf_filename'])
        if os.path.exists(src_pdf):
            shutil.copy2(src_pdf, dst_pdf)
            print(f"  -> Synced PDF {p['pdf_filename']} to docs/papers/")

    print("Generating Homepage index.html...")
    home_html = generate_homepage()
    with open(os.path.join(root_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(home_html)
    with open(os.path.join(docs_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(home_html)

    print("Generating sitemap.xml and robots.txt...")
    sitemap = generate_sitemap()
    robots = generate_robots()
    
    for d in [root_dir, docs_dir]:
        with open(os.path.join(d, "sitemap.xml"), "w", encoding="utf-8") as f:
            f.write(sitemap)
        with open(os.path.join(d, "robots.txt"), "w", encoding="utf-8") as f:
            f.write(robots)
        with open(os.path.join(d, ".nojekyll"), "w", encoding="utf-8") as f:
            f.write("")

    # Also copy interactive HTML simulators into docs/
    for h in ["air_defense_arena.html", "fast_trading_arena.html", "brick_breaker_arena.html", "heavyweight_llm_dashboard.html"]:
        src_h = os.path.join(root_dir, h)
        dst_h = os.path.join(docs_dir, h)
        if os.path.exists(src_h):
            shutil.copy2(src_h, dst_h)
            print(f"  -> Copied simulator {h} to docs/")

    print("All GitHub Pages files successfully generated!")

if __name__ == "__main__":
    main()
