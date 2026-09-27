#!/usr/bin/env python3
"""
Play Air Defense AI Arena (CLI Terminal Simulator)
Features:
- 5 Parallel Cities defended by 5 AI Architectures:
  * Jakarta: Laya Multilingual (421M GPU) - ~55ms
  * Surabaya: OpenJev (0.5B GPU Local Logits) - ~210ms
  * Bandung: TypeSafe JEV (Cloud Decision API) - ~160ms (Highlighted prominently)
  * Medan: Kev-0.8B (Local Ensemble) - ~950ms
  * Nusantara (IKN): Heavyweight LLM (Sahabat-AI / Qwen / Gemma) - ~1100-2800ms
- Real-Life Battery Firing Load Physics:
  * 20 missiles per pod magazine.
  * Ripple salvo spacing.
  * Reload cooldown cycle (3.5s) when empty.
- Progressive Wave Escalation (Hypersonic missiles, meteorite cratering, saturation bombardment).
- Threat-Specific Damage (Meteorite 45%, Missile 35%, Drone 12%, Shrapnel 3%).
- Periodic SITREP (Situation Report) printed at wave intervals.
"""

import sys
import time
import argparse
import urllib.request
import json
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text

from air_defense_engine import AirDefenseArena

console = Console()

def create_city_panel(city_data: dict):
    hp = city_data["city_hp"]
    name = city_data["city_name"].split(" (")[0]
    m_id = city_data["model_id"]
    lat = city_data["typical_latency_ms"]
    kills = city_data["intercepted_count"]
    hits = city_data["impact_missile_count"]
    civ = city_data["civilian_safe_count"]
    tok = city_data["tokens_burned"]
    ammo = city_data.get("ammo", 20)
    max_ammo = city_data.get("max_ammo", 20)
    is_reloading = city_data.get("is_reloading", False)
    reload_sec = city_data.get("reload_remaining_sec", 0.0)

    # Distinct model labeling to ensure JEV is prominent
    if m_id == "jev":
        model_display = "TypeSafe JEV (Cloud API)"
        model_style = "bold magenta"
    elif m_id == "openjev":
        model_display = "OpenJev (0.5B GPU Logits)"
        model_style = "bold cyan"
    elif m_id == "laya":
        model_display = "Laya (421M GPU)"
        model_style = "bold green"
    elif m_id == "kev":
        model_display = "Kev-0.8B (Local)"
        model_style = "bold yellow"
    else:
        model_display = city_data["model_label"].split(" (")[0]
        model_style = "bold red"

    hp_style = "bold green" if hp > 65 else ("bold yellow" if hp > 30 else "bold red")
    hp_bars = max(0, min(10, int(hp / 10)))
    hp_vis = f"[{hp_style}]{'█'*hp_bars}{'░'*(10-hp_bars)} {hp:.0f}%[/]"

    content = Text()
    content.append(f"{name}\n", style="bold white")
    content.append(f"{model_display}\n", style=model_style)
    content.append(f"Refleks: ~{lat:.0f} ms\n", style="dim")
    content.append(f"Integritas: ", style="dim")
    content.append(f"{hp:.0f}%\n", style=hp_style)

    # Ammo / Reload status
    content.append("Amunisi:   ", style="dim")
    if is_reloading:
        content.append(f"🔄 RELOAD ({reload_sec:.1f}s)\n", style="bold yellow")
    else:
        ammo_bars = max(0, min(10, int(ammo / 2)))
        ammo_style = "green" if ammo > 8 else ("yellow" if ammo > 3 else "red bold")
        content.append(f"{ammo}/{max_ammo} [{'█'*ammo_bars}{'░'*(10-ammo_bars)}]\n", style=ammo_style)

    content.append(f"Ditangkis: ", style="dim")
    content.append(f"{kills} rudal\n", style="green bold")
    content.append(f"Hantaman:  ", style="dim")
    content.append(f"{hits} lolos\n", style="red bold" if hits > 0 else "dim")
    content.append(f"Sipil:     ", style="dim")
    content.append(f"{civ} aman\n", style="cyan")
    content.append(f"Token:     ", style="dim")
    tok_style = "red bold" if tok > 0 else "green"
    content.append(f"{tok} tok", style=tok_style)

    border_color = "magenta" if m_id == "jev" else ("green" if hp > 65 else ("yellow" if hp > 30 else "red"))
    title_icon = "☁️" if m_id == "jev" else "🛡️"
    return Panel(content, title=f"{title_icon} {name}", border_style=border_color, width=28)

def print_sitrep_table(sitrep: dict):
    wave_num = sitrep["wave_completed"]
    table = Table(
        title=f"📊 SITREP REKAP TAKTIS PERTAHANAN UDARA (WAVE {wave_num} SELESAI)",
        header_style="bold cyan",
        border_style="yellow"
    )
    table.add_column("Kota Pertahanan", style="bold white")
    table.add_column("Arsitektur AI", style="bold")
    table.add_column("Integritas HP", justify="center")
    table.add_column("Amunisi", justify="center")
    table.add_column("Rudal Ditangkis", justify="center", style="green bold")
    table.add_column("Hantaman Kota (Lolos)", justify="center", style="red bold")
    table.add_column("Sipil Aman", justify="center", style="cyan")
    table.add_column("Token", justify="right")
    table.add_column("Status Kota", justify="center")

    for c in sitrep["cities"]:
        hp_val = c["hp"]
        hp_color = "[green]" if hp_val > 65 else ("[yellow]" if hp_val > 30 else "[red]")
        status_color = "[green]" if "PRIMA" in c["status"] or "OPERASIONAL" in c["status"] else ("[yellow]" if "WASPADA" in c["status"] else "[red]")
        tok_str = f"[red]{c['tokens']}[/red]" if c["tokens"] > 0 else "[green]0[/green]"

        # Distinct label for Jev vs others
        m_label = c["model"]
        if "TypeSafe JEV" in m_label:
            m_style = "[bold magenta]TypeSafe JEV (Cloud API)[/bold magenta]"
        elif "OpenJev" in m_label:
            m_style = "[cyan]OpenJev (0.5B GPU Logits)[/cyan]"
        elif "Laya" in m_label:
            m_style = "[green]Laya (421M GPU)[/green]"
        elif "Kev" in m_label:
            m_style = "[yellow]Kev-0.8B (Local)[/yellow]"
        else:
            m_style = f"[red]{m_label.split(' (')[0]}[/red]"

        table.add_row(
            c["city"].split(" (")[0],
            m_style,
            f"{hp_color}{hp_val:.0f}%[/]",
            c.get("ammo", "20/20"),
            str(c["intercepted"]),
            str(c["failed_impacts"]),
            str(c["civilian_safe"]),
            tok_str,
            f"{status_color}{c['status']}[/]"
        )

    console.print("\n")
    console.print(table)
    console.print("\n")

def test_live_decision_probe(model: str = "jev"):
    console.print(f"[bold cyan]🔌 Menguji Radar Telemetry Decision Probe ke Model: [bold magenta]{model.upper()}[/bold magenta] ...[/bold cyan]")
    try:
        req_data = {
            "model": model,
            "threat_type": "MISSILE",
            "speed_mach": 4.2,
            "altitude_km": 14.5,
            "time_to_impact_sec": 2.5,
            "iff_code": None
        }
        req = urllib.request.Request(
            "http://127.0.0.1:7860/api/air_defense/decision",
            data=json.dumps(req_data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=12) as resp:
            res = json.loads(resp.read().decode("utf-8"))
        lat = (time.perf_counter() - t0) * 1000
        console.print(f"[bold green]✓ Decision Radar Lock Diterima dalam {lat:.1f} ms![/bold green]")
        console.print_json(json.dumps(res, indent=2))
    except Exception as e:
        console.print(f"[bold red]✗ Gagal menghubungi API server: {e}[/bold red]")

def main():
    parser = argparse.ArgumentParser(description="Arena Air Defense AI (Iron Dome Paralel)")
    parser.add_argument("--ticks", type=int, default=60, help="Jumlah ticks simulasi pertahanan udara (default: 60)")
    parser.add_argument("--speed", type=str, default="normal", choices=["slow", "normal", "fast", "turbo"], help="Kecepatan simulasi")
    parser.add_argument("--llm", type=str, default="sahabatai", choices=["sahabatai", "qwen", "gemma", "gemma-2b"], help="Pilihan model LLM Kota 5")
    parser.add_argument("--probe", type=str, default=None, choices=["jev", "laya", "openjev", "kev", "sahabatai"], help="Uji 1 live decision probe ke model tertentu")
    parser.add_argument("--api", action="store_true", help="Uji 1 probe radar detection live ke TypeSafe JEV Cloud")
    args = parser.parse_args()

    if args.probe:
        test_live_decision_probe(args.probe)
        return

    if args.api:
        test_live_decision_probe("jev")
        return

    delay_map = {"slow": 0.35, "normal": 0.12, "fast": 0.04, "turbo": 0.01}
    delay = delay_map.get(args.speed, 0.12)

    arena = AirDefenseArena()
    arena.set_llm_model(args.llm)
    arena.ticks_per_wave = 25 # Every 25 ticks triggers SITREP in CLI

    llm_info = arena.LLM_MODELS_CATALOG.get(args.llm, arena.LLM_MODELS_CATALOG["sahabatai"])

    console.print(Panel(
        f"[bold red]🛡️ ARENA AIR DEFENSE AI (5 KOTA PARALEL - IRON DOME & C-RAM SIMULATOR)[/bold red]\n"
        f"Misi: Cegat Rudal & Meteorit | Lindungi Pesawat Sipil | Target Ticks: [yellow]{args.ticks}[/yellow]\n"
        f"Arsitektur Pertahanan:\n"
        f" • [green]Jakarta:[/green] Laya Multilingual (421M GPU) - ~55ms\n"
        f" • [cyan]Surabaya:[/cyan] OpenJev (0.5B GPU Local Logits) - ~210ms\n"
        f" • [bold magenta]Bandung: TypeSafe JEV (Cloud Decision API SaaS) - ~160ms[/bold magenta] [bold green]★ AKTIF[/bold green]\n"
        f" • [yellow]Medan:[/yellow] Kev-0.8B (Local Ensemble) - ~950ms\n"
        f" • [red]Nusantara (IKN):[/red] {llm_info['label']} - ~{llm_info['latency_ms']:.0f}ms\n"
        f"Karakteristik Fisik: [bold white]Pod Magazine 20 Rudal/Kota[/bold white] | [bold white]Siklus Reload 3.5s[/bold white] | [bold white]Kalkulasi Kerusakan Riil[/bold white]",
        border_style="red"
    ))

    with Live(console=console, refresh_per_second=15) as live:
        for t in range(1, args.ticks + 1):
            step_res = arena.step()
            wave = step_res["wave"]

            # Header info
            header_text = Text()
            header_text.append(f"DEFCON 1: AIR RAID ACTIVE  ", style="bold red")
            header_text.append(f"Wave: {wave}  ", style="bold yellow")
            header_text.append(f"Sasaran di Radar: {step_res['objects_count']} objek  ", style="cyan")
            header_text.append(f"Tick: {t}/{args.ticks}\n", style="white")

            # Active objects summary
            hostiles = [o for o in step_res["sky_objects"] if o["is_hostile"]]
            civilians = [o for o in step_res["sky_objects"] if not o["is_hostile"]]
            header_text.append(f"🔥 Ancaman: {len(hostiles)} (Rudal/Meteor) | ", style="red")
            header_text.append(f"🛡️ Sipil di Langit: {len(civilians)} (Pesawat Komersil) | ", style="green")
            header_text.append(f"☁️ TypeSafe JEV: AKTIF (Cloud Sub-200ms)", style="bold magenta")

            header_panel = Panel(header_text, title="📡 Tactical Doppler Radar Command", border_style="cyan")

            # City cards
            cards = [create_city_panel(c) for c in step_res["cities"].values()]
            cards_col = Columns(cards, equal=True)

            live.update(Group(
                header_panel,
                cards_col
            ))

            if step_res.get("wave_advanced"):
                sitrep = arena.generate_sitrep()
                live.stop()
                print_sitrep_table(sitrep)
                live.start()

            time.sleep(delay)

    # Final SITREP Summary
    final_sitrep = arena.generate_sitrep()
    print_sitrep_table(final_sitrep)

    console.print(Panel(
        "[bold cyan]KESIMPULAN ARSITEKTURAL AIR DEFENSE MILIDETIK:[/bold cyan]\n"
        "1. [bold green]Refleks Sub-100ms Selamatkan Kota:[/bold green] Decision Models (System 1: Laya ~55ms) mengunci dan menembak rudal saat masih di ketinggian aman.\n"
        "2. [bold magenta]TypeSafe JEV Cloud API Handal:[/bold magenta] Memberikan inferensi deterministik ~160ms dengan zero token waste tanpa membebani GPU lokal.\n"
        "3. [bold red]Decision Lag Bawa Kehancuran:[/bold red] Heavyweight LLM lambat hingga 2.8s per siklus sehingga interceptor baru meluncur saat rudal telah meledak di kota.\n"
        "4. [bold yellow]Manajemen Amunisi Riil:[/bold yellow] Kapasitas pod 20 rudal mengharuskan efisiensi tembakan agar kota tidak diserbu saat siklus reload 3.5s!",
        border_style="yellow",
        title="💡 MENGAPA HARUS DECISION MODEL DI SISTEM PERTAHANAN?"
    ))

if __name__ == "__main__":
    main()
