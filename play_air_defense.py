#!/usr/bin/env python3
"""
Play Air Defense AI Arena (CLI Terminal Simulator)
Features:
- 5 Parallel Cities defended by 5 AI Models (Jakarta, Surabaya, Bandung, Medan, Nusantara).
- Progressive Wave Escalation (Difficulty increases with time: Hypersonic missiles, saturation bombardment).
- Periodic SITREP (Situation Report) printed at wave intervals summarizing:
  * Intercepted missiles (Ditangkis)
  * City Ground Impacts (Lolos)
  * Civilian airliners safely protected
  * Token waste comparison (Decision Models: 0 vs LLM: thousands)
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
    model = city_data["model_label"].split(" (")[0]
    lat = city_data["typical_latency_ms"]
    kills = city_data["intercepted_count"]
    hits = city_data["impact_missile_count"]
    civ = city_data["civilian_safe_count"]
    tok = city_data["tokens_burned"]

    hp_style = "bold green" if hp > 65 else ("bold yellow" if hp > 30 else "bold red")
    hp_bars = int(hp / 10)
    hp_vis = f"[{hp_style}]{'█'*hp_bars}{'░'*(10-hp_bars)} {hp:.0f}%[/]"

    content = Text()
    content.append(f"{name}\n", style="bold white")
    content.append(f"{model} (~{lat:.0f}ms)\n", style="cyan")
    content.append(f"Integritas: ", style="dim")
    content.append(f"{hp:.0f}%\n", style=hp_style)
    content.append(f"Ditangkis:  ", style="dim")
    content.append(f"{kills} rudal\n", style="green bold")
    content.append(f"Hantaman:   ", style="dim")
    content.append(f"{hits} lolos\n", style="red bold" if hits > 0 else "dim")
    content.append(f"Sipil Aman: ", style="dim")
    content.append(f"{civ} pesawat\n", style="cyan")
    content.append(f"Token:      ", style="dim")
    tok_style = "red bold" if tok > 0 else "green"
    content.append(f"{tok} tokens", style=tok_style)

    border_color = "green" if hp > 65 else ("yellow" if hp > 30 else "red")
    return Panel(content, title=f"🛡️ {name}", border_style=border_color, width=28)

def print_sitrep_table(sitrep: dict):
    wave_num = sitrep["wave_completed"]
    table = Table(
        title=f"📊 SITREP REKAP TAKTIS PERTAHANAN UDARA (WAVE {wave_num} SELESAI)",
        header_style="bold cyan",
        border_style="yellow"
    )
    table.add_column("Kota Pertahanan", style="bold white")
    table.add_column("Model AI", style="cyan")
    table.add_column("Integritas HP", justify="center")
    table.add_column("Rudal Ditangkis", justify="center", style="green bold")
    table.add_column("Hantaman Kota (Lolos)", justify="center", style="red bold")
    table.add_column("Sipil Aman", justify="center", style="cyan")
    table.add_column("Token Terbuang", justify="right")
    table.add_column("Status Kota", justify="center")

    for c in sitrep["cities"]:
        hp_val = c["hp"]
        hp_color = "[green]" if hp_val > 65 else ("[yellow]" if hp_val > 30 else "[red]")
        status_color = "[green]" if "AMAN" in c["status"] or "OPERASIONAL" in c["status"] else ("[yellow]" if "KRITIS" in c["status"] else "[red]")
        tok_str = f"[red]{c['tokens']}[/red]" if c["tokens"] > 0 else "[green]0[/green]"

        table.add_row(
            c["city"].split(" (")[0],
            c["model"].split(" (")[0],
            f"{hp_color}{hp_val:.0f}%[/]",
            str(c["intercepted"]),
            str(c["failed_impacts"]),
            str(c["civilian_safe"]),
            tok_str,
            f"{status_color}{c['status']}[/]"
        )

    console.print("\n")
    console.print(table)
    console.print("\n")

def main():
    parser = argparse.ArgumentParser(description="Arena Air Defense AI (Iron Dome Paralel)")
    parser.add_argument("--ticks", type=int, default=60, help="Jumlah ticks simulasi pertahanan udara (default: 60)")
    parser.add_argument("--speed", type=str, default="normal", choices=["slow", "normal", "fast", "turbo"], help="Kecepatan simulasi")
    parser.add_argument("--api", action="store_true", help="Uji 1 probe radar detection live ke backend server")
    args = parser.parse_args()

    if args.api:
        console.print("[bold cyan]🔌 Menguji Probe Radar Air Defense ke http://127.0.0.1:7860/api/predict ...[/bold cyan]")
        try:
            req_data = {
                "state": "Radar Doppler: Sasaran terdeteksi pada ketinggian 12 km menukik lurus ke Kota Jakarta dengan laju Mach 3.8. Sinyal IFF: TIDAK TERDETEKSI (Objek tak dikenal / Musuh). Perkiraan waktu hantaman: 2.8 detik.",
                "questions": {
                    "action": {
                        "type": "choice",
                        "instructions": "Aksi taktis baterai pertahanan udara?",
                        "criteria": {
                            "FIRE": "Kunci sasaran & tembakkan rudal pencegat darurat",
                            "HOLD": "Pantau terus / jangan tembak (bukan ancaman daratan)",
                            "SAFE_PASS": "Izinkan melintas (pesawat sipil terverifikasi)"
                        }
                    }
                },
                "model": "laya"
            }
            req = urllib.request.Request(
                "http://127.0.0.1:7860/api/predict",
                data=json.dumps(req_data).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=10) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            lat = (time.perf_counter() - t0) * 1000
            console.print(f"[bold green]✓ Radar Lock & Keputusan diterima dalam {lat:.1f} ms![/bold green]")
            console.print_json(json.dumps(res))
        except Exception as e:
            console.print(f"[bold red]✗ Gagal menghubungi API server: {e}[/bold red]")
        return

    delay_map = {"slow": 0.35, "normal": 0.12, "fast": 0.04, "turbo": 0.01}
    delay = delay_map.get(args.speed, 0.12)

    arena = AirDefenseArena()
    arena.ticks_per_wave = 25 # Every 25 ticks triggers SITREP in CLI

    console.print(Panel(
        f"[bold red]🛡️ ARENA AIR DEFENSE AI (5 KOTA PARALEL - IRON DOME SIMULATOR)[/bold red]\n"
        f"Misi: Cegat Rudal & Meteorit | Lindungi Pesawat Sipil | Target Ticks: [yellow]{args.ticks}[/yellow]\n"
        f"Kota: [green]Jakarta (Laya)[/green], [cyan]Surabaya (OpenJev)[/cyan], [purple]Bandung (Jev)[/purple], [yellow]Medan (Kev)[/yellow], [red]Nusantara (LLM 8B)[/red]",
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
            header_text.append(f"🔥 Ancaman Aktif: {len(hostiles)} (Rudal/Meteor) | ", style="red")
            header_text.append(f"🛡️ Sipil di Langit: {len(civilians)} (Pesawat Komersil)", style="green")

            header_panel = Panel(header_text, title="📡 Tactical Doppler Radar Command", border_style="cyan")

            # City cards
            cards = [create_city_panel(c) for c in step_res["cities"].values()]
            cards_col = Columns(cards, equal=True)

            live.update(Group(
                header_panel,
                cards_col
            ))

            if step_res.get("wave_advanced"):
                # Print periodic SITREP table
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
        "1. [bold green]Refleks Sub-100ms Selamatkan Kota:[/bold green] Decision Models (System 1: Laya ~55ms) mengunci dan menembak rudal musuh saat masih di ketinggian aman.\n"
        "2. [bold red]Decision Lag Bawa Kehancuran:[/bold red] Heavyweight LLM (Sahabat-AI 8B) lambat 2.5 detik per siklus sehingga interceptor baru meluncur saat rudal telah meledak di daratan.\n"
        "3. [bold yellow]Zero Friendly Fire & Zero Token Waste:[/bold yellow] Decision Model membedakan pesawat komersil secara deterministik tanpa halusinasi dan berbiaya [bold green]0 token[/bold green].",
        border_style="yellow",
        title="💡 MENGAPA HARUS DECISION MODEL DI SISTEM PERTAHANAN?"
    ))

if __name__ == "__main__":
    main()
