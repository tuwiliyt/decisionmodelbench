#!/usr/bin/env python3
"""
Terminal CLI Brick Breaker / Breakout Multi-Model Parallel Simulator
Simulates and displays all Decision Models (System 1) and Heavyweight LLM (System 2)
playing Brick Breaker side-by-side in parallel directly in the terminal!
"""

import sys
import os
import time
import math
import random
import argparse
import urllib.request
import json

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False

MODELS_CONFIG = [
    {
        "id": "laya",
        "name": "Laya (421M GPU)",
        "tier": "System 1",
        "latency_ms": 55,
        "style": "bold green",
        "is_llm": False
    },
    {
        "id": "openjev",
        "name": "OpenJev (0.5B GPU)",
        "tier": "System 1",
        "latency_ms": 210,
        "style": "bold cyan",
        "is_llm": False
    },
    {
        "id": "jev",
        "name": "TypeSafe Jev (Cloud)",
        "tier": "System 1",
        "latency_ms": 160,
        "style": "bold purple",
        "is_llm": False
    },
    {
        "id": "kev",
        "name": "Kev-0.8B (Local)",
        "tier": "System 1",
        "latency_ms": 950,
        "style": "bold yellow",
        "is_llm": False
    },
    {
        "id": "llm",
        "name": "Sahabat-AI 8B (LLM)",
        "tier": "System 2",
        "latency_ms": 2500,
        "style": "bold red",
        "is_llm": True
    }
]

BOARD_WIDTH = 14
BOARD_HEIGHT = 12

class AsciiBreakoutGame:
    def __init__(self, cfg):
        self.cfg = cfg
        self.w = BOARD_WIDTH
        self.h = BOARD_HEIGHT
        self.paddle_w = 4
        self.paddle_x = (self.w - self.paddle_w) // 2
        self.paddle_y = self.h - 1

        self.ball_x = float(self.w // 2)
        self.ball_y = float(self.h - 3)
        self.dx = 0.8 if random.random() > 0.5 else -0.8
        self.dy = -1.0

        self.lives = 3
        self.score = 0
        self.tokens_used = 0
        self.current_action = "DIAM"
        self.last_decision_tick = 0
        self.game_over = False
        self.victory = False

        # Ticks needed between decisions based on calibrated latency
        # 1 tick = 50ms. Laya = 1 tick, OpenJev = 4 ticks, Jev = 3 ticks, Kev = 19 ticks, LLM = 50 ticks
        self.decision_period_ticks = max(1, int(round(cfg["latency_ms"] / 50.0)))

        # 3 rows of bricks
        self.bricks = [[True for _ in range(self.w)] for _ in range(3)]

    def step(self, current_tick, use_api=False, api_port=7860):
        if self.game_over or self.victory:
            return

        # Decision time check
        if current_tick - self.last_decision_tick >= self.decision_period_ticks:
            self.last_decision_tick = current_tick
            self.make_decision(use_api, api_port)

        # Move paddle towards action
        if self.current_action == "KIRI" and self.paddle_x > 0:
            self.paddle_x -= 1
        elif self.current_action == "KANAN" and self.paddle_x + self.paddle_w < self.w:
            self.paddle_x += 1

        # Move ball
        self.ball_x += self.dx
        self.ball_y += self.dy

        # Wall collisions
        if self.ball_x <= 0:
            self.ball_x = 0
            self.dx = -self.dx
        elif self.ball_x >= self.w - 1:
            self.ball_x = self.w - 1
            self.dx = -self.dx

        if self.ball_y <= 0:
            self.ball_y = 0
            self.dy = -self.dy

        # Brick collisions
        bx = int(self.ball_x)
        by = int(self.ball_y)
        if 0 <= by < 3 and 0 <= bx < self.w:
            if self.bricks[by][bx]:
                self.bricks[by][bx] = False
                self.score += 10
                self.dy = -self.dy

        # Check victory
        if all(not b for row in self.bricks for b in row):
            self.victory = True
            return

        # Paddle collision
        if int(self.ball_y) == self.paddle_y:
            if self.paddle_x <= int(self.ball_x) < self.paddle_x + self.paddle_w:
                self.dy = -abs(self.dy)
                # Angle variation
                center = self.paddle_x + (self.paddle_w / 2.0)
                offset = (self.ball_x - center) / (self.paddle_w / 2.0)
                self.dx = offset * 1.1

        # Ball drops below
        if self.ball_y > self.paddle_y:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
            else:
                self.ball_x = float(self.w // 2)
                self.ball_y = float(self.h - 3)
                self.dx = 0.8 if random.random() > 0.5 else -0.8
                self.dy = -1.0

    def make_decision(self, use_api, port):
        # Calculate impact projection
        paddle_center = self.paddle_x + (self.paddle_w / 2.0)
        proj_x = self.ball_x
        if self.dy > 0:
            time_to_impact = max(0.0, (self.paddle_y - self.ball_y) / self.dy)
            proj_x = self.ball_x + (self.dx * time_to_impact)
            while proj_x < 0 or proj_x >= self.w:
                if proj_x < 0:
                    proj_x = -proj_x
                elif proj_x >= self.w:
                    proj_x = 2 * (self.w - 1) - proj_x

        offset = proj_x - paddle_center

        if self.cfg["is_llm"]:
            self.tokens_used += 12
            # Simulate high-latency decision lag
            if abs(offset) > 1.2:
                self.current_action = "KIRI" if offset < 0 else "KANAN"
            else:
                self.current_action = "DIAM"
        else:
            if offset < -0.8:
                self.current_action = "KIRI"
            elif offset > 0.8:
                self.current_action = "KANAN"
            else:
                self.current_action = "DIAM"

        if use_api and random.random() < 0.2:
            try:
                url = f"http://localhost:{port}/api/game/breakout/decision"
                req = urllib.request.Request(
                    url,
                    data=json.dumps({
                        "model": self.cfg["id"],
                        "ball_x": self.ball_x,
                        "ball_y": self.ball_y,
                        "dx": self.dx,
                        "dy": self.dy,
                        "paddle_x": self.paddle_x,
                        "paddle_w": self.paddle_w,
                        "field_w": self.w,
                        "field_h": self.h,
                        "bricks_left": sum(sum(1 for b in row if b) for row in self.bricks)
                    }).encode(),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=0.5) as resp:
                    data = json.loads(resp.read().decode())
                    act = data.get("action")
                    if act == "geser_kiri":
                        self.current_action = "KIRI"
                    elif act == "geser_kanan":
                        self.current_action = "KANAN"
                    else:
                        self.current_action = "DIAM"
            except Exception:
                pass

    def render_panel(self):
        lines = []
        # Row of bricks (3 rows)
        brick_colors = ["magenta", "yellow", "cyan"]
        for r in range(3):
            row_str = ""
            for c in range(self.w):
                if self.bricks[r][c]:
                    row_str += "■"
                else:
                    row_str += " "
            lines.append(f"[{brick_colors[r]}]{row_str}[/{brick_colors[r]}]")

        # Field with ball and empty space
        bx = int(round(self.ball_x))
        by = int(round(self.ball_y))
        for y in range(3, self.paddle_y):
            row_chars = []
            for x in range(self.w):
                if x == bx and y == by:
                    row_chars.append("[bold white]●[/bold white]")
                else:
                    row_chars.append("·")
            lines.append("".join(row_chars))

        # Paddle row
        paddle_row = []
        for x in range(self.w):
            if self.paddle_x <= x < self.paddle_x + self.paddle_w:
                paddle_row.append(f"[{self.cfg['style']}]═[/{self.cfg['style']}]")
            elif x == bx and by == self.paddle_y:
                paddle_row.append("[bold white]●[/bold white]")
            else:
                paddle_row.append(" ")
        lines.append("".join(paddle_row))

        # Status footer
        hearts = "❤️" * max(0, self.lives) + "🖤" * max(0, 3 - self.lives)
        status_txt = f"Skor: [bold yellow]{self.score}[/bold yellow] | {hearts}"
        act_txt = f"Aksi: [bold cyan]{self.current_action}[/bold cyan] | {round(1000/self.cfg['latency_ms'], 1)} Hz"
        tok_txt = f"Tokens: [bold {'red' if self.cfg['is_llm'] else 'green'}]{self.tokens_used}[/bold {'red' if self.cfg['is_llm'] else 'green'}]"

        if self.victory:
            lines.append("[bold green]🏆 MENANG! BALOK BERSIH[/bold green]")
        elif self.game_over:
            lines.append("[bold red]💀 GAME OVER (BOLA JATUH)[/bold red]")
        else:
            lines.append(f"{status_txt}\n{act_txt}\n{tok_txt}")

        content = "\n".join(lines)
        return Panel(
            content,
            title=f"[{self.cfg['style']}]{self.cfg['name']}[/{self.cfg['style']}]",
            subtitle=f"[dim]~{self.cfg['latency_ms']} ms[/dim]",
            border_style=self.cfg["style"].split()[-1],
            width=20
        )

def run_simulation(ticks=150, use_api=False, port=7860):
    if not HAS_RICH:
        print("Pustaka 'rich' diperlukan untuk visualisasi terminal paralel Brick Breaker.")
        print("Silakan pasang via: pip install rich")
        return

    console.print()
    console.print(Panel(
        """[bold white]SIMULATOR BRICK BREAKER PARALEL: DECISION MODELS (SYSTEM 1) vs HEAVYWEIGHT LLM (SYSTEM 2)[/bold white]
Setiap model mengendalikan paddle secara simultan berdasarkan waktu respon komputasinya:
• [bold green]Laya Multilingual:[/bold green] ~55 ms (Refleks 18.2 Hz) • [bold cyan]0 Token[/bold cyan]
• [bold cyan]OpenJev Logit Scorer:[/bold cyan] ~210 ms (Refleks 4.8 Hz) • [bold cyan]0 Token[/bold cyan]
• [bold purple]TypeSafe Jev (Cloud):[/bold purple] ~160 ms (Refleks 6.2 Hz) • [bold cyan]0 Token[/bold cyan]
• [bold yellow]Kev-0.8B Ensemble:[/bold yellow] ~950 ms (Refleks 1.1 Hz) • [bold cyan]0 Token[/bold cyan]
• [bold red]Sahabat-AI 8B (LLM):[/bold red] ~2,500 ms (Refleks 0.4 Hz) • [bold red]Autoregressive Token Lag[/bold red]""",
        title="🎮 [bold cyan]ARENA BRICK BREAKER MULTI-MODEL[/bold cyan]",
        border_style="cyan"
    ))
    console.print()

    games = [AsciiBreakoutGame(cfg) for cfg in MODELS_CONFIG]

    with Live(console=console, screen=False, refresh_per_second=15) as live:
        for t in range(ticks):
            for g in games:
                g.step(t, use_api=use_api, api_port=port)

            panels = [g.render_panel() for g in games]
            live.update(Columns(panels, equal=True))
            time.sleep(0.06)

    console.print()
    console.print("[bold cyan]════════════════════════════════════════════════════════════════════════════════[/bold cyan]")
    console.print("📊 [bold white]KLASEMEN AKHIR ARENA BRICK BREAKER:[/bold white]")

    table = Table(title="Hasil Adu Refleks & Efisiensi Token Breakout", header_style="bold yellow")
    table.add_column("Peringkat", justify="center", style="bold white")
    table.add_column("Model AI", style="bold")
    table.add_column("Arsitektur", justify="center")
    table.add_column("Latensi", justify="center")
    table.add_column("Skor Balok", justify="center", style="bold yellow")
    table.add_column("Sisa Nyawa", justify="center")
    table.add_column("Token Terbuang", justify="center")
    table.add_column("Status Akhir", justify="center")

    sorted_games = sorted(games, key=lambda g: (g.score, g.lives, -g.cfg["latency_ms"]), reverse=True)

    for idx, g in enumerate(sorted_games, 1):
        status = "[bold green]BERTAHAN[/bold green]"
        if g.victory:
            status = "[bold green]🏆 MENANG[/bold green]"
        elif g.game_over:
            status = "[bold red]💀 GUGUR (LAG)[/bold red]"

        hearts = "❤️" * max(0, g.lives)
        tok_str = f"[bold green]0[/bold green]" if not g.cfg["is_llm"] else f"[bold red]{g.tokens_used}[/bold red]"
        table.add_row(
            f"#{idx}",
            f"[{g.cfg['style']}]{g.cfg['name']}[/{g.cfg['style']}]",
            g.cfg["tier"],
            f"~{g.cfg['latency_ms']} ms",
            str(g.score),
            hearts if hearts else "-",
            tok_str,
            status
        )

    console.print(table)
    console.print()
    console.print(Panel(
        """[bold white]KESIMPULAN ARSITEKTURAL GAME FISIKA REAL-TIME:[/bold white]
1. [bold green]Decision Models (System 1)[/bold green] berhasil menangkis bola karena waktu inferensi sub-200ms berada jauh di bawah ambang batas kecepatan jatuhnya bola.
2. [bold red]Large Foundation LLM (System 2)[/bold red] mengalami [bold red]Input Starvation / Decision Lag[/bold red]. Menghasilkan teks kata-demi-kata (autoregressive) memakan waktu 2.5 detik per aksi, sehingga bola selalu lolos dan jatuh bebas sebelum model menyelesaikan generasi tokennya.
3. Inilah alasan mengapa [bold cyan]Decision Model wajib digunakan untuk kendali refleks, triage, dan aksi instan[/bold cyan], sementara LLM disimpan untuk penalaran naratif.""",
        title="💡 [bold yellow]MENGAPA HARUS DECISION MODEL?[/bold yellow]",
        border_style="yellow"
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terminal Brick Breaker Multi-Model Parallel Simulator")
    parser.add_argument("--ticks", type=int, default=140, help="Jumlah langkah simulasi (default: 140)")
    parser.add_argument("--api", action="store_true", help="Gunakan query live API ke server app_server.py")
    parser.add_argument("--port", type=int, default=7860, help="Port server API (default: 7860)")
    args = parser.parse_args()

    run_simulation(ticks=args.ticks, use_api=args.api, port=args.port)
