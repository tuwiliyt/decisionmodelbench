#!/usr/bin/env python3
"""
Play Fast Trading Arena (CLI Terminal Simulator)
High-Frequency / Fast Dummy Trading Simulation ($10,000 Dummy Balance).
Tests 5 models in parallel:
1. Laya Multilingual (421M GPU) - ~55 ms, 0 tokens
2. OpenJev (0.5B GPU Logit Scorer) - ~210 ms, 0 tokens
3. TypeSafe Jev (Cloud SaaS) - ~160 ms, 0 tokens
4. Kev-0.8B (Local Ensemble) - ~950 ms, 0 tokens
5. Sahabat-AI 8B (Heavyweight LLM) - ~2,500 ms, ~32 tokens, High Slippage

Usage:
  python3 play_fast_trading.py
  python3 play_fast_trading.py --ticks 40 --speed fast
  python3 play_fast_trading.py --regime FLASH_CRASH
  python3 play_fast_trading.py --api
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
from rich.layout import Layout

from trading_engine import TradingArena, MarketTick

console = Console()

def render_ascii_sparkline(prices, width=28):
    if not prices:
        return "·" * width
    pts = prices[-width:]
    min_p = min(pts)
    max_p = max(pts)
    if max_p == min_p:
        return "▄" * len(pts)
    bars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    res = ""
    for p in pts:
        idx = int(((p - min_p) / (max_p - min_p)) * (len(bars) - 1))
        res += bars[max(0, min(len(bars) - 1, idx))]
    return res

def create_model_panel(p, current_price: float):
    d = p.to_dict(current_price)
    eq = d["total_equity"]
    pnl = d["total_pnl"]
    roi = d["roi_pct"]
    wr = d["win_rate_pct"]
    tr = d["total_trades"]
    tok = d["total_tokens"]
    act = d["last_action"]

    pnl_style = "bold green" if pnl >= 0 else "bold red"
    pnl_sign = "+" if pnl >= 0 else ""

    act_style = "bold white on green" if "BUY" in act else ("bold white on red" if "SELL" in act else "dim white")

    content = Text()
    content.append(f"Equity: ", style="dim")
    content.append(f"${eq:,.2f}\n", style="bold white")
    content.append(f"P&L:    ", style="dim")
    content.append(f"{pnl_sign}${pnl:,.2f} ({pnl_sign}{roi:.1f}%)\n", style=pnl_style)
    content.append(f"Posisi: ", style="dim")
    pos_str = f"{d['position_qty']:.4f} BTC" if d['position_qty'] > 0 else "FLAT"
    content.append(f"{pos_str}\n", style="cyan")
    content.append(f"WinRate:", style="dim")
    content.append(f" {wr:.1f}% ({tr} trades)\n", style="yellow")
    content.append(f"Token:  ", style="dim")
    tok_style = "red bold" if tok > 0 else "green"
    content.append(f"{tok} tokens\n", style=tok_style)
    content.append(f"Aksi:   ", style="dim")
    content.append(f" {act} ", style=act_style)

    border_color = "green" if "Laya" in p.name else ("cyan" if "OpenJev" in p.name else ("magenta" if "TypeSafe" in p.name else ("yellow" if "Kev" in p.name else "red")))

    title = f"{p.name.split(' (')[0]} (~{p.typical_latency_ms:.0f}ms)"
    return Panel(content, title=title, border_style=border_color, width=30)

def main():
    parser = argparse.ArgumentParser(description="Arena Simulasi Trading Cepat AI")
    parser.add_argument("--ticks", type=int, default=60, help="Jumlah tick pasar untuk disimulasikan (default: 60)")
    parser.add_argument("--speed", type=str, default="normal", choices=["slow", "normal", "fast", "turbo"], help="Kecepatan tick")
    parser.add_argument("--regime", type=str, default="NORMAL", choices=["NORMAL", "BULL_RUN", "FLASH_CRASH", "SIDEWAYS", "WHIPSAW"], help="Regime pasar")
    parser.add_argument("--api", action="store_true", help="Uji 1 probe keputusan trading live ke backend FastAPI server")
    args = parser.parse_args()

    if args.api:
        console.print("[bold cyan]🔌 Menguji Probe API Trading ke http://127.0.0.1:7860/api/predict ...[/bold cyan]")
        try:
            req_data = {
                "state": "Pasar BTC/USDT spot. Harga $65,200. Golden Cross EMA-9 > EMA-21, RSI 62.1. Order book 74% Bid. Saldo Dummy $10,000 USD. Posisi FLAT.",
                "questions": {
                    "action": {
                        "type": "choice",
                        "instructions": "Keputusan trading instan?",
                        "criteria": {
                            "BUY": "Beli (Long) momentum breakout",
                            "SELL": "Jual (Short) antisipasi koreksi",
                            "HOLD": "Tahan / jangan ambil aksi"
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
            console.print(f"[bold green]✓ Response diterima dalam {lat:.1f} ms![/bold green]")
            console.print_json(json.dumps(res))
        except Exception as e:
            console.print(f"[bold red]✗ Gagal menghubungi API server: {e}[/bold red]")
            console.print("[yellow]Pastikan uvicorn berjalan di port 7860.[/yellow]")
        return

    delay_map = {"slow": 0.4, "normal": 0.15, "fast": 0.05, "turbo": 0.01}
    delay = delay_map.get(args.speed, 0.15)

    arena = TradingArena(initial_cash=10000.0)
    if args.regime != "NORMAL":
        arena.market.set_regime(args.regime, duration=args.ticks + 10)

    console.print(Panel(
        f"[bold green]⚡ ARENA SIMULASI TRADING CEPAT AI (5 MODEL PARALEL)[/bold green]\n"
        f"Saldo Dummy Awal: [bold white]$10,000.00 USD[/bold white] | Aset: [cyan]BTC/USDT Spot[/cyan] | Target Ticks: [yellow]{args.ticks}[/yellow]\n"
        f"Regime: [bold cyan]{args.regime}[/bold cyan] | Model: [green]Laya[/green], [cyan]OpenJev[/cyan], [magenta]Jev[/magenta], [yellow]Kev[/yellow] vs [red]Sahabat-AI 8B[/red]",
        border_style="cyan"
    ))

    trade_logs = []
    prices_history = []

    with Live(console=console, refresh_per_second=15) as live:
        for t in range(1, args.ticks + 1):
            tick = arena.market.step()
            prices_history.append(tick.price)

            # Evaluate each model
            for m_id, portfolio in arena.portfolios.items():
                action = arena.evaluate_algorithmic_decision(tick, m_id)
                res = portfolio.execute_decision(action, tick.price)
                if res.get("executed"):
                    type_str = "[green]BUY[/green]" if res["action"] == "BUY" else "[red]SELL[/red]"
                    pnl_str = f" [green]+${res.get('pnl', 0):.1f}[/green]" if res.get('pnl', 0) > 0 else (f" [red]${res.get('pnl', 0):.1f}[/red]" if res.get('pnl', 0) < 0 else "")
                    trade_logs.append(f"Tick {t:02d} | [bold]{portfolio.name.split(' (')[0]}[/bold]: {type_str} @ ${res['fill_price']:,.2f}{pnl_str}")
                    if len(trade_logs) > 4:
                        trade_logs.pop(0)

            # Build UI Layout
            # 1. Market Banner
            spark = render_ascii_sparkline(prices_history, width=32)
            cross_icon = "[green]Golden Cross ↗[/green]" if tick.ema9 > tick.ema21 else "[red]Death Cross ↘[/red]"
            rsi_style = "red bold" if tick.rsi > 70 else ("green bold" if tick.rsi < 30 else "yellow")
            
            # Bid/Ask visual bar
            bid_b = int(tick.bid_pct / 5)
            ask_b = 20 - bid_b
            imbalance_bar = f"[green]{'█'*bid_b}[/green][red]{'█'*ask_b}[/red]"

            header_text = Text()
            header_text.append(f"BTC/USDT: ", style="bold white")
            header_text.append(f"${tick.price:,.2f}  ", style="bold yellow")
            header_text.append(f"Regime: [{tick.regime}]  ", style="cyan")
            header_text.append(f"Chart: {spark}\n", style="white")
            header_text.append(f"EMA9: ${tick.ema9:,.1f} | EMA21: ${tick.ema21:,.1f} ({cross_icon})  ", style="dim")
            header_text.append(f"RSI(14): {tick.rsi:.1f}  ", style=rsi_style)
            header_text.append(f"Book: {imbalance_bar} ({tick.bid_pct:.0f}% Bid / {tick.ask_pct:.0f}% Ask)", style="dim")

            header_panel = Panel(header_text, title=f"⚡ Tick {t}/{args.ticks} - Live Market Tape", border_style="cyan")

            # 2. Model Cards Columns
            cards = [create_model_panel(p, tick.price) for p in arena.portfolios.values()]
            cards_col = Columns(cards, equal=True)

            # 3. Trade logs panel
            log_text = "\n".join(trade_logs) if trade_logs else "[dim]Menunggu konfirmasi sinyal teknikal...[/dim]"
            log_panel = Panel(log_text, title="📜 Riwayat Transaksi Eksekusi Terakhir", border_style="dim", height=6)

            live.update(Group(
                header_panel,
                cards_col,
                log_panel
            ))

            time.sleep(delay)

    # FINAL LEADERBOARD TABLE
    console.print("\n" + "=" * 80)
    console.print("[bold yellow]📊 KLASEMEN AKHIR ARENA TRADING CEPAT (SALDO DUMMY $10,000 USD):[/bold yellow]")
    
    table = Table(title="Hasil Akhir Evaluasi Portofolio AI", header_style="bold cyan", border_style="dim")
    table.add_column("Peringkat", justify="center", style="bold")
    table.add_column("Model AI", style="bold white")
    table.add_column("Arsitektur", style="cyan")
    table.add_column("Latensi", justify="center")
    table.add_column("Saldo Akhir", justify="right", style="bold")
    table.add_column("Total P&L ($)", justify="right")
    table.add_column("ROI (%)", justify="right")
    table.add_column("Win Rate", justify="center")
    table.add_column("Trades", justify="center")
    table.add_column("Token Terbuang", justify="right")

    final_price = arena.market.current_price
    sorted_ports = sorted(
        arena.portfolios.values(),
        key=lambda p: p.get_equity(final_price),
        reverse=True
    )

    medals = ["🥇 #1", "🥈 #2", "🥉 #3", "   #4", "   #5"]
    for i, p in enumerate(sorted_ports):
        d = p.to_dict(final_price)
        eq = d["total_equity"]
        pnl = d["total_pnl"]
        roi = d["roi_pct"]
        wr = d["win_rate_pct"]
        tr = d["total_trades"]
        tok = d["total_tokens"]

        pnl_style = "[green]" if pnl >= 0 else "[red]"
        pnl_str = f"{pnl_style}{'+' if pnl >= 0 else ''}${pnl:,.2f}[/]"
        roi_str = f"{pnl_style}{'+' if roi >= 0 else ''}{roi:.2f}%[/]"
        tok_str = f"[red]{tok}[/red]" if tok > 0 else "[green]0[/green]"

        table.add_row(
            medals[i],
            p.name,
            p.architecture.split(" (")[0],
            f"~{p.typical_latency_ms:.0f} ms",
            f"${eq:,.2f}",
            pnl_str,
            roi_str,
            f"{wr:.1f}%",
            str(tr),
            tok_str
        )

    console.print(table)

    console.print(Panel(
        "[bold cyan]KESIMPULAN ARSITEKTURAL HIGH-FREQUENCY / FAST TRADING:[/bold cyan]\n"
        "1. [bold green]Sub-100ms Latency Wins:[/bold green] Decision Models (System 1: Laya, OpenJev, Jev) mengeksekusi harga secara instan dengan [bold green]0% slippage[/bold green].\n"
        "2. [bold red]Decision Lag Kills LLM:[/bold red] Heavyweight LLM (Sahabat-AI 8B) butuh 2.5 detik per keputusan sehingga selalu membeli terlambat di pucuk dan cut-loss di dasar jurang (Slippage Parah).\n"
        "3. [bold yellow]Zero Token Waste:[/bold yellow] Ribuan tick trading diselesaikan Decision Model dengan [bold green]0 token[/bold green], sedangkan LLM membakar ribuan token sia-sia.",
        border_style="yellow",
        title="💡 MENGAPA HARUS DECISION MODEL DI INDUSTRI FINANSIAL?"
    ))

if __name__ == "__main__":
    main()
