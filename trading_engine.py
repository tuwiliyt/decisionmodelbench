#!/usr/bin/env python3
"""
Trading Engine for High-Frequency / Fast Dummy Trading Simulation.
Features:
1. Synthetic High-Frequency Tick & Candlestick Generator (BTC/USDT).
2. Real-time Technical Indicators: EMA-9, EMA-21, RSI-14, Order Book Imbalance.
3. Multi-Model Portfolio Accounting ($10,000 dummy balance per model).
4. Realistic Latency-Based Slippage Model (Sub-100ms Fast Execution vs 2.5s LLM Delay).
5. Fast Decision Model State & Question Formatter.
"""

import time
import math
import random
from typing import Dict, Any, List, Optional

class MarketTick:
    def __init__(
        self,
        timestamp: float,
        price: float,
        high: float,
        low: float,
        volume: float,
        ema9: float,
        ema21: float,
        rsi: float,
        bid_pct: float,
        ask_pct: float,
        spread: float,
        regime: str
    ):
        self.timestamp = timestamp
        self.price = price
        self.high = high
        self.low = low
        self.volume = volume
        self.ema9 = ema9
        self.ema21 = ema21
        self.rsi = rsi
        self.bid_pct = bid_pct
        self.ask_pct = ask_pct
        self.spread = spread
        self.regime = regime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 2),
            "price": round(self.price, 2),
            "high": round(self.high, 2),
            "low": round(self.low, 2),
            "volume": round(self.volume, 2),
            "ema9": round(self.ema9, 2),
            "ema21": round(self.ema21, 2),
            "rsi": round(self.rsi, 1),
            "bid_pct": round(self.bid_pct, 1),
            "ask_pct": round(self.ask_pct, 1),
            "spread": round(self.spread, 2),
            "regime": self.regime
        }

class MarketSimulator:
    def __init__(self, initial_price: float = 65000.0):
        self.initial_price = initial_price
        self.current_price = initial_price
        self.price_history: List[float] = [initial_price]
        self.tick_count = 0
        self.regime = "NORMAL" # NORMAL, BULL_RUN, FLASH_CRASH, SIDEWAYS, WHIPSAW
        self.regime_ticks_left = 30
        
        # Indicator buffers
        self.ema9 = initial_price
        self.ema21 = initial_price
        self.gains: List[float] = []
        self.losses: List[float] = []
        self.rsi = 50.0

    def set_regime(self, regime: str, duration: int = 50):
        self.regime = regime.upper()
        self.regime_ticks_left = duration

    def step(self) -> MarketTick:
        self.tick_count += 1
        self.regime_ticks_left -= 1
        
        if self.regime_ticks_left <= 0:
            # Randomly switch regimes
            choices = ["NORMAL", "BULL_RUN", "SIDEWAYS", "FLASH_CRASH", "WHIPSAW"]
            weights = [0.45, 0.20, 0.20, 0.08, 0.07]
            self.regime = random.choices(choices, weights=weights)[0]
            self.regime_ticks_left = random.randint(25, 60)

        # Volatility and drift parameters based on regime
        drift = 0.0
        volatility = 0.0015 # 0.15% per tick base
        
        if self.regime == "BULL_RUN":
            drift = 0.0035 # +0.35% drift upwards
            volatility = 0.0025
        elif self.regime == "FLASH_CRASH":
            drift = -0.0070 # -0.70% steep crash
            volatility = 0.0050
        elif self.regime == "SIDEWAYS":
            # Mean reversion to initial price
            drift = -0.001 * ((self.current_price - self.initial_price) / self.initial_price)
            volatility = 0.0008
        elif self.regime == "WHIPSAW":
            volatility = 0.0060 # Extreme choppy noise

        # Price innovation
        noise = random.gauss(0, 1)
        pct_change = drift + (volatility * noise)
        old_price = self.current_price
        new_price = max(1000.0, old_price * (1.0 + pct_change))
        self.current_price = new_price
        self.price_history.append(new_price)

        # High/Low for candle
        spread_val = round(random.uniform(0.05, 0.25) * (new_price / 65000.0), 2)
        high = max(old_price, new_price) + (abs(noise) * spread_val * 2.0)
        low = min(old_price, new_price) - (abs(noise) * spread_val * 2.0)
        volume = round(random.uniform(2.5, 25.0) * (1.0 + abs(pct_change) * 50.0), 2)

        # Update EMA-9 & EMA-21
        k9 = 2.0 / (9.0 + 1.0)
        k21 = 2.0 / (21.0 + 1.0)
        self.ema9 = (new_price * k9) + (self.ema9 * (1.0 - k9))
        self.ema21 = (new_price * k21) + (self.ema21 * (1.0 - k21))

        # Update RSI (14 period)
        diff = new_price - old_price
        gain = max(0.0, diff)
        loss = max(0.0, -diff)
        self.gains.append(gain)
        self.losses.append(loss)
        if len(self.gains) > 14:
            self.gains.pop(0)
            self.losses.pop(0)

        avg_gain = sum(self.gains) / max(1, len(self.gains))
        avg_loss = sum(self.losses) / max(1, len(self.losses))
        if avg_loss == 0:
            self.rsi = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            self.rsi = 100.0 - (100.0 / (1.0 + rs))

        # Order Book Imbalance
        # Bullish regimes produce more bids; crash produces heavy asks
        base_bid = 50.0
        if self.regime == "BULL_RUN":
            base_bid = 72.0
        elif self.regime == "FLASH_CRASH":
            base_bid = 24.0
        elif pct_change > 0:
            base_bid = 58.0
        else:
            base_bid = 42.0

        bid_pct = max(10.0, min(90.0, base_bid + random.gauss(0, 6)))
        ask_pct = 100.0 - bid_pct

        return MarketTick(
            timestamp=time.time(),
            price=new_price,
            high=high,
            low=low,
            volume=volume,
            ema9=self.ema9,
            ema21=self.ema21,
            rsi=self.rsi,
            bid_pct=bid_pct,
            ask_pct=ask_pct,
            spread=spread_val,
            regime=self.regime
        )

class ModelPortfolio:
    def __init__(
        self,
        model_id: str,
        name: str,
        architecture: str,
        typical_latency_ms: float,
        token_cost_per_trade: int,
        initial_cash: float = 10000.0
    ):
        self.model_id = model_id
        self.name = name
        self.architecture = architecture
        self.typical_latency_ms = typical_latency_ms
        self.token_cost_per_trade = token_cost_per_trade
        self.initial_cash = initial_cash

        self.cash = initial_cash
        self.position_qty = 0.0 # BTC amount
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.win_trades = 0
        self.loss_trades = 0
        self.total_tokens = 0
        self.peak_equity = initial_cash
        self.max_drawdown_pct = 0.0
        self.last_action = "HOLD"
        self.last_reason = "Init"
        self.trade_history: List[Dict[str, Any]] = []

    def get_equity(self, current_price: float) -> float:
        return self.cash + (self.position_qty * current_price)

    def get_unrealized_pnl(self, current_price: float) -> float:
        if self.position_qty <= 0:
            return 0.0
        return (current_price - self.avg_entry_price) * self.position_qty

    def execute_decision(
        self,
        action: str, # BUY, SELL, HOLD
        current_price: float,
        actual_latency_ms: Optional[float] = None,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        action = action.upper()
        lat_ms = actual_latency_ms if actual_latency_ms is not None else self.typical_latency_ms
        self.last_action = action
        
        # Calculate realistic slippage based on latency:
        # Latency of 50ms -> slippage ~0.02%
        # Latency of 2500ms (LLM) -> slippage ~0.45% - 1.2%
        lat_ratio = max(1.0, lat_ms / 50.0)
        slippage_pct = min(0.035, 0.00025 * math.sqrt(lat_ratio))
        
        result = {
            "model_id": self.model_id,
            "action": action,
            "signal_price": current_price,
            "fill_price": current_price,
            "slippage_pct": round(slippage_pct * 100, 3),
            "executed": False,
            "message": ""
        }

        # Deduct token cost
        if action in ("BUY", "SELL"):
            self.total_tokens += self.token_cost_per_trade

        if action == "BUY":
            # Buy with up to 40% of available cash
            fill_price = current_price * (1.0 + slippage_pct)
            alloc_cash = self.cash * 0.40
            if alloc_cash > 100.0:
                qty = alloc_cash / fill_price
                new_total_qty = self.position_qty + qty
                # Update weighted average entry price
                self.avg_entry_price = (
                    (self.position_qty * self.avg_entry_price) + (qty * fill_price)
                ) / new_total_qty
                self.position_qty = new_total_qty
                self.cash -= alloc_cash
                self.total_trades += 1
                result["executed"] = True
                result["fill_price"] = round(fill_price, 2)
                result["qty"] = round(qty, 4)
                result["message"] = f"Beli {qty:.4f} BTC @ ${fill_price:,.2f}"
                self.trade_history.append({
                    "time": time.time(),
                    "type": "BUY",
                    "price": fill_price,
                    "qty": qty,
                    "pnl": 0.0,
                    "tokens": self.token_cost_per_trade
                })
            else:
                result["message"] = "Kas tidak mencukupi untuk membuka posisi beli"

        elif action == "SELL":
            # Liquidate current position or cut loss
            if self.position_qty > 0.0001:
                fill_price = current_price * (1.0 - slippage_pct)
                proceeds = self.position_qty * fill_price
                trade_pnl = (fill_price - self.avg_entry_price) * self.position_qty
                self.cash += proceeds
                self.realized_pnl += trade_pnl
                self.total_trades += 1
                if trade_pnl > 0:
                    self.win_trades += 1
                else:
                    self.loss_trades += 1
                
                result["executed"] = True
                result["fill_price"] = round(fill_price, 2)
                result["qty"] = round(self.position_qty, 4)
                result["pnl"] = round(trade_pnl, 2)
                result["message"] = f"Jual {self.position_qty:.4f} BTC @ ${fill_price:,.2f} (P&L: ${trade_pnl:+,.2f})"

                self.trade_history.append({
                    "time": time.time(),
                    "type": "SELL",
                    "price": fill_price,
                    "qty": self.position_qty,
                    "pnl": trade_pnl,
                    "tokens": self.token_cost_per_trade
                })

                self.position_qty = 0.0
                self.avg_entry_price = 0.0
            else:
                result["message"] = "Tidak ada posisi BTC untuk dijual (FLAT)"

        elif action == "HOLD":
            result["message"] = "Tahan posisi (Wait & See)"

        # Update peak equity and drawdown
        equity = self.get_equity(current_price)
        if equity > self.peak_equity:
            self.peak_equity = equity
        dd = (self.peak_equity - equity) / max(1.0, self.peak_equity) * 100.0
        if dd > self.max_drawdown_pct:
            self.max_drawdown_pct = dd

        return result

    def to_dict(self, current_price: float) -> Dict[str, Any]:
        equity = self.get_equity(current_price)
        unrealized = self.get_unrealized_pnl(current_price)
        win_rate = (self.win_trades / self.total_trades * 100.0) if self.total_trades > 0 else 0.0
        total_pnl = equity - self.initial_cash
        total_roi_pct = (total_pnl / self.initial_cash) * 100.0

        return {
            "model_id": self.model_id,
            "name": self.name,
            "architecture": self.architecture,
            "typical_latency_ms": self.typical_latency_ms,
            "cash": round(self.cash, 2),
            "position_qty": round(self.position_qty, 4),
            "avg_entry_price": round(self.avg_entry_price, 2),
            "unrealized_pnl": round(unrealized, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "total_equity": round(equity, 2),
            "roi_pct": round(total_roi_pct, 2),
            "total_trades": self.total_trades,
            "win_trades": self.win_trades,
            "loss_trades": self.loss_trades,
            "win_rate_pct": round(win_rate, 1),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "total_tokens": self.total_tokens,
            "last_action": self.last_action
        }

class TradingArena:
    def __init__(self, initial_cash: float = 10000.0):
        self.market = MarketSimulator()
        self.initial_cash = initial_cash
        self.portfolios: Dict[str, ModelPortfolio] = {
            "laya": ModelPortfolio(
                model_id="laya",
                name="Laya Multilingual (421M GPU)",
                architecture="System 1 (ModernBERT RLCD)",
                typical_latency_ms=55.0,
                token_cost_per_trade=0,
                initial_cash=initial_cash
            ),
            "openjev": ModelPortfolio(
                model_id="openjev",
                name="OpenJev (0.5B GPU Logit Scorer)",
                architecture="System 1 (Logit Head Contrast)",
                typical_latency_ms=210.0,
                token_cost_per_trade=0,
                initial_cash=initial_cash
            ),
            "jev": ModelPortfolio(
                model_id="jev",
                name="TypeSafe Jev (Cloud SaaS)",
                architecture="System 1 (Cloud Decision Engine)",
                typical_latency_ms=160.0,
                token_cost_per_trade=0,
                initial_cash=initial_cash
            ),
            "kev": ModelPortfolio(
                model_id="kev",
                name="Kev-0.8B (Local Ensemble)",
                architecture="System 1 (LoRA Ensemble)",
                typical_latency_ms=950.0,
                token_cost_per_trade=0,
                initial_cash=initial_cash
            ),
            "sahabatai": ModelPortfolio(
                model_id="sahabatai",
                name="Heavyweight LLM (Sahabat-AI 8B)",
                architecture="System 2 (Autoregressive Decoder)",
                typical_latency_ms=2500.0,
                token_cost_per_trade=32, # ~32 tokens per response
                initial_cash=initial_cash
            )
        }
        self.history_ticks: List[Dict[str, Any]] = []

    def reset(self):
        self.market = MarketSimulator()
        for p in self.portfolios.values():
            p.cash = self.initial_cash
            p.position_qty = 0.0
            p.avg_entry_price = 0.0
            p.realized_pnl = 0.0
            p.total_trades = 0
            p.win_trades = 0
            p.loss_trades = 0
            p.total_tokens = 0
            p.peak_equity = self.initial_cash
            p.max_drawdown_pct = 0.0
            p.last_action = "HOLD"
            p.trade_history.clear()
        self.history_ticks.clear()

    def build_trading_prompt(self, tick: MarketTick, model_id: str) -> Dict[str, Any]:
        p = self.portfolios.get(model_id)
        pos_str = f"LONG ({p.position_qty:.4f} BTC, Floating PnL: ${p.get_unrealized_pnl(tick.price):+,.2f})" if p and p.position_qty > 0 else "FLAT (Tidak ada posisi)"
        cash_val = p.cash if p else 10000.0

        cross_status = "Golden Cross (EMA9 > EMA21)" if tick.ema9 > tick.ema21 else "Death Cross (EMA9 < EMA21)"
        rsi_label = "Overbought (Jenuh Beli)" if tick.rsi > 70 else ("Oversold (Jenuh Jual)" if tick.rsi < 30 else "Netral")
        imbalance = "Dominasi Tekanan Beli (Bid)" if tick.bid_pct > 60 else ("Dominasi Tekanan Jual (Ask)" if tick.ask_pct > 60 else "Order Book Seimbang")

        state = (
            f"Pasar BTC/USDT Real-Time: Harga ${tick.price:,.2f} ({tick.regime}). "
            f"Indikator Teknis: {cross_status}, RSI(14)={tick.rsi:.1f} [{rsi_label}]. "
            f"Order Book Depth: {tick.bid_pct:.1f}% Bid vs {tick.ask_pct:.1f}% Ask [{imbalance}]. "
            f"Spread: ${tick.spread:.2f}, Volume: {tick.volume:.1f} BTC. "
            f"Portofolio Anda: Saldo Kas ${cash_val:,.2f}, Posisi Saat Ini: {pos_str}."
        )

        questions = {
            "action": {
                "type": "choice",
                "instructions": "Keputusan eksekusi trading instan?",
                "criteria": {
                    "BUY": "Beli (Long) memanfaatkan momentum kenaikan harga atau oversold bounce",
                    "SELL": "Jual (Short/Cut Loss/Take Profit) menghindari koreksi atau mengunci keuntungan",
                    "HOLD": "Tahan / jangan ambil tindakan, tunggu konfirmasi sinyal berikutnya"
                }
            },
            "urgency": {
                "type": "score",
                "instructions": "Tingkat urgensi eksekusi saat ini",
                "criteria": ["santai", "waspada", "segera", "darurat_ekstrem"]
            }
        }

        return {"state": state, "questions": questions}

    def evaluate_algorithmic_decision(self, tick: MarketTick, model_id: str) -> str:
        """
        Pure technical baseline algorithm for synthetic simulation loops
        when full neural models are calibrated to real measured latency.
        """
        p = self.portfolios.get(model_id)
        has_pos = p and p.position_qty > 0.0001
        
        # Golden Cross & RSI healthy & Bid dominance -> BUY
        if tick.ema9 > tick.ema21 and tick.rsi < 68 and tick.bid_pct > 55 and not has_pos:
            return "BUY"
        # Extreme Oversold Bounce -> BUY
        elif tick.rsi < 28 and tick.bid_pct > 50 and not has_pos:
            return "BUY"
        # Death Cross or Flash Crash or RSI Overbought -> SELL
        elif (tick.ema9 < tick.ema21 or tick.rsi > 75 or tick.regime == "FLASH_CRASH") and has_pos:
            return "SELL"
        # Take profit if gain > 3%
        elif has_pos and p.get_unrealized_pnl(tick.price) > (p.cash * 0.05):
            return "SELL"
        # Stop loss if loss < -2.5%
        elif has_pos and p.get_unrealized_pnl(tick.price) < -(p.cash * 0.025):
            return "SELL"
        
        return "HOLD"
