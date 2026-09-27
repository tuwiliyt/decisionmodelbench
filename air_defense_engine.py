#!/usr/bin/env python3
"""
Air Defense Tactical Simulation Engine (Iron Dome / C-RAM AI Simulator)
Features:
1. Multi-Threat Sky Object Generator:
   - Hostile Targets: Hypersonic Missiles (🚀), Impact Meteorites (☄️), Kamikaze Drones (🛸).
   - Non-Threats / Friendly: Commercial Airliners (✈️, active IFF), Flocks of Birds (🦅), Atmosphere Miss Meteorites (🌠).
2. Progressive Wave Escalation (Difficulty increases over time):
   - Wave 1 to 5+: Speed multiplier, saturation density, reduced time-to-impact.
3. Real-Life Battery Load Physics:
   - Pod magazine limit: 20 interceptors per pod (Iron Dome Tamir / C-RAM standard).
   - Salvo ripple firing interval (prevents instant magazine dump).
   - Reload cooldown cycle when empty (35 ticks / ~3.5s vulnerable reload window).
4. Threat-Specific City Damage Calculation:
   - 🚀 Hypersonic Missile: 35.0% structural damage.
   - ☄️ Kinetic Meteorite: 45.0% cratering damage.
   - 🛸 Kamikaze Drone: 12.0% tactical payload damage.
   - 💥 Low-Altitude Shrapnel: 3.0% collateral blast damage.
5. Selectable LLM Models (System 2):
   - Sahabat-AI 8B, Qwen 2.5 7B, Gemma 2 9B, Gemma 2 2B.
6. 5 Parallel Cities Defended by 5 AI Architectures:
   - Jakarta: Laya Multilingual (421M GPU) - ~55 ms reflex
   - Surabaya: OpenJev (0.5B GPU Local Logits) - ~210 ms reflex
   - Bandung: TypeSafe JEV (Cloud Decision API) - ~160 ms reflex (Prominently featured)
   - Medan: Kev-0.8B (Local Ensemble) - ~950 ms reflex
   - Nusantara (IKN): Heavyweight LLM (Selectable: Sahabat-AI / Qwen / Gemma) - ~1100-2800 ms Decision Lag
7. Periodic SITREP (Situation Report):
   - Periodic recaps of intercepted threats vs city impacts vs civilian casualties vs ammo vs token costs.
"""

import time
import math
import random
from typing import Dict, Any, List, Optional, Set

class SkyObject:
    def __init__(
        self,
        obj_id: int,
        obj_type: str, # MISSILE, METEOR_IMPACT, DRONE, AIRLINER, BIRD_FLOCK, METEOR_MISS
        x: float,
        y: float,
        vx: float,
        vy: float,
        speed: float,
        iff_code: Optional[str] = None,
        altitude_km: float = 12.0
    ):
        self.obj_id = obj_id
        self.obj_type = obj_type
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.speed = speed
        self.iff_code = iff_code # E.g. "SQUAWK-7700", "GARUDA-GA402", None for hostile
        self.altitude_km = altitude_km
        self.is_hostile = obj_type in ("MISSILE", "METEOR_IMPACT", "DRONE")
        self.escaped = False

    def time_to_impact(self, ground_y: float = 380.0) -> float:
        if self.vy <= 0:
            return 999.0
        return max(0.0, (ground_y - self.y) / self.vy)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.obj_id,
            "type": self.obj_type,
            "is_hostile": self.is_hostile,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "vx": round(self.vx, 2),
            "vy": round(self.vy, 2),
            "speed_mach": round(self.speed, 1),
            "altitude_km": round(self.altitude_km, 1),
            "iff": self.iff_code,
            "escaped": self.escaped
        }

class CityDefenseSystem:
    # Threat-specific damage values
    DAMAGE_TABLE = {
        "METEOR_IMPACT": 45.0, # High kinetic hypervelocity cratering
        "MISSILE": 35.0,       # High-explosive warhead
        "DRONE": 12.0,         # Tactical shaped-charge payload
        "SHRAPNEL": 3.0        # Low-altitude blast collateral
    }

    def __init__(
        self,
        city_name: str,
        model_id: str,
        model_label: str,
        typical_latency_ms: float,
        token_cost: int
    ):
        self.city_name = city_name
        self.model_id = model_id
        self.model_label = model_label
        self.typical_latency_ms = typical_latency_ms
        self.token_cost = token_cost

        # Health & damage state
        self.city_hp = 100.0 # 0 - 100%
        self.intercepted_count = 0
        self.impact_missile_count = 0
        self.civilian_safe_count = 0
        self.friendly_fire_count = 0
        self.tokens_burned = 0

        # Real-life battery firing load mechanics
        self.max_ammo = 20 # Standard 20-cell Tamir launcher canister
        self.ammo = 20
        self.is_reloading = False
        self.reload_ticks_remaining = 0
        self.reload_duration_ticks = 35 # ~3.5 seconds reload cycle
        self.fire_cooldown_ticks = 0 # Ripple interval between launches (2 ticks)

        # Interceptors and targets tracking for fair independent evaluation
        self.interceptor_missiles: List[Dict[str, Any]] = []
        self.assigned_targets: Set[int] = set()
        self.destroyed_targets: Set[int] = set()
        self.impacted_targets: Set[int] = set()
        self.civilian_safe_ids: Set[int] = set()
        self.recent_events: List[str] = []

    def reset(self):
        self.city_hp = 100.0
        self.intercepted_count = 0
        self.impact_missile_count = 0
        self.civilian_safe_count = 0
        self.friendly_fire_count = 0
        self.tokens_burned = 0
        self.ammo = self.max_ammo
        self.is_reloading = False
        self.reload_ticks_remaining = 0
        self.fire_cooldown_ticks = 0
        self.interceptor_missiles.clear()
        self.assigned_targets.clear()
        self.destroyed_targets.clear()
        self.impacted_targets.clear()
        self.civilian_safe_ids.clear()
        self.recent_events.clear()

    def can_fire(self) -> bool:
        return self.ammo > 0 and not self.is_reloading and self.fire_cooldown_ticks <= 0

    def fire_interceptor(self, target: SkyObject, latency_penalty_ms: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if not self.can_fire():
            return None

        # Deduct ammo & burn tokens if LLM
        self.ammo -= 1
        self.fire_cooldown_ticks = 2 # Ripple spacing
        self.tokens_burned += self.token_cost

        # Trigger reload if magazine is depleted
        if self.ammo <= 0:
            self.is_reloading = True
            self.reload_ticks_remaining = self.reload_duration_ticks
            self.recent_events.append(f"⚠️ {self.city_name}: Amunisi habis! Memulai siklus Reload (3.5s)...")

        lat = latency_penalty_ms if latency_penalty_ms is not None else self.typical_latency_ms

        # Interceptor starts at city battery launcher (x=130, y=370)
        interceptor = {
            "x": 130.0,
            "y": 370.0,
            "target_id": target.obj_id,
            "target_type": target.obj_type,
            "speed": 8.5, # Rapid ascent
            "delay_ticks": max(1, int(lat / 50.0)), # Latency delay before rocket launch
            "active": True,
            "trail": []
        }
        self.interceptor_missiles.append(interceptor)

        event_msg = f"🚀 {self.model_label}: Menembakkan rudal pencegat ({self.ammo}/{self.max_ammo}) -> Sasaran #{target.obj_id}"
        self.recent_events.append(event_msg)
        if len(self.recent_events) > 6:
            self.recent_events.pop(0)

        return interceptor

    def tick_battery_load(self):
        """Ticks reload cooldown and salvo ripple cooldown."""
        if self.fire_cooldown_ticks > 0:
            self.fire_cooldown_ticks -= 1

        if self.is_reloading:
            self.reload_ticks_remaining -= 1
            if self.reload_ticks_remaining <= 0:
                self.ammo = self.max_ammo
                self.is_reloading = False
                self.recent_events.append(f"🔄 {self.city_name}: Baterai terisi penuh (20/20) Siap Tempur!")
                if len(self.recent_events) > 6:
                    self.recent_events.pop(0)

    def to_dict(self) -> Dict[str, Any]:
        total_hostiles = self.intercepted_count + self.impact_missile_count
        intercept_rate = (self.intercepted_count / total_hostiles * 100.0) if total_hostiles > 0 else 100.0
        civ_total = self.civilian_safe_count + self.friendly_fire_count
        civ_safety_rate = (self.civilian_safe_count / civ_total * 100.0) if civ_total > 0 else 100.0

        if self.city_hp > 65:
            status_text = "OPERASIONAL PRIMA"
        elif self.city_hp > 30:
            status_text = "RUSAK WASPADA"
        elif self.city_hp > 0:
            status_text = "KRITIS MENDEKATI HANCUR"
        else:
            status_text = "KOTA HANCUR TOTAL"

        return {
            "city_name": self.city_name,
            "model_id": self.model_id,
            "model_label": self.model_label,
            "typical_latency_ms": self.typical_latency_ms,
            "city_hp": round(self.city_hp, 1),
            "ammo": self.ammo,
            "max_ammo": self.max_ammo,
            "is_reloading": self.is_reloading,
            "reload_remaining_sec": round(self.reload_ticks_remaining * 0.1, 1) if self.is_reloading else 0.0,
            "intercepted_count": self.intercepted_count,
            "impact_missile_count": self.impact_missile_count,
            "intercept_rate_pct": round(intercept_rate, 1),
            "civilian_safe_count": self.civilian_safe_count,
            "friendly_fire_count": self.friendly_fire_count,
            "civ_safety_rate_pct": round(civ_safety_rate, 1),
            "tokens_burned": self.tokens_burned,
            "active_interceptors": len([i for i in self.interceptor_missiles if i.get("active")]),
            "status": status_text,
            "recent_events": self.recent_events[-3:]
        }

class AirDefenseArena:
    # Catalog of selectable heavyweight & lightweight LLM models for City 5 (IKN)
    LLM_MODELS_CATALOG = {
        "sahabatai": {
            "name": "Nusantara (IKN)",
            "label": "Heavyweight LLM (Sahabat-AI 8B Instruct)",
            "latency_ms": 2500.0,
            "token_cost": 36,
            "desc": "Indonesian Sovereign LLM • 28 t/s • Decision Lag ~2.5s"
        },
        "qwen": {
            "name": "Nusantara (IKN)",
            "label": "Heavyweight LLM (Qwen 2.5 7B Instruct)",
            "latency_ms": 2200.0,
            "token_cost": 36,
            "desc": "Alibaba Reasoning LLM • 32 t/s • Decision Lag ~2.2s"
        },
        "gemma": {
            "name": "Nusantara (IKN)",
            "label": "Heavyweight LLM (Gemma 2 9B Instruct)",
            "latency_ms": 2800.0,
            "token_cost": 36,
            "desc": "Google DeepMind LLM • 25 t/s • Decision Lag ~2.8s"
        },
        "gemma-2b": {
            "name": "Nusantara (IKN)",
            "label": "Lightweight LLM (Gemma 2 2B Instruct)",
            "latency_ms": 1100.0,
            "token_cost": 32,
            "desc": "Google Compact LLM • 60 t/s • Decision Lag ~1.1s"
        }
    }

    def __init__(self):
        self.wave = 1
        self.wave_tick = 0
        self.ticks_per_wave = 120 # ~30-40 seconds per wave
        self.obj_counter = 0
        self.current_llm_key = "sahabatai"

        self.cities: Dict[str, CityDefenseSystem] = {
            "laya": CityDefenseSystem(
                city_name="Jakarta (Pusat Pemerintahan)",
                model_id="laya",
                model_label="Laya Multilingual (421M GPU)",
                typical_latency_ms=55.0,
                token_cost=0
            ),
            "openjev": CityDefenseSystem(
                city_name="Surabaya (Pangkalan AL)",
                model_id="openjev",
                model_label="OpenJev (0.5B GPU Local Logits)",
                typical_latency_ms=210.0,
                token_cost=0
            ),
            "jev": CityDefenseSystem(
                city_name="Bandung (Pusat Komando JEV Cloud)",
                model_id="jev",
                model_label="TypeSafe JEV (Cloud Decision API)",
                typical_latency_ms=160.0,
                token_cost=0
            ),
            "kev": CityDefenseSystem(
                city_name="Medan (Radar Terdepan)",
                model_id="kev",
                model_label="Kev-0.8B (Local Ensemble)",
                typical_latency_ms=950.0,
                token_cost=0
            ),
            "sahabatai": CityDefenseSystem(
                city_name="Nusantara / IKN (Ibu Kota)",
                model_id="sahabatai",
                model_label="Heavyweight LLM (Sahabat-AI 8B Instruct)",
                typical_latency_ms=2500.0,
                token_cost=36
            )
        }

        # Synchronized sky object stream shared by all 5 cities
        self.sky_objects: List[SkyObject] = []
        self.sitrep_history: List[Dict[str, Any]] = []

    def set_llm_model(self, llm_key: str) -> Dict[str, Any]:
        """Swaps City 5 LLM model dynamically."""
        if llm_key not in self.LLM_MODELS_CATALOG:
            return {"status": "error", "message": f"Model '{llm_key}' not found"}

        spec = self.LLM_MODELS_CATALOG[llm_key]
        self.current_llm_key = llm_key
        city5 = self.cities.get("sahabatai")
        if city5:
            city5.model_label = spec["label"]
            city5.typical_latency_ms = spec["latency_ms"]
            city5.token_cost = spec["token_cost"]

        return {
            "status": "success",
            "model_key": llm_key,
            "label": spec["label"],
            "latency_ms": spec["latency_ms"],
            "token_cost": spec["token_cost"]
        }

    def reset(self):
        self.wave = 1
        self.wave_tick = 0
        self.obj_counter = 0
        self.sky_objects.clear()
        self.sitrep_history.clear()
        for city in self.cities.values():
            city.reset()

    def spawn_object(self) -> SkyObject:
        self.obj_counter += 1
        
        # Difficulty multipliers based on wave
        speed_mult = 1.0 + (self.wave - 1) * 0.22
        hostile_prob = min(0.92, 0.60 + (self.wave - 1) * 0.08)

        is_hostile = random.random() < hostile_prob

        x = random.uniform(30.0, 230.0)
        y = random.uniform(-30.0, 5.0) # Spawns above canvas
        
        target_x = 130.0 + random.uniform(-80.0, 80.0)
        target_y = 380.0

        if is_hostile:
            threat_type = random.choices(
                ["MISSILE", "METEOR_IMPACT", "DRONE"],
                weights=[0.55, 0.30, 0.15]
            )[0]
            
            base_speed = random.uniform(2.6, 4.2) * speed_mult
            angle = math.atan2(target_y - y, target_x - x)
            vx = base_speed * math.cos(angle)
            vy = base_speed * math.sin(angle)
            
            mach = round(base_speed * 1.5, 1)
            altitude = random.uniform(8.0, 25.0)

            return SkyObject(
                obj_id=self.obj_counter,
                obj_type=threat_type,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                speed=mach,
                iff_code=None,
                altitude_km=altitude
            )
        else:
            civ_type = random.choices(
                ["AIRLINER", "BIRD_FLOCK", "METEOR_MISS"],
                weights=[0.60, 0.25, 0.15]
            )[0]

            if civ_type == "AIRLINER":
                vx = random.choice([-1.2, 1.2]) * random.uniform(0.8, 1.4)
                vy = random.uniform(0.1, 0.25)
                x = 10.0 if vx > 0 else 250.0
                y = random.uniform(60.0, 140.0)
                mach = 0.85
                callsigns = ["GARUDA-GA402", "LION-JT610", "CITILINK-QG801", "BATIK-ID652", "SQUAWK-7700"]
                iff = random.choice(callsigns)
                altitude = 10.5
            elif civ_type == "BIRD_FLOCK":
                vx = random.uniform(-0.5, 0.5)
                vy = random.uniform(0.2, 0.4)
                mach = 0.1
                iff = "BIO-RCS-LOW"
                altitude = 1.2
            else: # METEOR_MISS
                vx = random.uniform(3.0, 5.0) * random.choice([-1, 1])
                vy = random.uniform(0.5, 1.2)
                mach = 4.5
                iff = "TRAJECTORY-ORBITAL-MISS"
                altitude = 45.0

            return SkyObject(
                obj_id=self.obj_counter,
                obj_type=civ_type,
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                speed=mach,
                iff_code=iff,
                altitude_km=altitude
            )

    def step(self) -> Dict[str, Any]:
        self.wave_tick += 1

        # Check wave escalation
        wave_advanced = False
        if self.wave_tick >= self.ticks_per_wave:
            self.wave += 1
            self.wave_tick = 0
            wave_advanced = True
            sitrep = self.generate_sitrep()
            self.sitrep_history.append(sitrep)

        # Spawn frequency escalates with wave:
        spawn_rate = max(4, int(18 - (self.wave * 2.5)))
        if self.wave_tick % spawn_rate == 0:
            new_obj = self.spawn_object()
            self.sky_objects.append(new_obj)

        # Physical update of sky objects
        for obj in self.sky_objects:
            if obj.escaped:
                continue

            obj.x += obj.vx
            obj.y += obj.vy
            obj.altitude_km = max(0.0, obj.altitude_km - (obj.vy * 0.05))

            if obj.x < -50 or obj.x > 310 or obj.y > 420:
                obj.escaped = True

        # Process each city's defense AI independently
        for m_id, city in self.cities.items():
            city.tick_battery_load()

            # 1. Update active interceptors for this city
            for inc in city.interceptor_missiles:
                if not inc.get("active"):
                    continue

                if inc["delay_ticks"] > 0:
                    inc["delay_ticks"] -= 1
                    continue

                target = next((o for o in self.sky_objects if o.obj_id == inc["target_id"]), None)
                if not target or target.escaped or target.obj_id in city.destroyed_targets or target.obj_id in city.impacted_targets:
                    inc["active"] = False
                    continue

                dx = target.x - inc["x"]
                dy = target.y - inc["y"]
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < 15.0: # Interception Hit!
                    inc["active"] = False
                    if target.obj_id not in city.destroyed_targets:
                        city.destroyed_targets.add(target.obj_id)
                        if target.is_hostile:
                            city.intercepted_count += 1
                            # Check low altitude shrapnel damage
                            if target.y > 280.0 or target.altitude_km < 3.0:
                                city.city_hp = max(0.0, city.city_hp - CityDefenseSystem.DAMAGE_TABLE["SHRAPNEL"])
                                city.recent_events.append(f"⚠️ Serpihan meledak terlalu rendah: -{CityDefenseSystem.DAMAGE_TABLE['SHRAPNEL']}% HP")
                        else:
                            # Friendly Fire!
                            city.friendly_fire_count += 1
                            city.recent_events.append(f"❌ SALAH TEMBAK: Menghancurkan Pesawat Sipil #{target.obj_id}!")
                else:
                    inc["x"] += (dx / dist) * inc["speed"]
                    inc["y"] += (dy / dist) * inc["speed"]

            # 2. AI Threat Assessment & Engagement
            for obj in self.sky_objects:
                if obj.escaped or obj.obj_id in city.destroyed_targets or obj.obj_id in city.impacted_targets:
                    continue

                # Hostile engagement
                if obj.is_hostile:
                    if obj.obj_id not in city.assigned_targets:
                        if city.can_fire():
                            city.assigned_targets.add(obj.obj_id)
                            city.fire_interceptor(obj)
                    
                    # Check ground impact on this city
                    if obj.y >= 370.0:
                        city.impacted_targets.add(obj.obj_id)
                        dmg = CityDefenseSystem.DAMAGE_TABLE.get(obj.obj_type, 30.0)
                        city.city_hp = max(0.0, city.city_hp - dmg)
                        city.impact_missile_count += 1
                        city.recent_events.append(f"💥 HANTAMAN {obj.obj_type}! Kerusakan Kota: -{dmg:.0f}%")
                        if len(city.recent_events) > 6:
                            city.recent_events.pop(0)

                # Civilian protection check
                else:
                    if obj.y > 280.0 and obj.obj_id not in city.civilian_safe_ids:
                        city.civilian_safe_ids.add(obj.obj_id)
                        city.civilian_safe_count += 1

        # Clean up escaped objects
        self.sky_objects = [o for o in self.sky_objects if not o.escaped]

        return {
            "wave": self.wave,
            "wave_tick": self.wave_tick,
            "wave_advanced": wave_advanced,
            "objects_count": len(self.sky_objects),
            "current_llm": self.current_llm_key,
            "sky_objects": [o.to_dict() for o in self.sky_objects[:25]],
            "cities": {m: c.to_dict() for m, c in self.cities.items()}
        }

    def generate_sitrep(self) -> Dict[str, Any]:
        """Generates Situation Report (Rekap Berkala Pertahanan Udara)."""
        recap = {
            "timestamp": time.time(),
            "wave_completed": self.wave - 1,
            "current_llm": self.current_llm_key,
            "cities": []
        }
        for m, c in self.cities.items():
            d = c.to_dict()
            recap["cities"].append({
                "city": c.city_name,
                "model": c.model_label,
                "hp": d["city_hp"],
                "ammo": f"{d['ammo']}/{d['max_ammo']}",
                "intercepted": d["intercepted_count"],
                "failed_impacts": d["impact_missile_count"],
                "intercept_rate": d["intercept_rate_pct"],
                "civilian_safe": d["civilian_safe_count"],
                "friendly_fire": d["friendly_fire_count"],
                "tokens": d["tokens_burned"],
                "status": d["status"]
            })
        return recap
