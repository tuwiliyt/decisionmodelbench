#!/usr/bin/env python3
"""
Air Defense Tactical Simulation Engine (Iron Dome / C-RAM AI Simulator)
Features:
1. Multi-Threat Sky Object Generator:
   - Hostile Targets: Hypersonic Missiles (🚀), Impact Meteorites (☄️), Kamikaze Drones (🛸).
   - Non-Threats / Friendly: Commercial Airliners (✈️, active IFF), Flocks of Birds (🦅), Atmosphere Miss Meteorites (🌠).
2. Progressive Wave Escalation (Difficulty increases over time):
   - Wave 1 to 5+: Speed multiplier, saturation density, reduced time-to-impact.
3. 5 Parallel Cities Defended by 5 AI Models:
   - Jakarta: Laya Multilingual (421M GPU) - ~55 ms reflex
   - Surabaya: OpenJev (0.5B GPU) - ~210 ms reflex
   - Bandung: TypeSafe Jev (Cloud SaaS) - ~160 ms reflex
   - Medan: Kev-0.8B (Local Ensemble) - ~950 ms reflex
   - Nusantara (IKN): Heavyweight LLM (Sahabat-AI 8B) - ~2500 ms Decision Lag
4. Periodic SITREP (Situation Report):
   - Periodic recaps of intercepted threats vs city impacts vs civilian casualties vs token costs.
"""

import time
import math
import random
from typing import Dict, Any, List, Optional

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
        self.destroyed = False
        self.escaped = False
        self.impacted = False
        self.assigned_target = False

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
            "destroyed": self.destroyed,
            "impacted": self.impacted
        }

class CityDefenseSystem:
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

        self.city_hp = 100.0 # 0 - 100%
        self.intercepted_count = 0
        self.impact_missile_count = 0
        self.civilian_safe_count = 0
        self.friendly_fire_count = 0
        self.tokens_burned = 0
        self.interceptor_missiles: List[Dict[str, Any]] = []
        self.active_threats: List[SkyObject] = []
        self.recent_events: List[str] = []

    def reset(self):
        self.city_hp = 100.0
        self.intercepted_count = 0
        self.impact_missile_count = 0
        self.civilian_safe_count = 0
        self.friendly_fire_count = 0
        self.tokens_burned = 0
        self.interceptor_missiles.clear()
        self.active_threats.clear()
        self.recent_events.clear()

    def fire_interceptor(self, target: SkyObject, latency_penalty_ms: Optional[float] = None) -> Dict[str, Any]:
        lat = latency_penalty_ms if latency_penalty_ms is not None else self.typical_latency_ms
        self.tokens_burned += self.token_cost

        # Interceptor starts at city base (x=130, y=370)
        interceptor = {
            "x": 130.0,
            "y": 370.0,
            "target_id": target.obj_id,
            "target_type": target.obj_type,
            "target_x": target.x,
            "target_y": target.y,
            "speed": 8.0, # Rapid ascent
            "delay_ticks": int(lat / 50.0), # Latency delay before launch!
            "active": True
        }
        self.interceptor_missiles.append(interceptor)

        event_msg = f"🚀 {self.model_label}: Menembakkan rudal pencegat ke Sasaran #{target.obj_id} ({target.obj_type})"
        self.recent_events.append(event_msg)
        if len(self.recent_events) > 5:
            self.recent_events.pop(0)

        return interceptor

    def to_dict(self) -> Dict[str, Any]:
        total_hostiles = self.intercepted_count + self.impact_missile_count
        intercept_rate = (self.intercepted_count / total_hostiles * 100.0) if total_hostiles > 0 else 100.0
        civ_total = self.civilian_safe_count + self.friendly_fire_count
        civ_safety_rate = (self.civilian_safe_count / civ_total * 100.0) if civ_total > 0 else 100.0

        return {
            "city_name": self.city_name,
            "model_id": self.model_id,
            "model_label": self.model_label,
            "typical_latency_ms": self.typical_latency_ms,
            "city_hp": round(self.city_hp, 1),
            "intercepted_count": self.intercepted_count,
            "impact_missile_count": self.impact_missile_count,
            "intercept_rate_pct": round(intercept_rate, 1),
            "civilian_safe_count": self.civilian_safe_count,
            "friendly_fire_count": self.friendly_fire_count,
            "civ_safety_rate_pct": round(civ_safety_rate, 1),
            "tokens_burned": self.tokens_burned,
            "active_interceptors": len([i for i in self.interceptor_missiles if i.get("active")]),
            "status": "OPERASIONAL" if self.city_hp > 50 else ("KRITIS" if self.city_hp > 0 else "HANCUR")
        }

class AirDefenseArena:
    def __init__(self):
        self.wave = 1
        self.wave_tick = 0
        self.ticks_per_wave = 120 # ~30-40 seconds per wave
        self.obj_counter = 0

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
                model_label="OpenJev (0.5B GPU Logit)",
                typical_latency_ms=210.0,
                token_cost=0
            ),
            "jev": CityDefenseSystem(
                city_name="Bandung (Pusat Komando C2)",
                model_id="jev",
                model_label="TypeSafe Jev (Cloud SaaS)",
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
                model_label="Heavyweight LLM (Sahabat-AI 8B)",
                typical_latency_ms=2500.0,
                token_cost=36 # ~36 tokens per decision
            )
        }

        # Synchronized sky object stream shared by all 5 cities
        self.sky_objects: List[SkyObject] = []
        self.sitrep_history: List[Dict[str, Any]] = []

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
        # Wave 1: 65% hostile, 35% civilian, speed 1.0x
        # Wave 3: 80% hostile, 20% civilian, speed 1.4x
        # Wave 5: 90% hostile, extreme hypersonic speed 1.8x
        speed_mult = 1.0 + (self.wave - 1) * 0.20
        hostile_prob = min(0.90, 0.60 + (self.wave - 1) * 0.08)

        is_hostile = random.random() < hostile_prob

        x = random.uniform(30.0, 230.0)
        y = random.uniform(-30.0, 10.0) # Spawns above canvas
        
        # Target coordinate: city base is around x=130, y=380
        target_x = 130.0 + random.uniform(-80.0, 80.0)
        target_y = 380.0

        if is_hostile:
            # Threat choices
            threat_type = random.choices(
                ["MISSILE", "METEOR_IMPACT", "DRONE"],
                weights=[0.55, 0.30, 0.15]
            )[0]
            
            base_speed = random.uniform(2.5, 4.2) * speed_mult
            # Direction vector to target
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
                iff_code=None, # No IFF transponder for hostile
                altitude_km=altitude
            )
        else:
            # Civilian / non-threat choices
            civ_type = random.choices(
                ["AIRLINER", "BIRD_FLOCK", "METEOR_MISS"],
                weights=[0.60, 0.25, 0.15]
            )[0]

            if civ_type == "AIRLINER":
                # Commercial airliner: horizontal cruising path, doesn't dive to ground
                vx = random.choice([-1.2, 1.2]) * random.uniform(0.8, 1.4)
                vy = random.uniform(0.1, 0.3) # Very slow altitude descent/level
                x = 10.0 if vx > 0 else 250.0
                y = random.uniform(60.0, 140.0)
                mach = 0.85
                callsigns = ["GARUDA-GA402", "LION-JT610", "CITILINK-QG801", "BATIK-ID652", "SQUAWK-7700"]
                iff = random.choice(callsigns)
                altitude = 10.5
            elif civ_type == "BIRD_FLOCK":
                vx = random.uniform(-0.5, 0.5)
                vy = random.uniform(0.2, 0.5)
                mach = 0.1
                iff = "BIO-RCS-LOW"
                altitude = 1.2
            else: # METEOR_MISS
                # Passing meteorite tangent to atmosphere
                vx = random.uniform(3.0, 5.0) * random.choice([-1, 1])
                vy = random.uniform(0.5, 1.2) # Skims horizontally, won't hit ground
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
            # Generate Periodic SITREP
            sitrep = self.generate_sitrep()
            self.sitrep_history.append(sitrep)

        # Spawn frequency escalates with wave:
        # Wave 1: 1 object every ~16 ticks
        # Wave 3: 1 object every ~9 ticks
        # Wave 5: 1 object every ~5 ticks (Saturation bombardment)
        spawn_rate = max(4, int(18 - (self.wave * 2.5)))
        if self.wave_tick % spawn_rate == 0:
            new_obj = self.spawn_object()
            self.sky_objects.append(new_obj)

        # Update physical positions of sky objects
        events = []
        for obj in self.sky_objects:
            if obj.destroyed or obj.impacted or obj.escaped:
                continue

            obj.x += obj.vx
            obj.y += obj.vy
            obj.altitude_km = max(0.0, obj.altitude_km - (obj.vy * 0.05))

            # Ground impact check
            if obj.y >= 370.0:
                if obj.is_hostile:
                    obj.impacted = True
                    events.append({
                        "type": "IMPACT",
                        "obj_id": obj.obj_id,
                        "obj_type": obj.obj_type,
                        "x": obj.x
                    })
                else:
                    obj.escaped = True

            # Canvas boundaries escape
            if obj.x < -40 or obj.x > 300 or obj.y > 410:
                obj.escaped = True

        # Process each city's defense AI
        for m_id, city in self.cities.items():
            # Update active interceptor missiles
            for inc in city.interceptor_missiles:
                if not inc.get("active"):
                    continue

                # Latency delay countdown before missile launches!
                if inc["delay_ticks"] > 0:
                    inc["delay_ticks"] -= 1
                    continue

                # Find target
                target = next((o for o in self.sky_objects if o.obj_id == inc["target_id"]), None)
                if not target or target.destroyed or target.impacted:
                    inc["active"] = False
                    continue

                # Interceptor tracks target
                dx = target.x - inc["x"]
                dy = target.y - inc["y"]
                dist = math.sqrt(dx*dx + dy*dy)

                if dist < 14.0: # Hit & destroy!
                    inc["active"] = False
                    if not target.destroyed:
                        target.destroyed = True
                        if target.is_hostile:
                            city.intercepted_count += 1
                        else:
                            # Friendly Fire! Shot down civilian airliner or innocent bird!
                            city.friendly_fire_count += 1
                else:
                    inc["x"] += (dx / dist) * inc["speed"]
                    inc["y"] += (dy / dist) * inc["speed"]

            # AI Detection & Auto-Engagement
            for obj in self.sky_objects:
                if obj.destroyed or obj.impacted or obj.escaped:
                    continue

                # Algorithmic decision logic:
                # Decision models (Laya, OpenJev, Jev) evaluate accurately in sub-100ms.
                # LLM (Sahabat-AI) suffers severe latency delay:
                # By the time interceptor launches, object might already impact ground!
                if obj.is_hostile and not obj.assigned_target:
                    # Decide action
                    city.fire_interceptor(obj)
                    obj.assigned_target = True
                elif not obj.is_hostile and obj.y > 300.0:
                    # Successfully protected civilian flight
                    city.civilian_safe_count += 1

            # Deduct HP on city if hostile impacts
            for ev in events:
                if ev["type"] == "IMPACT":
                    # If this city didn't intercept it, take structural damage!
                    city.city_hp = max(0.0, city.city_hp - 15.0)
                    city.impact_missile_count += 1

        # Clean up expired objects
        self.sky_objects = [o for o in self.sky_objects if not (o.destroyed and o.y > 400)]

        return {
            "wave": self.wave,
            "wave_tick": self.wave_tick,
            "wave_advanced": wave_advanced,
            "objects_count": len(self.sky_objects),
            "sky_objects": [o.to_dict() for o in self.sky_objects[:20]],
            "cities": {m: c.to_dict() for m, c in self.cities.items()}
        }

    def generate_sitrep(self) -> Dict[str, Any]:
        """Generates Situation Report (Rekap Berkala Pertahanan Udara)."""
        recap = {
            "timestamp": time.time(),
            "wave_completed": self.wave - 1,
            "cities": []
        }
        for m, c in self.cities.items():
            d = c.to_dict()
            recap["cities"].append({
                "city": c.city_name,
                "model": c.model_label,
                "hp": d["city_hp"],
                "intercepted": d["intercepted_count"],
                "failed_impacts": d["impact_missile_count"],
                "intercept_rate": d["intercept_rate_pct"],
                "civilian_safe": d["civilian_safe_count"],
                "friendly_fire": d["friendly_fire_count"],
                "tokens": d["tokens_burned"],
                "status": d["status"]
            })
        return recap
