"""
data_generator/server_simulator.py

Simulates real-time telemetry from N servers in a data center.
Each server belongs to a rack and emits: CPU load, memory, temperatures,
power draw, and derived metrics like PUE contribution.

Designed to feed the Kafka producer (ingestion layer).
"""

import uuid
import random
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Generator
from loguru import logger


# ---------------------------------------------------------------------------
# Server Profiles — workload archetypes found in real data centers
# ---------------------------------------------------------------------------
SERVER_PROFILES = {
    "compute":  {"base_power_w": 300, "cpu_tdp": 250, "ram_gb": 128},
    "storage":  {"base_power_w": 120, "cpu_tdp":  80, "ram_gb":  64},
    "network":  {"base_power_w":  80, "cpu_tdp":  45, "ram_gb":  32},
    "gpu":      {"base_power_w": 600, "cpu_tdp": 300, "ram_gb": 256},
}

COOLING_STATES = ["normal", "elevated", "warning", "critical"]


@dataclass
class ServerTelemetry:
    """Single telemetry record emitted by one server."""
    event_id:           str
    server_id:          str
    rack_id:            str
    datacenter_zone:    str
    server_profile:     str
    timestamp_utc:      str

    # Power metrics (Watts)
    power_draw_w:       float
    cpu_power_w:        float
    ram_power_w:        float
    idle_power_w:       float

    # Thermal metrics (°C)
    cpu_temp_c:         float
    inlet_temp_c:       float
    outlet_temp_c:      float

    # Utilization (0–1)
    cpu_utilization:    float
    memory_utilization: float
    disk_io_utilization:float
    network_utilization:float

    # Derived
    pue_contribution:   float   # partial PUE of this server
    cooling_state:      str
    fan_speed_rpm:      int
    is_anomaly:         bool    # injected fault flag (for ML labeling)


class ServerSimulator:
    """
    Simulates a fleet of servers with realistic workload patterns:
      - Diurnal cycles (business hours load spike)
      - Rack-level thermal coupling (hot servers heat neighbors)
      - Stochastic faults (thermal runaway, power spike, zombie processes)
    """

    def __init__(
        self,
        num_servers: int = 100,
        num_racks: int = 10,
        datacenter_zone: str = "ZONE-A",
        fault_probability: float = 0.01,
        random_seed: int = 42,
    ):
        self.num_servers = num_servers
        self.num_racks = num_racks
        self.datacenter_zone = datacenter_zone
        self.fault_probability = fault_probability
        random.seed(random_seed)

        self.servers = self._initialize_servers()
        logger.info(
            f"ServerSimulator initialized: {num_servers} servers across "
            f"{num_racks} racks in {datacenter_zone}"
        )

    def _initialize_servers(self) -> list[dict]:
        """Assign each server a fixed rack, profile, and persistent state."""
        profiles = list(SERVER_PROFILES.keys())
        servers = []
        for i in range(self.num_servers):
            profile_name = profiles[i % len(profiles)]
            profile = SERVER_PROFILES[profile_name]
            servers.append({
                "server_id":  f"SRV-{str(uuid.uuid4())[:8].upper()}",
                "rack_id":    f"RACK-{(i % self.num_racks) + 1:02d}",
                "profile":    profile_name,
                "base_power": profile["base_power_w"],
                "cpu_tdp":    profile["cpu_tdp"],
                "ram_gb":     profile["ram_gb"],
                # Persistent noise seeds for correlated behaviour
                "_phase_offset": random.uniform(0, 2 * math.pi),
                "_noise_scale":  random.uniform(0.05, 0.15),
            })
        return servers

    # ------------------------------------------------------------------
    # Workload patterns
    # ------------------------------------------------------------------
    def _diurnal_load(self, hour: float, phase_offset: float) -> float:
        """
        Returns a [0,1] load factor representing business-hour patterns.
        Peak at ~14:00, trough at ~04:00.
        """
        base = 0.5 + 0.35 * math.sin(
            (2 * math.pi / 24) * (hour - 6) + phase_offset
        )
        return max(0.05, min(0.98, base))

    def _rack_thermal_coupling(self, rack_id: str, servers: list[dict]) -> float:
        """Rack-level ambient temperature penalty from neighboring servers."""
        rack_servers = [s for s in servers if s.get("_last_power", 0) > 0
                        and s["rack_id"] == rack_id]
        if not rack_servers:
            return 0.0
        avg_load = sum(s.get("_last_cpu_util", 0.5) for s in rack_servers) / len(rack_servers)
        return avg_load * 4.0  # up to +4°C from rack neighbors

    # ------------------------------------------------------------------
    # Fault injection
    # ------------------------------------------------------------------
    def _inject_fault(self, record: dict) -> dict:
        """Randomly inject one of several failure modes."""
        fault = random.choice(["thermal_runaway", "power_spike", "cpu_zombie"])
        if fault == "thermal_runaway":
            record["cpu_temp_c"] += random.uniform(20, 45)
            record["outlet_temp_c"] += random.uniform(15, 30)
            record["cooling_state"] = "critical"
            record["fan_speed_rpm"] = 6000
        elif fault == "power_spike":
            record["power_draw_w"] *= random.uniform(1.4, 2.1)
            record["cpu_power_w"] *= 1.8
        elif fault == "cpu_zombie":
            record["cpu_utilization"] = random.uniform(0.92, 1.0)
            record["cpu_temp_c"] += random.uniform(10, 25)
        record["is_anomaly"] = True
        return record

    # ------------------------------------------------------------------
    # Telemetry generation
    # ------------------------------------------------------------------
    def generate_snapshot(self, timestamp: datetime | None = None) -> list[ServerTelemetry]:
        """
        Generate one telemetry record per server for the given timestamp.
        Returns a list ready to be serialized and sent to Kafka.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        hour = timestamp.hour + timestamp.minute / 60.0
        records = []

        for srv in self.servers:
            load = self._diurnal_load(hour, srv["_phase_offset"])
            noise = random.gauss(0, srv["_noise_scale"])
            cpu_util = max(0.02, min(0.99, load + noise))
            mem_util = max(0.10, min(0.99, 0.4 + cpu_util * 0.45 + random.gauss(0, 0.05)))

            # Power model: P = idle + (TDP * utilization)
            idle_power = srv["base_power"] * 0.30
            cpu_power  = srv["cpu_tdp"] * cpu_util
            ram_power  = srv["ram_gb"] * 0.375  # ~0.375W per GB
            total_power = idle_power + cpu_power + ram_power

            # Thermal model
            rack_penalty = self._rack_thermal_coupling(srv["rack_id"], self.servers)
            inlet_temp   = 22.0 + rack_penalty * 0.5 + random.gauss(0, 0.5)
            cpu_temp     = inlet_temp + (cpu_util * 45) + random.gauss(0, 2)
            outlet_temp  = inlet_temp + (total_power / 100) + random.gauss(0, 1)

            # Cooling state
            if cpu_temp < 60:
                cooling_state = "normal"
                fan_rpm = int(1200 + cpu_util * 1800)
            elif cpu_temp < 75:
                cooling_state = "elevated"
                fan_rpm = int(2500 + cpu_util * 1500)
            elif cpu_temp < 85:
                cooling_state = "warning"
                fan_rpm = int(3800 + cpu_util * 1200)
            else:
                cooling_state = "critical"
                fan_rpm = 6000

            # PUE contribution (server power / IT load ratio approximation)
            # Clamp to >= 1.0 — physically impossible for PUE < 1
            pue_contrib = max(1.0, 1.0 + (0.2 * (outlet_temp - inlet_temp) / max(total_power, 1)))

            rec = {
                "event_id":           str(uuid.uuid4()),
                "server_id":          srv["server_id"],
                "rack_id":            srv["rack_id"],
                "datacenter_zone":    self.datacenter_zone,
                "server_profile":     srv["profile"],
                "timestamp_utc":      timestamp.isoformat(),
                "power_draw_w":       round(total_power, 2),
                "cpu_power_w":        round(cpu_power, 2),
                "ram_power_w":        round(ram_power, 2),
                "idle_power_w":       round(idle_power, 2),
                "cpu_temp_c":         round(cpu_temp, 2),
                "inlet_temp_c":       round(inlet_temp, 2),
                "outlet_temp_c":      round(outlet_temp, 2),
                "cpu_utilization":    round(cpu_util, 4),
                "memory_utilization": round(mem_util, 4),
                "disk_io_utilization":round(random.uniform(0.01, cpu_util * 0.6), 4),
                "network_utilization":round(random.uniform(0.01, cpu_util * 0.4), 4),
                "pue_contribution":   round(pue_contrib, 4),
                "cooling_state":      cooling_state,
                "fan_speed_rpm":      fan_rpm,
                "is_anomaly":         False,
            }

            # Persist for rack thermal coupling next tick
            srv["_last_power"]    = total_power
            srv["_last_cpu_util"] = cpu_util

            # Fault injection
            if random.random() < self.fault_probability:
                rec = self._inject_fault(rec)

            records.append(ServerTelemetry(**rec))

        return records

    def stream(
        self,
        interval_seconds: float = 5.0,
        max_records: int | None = None,
    ) -> Generator[list[ServerTelemetry], None, None]:
        """
        Infinite (or bounded) generator — yields one snapshot per interval.
        Use in the Kafka producer main loop.
        """
        count = 0
        while True:
            snapshot = self.generate_snapshot()
            yield snapshot
            count += 1
            if max_records and count >= max_records:
                break
            time.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    sim = ServerSimulator(num_servers=10, num_racks=2, fault_probability=0.05)
    snapshot = sim.generate_snapshot()

    print(f"\n=== Snapshot: {len(snapshot)} servers ===")
    for rec in snapshot[:3]:
        print(json.dumps(asdict(rec), indent=2))

    anomalies = [r for r in snapshot if r.is_anomaly]
    print(f"\nAnomalies injected: {len(anomalies)}")
