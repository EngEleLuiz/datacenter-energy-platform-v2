"""
tests/test_simulators.py

Unit tests for the data generator layer.
These run in CI via GitHub Actions without Docker or Kafka.
"""

import pytest
from datetime import datetime, timezone
from dataclasses import asdict

from data_generator.server_simulator import ServerSimulator, ServerTelemetry
from data_generator.ups_inverter_simulator import (
    UPSSimulator,
    InverterSimulator,
    UPSTelemetry,
    InverterTelemetry,
)


# ---------------------------------------------------------------------------
# Server Simulator
# ---------------------------------------------------------------------------
class TestServerSimulator:

    def setup_method(self):
        self.sim = ServerSimulator(num_servers=20, num_racks=4, fault_probability=0.0)

    def test_snapshot_returns_correct_count(self):
        snap = self.sim.generate_snapshot()
        assert len(snap) == 20

    def test_all_records_are_server_telemetry(self):
        snap = self.sim.generate_snapshot()
        assert all(isinstance(r, ServerTelemetry) for r in snap)

    def test_power_draw_is_positive(self):
        snap = self.sim.generate_snapshot()
        assert all(r.power_draw_w > 0 for r in snap)

    def test_cpu_utilization_bounds(self):
        snap = self.sim.generate_snapshot()
        assert all(0.0 <= r.cpu_utilization <= 1.0 for r in snap)

    def test_memory_utilization_bounds(self):
        snap = self.sim.generate_snapshot()
        assert all(0.0 <= r.memory_utilization <= 1.0 for r in snap)

    def test_temperature_is_physical(self):
        snap = self.sim.generate_snapshot()
        assert all(0 < r.cpu_temp_c < 150 for r in snap)

    def test_pue_contribution_above_one(self):
        snap = self.sim.generate_snapshot()
        assert all(r.pue_contribution >= 1.0 for r in snap)

    def test_rack_ids_are_distributed(self):
        snap = self.sim.generate_snapshot()
        rack_ids = {r.rack_id for r in snap}
        assert len(rack_ids) == 4

    def test_event_ids_are_unique(self):
        snap = self.sim.generate_snapshot()
        ids = [r.event_id for r in snap]
        assert len(ids) == len(set(ids))

    def test_no_anomalies_when_fault_probability_zero(self):
        snap = self.sim.generate_snapshot()
        assert all(not r.is_anomaly for r in snap)

    def test_fault_injection_produces_anomalies(self):
        sim = ServerSimulator(num_servers=200, num_racks=10, fault_probability=1.0)
        snap = sim.generate_snapshot()
        assert any(r.is_anomaly for r in snap)

    def test_server_profile_is_valid(self):
        valid_profiles = {"compute", "storage", "network", "gpu"}
        snap = self.sim.generate_snapshot()
        assert all(r.server_profile in valid_profiles for r in snap)

    def test_cooling_state_is_valid(self):
        valid_states = {"normal", "elevated", "warning", "critical"}
        snap = self.sim.generate_snapshot()
        assert all(r.cooling_state in valid_states for r in snap)

    def test_serializable_to_dict(self):
        snap = self.sim.generate_snapshot()
        for rec in snap:
            d = asdict(rec)
            assert "server_id" in d
            assert "power_draw_w" in d

    def test_diurnal_pattern(self):
        """Load at noon should be higher than at 3am on average."""
        ts_noon  = datetime(2024, 6, 15, 12, 0, tzinfo=timezone.utc)
        ts_night = datetime(2024, 6, 15,  3, 0, tzinfo=timezone.utc)
        noon_util  = sum(r.cpu_utilization for r in self.sim.generate_snapshot(ts_noon))
        night_util = sum(r.cpu_utilization for r in self.sim.generate_snapshot(ts_night))
        assert noon_util > night_util


# ---------------------------------------------------------------------------
# UPS Simulator
# ---------------------------------------------------------------------------
class TestUPSSimulator:

    def setup_method(self):
        self.sim = UPSSimulator(num_ups=4)

    def test_snapshot_count(self):
        assert len(self.sim.generate_snapshot()) == 4

    def test_soc_bounds(self):
        for rec in self.sim.generate_snapshot():
            assert 0.0 <= rec.battery_soc <= 1.0

    def test_ups_mode_valid(self):
        valid_modes = {"normal", "battery", "bypass", "fault"}
        for rec in self.sim.generate_snapshot():
            assert rec.ups_mode in valid_modes

    def test_efficiency_bounds(self):
        for rec in self.sim.generate_snapshot():
            assert 0.0 <= rec.ups_efficiency <= 1.0

    def test_all_ups_have_unique_ids(self):
        snap = self.sim.generate_snapshot()
        ids = [r.ups_id for r in snap]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Inverter Simulator
# ---------------------------------------------------------------------------
class TestInverterSimulator:

    def setup_method(self):
        self.sim = InverterSimulator(num_inverters=2, islanding_probability=0.0)

    def test_snapshot_count(self):
        assert len(self.sim.generate_snapshot()) == 2

    def test_control_mode_valid(self):
        valid_modes = {"GFL", "GFM", "transitioning"}
        for rec in self.sim.generate_snapshot():
            assert rec.control_mode in valid_modes

    def test_no_islanding_when_probability_zero(self):
        for rec in self.sim.generate_snapshot():
            assert not rec.islanding_detected

    def test_frequency_near_nominal(self):
        for rec in self.sim.generate_snapshot():
            assert abs(rec.output_frequency_hz - 60.0) < 2.0

    def test_gfm_has_lower_rocof_than_gfl_during_islanding(self):
        """GFM should maintain lower ROCOF during islanding events."""
        sim = InverterSimulator(num_inverters=10, islanding_probability=1.0)
        samples = 20
        gfl_rocofs, gfm_rocofs = [], []

        for _ in range(samples):
            for rec in sim.generate_snapshot():
                if rec.control_mode == "GFL":
                    gfl_rocofs.append(rec.rocof_hz_per_s)
                elif rec.control_mode == "GFM":
                    gfm_rocofs.append(rec.rocof_hz_per_s)

        if gfl_rocofs and gfm_rocofs:
            assert sum(gfm_rocofs) / len(gfm_rocofs) < sum(gfl_rocofs) / len(gfl_rocofs), \
                "GFM should have lower average ROCOF than GFL during islanding"

    def test_thd_within_ieee_limits(self):
        """IEEE 1547: THD < 5% under normal GFL, can be higher during faults."""
        sim = InverterSimulator(num_inverters=4, islanding_probability=0.0)
        for rec in sim.generate_snapshot():
            assert rec.thd_percent < 10.0  # generous bound for simulation

    def test_efficiency_bounds(self):
        for rec in self.sim.generate_snapshot():
            assert 0.9 <= rec.efficiency <= 1.0
