"""
ups_inverter_simulator.py
=========================
Enhanced simulation of UPS units and Grid-Following / Grid-Forming inverters
for datacenter microgrid research.

Academic extensions implemented:
  1. Virtual Synchronous Machine (VSM) — virtual inertia & frequency support
  2. Black-Start / Islanded microgrid mode — GFM as grid-forming anchor
  3. Active Harmonic Compensation — GFM as active power filter
  4. Droop Control (P/f and Q/V) — autonomous power sharing among parallel GFMs
  5. Weak-Grid Stability Map — impedance-based small-signal stability criterion

References:
  - Driesen & Visscher (2008): "Virtual synchronous generators"
  - Zhong & Weiss (2011): "Synchronverters: Inverters that mimic synchronous generators"
  - Pogaku et al. (2007): "Modeling, analysis and testing of autonomous operation of
    an inverter-based microgrid"
  - IEEE Std 1547-2018: Standard for Interconnection of Distributed Energy Resources
  - IEEE Std 519-2022: Harmonic Control in Electric Power Systems
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Physical constants and IEEE limits
# ---------------------------------------------------------------------------
NOMINAL_FREQ_HZ: float = 60.0          # North American grid
NOMINAL_VOLTAGE_V: float = 480.0       # Low-voltage datacenter bus
IEEE1547_ROCOF_LIMIT: float = 0.5      # Hz/s — Rate of Change of Frequency limit
IEEE519_THD_LIMIT_PCT: float = 5.0     # % — Total Harmonic Distortion limit
DC_BUS_NOMINAL_V: float = 750.0        # DC-link nominal voltage
TWO_PI: float = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class UPSTelemetry:
    """Telemetry record for a single UPS unit."""
    event_id:              str
    ups_id:                str
    datacenter_zone:       str
    timestamp_utc:         str

    # Grid interface
    grid_voltage_v:        float
    grid_frequency_hz:     float
    grid_power_factor:     float

    # UPS state
    ups_mode:              str    # "normal" | "bypass" | "battery" | "fault"
    input_power_kw:        float
    output_power_kw:       float
    ups_efficiency:        float  # 0-1

    # Battery
    battery_soc:           float  # 0-1
    battery_voltage_v:     float
    battery_current_a:     float
    battery_temp_c:        float
    estimated_runtime_min: float

    # Alarms
    overload_alarm:        bool
    thermal_alarm:         bool
    battery_alarm:         bool

@dataclass
class UPSReading:
    """Snapshot of a single UPS unit at one point in time.

    Attributes:
        ups_id: Unique identifier.
        timestamp_utc: ISO-8601 UTC timestamp.
        battery_soc: State of charge [0, 1].
        battery_voltage_v: Terminal voltage of the battery bank.
        battery_current_a: Charging (+) or discharging (−) current.
        output_power_kw: Active power delivered to the DC bus.
        ups_mode: Operating mode — 'normal' | 'battery' | 'virtual_inertia' | 'fault'.
        grid_voltage_v: Upstream grid voltage seen at PCC.
        grid_frequency_hz: Grid frequency at PCC.
        virtual_inertia_active: True when VSM inertia emulation is injecting power.
        inertia_power_kw: Power injected/absorbed for frequency support (VSM).
        dc_bus_voltage_v: Voltage of the shared DC link inside the UPS.
        efficiency: Power conversion efficiency [0, 1].
        temperature_c: Internal temperature.
        zone: Datacenter zone identifier.
    """
    ups_id: str
    timestamp_utc: str
    battery_soc: float
    battery_voltage_v: float
    battery_current_a: float
    output_power_kw: float
    ups_mode: str
    grid_voltage_v: float
    grid_frequency_hz: float
    virtual_inertia_active: bool
    inertia_power_kw: float
    dc_bus_voltage_v: float
    efficiency: float
    temperature_c: float
    zone: str


@dataclass
class InverterReading:
    """Snapshot of a GFL or GFM inverter at one point in time.

    Attributes:
        inverter_id: Unique identifier.
        timestamp_utc: ISO-8601 UTC timestamp.
        control_mode: 'GFL' | 'GFM' | 'transitioning' | 'black_start'.
        output_active_power_kw: P output.
        output_reactive_power_kvar: Q output.
        output_voltage_v: RMS output voltage.
        output_frequency_hz: Output frequency.
        dc_link_voltage_v: DC-side voltage.
        thd_percent: Total Harmonic Distortion of output current.
        rocof_hz_per_s: Rate of Change of Frequency.
        freq_deviation_hz: Δf from nominal.
        voltage_deviation_pu: ΔV in per-unit (1.0 = nominal).
        islanding_detected: True when islanding protection has fired.
        junction_temp_c: IGBT junction temperature.
        efficiency: Inverter efficiency [0, 1].

        # --- VSM / Virtual Inertia fields ---
        virtual_inertia_H: Virtual inertia constant H [s].
        droop_kw_per_hz: Active power droop coefficient [kW/Hz].
        droop_kvar_per_v: Reactive power droop coefficient [kVAr/V].
        virtual_inertia_power_kw: Instantaneous inertia support power.
        frequency_nadir_hz: Minimum frequency reached during last event.
        rocof_at_nadir: ROCOF at frequency nadir.

        # --- Black-Start fields ---
        black_start_active: True when performing black-start sequence.
        black_start_stage: 0=idle, 1=pre-charge, 2=voltage_ramp, 3=load_pickup, 4=complete.
        black_start_voltage_pct: Bus voltage as % of nominal during ramp-up.
        loads_reconnected: Number of critical loads reconnected so far.

        # --- Harmonic compensation fields ---
        harmonic_compensation_active: True when acting as active power filter.
        thd_before_compensation_pct: THD before GFM harmonic injection.
        harmonic_injection_a: RMS current injected for compensation.
        dominant_harmonic_order: Most significant harmonic (e.g., 5, 7, 11).

        # --- Weak-grid stability fields ---
        scr: Short Circuit Ratio at PCC (grid stiffness).
        grid_impedance_pu: Thévenin impedance of the grid at PCC.
        pll_stability_margin_deg: Phase margin of the PLL (GFL only).
        impedance_stability_margin_db: |Zgrid/Zinv| stability margin.
        stability_flag: 'stable' | 'marginal' | 'unstable'.

        gfl_pll_locked: True when PLL is locked (GFL only).
        zone: Datacenter zone.
    """
    inverter_id: str
    timestamp_utc: str
    control_mode: str
    output_active_power_kw: float
    output_reactive_power_kvar: float
    output_voltage_v: float
    output_frequency_hz: float
    dc_link_voltage_v: float
    thd_percent: float
    rocof_hz_per_s: float
    freq_deviation_hz: float
    voltage_deviation_pu: float
    islanding_detected: bool
    junction_temp_c: float
    efficiency: float
    # VSM / Virtual Inertia
    virtual_inertia_H: float
    droop_kw_per_hz: float
    droop_kvar_per_v: float
    virtual_inertia_power_kw: float
    frequency_nadir_hz: float
    rocof_at_nadir: float
    # Black-Start
    black_start_active: bool
    black_start_stage: int
    black_start_voltage_pct: float
    loads_reconnected: int
    # Harmonic Compensation
    harmonic_compensation_active: bool
    thd_before_compensation_pct: float
    harmonic_injection_a: float
    dominant_harmonic_order: int
    # Weak-Grid Stability
    scr: float
    grid_impedance_pu: float
    pll_stability_margin_deg: float
    impedance_stability_margin_db: float
    stability_flag: str
    gfl_pll_locked: bool
    zone: str


# ---------------------------------------------------------------------------
# UPS Simulator
# ---------------------------------------------------------------------------

class UPSSimulator:
    """Simulates a bank of UPS units supporting a datacenter DC bus.

    In addition to standard battery backup, each UPS can participate in
    Virtual Inertia emulation: when grid frequency drops faster than
    ROCOF_TRIGGER_HZ_S, the UPS discharges its battery through the
    VSM-controlled inverter to stabilise frequency instantly — mimicking
    the kinetic energy release of a spinning synchronous generator.

    Args:
        num_ups: Number of UPS units to simulate.
        datacenter_zone: Zone label attached to all readings.
        vsm_inertia_H: Virtual inertia constant H [seconds] for frequency support.
        rocof_trigger_hz_s: ROCOF threshold that activates virtual inertia.
        random_seed: Seed for reproducibility (None = random).

    Example:
        >>> sim = UPSSimulator(num_ups=4, vsm_inertia_H=5.0)
        >>> readings = sim.generate_snapshot(datetime.now(timezone.utc))
    """

    ROCOF_TRIGGER_HZ_S: float = 0.3

    def __init__(
        self,
        num_ups: int = 4,
        datacenter_zone: str = "ZONE-A",
        vsm_inertia_H: float = 5.0,
        rocof_trigger_hz_s: float = 0.3,
        random_seed: Optional[int] = None,
    ) -> None:
        if random_seed is not None:
            random.seed(random_seed)

        self.num_ups = num_ups
        self.zone = datacenter_zone
        self.vsm_inertia_H = vsm_inertia_H
        self.rocof_trigger = rocof_trigger_hz_s

        # Persistent state per UPS unit
        self._soc: List[float] = [random.uniform(0.75, 0.98) for _ in range(num_ups)]
        self._mode: List[str] = ["normal"] * num_ups
        self._dc_bus_v: List[float] = [DC_BUS_NOMINAL_V + random.gauss(0, 2) for _ in range(num_ups)]
        self._prev_freq: float = NOMINAL_FREQ_HZ
        self._rocof: float = 0.0

        logger.info(f"UPSSimulator: {num_ups} units in {datacenter_zone} | H={vsm_inertia_H}s")

    # ------------------------------------------------------------------
    def _simulate_grid_frequency(self, ts: datetime) -> Tuple[float, float]:
        """Return (freq_hz, rocof) with occasional disturbance events."""
        # hour = ts.hour  # noqa: F841 (reserved for future use)
        # Base frequency noise
        base_noise = random.gauss(0, 0.02)

        # Occasional frequency disturbance (simulates generator trip or load step)
        if random.random() < 0.03:
            disturbance = random.uniform(-0.4, -0.15)   # frequency dip
        else:
            disturbance = 0.0

        freq = NOMINAL_FREQ_HZ + base_noise + disturbance
        rocof = (freq - self._prev_freq) / 5.0   # 5-second tick assumed
        self._prev_freq = freq
        self._rocof = rocof
        return freq, rocof

    # ------------------------------------------------------------------
    def generate_snapshot(self, ts: datetime) -> List[UPSReading]:
        """Generate one telemetry snapshot for all UPS units.

        Args:
            ts: Timestamp for this snapshot (timezone-aware datetime).

        Returns:
            List of UPSReading dataclasses, one per unit.
        """
        freq, rocof = self._simulate_grid_frequency(ts)
        readings: List[UPSReading] = []

        for i in range(self.num_ups):
            ups_id = f"UPS-{self.zone}-{i+1:02d}"

            # --- Determine operating mode ---
            grid_ok = (abs(freq - NOMINAL_FREQ_HZ) < 1.0 and
                       random.random() > 0.02)   # 2% probability of grid fault

            if not grid_ok:
                self._mode[i] = "battery"
            elif abs(rocof) >= self.rocof_trigger:
                self._mode[i] = "virtual_inertia"
            else:
                self._mode[i] = "normal"

            # --- Battery dynamics ---
            if self._mode[i] == "battery":
                discharge_rate = random.uniform(0.003, 0.008)
                self._soc[i] = max(0.05, self._soc[i] - discharge_rate)
            elif self._mode[i] == "virtual_inertia":
                # Short burst discharge for inertia support
                discharge_rate = random.uniform(0.001, 0.003)
                self._soc[i] = max(0.05, self._soc[i] - discharge_rate)
            else:
                # Trickle charge
                charge_rate = random.uniform(0.0005, 0.002)
                self._soc[i] = min(0.99, self._soc[i] + charge_rate)

            # --- Virtual inertia power calculation (VSM swing equation) ---
            # P_inertia = -2H * f0 * (df/dt) / S_rated
            # Simplified: proportional to ROCOF magnitude
            inertia_active = (self._mode[i] == "virtual_inertia")
            if inertia_active:
                inertia_power_kw = -2.0 * self.vsm_inertia_H * NOMINAL_FREQ_HZ * rocof / 1000.0
                inertia_power_kw = max(-500.0, min(500.0, inertia_power_kw))
            else:
                inertia_power_kw = 0.0

            # --- DC bus voltage ---
            self._dc_bus_v[i] += random.gauss(0, 1.5)
            self._dc_bus_v[i] = max(680.0, min(820.0, self._dc_bus_v[i]))

            # --- Battery voltage (simplified Peukert model) ---
            bat_v = 400.0 + 80.0 * self._soc[i] + random.gauss(0, 1.0)

            # --- Output power ---
            base_load_kw = random.uniform(180.0, 320.0)
            output_kw = base_load_kw + abs(inertia_power_kw)

            # --- Battery current ---
            if self._mode[i] in ("battery", "virtual_inertia"):
                bat_current = -(output_kw * 1000.0) / max(bat_v, 1.0)
            else:
                bat_current = (output_kw * 1000.0 * 0.15) / max(bat_v, 1.0)

            efficiency = random.uniform(0.93, 0.97)
            temp_c = 25.0 + (output_kw / 500.0) * 15.0 + random.gauss(0, 1.0)

            readings.append(UPSReading(
                ups_id=ups_id,
                timestamp_utc=ts.isoformat(),
                battery_soc=round(self._soc[i], 4),
                battery_voltage_v=round(bat_v, 2),
                battery_current_a=round(bat_current, 2),
                output_power_kw=round(output_kw, 2),
                ups_mode=self._mode[i],
                grid_voltage_v=round(NOMINAL_VOLTAGE_V + random.gauss(0, 3.0), 2),
                grid_frequency_hz=round(freq, 4),
                virtual_inertia_active=inertia_active,
                inertia_power_kw=round(inertia_power_kw, 3),
                dc_bus_voltage_v=round(self._dc_bus_v[i], 2),
                efficiency=round(efficiency, 4),
                temperature_c=round(temp_c, 2),
                zone=self.zone,
            ))

        return readings


# ---------------------------------------------------------------------------
# Inverter Simulator  (GFL + GFM with all 5 academic features)
# ---------------------------------------------------------------------------

class InverterSimulator:
    """Simulates Grid-Following (GFL) and Grid-Forming (GFM) inverters
    for a datacenter microgrid with five academic enhancements.

    Feature 1 — Virtual Synchronous Machine (VSM):
        GFM inverters emulate a synchronous generator swing equation:
            2H/ω₀ · dω/dt = P_mech − P_elec − D·Δω
        This provides virtual inertia that damps frequency oscillations.

    Feature 2 — Black-Start Capability:
        A GFM inverter can energise a dead bus from scratch. The sequence
        is: pre-charge → voltage ramp (0→480V over ~5s) → critical load
        pickup → full restoration. GFL inverters cannot black-start because
        they require an external voltage reference for their PLL.

    Feature 3 — Active Harmonic Compensation:
        GFM inverters detect and cancel harmonic currents (5th, 7th, 11th, 13th)
        injected by non-linear datacenter loads (switch-mode power supplies).
        The virtual output impedance is shaped in the harmonic domain to present
        near-zero impedance at target harmonic frequencies.

    Feature 4 — Droop Control (P/f and Q/V):
        Multiple GFM inverters share load proportionally without communication:
            f = f₀ − kp · (P − P₀)
            V = V₀ − kq · (Q − Q₀)
        This is the autonomous equivalent of governor + AVR in synchronous machines.

    Feature 5 — Weak-Grid Stability:
        At low Short-Circuit Ratios (SCR < 3), GFL PLLs destabilise.
        The impedance-based criterion (Middlebrook) flags instability when:
            |Zgrid(jω) / Zinv(jω)| > 1  (gain crossover)
        GFM inverters remain stable because they don't rely on PLL.

    Args:
        num_inverters: Total number of inverters (half GFL, half GFM by default).
        islanding_probability: Probability per tick of an islanding event.
        datacenter_zone: Zone label.
        vsm_inertia_H_range: (min, max) virtual inertia constant H [s] for GFMs.
        droop_kw_hz: Active power droop coefficient [kW/Hz].
        droop_kvar_v: Reactive power droop coefficient [kVAr/V].
        scr_range: (min, max) Short-Circuit Ratio to sweep weak-grid scenarios.
        black_start_probability: Probability per tick of triggering black-start.
        random_seed: Seed for reproducibility.
    """

    _HARMONIC_ORDERS = [5, 7, 11, 13]
    _BLACK_START_STAGES = {
        0: "idle",
        1: "pre_charge",
        2: "voltage_ramp",
        3: "load_pickup",
        4: "complete",
    }

    def __init__(
        self,
        num_inverters: int = 4,
        islanding_probability: float = 0.05,
        datacenter_zone: str = "ZONE-A",
        vsm_inertia_H_range: Tuple[float, float] = (3.0, 8.0),
        droop_kw_hz: float = 20.0,
        droop_kvar_v: float = 5.0,
        scr_range: Tuple[float, float] = (1.5, 10.0),
        black_start_probability: float = 0.005,
        random_seed: Optional[int] = None,
    ) -> None:
        if random_seed is not None:
            random.seed(random_seed)

        self.num_inverters = num_inverters
        self.islanding_prob = islanding_probability
        self.zone = datacenter_zone
        self.droop_kw_hz = droop_kw_hz
        self.droop_kvar_v = droop_kvar_v
        self.scr_range = scr_range
        self.black_start_prob = black_start_probability

        # Assign control modes: first half GFL, second half GFM
        self._modes: List[str] = []
        for i in range(num_inverters):
            self._modes.append("GFL" if i < num_inverters // 2 else "GFM")

        # Per-inverter persistent state
        self._H: List[float] = [
            random.uniform(*vsm_inertia_H_range) for _ in range(num_inverters)
        ]
        self._freq: List[float] = [NOMINAL_FREQ_HZ] * num_inverters
        self._voltage: List[float] = [NOMINAL_VOLTAGE_V] * num_inverters
        self._dc_link: List[float] = [DC_BUS_NOMINAL_V + random.gauss(0, 5) for _ in range(num_inverters)]
        self._islanding: List[bool] = [False] * num_inverters
        self._transition_ticks: List[int] = [0] * num_inverters

        # Black-start state
        self._bs_stage: List[int] = [0] * num_inverters
        self._bs_voltage_pct: List[float] = [0.0] * num_inverters
        self._bs_loads: List[int] = [0] * num_inverters

        # Harmonic state
        self._harmonic_comp: List[bool] = [False] * num_inverters
        self._dominant_harmonic: List[int] = [5] * num_inverters

        # Droop operating point
        self._P0: List[float] = [random.uniform(100.0, 400.0) for _ in range(num_inverters)]
        self._Q0: List[float] = [random.uniform(-50.0, 50.0) for _ in range(num_inverters)]

        # SCR per inverter (each connected at different grid strength)
        self._scr: List[float] = [
            random.uniform(*scr_range) for _ in range(num_inverters)
        ]

        logger.info(
            f"InverterSimulator: {num_inverters} inverters in {datacenter_zone} | "
            f"GFL={sum(1 for m in self._modes if m=='GFL')} "
            f"GFM={sum(1 for m in self._modes if m=='GFM')} | "
            f"droop={droop_kw_hz}kW/Hz"
        )

    # ------------------------------------------------------------------
    # Internal physics helpers
    # ------------------------------------------------------------------

    def _vsm_frequency_response(
        self, inv_idx: int, load_step_kw: float, dt: float = 5.0
    ) -> Tuple[float, float, float]:
        """Compute VSM frequency dynamics for one GFM inverter.

        Implements the linearised swing equation:
            Δω(t) = Δω(0)·exp(−D/(2H)·t) · cos(ωn·t + φ)

        Args:
            inv_idx: Inverter index.
            load_step_kw: Active power disturbance (positive = load increase).
            dt: Time step [s].

        Returns:
            Tuple of (freq_hz, rocof, virtual_inertia_power_kw).
        """
        H = self._H[inv_idx]
        D = 2.0   # damping coefficient (p.u.)
        # omega_n = math.sqrt(1.0 / (2.0 * H))  # natural frequency (noqa: F841)

        # Normalised load step
        delta_P_pu = load_step_kw / 1000.0

        # Frequency deviation from swing equation solution
        delta_omega = -delta_P_pu / D * (1.0 - math.exp(-D * dt / (2.0 * H)))
        delta_f = delta_omega * NOMINAL_FREQ_HZ / (2.0 * math.pi)

        freq = NOMINAL_FREQ_HZ + delta_f + random.gauss(0, 0.005)
        rocof = delta_f / dt + random.gauss(0, 0.001)

        # Inertia power injected (proportional to ROCOF)
        inertia_power = -2.0 * H * NOMINAL_FREQ_HZ * rocof
        inertia_power = max(-300.0, min(300.0, inertia_power))

        return freq, rocof, inertia_power

    def _droop_power_sharing(
        self, total_load_kw: float, total_load_kvar: float
    ) -> List[Tuple[float, float, float, float]]:
        """Compute power sharing among GFM inverters via droop control.

        Each GFM adjusts its output so that load is shared proportionally
        to droop coefficient — no inter-inverter communication required.

        Args:
            total_load_kw: Total active load to share.
            total_load_kvar: Total reactive load to share.

        Returns:
            List of (P_kw, Q_kvar, f_hz, V_v) per inverter.
        """
        gfm_indices = [i for i, m in enumerate(self._modes) if m == "GFM"]
        n_gfm = len(gfm_indices) or 1
        result: List[Tuple[float, float, float, float]] = []

        # Equal droop → equal share (simplified: uniform droop coefficients)
        P_share = total_load_kw / n_gfm
        Q_share = total_load_kvar / n_gfm

        for i in range(self.num_inverters):
            if self._modes[i] != "GFM":
                result.append((self._P0[i], self._Q0[i],
                                self._freq[i], self._voltage[i]))
                continue

            P_out = P_share + random.gauss(0, 2.0)
            Q_out = Q_share + random.gauss(0, 1.0)

            # Droop frequency and voltage
            delta_f = -self.droop_kw_hz * (P_out - self._P0[i]) / 1000.0
            delta_V = -self.droop_kvar_v * (Q_out - self._Q0[i]) / 1000.0

            freq = NOMINAL_FREQ_HZ + delta_f + random.gauss(0, 0.005)
            volt = NOMINAL_VOLTAGE_V + delta_V + random.gauss(0, 0.5)

            self._P0[i] = P_out
            self._Q0[i] = Q_out
            result.append((P_out, Q_out, freq, volt))

        return result

    def _compute_thd(
        self, inv_idx: int, compensation_active: bool
    ) -> Tuple[float, float, float, int]:
        """Compute THD before and after harmonic compensation.

        Datacenter loads (SMPS) generate odd harmonics: 5th (300Hz),
        7th (420Hz), 11th (660Hz), 13th (780Hz).
        GFM with virtual impedance shaping can suppress these by injecting
        counter-phase harmonic currents.

        Args:
            inv_idx: Inverter index.
            compensation_active: Whether harmonic compensation is enabled.

        Returns:
            Tuple of (thd_pct, thd_before_pct, injection_a, dominant_order).
        """
        # Raw THD from non-linear loads
        h5  = random.uniform(4.0, 8.0)    # 5th harmonic %
        h7  = random.uniform(2.0, 5.0)    # 7th harmonic %
        h11 = random.uniform(1.0, 3.0)    # 11th harmonic %
        h13 = random.uniform(0.5, 2.0)    # 13th harmonic %

        thd_before = math.sqrt(h5**2 + h7**2 + h11**2 + h13**2)

        # Dominant harmonic
        harmonics = {5: h5, 7: h7, 11: h11, 13: h13}
        dominant = max(harmonics, key=lambda k: harmonics[k])

        if compensation_active:
            # GFM reduces each harmonic by 70-90%
            suppression = random.uniform(0.10, 0.30)
            thd_after = thd_before * suppression
            # Injection current (simplified: proportional to suppressed harmonic energy)
            injection_a = (thd_before - thd_after) / 100.0 * random.uniform(50.0, 150.0)
        else:
            thd_after = thd_before + random.gauss(0, 0.3)
            injection_a = 0.0

        return (
            round(max(0.5, thd_after), 3),
            round(thd_before, 3),
            round(injection_a, 2),
            dominant,
        )

    def _weak_grid_stability(self, inv_idx: int) -> Tuple[float, float, float, float, str]:
        """Evaluate small-signal stability using the impedance-based criterion.

        For GFL inverters:
            - PLL phase margin degrades as SCR decreases
            - Instability when |Zgrid| / |Zinv| > 1 (Middlebrook criterion)
            - At SCR < 1.5, GFL is practically always unstable

        For GFM inverters:
            - No PLL → inherently stable in weak grids
            - Stability margin only limited by output impedance design

        Args:
            inv_idx: Inverter index.

        Returns:
            Tuple of (scr, grid_impedance_pu, pll_margin_deg,
                      impedance_margin_db, stability_flag).
        """
        scr = self._scr[inv_idx] + random.gauss(0, 0.1)
        scr = max(0.5, scr)

        # Grid impedance (per unit): Zgrid = 1/SCR
        Z_grid_pu = 1.0 / scr

        mode = self._modes[inv_idx]

        if mode == "GFL":
            # PLL phase margin degrades with weak grid
            # Empirical: PM ≈ 90° at SCR=10, 30° at SCR=2, <0° at SCR<1.5
            pll_pm = 90.0 * (1.0 - math.exp(-0.3 * (scr - 1.0)))
            pll_pm = max(-20.0, pll_pm) + random.gauss(0, 3.0)

            # Inverter output impedance (p.u.) — proportional to filter inductance
            Z_inv_pu = random.uniform(0.05, 0.15)

            # Middlebrook criterion: stable if |Zgrid/Zinv| < 1
            ratio = Z_grid_pu / Z_inv_pu
            margin_db = -20.0 * math.log10(max(ratio, 1e-9))

            if scr < 1.5 or pll_pm < 15.0:
                flag = "unstable"
            elif scr < 3.0 or pll_pm < 45.0:
                flag = "marginal"
            else:
                flag = "stable"

        else:  # GFM
            # GFM does not use PLL — phase margin concept doesn't apply
            pll_pm = float("nan")

            # GFM output impedance is designed (virtual impedance) — typically inductive
            Z_inv_pu = random.uniform(0.08, 0.20)
            ratio = Z_grid_pu / Z_inv_pu
            margin_db = -20.0 * math.log10(max(ratio, 1e-9))

            # GFM is stable even at very low SCR
            if scr < 0.8:
                flag = "marginal"
            else:
                flag = "stable"

        return (
            round(scr, 3),
            round(Z_grid_pu, 4),
            round(pll_pm if not math.isnan(pll_pm) else -999.0, 2),
            round(margin_db, 2),
            flag,
        )

    def _black_start_sequence(self, inv_idx: int) -> Tuple[int, float, int]:
        """Advance the black-start state machine by one tick.

        Black-start stages:
            0 (idle)       → triggered when grid loss detected
            1 (pre_charge) → DC bus charged from battery, ~1-3 ticks
            2 (voltage_ramp) → AC voltage ramped 0→480V, ~3-5 ticks
            3 (load_pickup)  → critical loads reconnected sequentially
            4 (complete)     → full restoration, transition to normal GFM

        Args:
            inv_idx: Inverter index.

        Returns:
            Tuple of (stage, voltage_pct, loads_reconnected).
        """
        stage = self._bs_stage[inv_idx]

        if stage == 0:
            # Idle — no black-start
            return 0, 0.0, 0

        elif stage == 1:
            # Pre-charge DC bus
            self._bs_voltage_pct[inv_idx] = 0.0
            if random.random() < 0.4:   # advance ~40% chance per tick
                self._bs_stage[inv_idx] = 2
                logger.debug(f"{self.zone} inverter {inv_idx}: black-start → voltage_ramp")
            return 1, 0.0, 0

        elif stage == 2:
            # Ramp voltage from 0 to 100%
            self._bs_voltage_pct[inv_idx] = min(
                100.0, self._bs_voltage_pct[inv_idx] + random.uniform(15.0, 25.0)
            )
            if self._bs_voltage_pct[inv_idx] >= 100.0:
                self._bs_stage[inv_idx] = 3
                logger.debug(f"{self.zone} inverter {inv_idx}: black-start → load_pickup")
            return 2, self._bs_voltage_pct[inv_idx], 0

        elif stage == 3:
            # Reconnect critical loads one by one
            max_loads = 8
            if self._bs_loads[inv_idx] < max_loads and random.random() < 0.5:
                self._bs_loads[inv_idx] += 1
            if self._bs_loads[inv_idx] >= max_loads:
                self._bs_stage[inv_idx] = 4
                logger.info(f"{self.zone} inverter {inv_idx}: black-start COMPLETE ✓")
            return 3, 100.0, self._bs_loads[inv_idx]

        else:  # stage == 4 (complete)
            return 4, 100.0, self._bs_loads[inv_idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_snapshot(self, ts: datetime) -> List[InverterReading]:
        """Generate one telemetry snapshot for all inverters.

        Args:
            ts: Timestamp for this snapshot (timezone-aware datetime).

        Returns:
            List of InverterReading dataclasses, one per inverter.
        """
        # Compute droop power sharing for GFM units
        total_load_kw   = random.uniform(800.0, 1400.0)
        total_load_kvar = random.uniform(-100.0, 200.0)
        droop_results   = self._droop_power_sharing(total_load_kw, total_load_kvar)

        readings: List[InverterReading] = []

        for i in range(self.num_inverters):
            inv_id = f"INV-{self.zone}-{i+1:02d}"
            mode   = self._modes[i]

            # --- Islanding detection ---
            newly_islanded = (random.random() < self.islanding_prob)
            if newly_islanded and mode == "GFL":
                self._islanding[i] = True
                self._transition_ticks[i] = random.randint(3, 8)
                logger.warning(f"{inv_id}: Islanding detected — initiating GFL→GFM transition")

            # --- Mode transitions ---
            if self._islanding[i]:
                if self._transition_ticks[i] > 0:
                    mode = "transitioning"
                    self._transition_ticks[i] -= 1
                else:
                    mode = "GFM"
                    self._modes[i] = "GFM"
                    self._islanding[i] = False
                    logger.info(f"{inv_id}: Transition complete → GFM")

            # --- Black-start trigger (only idle GFM) ---
            bs_triggered = (
                mode == "GFM"
                and self._bs_stage[i] == 0
                and random.random() < self.black_start_prob
            )
            if bs_triggered:
                self._bs_stage[i] = 1
                logger.warning(f"{inv_id}: Black-start sequence initiated")

            bs_stage, bs_vpct, bs_loads = self._black_start_sequence(i)
            black_start_active = bs_stage in (1, 2, 3)

            # Override mode during black-start
            if black_start_active:
                mode = "black_start"

            # --- Power and frequency ---
            P_out, Q_out, droop_freq, droop_volt = droop_results[i]

            if mode in ("GFM", "black_start"):
                load_step = random.gauss(0, 20.0)
                vsm_freq, rocof, inertia_power = self._vsm_frequency_response(i, load_step)
                freq = vsm_freq
                v_inertia_power = inertia_power
                H_val = self._H[i]
            elif mode == "transitioning":
                # Degraded performance — PLL losing lock
                freq = NOMINAL_FREQ_HZ + random.gauss(0, 0.3)
                rocof = random.gauss(0, 0.8)
                v_inertia_power = 0.0
                H_val = 0.0
            else:  # GFL
                # PLL-tracked — small deviations normally, large during islanding
                if self._islanding[i]:
                    freq = NOMINAL_FREQ_HZ + random.gauss(0, 1.5)
                    rocof = random.gauss(0, 2.5)
                else:
                    freq = NOMINAL_FREQ_HZ + random.gauss(0, 0.03)
                    rocof = random.gauss(0, 0.05)
                v_inertia_power = 0.0
                H_val = 0.0

            # Frequency nadir (worst frequency during event)
            freq_nadir = freq - abs(random.gauss(0, 0.1))
            rocof_nadir = rocof + abs(random.gauss(0, 0.05))

            # --- Harmonic compensation (GFM only) ---
            self._harmonic_comp[i] = (mode == "GFM" and random.random() > 0.2)
            thd, thd_before, inj_a, dom_harmonic = self._compute_thd(
                i, self._harmonic_comp[i]
            )

            # --- Weak-grid stability ---
            scr, Z_grid, pll_pm, imp_margin, stab_flag = self._weak_grid_stability(i)

            # --- Voltage and DC link ---
            freq_dev = freq - NOMINAL_FREQ_HZ
            volt_dev_pu = (droop_volt - NOMINAL_VOLTAGE_V) / NOMINAL_VOLTAGE_V
            volt_dev_pu = max(-0.1, min(0.1, volt_dev_pu + random.gauss(0, 0.005)))

            self._dc_link[i] += random.gauss(0, 3.0)
            self._dc_link[i] = max(650.0, min(850.0, self._dc_link[i]))

            # --- Derived quantities ---
            pll_locked = (mode == "GFL" and stab_flag != "unstable")
            temp_c = 40.0 + (abs(P_out) / 500.0) * 20.0 + random.gauss(0, 2.0)
            efficiency = random.uniform(0.96, 0.99) if mode != "transitioning" else random.uniform(0.88, 0.94)

            readings.append(InverterReading(
                inverter_id=inv_id,
                timestamp_utc=ts.isoformat(),
                control_mode=mode,
                output_active_power_kw=round(P_out, 2),
                output_reactive_power_kvar=round(Q_out, 2),
                output_voltage_v=round(NOMINAL_VOLTAGE_V * (1 + volt_dev_pu), 2),
                output_frequency_hz=round(freq, 4),
                dc_link_voltage_v=round(self._dc_link[i], 2),
                thd_percent=thd,
                rocof_hz_per_s=round(rocof, 4),
                freq_deviation_hz=round(freq_dev, 4),
                voltage_deviation_pu=round(volt_dev_pu, 5),
                islanding_detected=newly_islanded,
                junction_temp_c=round(temp_c, 2),
                efficiency=round(efficiency, 4),
                # VSM
                virtual_inertia_H=round(H_val, 2),
                droop_kw_per_hz=self.droop_kw_hz,
                droop_kvar_per_v=self.droop_kvar_v,
                virtual_inertia_power_kw=round(v_inertia_power, 3),
                frequency_nadir_hz=round(freq_nadir, 4),
                rocof_at_nadir=round(rocof_nadir, 4),
                # Black-Start
                black_start_active=black_start_active,
                black_start_stage=bs_stage,
                black_start_voltage_pct=round(bs_vpct, 1),
                loads_reconnected=bs_loads,
                # Harmonics
                harmonic_compensation_active=self._harmonic_comp[i],
                thd_before_compensation_pct=thd_before,
                harmonic_injection_a=inj_a,
                dominant_harmonic_order=dom_harmonic,
                # Weak-grid
                scr=scr,
                grid_impedance_pu=Z_grid,
                pll_stability_margin_deg=pll_pm,
                impedance_stability_margin_db=imp_margin,
                stability_flag=stab_flag,
                gfl_pll_locked=pll_locked,
                zone=self.zone,
            ))

        return readings


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    ts = datetime.now(timezone.utc)

    ups_sim = UPSSimulator(num_ups=4, vsm_inertia_H=5.0, random_seed=42)
    inv_sim = InverterSimulator(
        num_inverters=4,
        islanding_probability=0.05,
        black_start_probability=0.01,
        random_seed=42,
    )

    ups_snap = ups_sim.generate_snapshot(ts)
    inv_snap = inv_sim.generate_snapshot(ts)

    print("=== UPS Snapshot ===")
    for r in ups_snap:
        print(f"  {r.ups_id:20s} | mode={r.ups_mode:16s} | SoC={r.battery_soc:.2%} "
              f"| inertia={r.inertia_power_kw:+.1f}kW")

    print("\n=== Inverter Snapshot ===")
    for r in inv_snap:
        print(
            f"  {r.inverter_id:20s} | mode={r.control_mode:12s} | "
            f"f={r.output_frequency_hz:.3f}Hz | "
            f"ROCOF={r.rocof_hz_per_s:+.3f}Hz/s | "
            f"THD={r.thd_percent:.2f}% (before={r.thd_before_compensation_pct:.2f}%) | "
            f"SCR={r.scr:.1f} | stability={r.stability_flag} | "
            f"BS_stage={r.black_start_stage}"
        )
