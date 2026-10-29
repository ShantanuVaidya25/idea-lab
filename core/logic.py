"""Decision logic: occupancy state machine, AC inference, appliance policy.

Pure Python, no hardware - easy to unit test.
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class State(str, Enum):
    INITIALIZING = "INITIALIZING"
    OCCUPIED = "OCCUPIED"
    VACANCY_PENDING = "VACANCY_PENDING"
    VACANT = "VACANT"
    AC_OBSERVATION = "AC_OBSERVATION"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"
    ERROR = "ERROR"


# ---------------------------------------------------------------- occupancy
class OccupancyMachine:
    """Turns per-frame person counts into a debounced occupancy state.

    Vacancy (report section 3.4): the room is VACANT once the detector has seen zero people in
    `vacancy_confirm_frames` CONSECUTIVE frames (and for at least `vacancy_confirm_s`). One frame with a
    person cancels it. person_count=None means "detection failed this frame": failures never cause VACANT;
    after `fail_timeout_s` the state becomes ERROR.
    """

    def __init__(self, vacancy_confirm_s: float = 300, occupied_confirm_frames: int = 2,
                 fail_timeout_s: float = 30, vacancy_confirm_frames: int = 0):
        self.vacancy_confirm_s = vacancy_confirm_s
        self.vacancy_confirm_frames = max(0, int(vacancy_confirm_frames))
        self.occupied_confirm_frames = max(1, occupied_confirm_frames)
        self.fail_timeout_s = fail_timeout_s
        self.state = State.INITIALIZING
        self.pending_since: float | None = None
        self._seen_streak = 0
        self._empty_streak = 0
        self._last_valid: float | None = None
        self._start: float | None = None

    def update(self, person_count: int | None, now: float, fresh: bool = True) -> State:
        """`fresh` = this count comes from a NEW camera frame (the same frame seen twice is not 'consecutive')."""
        if self._start is None:
            self._start = now

        if person_count is None:  # detection failure -> hold, then ERROR
            ref = self._last_valid if self._last_valid is not None else self._start
            if now - ref >= self.fail_timeout_s:
                self.state = State.ERROR
                self.pending_since = None
            return self.state

        self._last_valid = now
        if person_count > 0:
            if fresh:
                self._seen_streak += 1
            self._empty_streak = 0
            if self.state == State.VACANCY_PENDING:
                # someone came back -> cancel pending vacancy immediately
                self.state = State.OCCUPIED
                self.pending_since = None
            elif self._seen_streak >= self.occupied_confirm_frames:
                self.state = State.OCCUPIED
                self.pending_since = None
        else:
            self._seen_streak = 0
            if fresh:
                self._empty_streak += 1
            if self.state in (State.OCCUPIED, State.INITIALIZING, State.ERROR):
                if self.state != State.INITIALIZING or self._empty_streak >= self.occupied_confirm_frames:
                    self.state = State.VACANCY_PENDING
                    self.pending_since = now
            if self.state == State.VACANCY_PENDING and self.pending_since is not None:
                if (now - self.pending_since >= self.vacancy_confirm_s
                        and self._empty_streak >= self.vacancy_confirm_frames):
                    self.state = State.VACANT
                    self.pending_since = None
        return self.state

    def reset_detection(self) -> None:
        """Detection source was swapped: restart the failure timer and streaks (state is kept)."""
        self._last_valid = None
        self._start = None
        self._seen_streak = 0
        self._empty_streak = 0

    def seconds_to_vacant(self, now: float) -> float | None:
        if self.state == State.VACANCY_PENDING and self.pending_since is not None:
            return max(0.0, self.vacancy_confirm_s - (now - self.pending_since))
        return None

    def empty_frames(self) -> tuple[int, int]:
        """(consecutive empty frames so far, frames needed) for the dashboard."""
        return self._empty_streak, self.vacancy_confirm_frames


# ------------------------------------------------------------- AC inference
class ACStatus(str, Enum):
    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    AC_ACTIVE = "AC_ACTIVE"                        # report formula: mean regional |delta T| > threshold
    AC_INACTIVE = "AC_INACTIVE"
    COOLING_OBSERVED = "COOLING_OBSERVED"          # signed formula (cooling only)
    NO_SIGNIFICANT_COOLING_OBSERVED = "NO_SIGNIFICANT_COOLING_OBSERVED"
    INCONCLUSIVE = "INCONCLUSIVE"
    SENSOR_ERROR = "SENSOR_ERROR"


AC_ON_RESULTS = {ACStatus.AC_ACTIVE, ACStatus.COOLING_OBSERVED}


def classify_delta(delta_t: float, cooling_threshold: float, warming_threshold: float) -> ACStatus:
    """delta_t = T_final - T_initial (signed)."""
    if delta_t <= -abs(cooling_threshold):
        return ACStatus.COOLING_OBSERVED
    if delta_t >= abs(warming_threshold):
        return ACStatus.NO_SIGNIFICANT_COOLING_OBSERVED
    return ACStatus.INCONCLUSIVE


def as_regions(temps) -> dict[int, float]:
    """A single reading, a list (region 1..4) or a {region: temp} dict -> {region: temp} without missing values."""
    if temps is None:
        return {}
    if isinstance(temps, (int, float)) and not isinstance(temps, bool):
        return {1: float(temps)}
    if isinstance(temps, dict):
        return {int(k): float(v) for k, v in temps.items() if v is not None}
    return {i + 1: float(v) for i, v in enumerate(temps) if v is not None}


class ACInference:
    """Post-vacancy AC inference across the classroom regions (project report, section 3.4).

    1. The room is confirmed vacant -> the observation timer starts.
    2. t = 0  : initial temperature of every region, T_i,init
    3. t = dt : final temperature of every region,   T_i,final
    4. delta T_i = | T_i,final - T_i,init |            (mode "absolute", as in the report)
       (mode "signed" uses T_i,final - T_i,init, so only COOLING counts)
    5. mean delta T = average over the regions that have a sensor; AC active if mean delta T > threshold.

    Readings go through a moving-average filter: T_init / T_final are the average of the readings in the first /
    last few seconds of the window, so one noisy DHT11 reading (1 C steps) cannot decide the result. A few dropped
    readings are tolerated; only a long run of bad readings is a sensor error.
    """

    def __init__(self, observation_s: float = 120, cooling_threshold_c: float = 1.0,
                 warming_threshold_c: float = 1.0, max_invalid_s: float = 15.0, mode: str = "absolute",
                 avg_window_s: float | None = None):
        self.observation_s = observation_s
        self.cooling_threshold_c = cooling_threshold_c     # = delta_thresh of the report
        self.warming_threshold_c = warming_threshold_c
        self.max_invalid_s = max_invalid_s                 # this long without a valid reading -> SENSOR_ERROR
        self.mode = mode if mode in ("absolute", "signed") else "absolute"
        self.edge_window_s = avg_window_s or min(15.0, observation_s * 0.25)   # moving-average window at each end
        self.reset()

    @property
    def threshold_c(self) -> float:
        return self.cooling_threshold_c

    def reset(self) -> None:
        self.status = ACStatus.IDLE
        self.t_initial: float | None = None              # mean over regions (for the summary line)
        self.t_final: float | None = None
        self.delta_t: float | None = None                # mean regional delta T (the decision value)
        self.delta_signed: float | None = None           # mean of T_final - T_init (cooling is negative)
        self.regions: dict[int, dict] = {}               # region -> {init, final, delta}
        self.started_at: float | None = None
        self.samples: dict[int, list[tuple[float, float]]] = {}
        self._invalid_since: float | None = None

    def start(self, temps, sensor_ok: bool, now: float) -> ACStatus:
        self.reset()
        regs = as_regions(temps) if sensor_ok else {}
        if not regs:
            self.status = ACStatus.SENSOR_ERROR
            return self.status
        self.started_at = now
        self.samples = {i: [(now, v)] for i, v in regs.items()}
        self.t_initial = round(sum(regs.values()) / len(regs), 2)
        self.status = ACStatus.OBSERVING
        return self.status

    def update(self, temps, sensor_ok: bool, now: float, fresh: bool = True) -> ACStatus:
        """`fresh` = this is a NEW reading from the sensors (not the same one seen again)."""
        if self.status != ACStatus.OBSERVING:
            return self.status
        assert self.started_at is not None
        if fresh:
            regs = as_regions(temps) if sensor_ok else {}
            if not regs:
                if self._invalid_since is None:
                    self._invalid_since = now
            else:
                self._invalid_since = None
                for i, v in regs.items():
                    self.samples.setdefault(i, []).append((now, v))
        if self._invalid_since is not None and now - self._invalid_since > self.max_invalid_s:
            self.status = ACStatus.SENSOR_ERROR
            return self.status
        if now - self.started_at >= self.observation_s:
            self._finish(now)
        return self.status

    def _region_values(self, now: float) -> dict[int, dict]:
        """Moving-average T_init / T_final per region, and its delta T."""
        assert self.started_at is not None
        out = {}
        for i, pts in sorted(self.samples.items()):
            first = [v for t, v in pts if t - self.started_at <= self.edge_window_s]
            last = [v for t, v in pts if now - t <= self.edge_window_s]
            if not first or not last:
                continue
            init, final = sum(first) / len(first), sum(last) / len(last)
            signed = final - init
            out[i] = {"init": round(init, 2), "final": round(final, 2), "signed": round(signed, 2),
                      "delta": round(abs(signed) if self.mode == "absolute" else signed, 2)}
        return out

    def _finish(self, now: float) -> None:
        regs = self._region_values(now)
        if not regs:                                     # no valid reading near the end -> cannot judge
            self.status = ACStatus.SENSOR_ERROR
            return
        n = len(regs)
        self.regions = regs
        self.t_initial = round(sum(r["init"] for r in regs.values()) / n, 2)
        self.t_final = round(sum(r["final"] for r in regs.values()) / n, 2)
        self.delta_signed = round(sum(r["signed"] for r in regs.values()) / n, 2)
        self.delta_t = round(sum(r["delta"] for r in regs.values()) / n, 2)
        if self.mode == "absolute":
            self.status = ACStatus.AC_ACTIVE if self.delta_t > self.cooling_threshold_c else ACStatus.AC_INACTIVE
        else:
            self.status = classify_delta(self.delta_t, self.cooling_threshold_c, self.warming_threshold_c)

    def live_regions(self, now: float) -> dict[int, dict]:
        """Running per-region values while observing (the official result comes at the end)."""
        if self.status == ACStatus.OBSERVING and self.started_at is not None:
            return self._region_values(now)
        return dict(self.regions)

    def live_delta(self, now: float) -> float | None:
        """Running mean delta T while observing (for the dashboard)."""
        if self.status != ACStatus.OBSERVING or self.started_at is None:
            return None
        regs = self._region_values(now)
        return round(sum(r["delta"] for r in regs.values()) / len(regs), 2) if regs else None

    def seconds_left(self, now: float) -> float | None:
        if self.status == ACStatus.OBSERVING and self.started_at is not None:
            return max(0.0, self.observation_s - (now - self.started_at))
        return None


# ------------------------------------------------------ continuous AC monitor
class ACState(str, Enum):
    UNKNOWN = "UNKNOWN"
    ON = "ON"
    OFF = "OFF"


def slope_per_min(samples: list[tuple[float, float]]) -> float:
    """Least-squares temperature slope in degC per minute (robust to the DHT11's 1 degC steps)."""
    n = len(samples)
    if n < 2:
        return 0.0
    t0 = samples[0][0]
    xs = [(t - t0) / 60.0 for t, _ in samples]
    ys = [v for _, v in samples]
    mx, my = sum(xs) / n, sum(ys) / n
    var = sum((x - mx) ** 2 for x in xs)
    return 0.0 if var == 0 else sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var


class ACMonitor:
    """Continuous AC ON/OFF estimate from the temperature trend of one DHT11 (there is no AC power meter).

    ON  : over the last `window_s` the room cooled by at least `on_drop_c` and the slope is <= `on_slope`.
    OFF : after ON, the room warmed back by at least `off_rise_c` from its lowest point and the slope is
          >= `off_slope` (the AC stopped pulling heat out).
    A stable / warming room at start-up just sets the baseline to OFF (no event is reported for it).
    """

    def __init__(self, window_s: float = 180, on_drop_c: float = 1.0, on_slope_c_min: float = -0.25,
                 off_rise_c: float = 1.0, off_slope_c_min: float = 0.1, min_samples: int = 8,
                 stale_s: float = 30):
        self.window_s = window_s
        self.on_drop_c = on_drop_c
        self.on_slope = on_slope_c_min
        self.off_rise_c = off_rise_c
        self.off_slope = off_slope_c_min
        self.min_samples = min_samples
        self.stale_s = stale_s
        self.samples: deque[tuple[float, float]] = deque()
        self.state = ACState.UNKNOWN
        self.since: float | None = None            # when the current ON/OFF state started
        self.slope = 0.0                           # degC per minute over the window
        self.drop = 0.0                            # start-of-window minus end-of-window (positive = cooling)
        self._low: float | None = None             # lowest temperature seen while ON

    def add(self, temp_c: float, now: float) -> None:
        self.samples.append((now, temp_c))
        while self.samples and now - self.samples[0][0] > self.window_s:
            self.samples.popleft()

    def _edge_medians(self) -> tuple[float, float]:
        k = max(1, len(self.samples) // 4)
        vals = [v for _, v in self.samples]
        return statistics.median(vals[:k]), statistics.median(vals[-k:])

    def update(self, now: float) -> str | None:
        """Re-evaluate. Returns "ON" or "OFF" when the AC state CHANGES, otherwise None."""
        if not self.samples or now - self.samples[-1][0] > self.stale_s:
            self.state, self.since = ACState.UNKNOWN, None   # no recent readings: don't guess
            return None
        span = self.samples[-1][0] - self.samples[0][0]
        if len(self.samples) < self.min_samples or span < self.window_s * 0.6:
            return None
        pts = list(self.samples)
        self.slope = round(slope_per_min(pts), 3)
        first, last = self._edge_medians()
        self.drop = round(first - last, 2)

        if self.state == ACState.ON:
            self._low = last if self._low is None else min(self._low, last)
            if last - self._low >= self.off_rise_c and self.slope >= self.off_slope:
                return self._set(ACState.OFF, now, report=True)
            return None
        cooling = self.drop >= self.on_drop_c and self.slope <= self.on_slope
        if cooling:
            self._low = last
            return self._set(ACState.ON, now, report=True)      # also when the AC was already on at start-up
        if self.state == ACState.UNKNOWN:
            self._set(ACState.OFF, now, report=False)    # stable / warming room: baseline is OFF
        return None

    def _set(self, state: ACState, now: float, report: bool) -> str | None:
        self.state, self.since = state, now
        return state.value if report else None

    def info(self, now: float) -> dict:
        return {"state": self.state.value, "slope_c_min": self.slope, "drop_c": self.drop,
                "samples": len(self.samples), "for_s": None if self.since is None else round(now - self.since)}


# ------------------------------------------------------------ relay policy
def desired_relay_state(occ_state: State, relay_cfg: dict, current: int) -> int:
    """What a relay SHOULD be, given occupancy. Unknown/unsafe states hold current."""
    if not relay_cfg.get("auto", False):
        return current
    if occ_state == State.VACANT:
        return 0
    if occ_state == State.OCCUPIED:
        return int(relay_cfg.get("occupied_state", current))
    return current  # INITIALIZING / VACANCY_PENDING / ERROR -> do nothing


@dataclass
class SwitchGuard:
    """Enforces a minimum interval between switches per relay."""
    min_interval_s: float = 10
    last_switch: dict[int, float] = field(default_factory=dict)

    def can_switch(self, channel: int, now: float) -> bool:
        last = self.last_switch.get(channel)
        return last is None or now - last >= self.min_interval_s

    def mark(self, channel: int, now: float) -> None:
        self.last_switch[channel] = now
