"""Fan / light tracking and visual ON/OFF estimation from camera frames.

Fan   -> ON when its blades are moving (pixels change), or when they have blurred away completely
         (a fast fan at 30 fps looks like a still, smooth disc: only the motor housing stays visible).
Light -> ON when it glows (clearly brighter than its surroundings), OFF otherwise.

This is a VISUAL estimate. Relay state still comes from the ESP32.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ON, OFF, UNKNOWN = "ON", "OFF", "UNKNOWN"


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def inside(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Fraction of box `a` that lies inside box `b`."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    area = (a[2] - a[0]) * (a[3] - a[1])
    return max(0, ix2 - ix1) * max(0, iy2 - iy1) / area if area > 0 else 0.0


def dark_fraction(gray: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """Share of the box that is clearly darker than its own background (crisp fan blades / housing).
    A fast fan blurs its blades away, so this drops sharply when the fan spins."""
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    mx, my = int((x2 - x1) * 0.05), int((y2 - y1) * 0.05)
    roi = gray[y1 + my:y2 - my, x1 + mx:x2 - mx].astype(np.float32)
    if roi.size == 0:
        return 0.0
    return float((roi < float(np.median(roi)) - 40).mean())


def _center(b) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def glow_mask(gray: np.ndarray, c: dict | None = None) -> np.ndarray:
    """Pixels that GLOW: very bright, clearly brighter than the area around them, in the upper part of the
    picture, and not part of a big bright region (a window or a sunlit wall)."""
    import cv2
    c = c or {}
    H, W = gray.shape[:2]
    level = c.get("glow_min_level", 215)
    bg = cv2.blur(gray, (61, 61)).astype(np.int16)
    mask = (gray >= level) & (gray.astype(np.int16) - bg >= c.get("glow_min_lift", 35))
    mask[int(H * c.get("glow_region_top", 0.6)):, :] = False
    nb, blab, bst, _ = cv2.connectedComponentsWithStats((gray >= level).astype(np.uint8), 8)
    big = bst[:, cv2.CC_STAT_AREA] > c.get("glow_max_frac", 0.02) * H * W
    big[0] = False
    mask &= ~big[blab]
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)


def blade_metrics(gray: np.ndarray, housing: tuple[int, int, int, int], scale_w: float = 4.0,
                  flat: float = 0.45) -> tuple[float, float, float, tuple[int, int, int, int]] | None:
    """Look for fan blades in the (perspective-flattened) disc around a ceiling-fan housing.

    Returns (dark, edges, brightness, disc_box): `dark` = share of blade-dark pixels, `edges` = share of
    strong edges, both with the housing itself left out. Stopped fan: both high (about 0.4 / 0.2).
    Spinning fan: both near 0 - the blades have blurred into the ceiling.
    """
    import cv2
    x1, y1, x2, y2 = housing
    w = max(8, x2 - x1)
    cx, cy = (x1 + x2) / 2, y2 - 0.25 * (y2 - y1)
    rx, ry = scale_w * w / 2, scale_w * w / 2 * flat
    H, W = gray.shape[:2]
    bx1, bx2 = int(max(0, cx - rx)), int(min(W, cx + rx))
    by1, by2 = int(max(0, cy - ry)), int(min(H, cy + ry))
    if bx2 - bx1 < 10 or by2 - by1 < 6:
        return None
    crop = gray[by1:by2, bx1:bx2]
    yy, xx = np.mgrid[by1:by2, bx1:bx2]
    disc = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
    disc &= ~((xx >= x1 - 3) & (xx <= x2 + 3) & (yy >= y1 - 3) & (yy <= y2 + 3))
    vals = crop[disc].astype(np.float32)
    if vals.size < 50:
        return None
    bg = float(np.median(vals))
    gx = np.abs(cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3))
    return (float((vals < bg - 30).mean()), float(((gx + gy) > 80)[disc].mean()), float(vals.mean()),
            (bx1, by1, bx2, by2))


@dataclass
class Track:
    id: int
    cls: str                                   # "fan" | "light"
    bbox: tuple[int, int, int, int]
    conf: float
    last_seen: float
    hits: int = 1                              # detection passes that have seen it
    state: str = UNKNOWN
    metric: float = 0.0                        # fan: moving-pixel ratio, light: saturated ratio
    samples: deque = field(default_factory=deque)   # (time, value)
    housing: bool = False                      # fan seen only as its motor housing (blades blurred away)
    ttl: float | None = None                   # own time-to-live (glow lights stay listed after switching off)
    blades: deque = field(default_factory=deque)    # (time, dark, edges, motion) for housing fans


class ApplianceTracker:
    def __init__(self, cfg: dict | None = None):
        c = cfg or {}
        self.pixel_delta = c.get("motion_pixel_delta", 25)        # gray-level change that counts as "moved"
        self.fan_on_ratio = c.get("fan_motion_ratio", 0.08)       # >8% of the box changed over fan_lag_s -> fan ON
        # compare with frames this long ago and keep the strongest change: a 4-blade fan can turn exactly a
        # quarter-turn in one gap (looks identical), but not in every gap at once
        self.fan_lags_s = tuple(c.get("fan_lags_s", (0.1, 0.25, 0.4)))
        self.window_s = c.get("state_window_s", 2.0)              # decide over this much recent video
        self.light_sat_ratio = c.get("light_saturated_ratio", 0.08)
        self.light_peak = c.get("light_peak", 240)
        self.light_contrast = c.get("light_contrast", 25)
        self.ttl_s = c.get("track_ttl_s", 30)
        self.confirm_hits = c.get("confirm_hits", 1)               # must be seen this many passes to be shown
        self.drop_light_in_fan = c.get("ignore_light_inside_fan", True)
        self.light_glow_lift = c.get("light_glow_lift", 40)        # lit if its brightest part is this much above the area around it
        # the person/appliance model calls a spinning ceiling fan a "lamp" (only the housing is left): a hanging
        # "lamp" that does NOT glow is treated as a fan housing
        self.unlit_lamp_is_fan = c.get("unlit_lamp_is_fan", True)
        self.blades_visible = c.get("blades_visible", (0.12, 0.10))    # dark, edges -> blades clearly there = stopped
        self.blades_gone = c.get("blades_gone", (0.06, 0.08))          # dark, edges -> blades blurred away = spinning
        self.glow_ttl_s = c.get("glow_ttl_s", 120)                  # a light that went dark stays listed as OFF this long
        self.drop_unlit_lamps = c.get("drop_unlit_lamps", True)
        self.glow_cfg = c
        self.glow_min_area = c.get("glow_min_area", 20)
        self._glow: np.ndarray | None = None                       # glow map of the latest frame
        self.min_samples = 3
        # share of the picture that must change at once: still camera + people moving <= 0.08, a 40 px pan ~0.28
        self.camera_move_ratio = c.get("camera_move_ratio", 0.18)
        self.camera_moves = 0
        self._thumb: tuple[float, np.ndarray] | None = None
        self.tracks: list[Track] = []
        self._next_id = 1
        self._hist: deque = deque()                               # (time, gray) of the last ~1 s
        self._lock = threading.Lock()
        self.zones_path = c.get("zones_path")                      # saved fan/light positions + calibration
        self.zones: list[dict] = []                                # {id, cls, bbox, off, on}
        self._zrun: dict[str, dict] = {}                           # live state per zone
        self._load_zones()

    # ------------------------------------------------------------ saved zones
    def _load_zones(self) -> None:
        if self.zones_path and Path(self.zones_path).exists():
            try:
                self.zones = json.loads(Path(self.zones_path).read_text(encoding="utf-8"))
            except Exception:
                self.zones = []

    def _save_zones(self) -> None:
        if self.zones_path:
            Path(self.zones_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.zones_path).write_text(json.dumps(self.zones, indent=1), encoding="utf-8")

    def _new_zone_id(self) -> str:
        n = 1
        while f"z{n}" in {z["id"] for z in self.zones}:
            n += 1
        return f"z{n}"

    def _measure_dark(self, box) -> float | None:
        grays = [g for _, g in self._hist]
        return float(np.mean([dark_fraction(g, box) for g in grays])) if grays else None

    def add_zone(self, cls: str, bbox: tuple[int, int, int, int]) -> str:
        with self._lock:
            z = {"id": self._new_zone_id(), "cls": cls, "bbox": [int(v) for v in bbox], "off": None, "on": None}
            self.zones.append(z)
            self._save_zones()
            return z["id"]

    def zones_from_tracks(self) -> int:
        """Remember the fans/lights seen right now as permanent zones (do this with the fan STOPPED)."""
        with self._lock:
            made = 0
            for t in self.tracks:
                if t.hits < self.confirm_hits or any(max(iou(t.bbox, z["bbox"]), inside(t.bbox, z["bbox"])) >= 0.5
                                                     for z in self.zones):
                    continue
                z = {"id": self._new_zone_id(), "cls": t.cls, "bbox": list(t.bbox), "off": None, "on": None}
                if t.cls == "fan":
                    z["off"] = self._measure_dark(t.bbox)
                self.zones.append(z)
                made += 1
            if made:
                self.tracks = [t for t in self.tracks if not any(
                    max(iou(t.bbox, z["bbox"]), inside(t.bbox, z["bbox"])) >= 0.5 for z in self.zones)]
                self._save_zones()
            return made

    def mark_zone(self, zone_id: str, state: str) -> bool:
        """Calibrate: the fan in this zone is `state` (ON/OFF) RIGHT NOW. Learns how it looks in that state."""
        with self._lock:
            for z in self.zones:
                if z["id"] == zone_id and z["cls"] == "fan":
                    v = self._measure_dark(z["bbox"])
                    if v is None:
                        return False
                    z["on" if state == ON else "off"] = v
                    self._save_zones()
                    return True
            return False

    def clear_zones(self) -> None:
        with self._lock:
            self.zones, self._zrun = [], {}
            self._save_zones()

    def zone_info(self) -> list[dict]:
        with self._lock:
            return [{**z, "dark_now": self._zrun.get(z["id"], {}).get("dark_now")} for z in self.zones]

    # ------------------------------------------------------------ detections
    def update_detections(
        self,
        dets: list[tuple[str, float, tuple[int, int, int, int]]] | list[tuple[str, float, tuple[int, int, int, int], str]] | list[tuple],
        now: float,
    ) -> None:
        """dets: (cls, conf, (x1,y1,x2,y2)) from YOLO. Match to existing tracks, create new ones."""
        with self._lock:
            if self.zones:                   # a saved zone owns its area: ignore auto-detections there
                zb = [z["bbox"] for z in self.zones]
                dets = [d for d in dets if not any(max(iou(d[2], b), inside(d[2], b)) >= 0.5 for b in zb)]
            self.tracks = [t for t in self.tracks if not any(
                max(iou(t.bbox, z["bbox"]), inside(t.bbox, z["bbox"])) >= 0.5 for z in self.zones)]
            gray = self._hist[-1][1] if self._hist else None
            if gray is not None:
                self._glow = glow_mask(gray, self.glow_cfg)
            fixed: list[tuple[str, float, tuple[int, int, int, int], str]] = []
            for d in dets:
                cls, conf, box = d[0], d[1], tuple(int(v) for v in d[2])
                src = d[3] if len(d) > 3 else "model"
                if cls == "light" and src == "model" and gray is not None and self._light_value(gray, box) == 0.0:
                    # the model's "lamp" does not glow: a spinning fan's housing, or just a guess
                    if self.unlit_lamp_is_fan and self._looks_like_fan_housing(gray, box):
                        cls, src = "fan", "housing"
                    elif self.drop_unlit_lamps:
                        continue        # an unlit "lamp" can't be told from clutter; lights are found by their glow
                fixed.append((cls, conf, box, src))
            active_dets = fixed
            # a lit light found by its glow and the model's "lamp" box on the same spot are one light: keep the glow one
            glows = [d[2] for d in active_dets if d[3] == "glow"] + [t.bbox for t in self.tracks if t.cls == "light" and t.ttl]
            near = lambda a, b: max(iou(a, b), inside(a, b), inside(b, a)) >= 0.4
            active_dets = [d for d in active_dets if not (d[0] == "light" and d[3] != "glow" and any(near(d[2], g) for g in glows))]
            self.tracks = [t for t in self.tracks if not (t.cls == "light" and not t.ttl and any(near(t.bbox, g) for g in glows))]
            if self.drop_light_in_fan:       # a ceiling fan's motor housing looks like a lamp: not a separate light
                fans = [d[2] for d in active_dets if d[0] == "fan"] + [t.bbox for t in self.tracks if t.cls == "fan"]
                active_dets = [d for d in active_dets if not (d[0] == "light" and d[3] != "glow" and any(inside(d[2], f) >= 0.6 for f in fans))]
                self.tracks = [t for t in self.tracks
                               if not (t.cls == "light" and t.ttl is None and any(inside(t.bbox, f) >= 0.6 for f in fans))]
            # a housing inside a fan the model can see whole (blades visible) is the same fan
            whole = [d[2] for d in active_dets if d[0] == "fan" and d[3] != "housing"] + \
                    [t.bbox for t in self.tracks if t.cls == "fan" and not t.housing]
            active_dets = [d for d in active_dets if not (d[3] == "housing" and any(inside(d[2], f) >= 0.6 for f in whole))]
            used: set[int] = set()
            for cls, conf, box, src in active_dets:
                best, best_iou = None, 0.3
                for t in self.tracks:
                    if t.cls == cls and t.id not in used:
                        v = iou(t.bbox, box)
                        if src == "glow" and t.ttl:     # a glow box changes size with brightness: match by position
                            (ax, ay), (bx, by) = _center(t.bbox), _center(box)
                            reach = 0.75 * max(t.bbox[2] - t.bbox[0], t.bbox[3] - t.bbox[1], box[2] - box[0], box[3] - box[1])
                            if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 < reach:
                                v = max(v, best_iou + 0.01)
                        if v > best_iou:
                            best, best_iou = t, v
                if best is None:
                    best = Track(self._next_id, cls, tuple(box), conf, now, housing=src == "housing",
                                 ttl=self.glow_ttl_s if src == "glow" else None)
                    self._next_id += 1
                    self.tracks.append(best)
                else:   # smooth the box so it doesn't jitter
                    best.bbox = tuple(int(0.5 * o + 0.5 * n) for o, n in zip(best.bbox, box))
                    best.conf = conf
                    best.last_seen = now
                    best.hits += 1
                used.add(best.id)
            self.tracks = [t for t in self.tracks if now - t.last_seen <= (t.ttl or self.ttl_s)]
            # overlapping light tracks are the same light (its glow splits differently from frame to frame)
            kept: list[Track] = []
            for t in sorted(self.tracks, key=lambda t: (t.cls != "light", t.state != ON, -t.hits)):
                if t.cls == "light" and any(k.cls == "light" and near(t.bbox, k.bbox) for k in kept):
                    continue
                kept.append(t)
            self.tracks = kept

    def _looks_like_fan_housing(self, gray: np.ndarray, box: tuple[int, int, int, int]) -> bool:
        """Shape test for an unlit 'lamp': hanging in the upper part of the picture, taller than wide
        (down-rod + motor housing) = the housing of a spinning fan."""
        h = gray.shape[0]
        x1, y1, x2, y2 = box
        upper = (y1 + y2) / 2 < 0.6 * h
        tall = (y2 - y1) >= 1.1 * max(1, x2 - x1)            # down-rod + motor housing
        roi = gray[max(0, y1):y2, max(0, x1):x2]
        # a housing (white or dark) is never clearly BRIGHTER than the ceiling beside it - a light is
        not_bright = roi.size > 0 and float(np.median(roi)) <= self._side_median(gray, box) + 10
        return upper and tall and not_bright

    @staticmethod
    def _side_median(gray: np.ndarray, box: tuple[int, int, int, int], strip: int = 30) -> float:
        """Median brightness of the strips just left and right of a box."""
        x1, y1, x2, y2 = (max(0, box[0]), max(0, box[1]), min(gray.shape[1], box[2]), min(gray.shape[0], box[3]))
        sides = [gray[y1:y2, max(0, x1 - strip):x1], gray[y1:y2, x2:x2 + strip]]
        vals = np.concatenate([s.ravel() for s in sides if s.size])
        return float(np.median(vals)) if vals.size else float(np.median(gray))

    # ---------------------------------------------------------------- frames
    def _camera_moved(self, gray: np.ndarray, now: float) -> bool:
        """True when the whole picture shifted (camera turned / picked up), compared with ~0.5 s ago.

        Uses a tiny 64x48 copy of the frame. The median difference is removed first (auto-exposure making the whole
        picture brighter is not movement), and people walking change only part of the picture, which stays below
        the threshold.
        """
        import cv2
        thumb = cv2.resize(gray, (64, 48), interpolation=cv2.INTER_AREA).astype(np.int16)
        if self._thumb is None or thumb.shape != self._thumb[1].shape:
            self._thumb = (now, thumb)
            return False
        if now - self._thumb[0] < 0.5:
            return False
        d = thumb - self._thumb[1]
        d -= int(np.median(d))
        self._thumb = (now, thumb)
        return float((np.abs(d) > 25).mean()) > self.camera_move_ratio

    def update_frame(self, gray: np.ndarray, now: float) -> None:
        """Call for every camera frame with a grayscale image."""
        with self._lock:
            if self._camera_moved(gray, now):
                # everything found so far was at the OLD camera position: start again (saved zones stay)
                self.tracks = []
                self._hist.clear()
                self.camera_moves += 1
            self._hist.append((now, gray))
            while self._hist and now - self._hist[0][0] > max(self.fan_lags_s) + 0.3:
                self._hist.popleft()
            refs = [next((g for t, g in reversed(self._hist) if now - t >= lag), None) for lag in self.fan_lags_s]
            refs = [r for r in refs if r is not None and r.shape == gray.shape]
            h, w = gray.shape[:2]
            if any(t.cls == "light" for t in self.tracks) or any(z["cls"] == "light" for z in self.zones):
                self._glow = glow_mask(gray, self.glow_cfg)
            for t in self.tracks:
                x1, y1, x2, y2 = (max(0, t.bbox[0]), max(0, t.bbox[1]), min(w, t.bbox[2]), min(h, t.bbox[3]))
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                if t.cls == "fan" and t.housing:
                    self._update_housing_fan(t, gray, refs, now)
                    continue
                if t.cls == "fan":
                    if not refs:
                        continue
                    cur = gray[y1:y2, x1:x2].astype(np.int16)
                    best = 0.0
                    for ref in refs:
                        d = cur - ref[y1:y2, x1:x2].astype(np.int16)
                        d -= int(np.median(d))                    # ignore global brightness / auto-exposure shifts
                        best = max(best, float((np.abs(d) > self.pixel_delta).mean()))
                    self._add(t, now, best)
                    self._decide_fan(t)
                elif t.cls == "light":
                    self._add(t, now, self._light_value(gray, (x1, y1, x2, y2)))
                    self._decide_light(t)

            for z in self.zones:
                box = tuple(z["bbox"])
                x1, y1, x2, y2 = (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))
                if x2 - x1 < 8 or y2 - y1 < 8:
                    continue
                run = self._zrun.setdefault(z["id"], {"motion": deque(), "dark": deque(), "state": UNKNOWN,
                                                      "metric": 0.0, "dark_now": None})
                if z["cls"] == "fan":
                    roi = gray[y1:y2, x1:x2]
                    if roi.mean() < 30 or roi.std() < 4:
                        # too dark / blank (camera covered, privacy shutter, lights off): blades can't be judged,
                        # and "no visible blades" must not be mistaken for "blades blurred = ON"
                        run["motion"].clear()
                        run["dark"].clear()
                        run["state"], run["metric"], run["dark_now"] = UNKNOWN, 0.0, None
                        continue
                    if refs:
                        cur = roi.astype(np.int16)
                        best = 0.0
                        for ref in refs:
                            d = cur - ref[y1:y2, x1:x2].astype(np.int16)
                            d -= int(np.median(d))
                            best = max(best, float((np.abs(d) > self.pixel_delta).mean()))
                        self._push(run["motion"], now, best)
                    self._push(run["dark"], now, dark_fraction(gray, box))
                    self._decide_zone_fan(z, run)
                else:
                    self._push(run["motion"], now, self._light_value(gray, (x1, y1, x2, y2)))
                    if len(run["motion"]) >= self.min_samples:
                        run["metric"] = float(np.mean([v for _, v in run["motion"]]))
                        if run["metric"] >= 0.6:
                            run["state"] = ON
                        elif run["metric"] <= 0.4 or run["state"] == UNKNOWN:
                            run["state"] = OFF

    def _update_housing_fan(self, t: Track, gray: np.ndarray, refs: list, now: float) -> None:
        m = blade_metrics(gray, t.bbox)
        if m is None:
            return
        dark, edges, bright, (bx1, by1, bx2, by2) = m
        if bright < 30 or float(gray[by1:by2, bx1:bx2].std()) < 3:    # too dark / blank to judge
            t.blades.clear()
            t.state, t.metric = UNKNOWN, 0.0
            return
        motion = 0.0
        if refs:
            cur = gray[by1:by2, bx1:bx2].astype(np.int16)
            for ref in refs:
                d = cur - ref[by1:by2, bx1:bx2].astype(np.int16)
                d -= int(np.median(d))
                motion = max(motion, float((np.abs(d) > self.pixel_delta).mean()))
        t.blades.append((now, dark, edges, motion))
        while t.blades and now - t.blades[0][0] > self.window_s:
            t.blades.popleft()
        if len(t.blades) < self.min_samples:
            return
        d_m = float(np.mean([b[1] for b in t.blades]))
        e_m = float(np.mean([b[2] for b in t.blades]))
        mo = float(np.mean([b[3] for b in t.blades]))
        t.metric = round(max(d_m, e_m), 3)                      # blade visibility (0 = blurred away)
        visible = d_m >= self.blades_visible[0] or e_m >= self.blades_visible[1]
        gone = d_m <= self.blades_gone[0] and e_m <= self.blades_gone[1]
        if mo >= self.fan_on_ratio or gone:
            t.state = ON                                       # slow fan: blades move / fast fan: blades blurred
        elif visible:
            t.state = OFF                                      # crisp, still blades
        elif t.state == UNKNOWN:
            t.state = OFF

    def _push(self, dq: deque, now: float, value: float) -> None:
        dq.append((now, value))
        while dq and now - dq[0][0] > self.window_s:
            dq.popleft()

    def _decide_zone_fan(self, z: dict, run: dict) -> None:
        """ON if the fan is moving OR its blades have blurred away compared with how it looks when stopped."""
        if len(run["dark"]) < self.min_samples:
            return
        dark = float(np.mean([v for _, v in run["dark"]]))
        motion = float(np.mean([v for _, v in run["motion"]])) if run["motion"] else 0.0
        run["dark_now"], run["metric"] = dark, motion
        off, on = z.get("off"), z.get("on")
        thr = (off + on) / 2 if off is not None and on is not None else (off * 0.5 if off is not None else None)
        looks_on = thr is not None and dark < thr
        run["state"] = ON if (motion >= self.fan_on_ratio or looks_on) else OFF

    def _add(self, t: Track, now: float, value: float) -> None:
        t.samples.append((now, value))
        while t.samples and now - t.samples[0][0] > self.window_s:
            t.samples.popleft()

    def _decide_fan(self, t: Track) -> None:
        if len(t.samples) < self.min_samples:
            return
        t.metric = float(np.mean([v for _, v in t.samples]))
        if t.metric >= self.fan_on_ratio:
            t.state = ON
        elif t.metric <= self.fan_on_ratio * 0.6 or t.state == UNKNOWN:   # hysteresis (OFF below ~5%)
            t.state = OFF

    def _light_value(self, gray: np.ndarray, box: tuple[int, int, int, int]) -> float:
        """1.0 when the box looks lit, 0.0 when not (averaged over the window later)."""
        x1, y1, x2, y2 = box
        roi = gray[y1:y2, x1:x2]
        pad_x, pad_y = int((x2 - x1) * 0.4), int((y2 - y1) * 0.4)
        bx1, by1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        bx2, by2 = min(gray.shape[1], x2 + pad_x), min(gray.shape[0], y2 + pad_y)
        big = gray[by1:by2, bx1:bx2]
        ring_area = big.size - roi.size
        ring_mean = (float(big.sum()) - float(roi.sum())) / ring_area if ring_area > 0 else float(roi.mean())
        sat = float((roi >= 245).mean())
        peak = float(np.percentile(roi, 95))
        contrast = float(roi.mean()) - ring_mean
        # must be BRIGHTER THAN ITS SURROUNDINGS: a bright window / white wall is not a lit lamp
        # lit = it holds a glowing spot (the box may be much bigger than the light itself), or the box as a whole
        # is brighter than around it and partly blown out
        glowing = self._glow is not None and self._glow.shape == gray.shape and \
            int(self._glow[y1:y2, x1:x2].sum()) >= self.glow_min_area
        # over-exposed picture: the light has merged with a bright ceiling and has little headroom left to
        # "glow", but it is still clearly brighter than the area beside it and holds near-white pixels
        side_med = self._side_median(gray, box)
        brighter = float(np.median(roi)) >= side_med + 20 and float(np.percentile(roi, 99)) >= 230
        lit = glowing or brighter or (contrast >= self.light_contrast and
                                      (sat >= self.light_sat_ratio or peak >= self.light_peak))
        return 1.0 if lit else 0.0

    def _decide_light(self, t: Track) -> None:
        if len(t.samples) < self.min_samples:
            return
        t.metric = float(np.mean([v for _, v in t.samples]))
        if t.metric >= 0.6:
            t.state = ON
        elif t.metric <= 0.4 or t.state == UNKNOWN:
            t.state = OFF

    # ---------------------------------------------------------------- output
    def snapshot(self) -> list[dict]:
        with self._lock:
            out = [{"id": t.id, "cls": t.cls, "state": t.state, "conf": round(t.conf, 2),
                    "bbox": list(t.bbox), "metric": round(t.metric, 3)}
                   for t in self.tracks if t.hits >= self.confirm_hits or t.ttl]   # a glow light shows at once
            for z in self.zones:
                run = self._zrun.get(z["id"], {})
                out.append({"id": z["id"], "cls": z["cls"], "state": run.get("state", UNKNOWN), "conf": 1.0,
                            "bbox": list(z["bbox"]), "metric": round(run.get("metric", 0.0), 3), "zone": True})
            return out
