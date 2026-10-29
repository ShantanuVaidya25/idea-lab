"""Webcam + YOLO person / fan / light detection (runs in its own thread)."""
from __future__ import annotations

import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import dataclass, field

import numpy as np

from core.appliances import ApplianceTracker
from core.config import ROOT

log =logging.getLogger("vision")

COLORS = {"person": (0, 200, 0)}
STATE_COLORS = {"ON": (0, 200, 0), "OFF": (0, 0, 255), "UNKNOWN": (160, 160, 160)}   # BGR


@dataclass
class VisionResult:
    ok: bool = False
    error: str = ""
    person_count: int | None = None          # None = no valid detection
    appliances: list[dict] = field(default_factory=list)  # {cls, conf, bbox, ts}
    frame: np.ndarray | None = None          # annotated RGB frame
    ts: float = 0.0
    fps: float = 0.0


def find_glow_spots(gray: np.ndarray, cfg: dict | None = None,
                    exclude: list | None = None) -> list[tuple[int, int, int, int]]:
    """Lit lights found by brightness alone (the detector model cannot see a lit panel - it is just glare).

    A spot counts if it is very bright AND clearly brighter than the area around it, sits in the upper part of
    the picture (ceiling / upper walls) and is not window-sized. Bright spots touching each other are one light;
    a dimmer neighbour (its reflection on the wall) is dropped. Spots on a detected person (a white shirt
    in bright light) are ignored: pass the person boxes as `exclude`.
    """
    import cv2
    from core.appliances import glow_mask
    c = cfg or {}
    H, W = gray.shape[:2]
    max_area = c.get("glow_max_frac", 0.02) * H * W
    n, lab, st, _ = cv2.connectedComponentsWithStats(glow_mask(gray, c).astype(np.uint8), 8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, a = (int(v) for v in st[i])
        if a < c.get("glow_min_area", 20) or a > max_area:
            continue
        if min(w, h) < 5 and (x == 0 or y == 0 or x + w == W or y + h == H):
            continue                                   # a sliver of a light outside the picture
        blobs.append({"box": [x, y, x + w, y + h], "mean": float(gray[lab == i].mean())})
    # cluster blobs that touch (within 20 px); keep only the bright cores of each cluster
    clusters: list[list[dict]] = []
    for b in sorted(blobs, key=lambda b: -b["mean"]):
        for cl in clusters:
            if any(b["box"][0] - 20 <= o["box"][2] and o["box"][0] - 20 <= b["box"][2] and
                   b["box"][1] - 20 <= o["box"][3] and o["box"][1] - 20 <= b["box"][3] for o in cl):
                cl.append(b)
                break
        else:
            clusters.append([b])
    pad = c.get("glow_pad", 1.8)
    out = []
    for cl in clusters:
        top = max(o["mean"] for o in cl)
        core = [o["box"] for o in cl if o["mean"] >= top - 3]          # dimmer neighbours = reflections
        x1, y1 = min(b[0] for b in core), min(b[1] for b in core)
        x2, y2 = max(b[2] for b in core), max(b[3] for b in core)
        cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 8) * pad / 2, max(y2 - y1, 8) * pad / 2
        if any(e[0] <= cx <= e[2] and e[1] <= cy <= e[3] for e in exclude or []):
            continue
        out.append((int(max(0, cx - hw)), int(max(0, cy - hh)), int(min(W, cx + hw)), int(min(H, cy + hh))))
    return out


def next_exposure(current: int, median: float, target: tuple[float, float], lo: int, hi: int) -> int:
    """One step of the exposure controller: darker if the picture is too bright, brighter if too dark."""
    if median > target[1] and current > lo:
        return current - 1
    if median < target[0] and current < hi:
        return current + 1
    return current


def center_in_zones(box: tuple, zones: list) -> bool:
    """True if the centre of `box` lies inside any [x1, y1, x2, y2] zone (used to ignore posters / photos)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(z[0] <= cx <= z[2] and z[1] <= cy <= z[3] for z in zones or [])


def resolve_classes(model_names: dict[int, str], aliases: dict[str, list[str]]) -> dict[str, list[int]]:
    """Map logical classes (person/fan/light) to the model's REAL class ids."""
    lower = {i: n.lower().strip() for i, n in model_names.items()}
    out: dict[str, list[int]] = {}
    for logical, names in aliases.items():
        wanted = {n.lower().strip() for n in names}
        out[logical] = [i for i, n in lower.items() if n in wanted]
    return out


class _StreamHandler(BaseHTTPRequestHandler):
    """Serves the annotated camera frames as an MJPEG stream (a real live video for <img src=...>)."""

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/capture"):
            self._capture()
            return
        if not self.path.startswith("/stream"):
            self.send_error(404)
            return
        vision = self.server.vision
        # Wait briefly for camera warmup so the browser doesn't receive an empty stream
        wait_end = time.time() + 5.0
        while vision._jpeg is None and time.time() < wait_end and not vision._stop.is_set():
            time.sleep(0.05)
        if vision._jpeg is None:
            self.send_error(503, "Camera still initializing")
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.end_headers()
        last = None
        try:
            while not vision._stop.is_set():
                jpg = vision._jpeg
                if jpg is None or jpg is last:
                    time.sleep(0.02)
                    continue
                last = jpg
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                self.wfile.write(jpg + b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionError, OSError):
            pass

    def _capture(self) -> None:
        """One RAW camera frame (no boxes drawn) plus what the system detected at that moment (for the test set)."""
        import base64
        import json

        import cv2
        vision = self.server.vision
        raw, det = vision._raw, vision._det
        if raw is None:
            self.send_error(503, "no camera frame yet")
            return
        ok, buf = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
        body = json.dumps({
            "ts": time.time(),
            "width": int(raw.shape[1]), "height": int(raw.shape[0]),
            "image_jpg_b64": base64.b64encode(buf.tobytes()).decode() if ok else None,
            "persons": [[*box, round(conf, 3)] for cls, conf, box in (det or {}).get("boxes", []) if cls == "person"],
            "appliances": vision.tracker.snapshot(),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence per-request logging
        pass


class Vision:
    def __init__(self, cfg: dict):
        v = cfg["vision"]
        self.cfg = v
        self.result = VisionResult(error="Loading YOLO model / opening camera...")
        self.loading = True
        self._det: dict | None = None
        self._raw = None                      # newest raw camera frame
        self._jpeg: bytes | None = None       # newest annotated frame, JPEG-encoded for the stream
        self._server: ThreadingHTTPServer | None = None
        self.stream_url: str | None = None
        self.warnings: list[str] = []
        self.model_names: dict[int, str] = {}
        self.class_map: dict[str, list[int]] = {}
        self.device = "cpu"
        self.ap_cfg: dict = v.get("appliances", {})
        self.tracker = ApplianceTracker({**self.ap_cfg,
                                         "zones_path": str(ROOT / self.ap_cfg.get("zones_path", "data/zones.json"))})
        self._world = None                    # open-vocabulary model that finds fans / lights
        self._world_prompts: dict[str, str] = {}   # prompt text -> "fan" | "light"
        self._last_world = 0.0
        self._last_glow = 0.0
        self._exposure: int | None = None          # current manual exposure (None = camera auto)
        self._last_exposure_step = 0.0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._start_stream(int(v.get("stream_port", 8502)))
        threading.Thread(target=self._run, daemon=True, name="vision").start()

    def _start_stream(self, port: int) -> None:
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", port), _StreamHandler)
        except OSError as e:
            log.warning("Live stream disabled (port %s busy: %s)", port, e)
            return
        srv.daemon_threads = True
        srv.vision = self
        self._server = srv
        self.stream_url = f"http://127.0.0.1:{port}/stream"
        threading.Thread(target=srv.serve_forever, daemon=True, name="vision-stream").start()

    # ---------------------------------------------------------------- setup
    def _load_model(self):
        from ultralytics import YOLO
        model = YOLO(self.cfg["model_path"])
        self.model_names = dict(model.names)
        self.class_map = resolve_classes(self.model_names, self.cfg.get("class_aliases", {}))
        if not self.class_map.get("person"):
            # without a person class we cannot judge occupancy -> refuse (never report fake 0 people)
            raise RuntimeError(f"Model has no person class. Classes: {self.model_names}")
        dev = self.cfg.get("device", "auto")
        if dev == "auto":
            try:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev = "cpu"
        self.device = dev
        log.info("Model loaded: %s on %s, classes=%s", self.cfg["model_path"], dev, self.model_names)

        # Fans / lights: use the main model if it knows them, otherwise an open-vocabulary YOLO-World model
        missing = [lg for lg in ("fan", "light") if not self.class_map.get(lg)]
        if missing and self.ap_cfg.get("enabled", True):
            try:
                self._load_world(missing)
            except Exception as e:
                log.exception("appliance model failed")
                self.warnings.append(f"Fan/light model failed to load ({e}). Fan/light detection is DISABLED.")
        elif missing:
            self.warnings.append(f"Model has no {'/'.join(missing)} class and vision.appliances is disabled.")
        return model

    def _load_world(self, logicals: list[str]) -> None:
        from ultralytics import YOLO
        prompts = self.ap_cfg.get("prompts", {"fan": ["ceiling fan", "table fan"],
                                              "light": ["tube light", "light bulb", "lamp"]})
        self._world_prompts = {p.lower(): lg for lg in logicals for p in prompts.get(lg, [])}
        world = YOLO(self.ap_cfg.get("model_path", "yolov8s-worldv2.pt"))
        world.set_classes(list(self._world_prompts))
        self._world = world
        log.info("Appliance model loaded, prompts=%s", list(self._world_prompts))

    # ----------------------------------------------------------------- loop
    def _run(self) -> None:
        import cv2
        try:
            model = self._load_model()
        except Exception as e:
            self._set_error(f"Model load failed: {e}")
            return

        threading.Thread(target=self._detect_loop, args=(model,), daemon=True, name="vision-detect").start()
        cap = None
        try:
            while not self._stop.is_set():
                if cap is None or not cap.isOpened():
                    cap = self._open_camera(cv2)
                    if cap is None:
                        self._set_error(f"Cannot open camera source {self.cfg.get('source')} "
                                        f"(no backend gave a usable picture - close other apps using the camera)")
                        self._stop.wait(3)
                        continue
                ok, frame = cap.read()
                if not ok:
                    self._set_error("Camera frame read failed")
                    cap.release()
                    cap = None
                    self._stop.wait(1)
                    continue
                self._raw = frame
                self._exposure_control(cap, cv2, frame, time.time())
                self.tracker.update_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), time.time())
                self._publish(frame.copy(), cv2)      # live video at camera speed, independent of YOLO speed
        finally:
            if cap is not None:
                cap.release()

    def _apply_exposure(self, cap, cv2, value) -> None:
        """Fixed exposure instead of the camera's auto-exposure, which over-brightens the ceiling whenever a face
        is close to the lens until the lights wash out (and cannot be told from the ceiling)."""
        if value in (None, "auto"):
            return
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)       # DirectShow: manual
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)          # Media Foundation: manual
            cap.set(cv2.CAP_PROP_EXPOSURE, int(value))
            self._exposure = int(value)
            log.info("Camera exposure fixed at %s", value)
        except Exception as e:                              # camera without manual exposure: keep auto
            log.warning("Manual exposure not supported (%s)", e)

    def _exposure_control(self, cap, cv2, frame, now: float) -> None:
        """Keep the picture's median brightness in vision.exposure_target by stepping the exposure (every 2 s)."""
        if self._exposure is None or now - self._last_exposure_step < 2.0:
            return
        self._last_exposure_step = now
        lo, hi = self.cfg.get("exposure_range", [-8, -1])
        target = self.cfg.get("exposure_target", [70, 150])
        median = float(np.median(frame[::8, ::8]))
        new = next_exposure(self._exposure, median, target, lo, hi)
        if new != self._exposure:
            cap.set(cv2.CAP_PROP_EXPOSURE, new)
            self._exposure = new

    def _open_camera(self, cv2):
        """Open the camera, trying each backend until one gives a real (non-black) picture."""
        raw_src = self.cfg.get("source", 0)
        sources = [raw_src]
        if isinstance(raw_src, int) and raw_src != 0:
            sources.append(0)  # fallback to camera 0 if chosen index doesn't exist

        for src in sources:
            if isinstance(src, int) and sys.platform == "win32":
                backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]     # either one can get stuck after a hard kill
            else:
                backends = [cv2.CAP_ANY]
            for be in backends:
                cap = cv2.VideoCapture(src, be)
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always the newest frame (live, no lag)
                # webcams deliver black frames until auto-exposure settles: discard them, then check we see light
                good = False
                end = time.time() + 3.0
                while time.time() < end and not self._stop.is_set():
                    ok, f = cap.read()
                    if ok and time.time() > end - 1.0 and f.max() > 10:     # a dark room still has sensor noise; a dead backend is pure 0
                        good = True
                        break
                if good:
                    log.info("Camera %s opened with backend %s", src, cap.getBackendName())
                    self._exposure = None
                    self._apply_exposure(cap, cv2, self.cfg.get("exposure", "auto"))
                    return cap
                log.warning("Camera %s backend %s gave no usable picture", src, be)
                cap.release()
        return None

    def _detect_loop(self, model) -> None:
        """YOLO runs here on the newest frame, so a slow CPU never freezes the live video."""
        last = None
        while not self._stop.is_set():
            frame = self._raw
            if frame is None or frame is last:
                self._stop.wait(0.05)
                continue
            last = frame
            start = time.time()
            try:
                self._detect(model, frame, start)
            except Exception as e:
                self._set_error(f"Inference failed: {e}")
                self._stop.wait(1)
                continue
            self._stop.wait(max(0.0, self.cfg.get("inference_interval_s", 0.5) - (time.time() - start)))

    def _detect(self, model, frame, now: float) -> None:
        t0 = time.time()
        imgsz = self.cfg.get("imgsz") or (320 if self.device == "cpu" else 640)
        wanted = sorted({i for ids in self.class_map.values() for i in ids})
        res = model.predict(frame, conf=self.cfg.get("confidence", 0.45), imgsz=imgsz,
                            device=self.device, classes=wanted or None, verbose=False)[0]
        id_to_logical = {i: lg for lg, ids in self.class_map.items() for i in ids}
        boxes, cands, persons = [], [], 0
        for box in res.boxes:
            cid = int(box.cls[0])
            logical = id_to_logical.get(cid)
            if logical is None:
                continue
            xyxy = tuple(int(x) for x in box.xyxy[0].tolist())
            conf = float(box.conf[0])
            if logical == "person":
                if center_in_zones(xyxy, self.cfg.get("ignore_person_zones")):
                    continue                     # e.g. a photo / poster on the wall that looks like a person
                persons += 1
                boxes.append(("person", conf, xyxy))
            else:                      # fan / light from the main (custom) model
                cands.append((logical, conf, xyxy))
        update_tracks = any(self.class_map.get(lg) for lg in ("fan", "light"))   # custom model sees them every pass

        # open-vocabulary fan/light finder runs less often (fans/lights don't move around)
        if self._world is not None and now - self._last_world >= self.ap_cfg.get("interval_s", 2.0):
            self._last_world = now
            min_conf = {"fan": self.ap_cfg.get("fan_confidence", 0.25), "light": self.ap_cfg.get("light_confidence", 0.25)}
            min_frac = {"fan": self.ap_cfg.get("fan_min_size", 0.02), "light": self.ap_cfg.get("light_min_size", 0.002)}
            max_frac = self.ap_cfg.get("max_size", 0.5)
            fh, fw = frame.shape[:2]
            wres = self._world.predict(frame, conf=min(min_conf.values()), imgsz=imgsz,
                                       device=self.device, verbose=False)[0]
            for b in wres.boxes:
                logical = self._world_prompts.get(str(self._world.names[int(b.cls[0])]).lower())
                if not logical:
                    continue
                box = tuple(int(x) for x in b.xyxy[0].tolist())
                frac = (box[2] - box[0]) * (box[3] - box[1]) / (fw * fh)       # share of the picture it covers
                if float(b.conf[0]) >= min_conf[logical] and min_frac[logical] <= frac <= max_frac:
                    cands.append((logical, float(b.conf[0]), box))
            update_tracks = True
        # lit lights by glow (runs with the open-vocabulary finder, or on its own every interval)
        if self.ap_cfg.get("glow_lights", True) and (update_tracks or now - self._last_glow >= self.ap_cfg.get("interval_s", 2.0)):
            self._last_glow = now
            import cv2
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cands += [("light", 0.9, box, "glow") for box in find_glow_spots(gray, self.ap_cfg, [b[2] for b in boxes])]
            update_tracks = True
        if update_tracks:
            self.tracker.update_detections(cands, now)

        took = time.time() - t0
        if took > 3:
            log.warning("Slow detection pass: %.1f s (CPU busy?)", took)
        self._det = {"boxes": boxes, "persons": persons, "ts": time.time(),   # time the result was READY
                     "fps": round(1 / max(took, 1e-3), 1)}

    def _publish(self, frame, cv2) -> None:
        """Draw the latest detections on the current live frame; update the stream and the result."""
        d = self._det
        appliances = self.tracker.snapshot()
        for a in appliances:        # fans / lights: green = ON, red = OFF, grey = still measuring
            x1, y1, x2, y2 = a["bbox"]
            c = STATE_COLORS[a["state"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            cv2.putText(frame, f"{a['cls']} {a['state']}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        if d is not None:
            for logical, conf, (x1, y1, x2, y2) in d["boxes"]:
                c = COLORS.get(logical, (200, 200, 200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                cv2.putText(frame, f"{logical} {conf:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
            cv2.putText(frame, f"People: {d['persons']}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self._jpeg = buf.tobytes()
        if d is None:
            return
        if time.time() - d["ts"] > 10:      # detector stalled: never report a stale count as valid
            self._set_error("Detection stalled (no YOLO result for 10 s)")
            return
        self.loading = False
        with self._lock:
            self.result = VisionResult(ok=True, person_count=d["persons"], appliances=appliances,
                                       frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ts=d["ts"], fps=d["fps"])

    def _set_error(self, msg: str) -> None:
        log.error(msg)
        self.loading = False
        with self._lock:
            self.result = VisionResult(ok=False, error=msg, ts=time.time())

    def snapshot(self) -> VisionResult:
        with self._lock:
            return self.result

    def close(self) -> None:
        self._stop.set()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


class SimulatedVision:
    """No camera: person count is set from the dashboard (clearly marked SIMULATED)."""

    def __init__(self, cfg: dict):
        self.warnings = ["Vision is SIMULATED - no camera / YOLO in use."]
        self.model_names: dict[int, str] = {}
        self.class_map: dict[str, list[int]] = {}
        self.device = "sim"
        self.person_count = 0

    def snapshot(self) -> VisionResult:
        frame = np.full((360, 640, 3), 40, dtype=np.uint8)
        return VisionResult(ok=True, person_count=self.person_count,
                            frame=frame, ts=time.time())

    def close(self) -> None:
        pass
