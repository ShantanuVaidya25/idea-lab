"""Alert & notification system.

Every alert goes to:
  1. the dashboard (alert panel + banner)
  2. the SQLite log
  3. optional Windows desktop popup (plyer)
  4. optional mobile push message via Telegram bot

A cooldown stops the same alert from spamming the phone.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

log = logging.getLogger("alerts")

LEVELS = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}

# the ONLY alerts that produce a popup / phone message (everything else is just logged)
TURN_OFF_ALERTS = {"VISUAL_ENERGY_WASTE", "ENERGY_WASTE", "AC_LEFT_ON", "AC_ON", "AC_OFF"}

# readable names for popups / phone messages / dashboard banners (the raw code is still logged)
TITLES = {
    "TEST_ALERT": "Test notification (notifications are working)",
    "VISUAL_ENERGY_WASTE": "Room empty - fan/light still ON",
    "ENERGY_WASTE": "Room empty - appliance relay still ON",
    "AC_LEFT_ON": "AC may be left ON in an empty room",
    "AC_ON": "AC switched ON",
    "AC_OFF": "AC switched OFF",
    "AUTO_SHUTDOWN": "Automatically switched OFF",
    "VACANCY_CONFIRMED": "Classroom confirmed empty",
    "HIGH_TEMPERATURE": "High room temperature",
    "SENSOR_FAULT": "Temperature sensor fault",
    "ESP32_DISCONNECTED": "ESP32 controller disconnected",
    "CAMERA_FAULT": "Camera / detection problem",
    "RELAY_FAULT": "Relay not responding",
}


@dataclass
class Alert:
    ts: float
    level: str      # INFO / WARNING / CRITICAL
    code: str       # e.g. ENERGY_WASTE
    message: str
    custom_title: str = ""      # action-style title, e.g. "Turn OFF: Fan, Light"

    @property
    def time_str(self) -> str:
        return datetime.fromtimestamp(self.ts).strftime("%Y-%m-%d %H:%M:%S")

    @property
    def title(self) -> str:
        return self.custom_title or TITLES.get(self.code, self.code.replace("_", " ").capitalize())

    def text(self) -> str:
        return f"[{self.level}] {self.title}\n{self.message}\n{self.time_str}"


# ---------------------------------------------------------------- notifiers
class TelegramNotifier:
    """Sends alert text to your phone using a Telegram bot (free)."""

    def __init__(self, bot_token: str, chat_id: str, min_level: str = "WARNING"):
        self.url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        self.chat_id = chat_id
        self.min_level = LEVELS.get(min_level, 1)

    def __call__(self, alert: Alert) -> None:
        if LEVELS.get(alert.level, 0) < self.min_level:
            return
        import requests
        text = "Smart Classroom\n" + alert.text()
        r = requests.post(self.url, data={"chat_id": self.chat_id, "text": text}, timeout=10)
        r.raise_for_status()


class DesktopNotifier:
    """Windows notification popup (best effort)."""

    def __call__(self, alert: Alert) -> None:
        from plyer import notification
        notification.notify(title=f"{alert.level.capitalize()}: {alert.title}", message=alert.message[:250],
                            app_name="Smart Classroom", timeout=8)


# ------------------------------------------------------------------ manager
class AlertManager:
    def __init__(self, cooldown_s: float = 300, max_keep: int = 200):
        self.cooldown_s = cooldown_s
        self.history: deque[Alert] = deque(maxlen=max_keep)
        self.active: dict[str, Alert] = {}          # conditions currently true
        self._last_sent: dict[str, float] = {}
        self.notifiers: list[Callable[[Alert], None]] = []
        self.on_alert: list[Callable[[Alert], None]] = []  # e.g. DB logger
        self.notify_codes: set[str] = set(TURN_OFF_ALERTS)   # codes allowed to notify (popup / phone)
        self._lock = threading.Lock()

    def raise_alert(self, code: str, level: str, message: str, now: float | None = None, title: str = "") -> bool:
        """Raise an alert. Returns True if it was emitted (not suppressed by cooldown)."""
        now = time.time() if now is None else now
        with self._lock:
            alert = Alert(now, level, code, message, title)
            if level != "INFO":           # INFO alerts are one-shot events, not conditions
                self.active[code] = alert
            last = self._last_sent.get(code)
            if last is not None and now - last < self.cooldown_s:
                return False
            self._last_sent[code] = now
            self.history.appendleft(alert)
        for cb in self.on_alert:
            try:
                cb(alert)
            except Exception as e:  # never let logging break the app
                log.warning("alert callback failed: %s", e)
        if code not in self.notify_codes:
            return True                 # logged only, no popup / phone message
        for n in self.notifiers:  # send in background so we never block
            threading.Thread(target=self._safe_send, args=(n, alert), daemon=True).start()
        return True

    def clear(self, code: str) -> None:
        """Condition no longer true -> remove from active list (and allow re-alert)."""
        with self._lock:
            if self.active.pop(code, None) is not None:
                self._last_sent.pop(code, None)

    @staticmethod
    def _safe_send(notifier: Callable[[Alert], None], alert: Alert) -> None:
        try:
            notifier(alert)
        except Exception as e:
            log.warning("notification failed (%s): %s", type(notifier).__name__, e)


def build_alert_manager(cfg: dict) -> AlertManager:
    a = cfg.get("alerts", {})
    mgr = AlertManager(cooldown_s=a.get("cooldown_s", 300))
    mgr.notify_codes = set(a.get("notify_only", TURN_OFF_ALERTS))
    if a.get("desktop_popup"):
        mgr.notifiers.append(DesktopNotifier())
    tg = a.get("telegram", {})
    if tg.get("enabled") and tg.get("bot_token") and tg.get("chat_id"):
        mgr.notifiers.append(TelegramNotifier(tg["bot_token"], str(tg["chat_id"]), tg.get("min_level", "WARNING")))
    return mgr
