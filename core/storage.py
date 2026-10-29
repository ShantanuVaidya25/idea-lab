"""SQLite logging (schema follows report Table 4.2 + an events table)."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path


class Storage:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._con.executescript("""
            CREATE TABLE IF NOT EXISTS log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, occupancy_count INTEGER, state TEXT,
                temp_c REAL, sensor_ok INTEGER, simulated INTEGER,
                appliance_status TEXT, ac_inferred_state TEXT);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT, kind TEXT, level TEXT, message TEXT);
            """)
            # columns added later: add them to an existing database (keeps old data)
            have = {r[1] for r in self._con.execute("PRAGMA table_info(log)")}
            for col, typ in (("humidity", "REAL"), ("visual_status", "TEXT"), ("ac_monitor", "TEXT"),
                             ("region_temp_1", "REAL"), ("region_temp_2", "REAL"),        # report Table 4.2
                             ("region_temp_3", "REAL"), ("region_temp_4", "REAL")):
                if col not in have:
                    self._con.execute(f"ALTER TABLE log ADD COLUMN {col} {typ}")
            self._con.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log_status(self, count, state, temp, sensor_ok, simulated, appliance_status, ac_state,
                   humidity=None, visual_status="", ac_monitor="", region_temps=None) -> None:
        r = (list(region_temps or []) + [None] * 4)[:4]
        with self._lock:
            self._con.execute(
                "INSERT INTO log (timestamp, occupancy_count, state, temp_c, sensor_ok, simulated,"
                " appliance_status, ac_inferred_state, humidity, visual_status, ac_monitor,"
                " region_temp_1, region_temp_2, region_temp_3, region_temp_4)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self._now(), count, state, temp, int(bool(sensor_ok)), int(simulated), appliance_status, ac_state,
                 humidity, visual_status, ac_monitor, *r))
            self._con.commit()

    def log_event(self, kind: str, level: str, message: str) -> None:
        with self._lock:
            self._con.execute("INSERT INTO events (timestamp, kind, level, message) VALUES (?,?,?,?)",
                              (self._now(), kind, level, message))
            self._con.commit()

    def recent_events(self, n: int = 100) -> list[tuple]:
        with self._lock:
            return self._con.execute(
                "SELECT timestamp, kind, level, message FROM events ORDER BY id DESC LIMIT ?", (n,)).fetchall()

    # ------------------------------------------------------------ log viewer
    def query_events(self, levels: list[str] | None = None, kinds: list[str] | None = None,
                     text: str = "", since: str | None = None, limit: int = 1000) -> list[tuple]:
        """Filtered event log, newest first. `since` is 'YYYY-MM-DD HH:MM:SS'."""
        sql, params = "SELECT timestamp, kind, level, message FROM events WHERE 1=1", []
        if levels:
            sql += f" AND level IN ({','.join('?' * len(levels))})"
            params += levels
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params += kinds
        if text:
            sql += " AND (message LIKE ? OR kind LIKE ?)"
            params += [f"%{text}%", f"%{text}%"]
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return self._con.execute(sql, params).fetchall()

    def event_kinds(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._con.execute("SELECT DISTINCT kind FROM events ORDER BY kind")]

    def event_counts(self, since: str | None = None) -> dict[str, int]:
        sql, params = "SELECT level, COUNT(*) FROM events", []
        if since:
            sql += " WHERE timestamp >= ?"
            params.append(since)
        with self._lock:
            return dict(self._con.execute(sql + " GROUP BY level", params).fetchall())

    def query_status(self, since: str | None = None, limit: int = 3000) -> list[tuple]:
        """Periodic status snapshots (people, state, temperature, appliances), newest first."""
        sql, params = ("SELECT timestamp, occupancy_count, state, temp_c, sensor_ok, simulated, appliance_status, "
                       "ac_inferred_state FROM log"), []
        if since:
            sql += " WHERE timestamp >= ?"
            params.append(since)
        sql += " ORDER BY log_id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return self._con.execute(sql, params).fetchall()

    def query_status_full(self, since: str | None = None, limit: int = 200000) -> list[tuple]:
        """All status columns (for the Analysis page), newest first."""
        sql, params = ("SELECT timestamp, occupancy_count, state, temp_c, sensor_ok, simulated, appliance_status, "
                       "ac_inferred_state, humidity, visual_status, ac_monitor, "
                       "region_temp_1, region_temp_2, region_temp_3, region_temp_4 FROM log"), []
        if since:
            sql += " WHERE timestamp >= ?"
            params.append(since)
        sql += " ORDER BY log_id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return self._con.execute(sql, params).fetchall()
