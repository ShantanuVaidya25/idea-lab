# Quick Start: Smart Classroom

How to start the dashboard. The full documentation is in `README.md`.

## 1. One-time setup (PowerShell)

```powershell
cd "C:\Users\shant\Downloads\idea lab\SmartClassroom\SmartClassroom"
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
```

Use **Python 3.9 to 3.12** if you can. The project was tested on 3.13 in simulation mode.
`ultralytics` (YOLO) is only needed for real camera detection, so you can skip it for the demo.

## 2. Check that it works

```powershell
python -m pytest                  # expect: 27 passed
```

## 3. Start the dashboard

```powershell
streamlit run app.py
```

Open **http://localhost:8501** in your browser. Stop it with `Ctrl+C`.

## Demo mode (no camera, no ESP32)

Edit `config.yaml`:

```yaml
esp32:
  port: ""            # empty = simulated ESP32
vision:
  simulate: true      # no camera or YOLO needed
occupancy:
  vacancy_confirm_s: 20   # fast demo (default 300)
ac_inference:
  observation_s: 30       # fast demo (default 120)
```

Restart the app, then in the sidebar:

1. Turn on **Automation enabled**.
2. Set **Simulated people in room** to `0`. The state goes `VACANCY_PENDING` then `VACANT`.
3. Tick **Simulated AC running**. After the observation window you get a red `AC_LEFT_ON` alert.
4. Set people back to `3`. The room returns to `OCCUPIED`.

## Real hardware

| Item | What to do |
|---|---|
| Webcam + YOLO | `vision.simulate: false`, set `vision.model_path` (e.g. `models/best.pt`), install `ultralytics` |
| ESP32 | Upload `firmware/esp32_controller/esp32_controller.ino`, run `python tools/list_ports.py`, set `esp32.port: "COM5"` |
| Telegram alerts | Put `bot_token` and `chat_id` under `alerts.telegram` in `config.yaml` |

Close the Arduino Serial Monitor before starting the app (only one program can use the COM port).

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` not found | Activate the venv first: `.\.venv\Scripts\Activate.ps1` |
| `Cannot open camera 0` | Set `vision.source: 1`, or close Zoom/Teams |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
| Settings changes ignored | Config is read at startup, so restart the app |
