# ⚡ Smart Classroom Electricity Management System
> **An Edge-AI & IoT Fusion for Automated Classroom Energy Conservation, Post-Vacancy Thermal AC Inference, and Facility Telemetry**

---

## 📑 Quick Navigation for Presentation
- [1. Executive Summary & 30-Second Pitch](#1-executive-summary--30-second-pitch)
- [2. Problem Statement & Motivation](#2-problem-statement--motivation)
- [3. System Architecture & Block Diagram](#3-system-architecture--block-diagram)
- [4. Key Technical Innovations & Algorithms](#4-key-technical-innovations--algorithms)
  - [4.1 Debounced Occupancy Finite State Machine (FSM)](#41-debounced-occupancy-finite-state-machine-fsm)
  - [4.2 Non-Invasive Post-Vacancy AC Thermal Inference](#42-non-invasive-post-vacancy-ac-thermal-inference)
  - [4.3 Continuous Trend Slope Monitoring (Least-Squares Regression)](#43-continuous-trend-slope-monitoring-least-squares-regression)
  - [4.4 SwitchGuard Hardware Protection](#44-switchguard-hardware-protection)
- [5. Hardware Specifications & Circuit Design](#5-hardware-specifications--circuit-design)
- [6. Software Architecture & Technology Stack](#6-software-architecture--technology-stack)
- [7. Energy Analytics, Tariff Calculation & ROI](#7-energy-analytics-tariff-calculation--roi)
- [8. Step-by-Step Live Demo Presentation Script](#8-step-by-step-live-demo-presentation-script)
- [9. Viva & Staff Q&A Defense Guide](#9-viva--staff-qa-defense-guide)
- [10. Quickstart & Deployment Guide](#10-quickstart--deployment-guide)

---

## 1. Executive Summary & 30-Second Pitch

> **"Good morning/afternoon, Respected Faculty and Staff.**  
> Classrooms and university lecture halls waste thousands of kilowatt-hours every semester because lights, fans, and high-wattage air conditioners are routinely left running after lectures end. Traditional PIR motion sensors fail because students sitting still reading or taking exams get plunged into darkness, and timers fail because academic schedules are dynamic.  
> 
> Our project, **Smart Classroom Electricity Management**, solves this with a two-tier edge computing architecture:
> 1. **Computer Vision (YOLOv8)** continuously detects human occupancy across **4 spatial classroom quadrants** with intelligent debouncing and fail-safe logic.
> 2. **Microcontroller Hardware (ESP32)** coordinates an optocoupled relay system to automatically switch off redundant lighting and ventilation upon verified vacancy.
> 3. **Non-Invasive Thermal AC Inference**: Without voiding AC warranties or tapping hazardous 230V mains, our system monitors ambient regional temperatures ($T_{final} - T_{initial}$). If room temperature continues dropping after students leave, it identifies that an AC was forgotten and immediately pushes actionable alerts to the facility manager’s phone via Telegram, desktop notifications, and a real-time Streamlit dashboard."

---

## 2. Problem Statement & Motivation

### The Campus Energy Crisis
1. **Human Negligence**: In colleges, the last student or professor leaving a classroom rarely checks every single fan, light, and air conditioner switch.
2. **High-Power Inefficiencies**: While LED lights consume 20W–40W, classroom split ACs consume **1,500W to 3,000W each**. Leaving even two ACs running unattended in an empty hall overnight costs an institution hundreds of dollars and wastes massive amounts of grid power.

### Why Existing Solutions Fail
| Solution | Working Principle | Why It Fails in Academic Settings |
|---|---|---|
| **Timer Switches** | Turns power off after fixed schedule (e.g., 5:00 PM) | Disrupts extra classes, seminars, or student study sessions; wastes power if class finishes 45 minutes early. |
| **PIR Motion Sensors** | Detects IR movement | **False Negatives**: Stationary students reading, coding, or taking tests do not move enough; lights shut off while people are inside. |
| **Current / CT Clamps** | Clamps onto AC phase wire | Hazardous high-voltage wiring, requires electrical panel redesign, voiding appliance warranty. |
| **Our System (YOLO + ESP32)** | **Object detection vision + Thermal gradient inference** | **Zero false-negatives** for stationary occupants, debounced state machine, zero 230V mains tampering, real-time spatial analytics. |

---

## 3. System Architecture & Block Diagram

```
 +─────────────────────────────────────────────────────────────────────────+
 │                              EDGE VISION LAYER                          │
 │  Webcam / USB Camera (0/1) ──► YOLOv8 Detection Thread (Ultralytics)    │
 │  - Real-time Human Bounding Box Detection                               │
 │  - 4-Quadrant Spatial Region Segmentation (Q1, Q2, Q3, Q4)              │
 +────────────────────────────────────┬────────────────────────────────────+
                                      │ Occupancy Count per Region
                                      ▼
 +─────────────────────────────────────────────────────────────────────────+
 │                       CENTRAL DECISION ENGINE                           │
 │                                                                         │
 │   ┌───────────────────────┐         ┌───────────────────────────────┐   │
 │   │ Occupancy State       │         │ Post-Vacancy Thermal AC       │   │
 │   │ Machine (FSM)         │         │ Inference Engine (ΔT Analysis)│   │
 │   └──────────┬────────────┘         └───────────────┬───────────────┘   │
 │              │                                      │                   │
 │              ▼                                      ▼                   │
 │   ┌───────────────────────┐         ┌───────────────────────────────┐   │
 │   │ Appliance Policy &    │         │ Multi-Tier Alert Manager      │   │
 │   │ SwitchGuard Hysteresis│         │ (Telegram, Desktop, UI Banner)│   │
 │   └──────────┬────────────┘         └───────────────┬───────────────┘   │
 +──────────────┼──────────────────────────────────────┼───────────────────+
                │ Bidirectional Serial (JSON Lines)    │ SQLite Logging
                ▼                                      ▼
 +───────────────────────────────+      +──────────────────────────────────+
 │       IOT HARDWARE LAYER      │      │        PRESENTATION LAYER        │
 │  ESP32 Dev Module (115200)    │      │  Streamlit Multi-Page Dashboard  │
 │  ├── 4x DHT11 Temp/Humidity   │      │  ├── Live Camera & Overlays      │
 │  └── 4-Channel Opto-Relay     │      │  ├── Energy & Tariff Analytics   │
 │      (Lights, Fan, AC Demo)   │      │  └── Auditable SQLite Log Viewer │
 +───────────────────────────────+      +──────────────────────────────────+
```

---

## 4. Key Technical Innovations & Algorithms

### 4.1 Debounced Occupancy Finite State Machine (FSM)
To ensure reliable operation without flickering lights, `core/logic.py` implements a formal 7-state finite state machine:

```
              ┌────────────────┐
              │  INITIALIZING  │
              └───────┬────────┘
                      │ Camera ready
                      ▼
         ┌─────────────────────────┐
    ┌───►│        OCCUPIED         │◄──┐
    │    └────────────┬────────────┘   │ Person returns
    │                 │ Count == 0     │ (Streak reset)
    │                 ▼                │
    │    ┌─────────────────────────┐   │
    │    │     VACANCY_PENDING     ├───┘
    │    └────────────┬────────────┘
    │                 │ Vacancy Timeout Reached (default 300s / demo 20s)
    │                 ▼
    │    ┌─────────────────────────┐
    │    │         VACANT          │──► [Auto-shutdown Lights & Fans]
    │    └────────────┬────────────┘
    │                 │ Trigger observation window
    │                 ▼
    │    ┌─────────────────────────┐
    │    │     AC_OBSERVATION      │──► [Record T_initial -> T_final (120s/30s)]
    └─── │                         │
         └─────────────────────────┘
          Special States:
          • MANUAL_OVERRIDE : User toggled a relay manually; automation halted.
          • ERROR           : Camera disconnect or YOLO failure; all loads freeze!
```

#### Fail-Safe Design Rule (Crucial for Staff Questions)
> **If the camera fails, disconnects, or crashes, does the room go dark?**  
> **NO.** The engine holds state for `fail_timeout_s` (30s) and then transitions to `State.ERROR`. The policy strictly forbids turning off appliances in `ERROR`. It assumes students might still be present, ensuring student safety and uninterrupted classroom activities.

---

### 4.2 Non-Invasive Post-Vacancy AC Thermal Inference
Directly wiring an optocoupled relay to an institutional 2-ton Split AC is dangerous and unrealistic:
1. High inductive inrush currents weld small relay contacts.
2. Sudden hard power cuts bypass compressor shutdown cycles, damaging equipment.
3. Tampering with institutional electrical installations violates campus safety codes.

**Our Mathematical Solution**:
When the room transitions to `VACANT`, the system initiates an observation timer ($t_{obs} = 120\text{ s}$ standard, $30\text{ s}$ demo):
1. **Baseline Temperature**: Average temperature across active sensor regions during the initial window ($t \in [0, \tau_{edge}]$):
   $$\bar{T}_{\text{initial}} = \frac{1}{N} \sum_{i=1}^{N} T_{i, \text{init}}$$
2. **Final Temperature**: Average temperature across regions during the closing window ($t \in [t_{obs} - \tau_{edge}, t_{obs}]$):
   $$\bar{T}_{\text{final}} = \frac{1}{N} \sum_{i=1}^{N} T_{i, \text{final}}$$
3. **Thermal Gradient**:
   $$\Delta T = \bar{T}_{\text{final}} - \bar{T}_{\text{initial}}$$
4. **Classification Logic**:
   - If $\Delta T \le -1.0^\circ\text{C}$ (`cooling_threshold_c`): **`COOLING_OBSERVED` / `AC_LEFT_ON`** $\rightarrow$ Dispatches Critical Alert!
   - If $\Delta T \ge +1.0^\circ\text{C}$: **`NO_SIGNIFICANT_COOLING`** $\rightarrow$ AC is confirmed OFF (room warming naturally).
   - If $-1.0^\circ\text{C} < \Delta T < +1.0^\circ\text{C}$: **`INCONCLUSIVE`** $\rightarrow$ Ambient thermal equilibrium.

*Noise Mitigation*: DHT11 sensors output in integer $1^\circ\text{C}$ increments. Single noisy pulses are eliminated using moving-average edge buffers ($\tau_{edge} = 15\text{ s}$).

---

### 4.3 Continuous Trend Slope Monitoring (Least-Squares Regression)
In addition to the post-vacancy observation window, the system runs a continuous background monitor (`ACMonitor` in `core/logic.py`) using **Ordinary Least Squares (OLS) Linear Regression** over rolling temperature samples:

$$\text{Slope } \beta = \frac{\sum (t_i - \bar{t})(T_i - \bar{T})}{\sum (t_i - \bar{t})^2} \quad \left[^\circ\text{C / min}\right]$$

- If room temperature drop rate $\beta \le -0.25^\circ\text{C}/\text{min}$ with net drop $\ge 1.0^\circ\text{C}$, AC state is classified as **`ON`**.
- If temperature rises from its minimum by $\ge 1.0^\circ\text{C}$ with slope $\beta \ge +0.10^\circ\text{C}/\text{min}$, AC state transitions to **`OFF`**.

---

### 4.4 SwitchGuard Hardware Protection
Frequent mechanical switching destroys relay contacts and induces electromagnetic interference (EMI). `SwitchGuard` enforces a strict **10-second minimum dwell time** per relay channel. Rapid oscillating triggers are absorbed and suppressed.

---

## 5. Hardware Specifications & Circuit Design

### Bill of Materials (BOM)
| Component | Specification | Function in System |
|---|---|---|
| **ESP32 NodeMCU DevKit** | 30-pin, Dual-Core 240MHz, 520KB SRAM | Microcontroller, telemetry aggregation, relay driver |
| **4-Channel Relay Module** | 5V Optocoupled, Active-LOW, 10A 250VAC | Controls test loads (Light 1, Light 2, Fan, AC Demo) |
| **DHT11 Sensors (x1 to x4)** | Single-bus digital, 20–90% RH, 0–50°C | Ambient temperature/humidity per spatial quadrant |
| **Demo Loads** | 4x LEDs + 220Ω Current Limiting Resistors | Safe low-voltage physical demonstration of appliances |
| **USB Data Cable** | High-speed Micro-USB | Serial UART communication (115200 baud) & 5V power |

### Wiring & GPIO Pinout Table
```
 +--------------------------------------------------------------------------+
 │ ESP32 Pin │ Component Pin   │ Target Appliance / Function                │
 +-----------+-----------------+--------------------------------------------+
 │ GPIO 4    │ DHT11 #1 DATA   │ Region 1 (Front-Left) Temperature Sensor   │
 │ GPIO 16   │ DHT11 #2 DATA   │ Region 2 (Front-Right) Temperature Sensor  │
 │ GPIO 21   │ DHT11 #3 DATA   │ Region 3 (Back-Left) Temperature Sensor    │
 │ GPIO 22   │ DHT11 #4 DATA   │ Region 4 (Back-Right) Temperature Sensor   │
 │ 3.3V      │ VCC (all DHT11) │ Sensor Power                               │
 │ GND       │ GND (all DHT11) │ Common Ground                              │
 +-----------+-----------------+--------------------------------------------+
 │ GPIO 17   │ Relay IN1       │ Light 1 (Region 1 & 3 Lighting)            │
 │ GPIO 5    │ Relay IN2       │ Light 2 (Region 2 & 4 Lighting)            │
 │ GPIO 18   │ Relay IN3       │ Ceiling Fan Relay                          │
 │ GPIO 19   │ Relay IN4       │ AC Indicator Load (Demo LED)               │
 │ VIN (5V)  │ Relay VCC       │ Coil Operating Voltage                     │
 │ GND       │ Relay GND       │ Common Ground                              │
 +--------------------------------------------------------------------------+
```

> **Safety Warning for Staff**: The relay outputs use low-voltage 5V DC LEDs for viva safety. The design is galvanically isolated through optocouplers, ready for industrial contactors.

### Bidirectional Serial JSON Protocol
- **Telemetry (ESP32 $\rightarrow$ Host PC every 2s)**:
  ```json
  {"type":"telemetry","temps":[28.0,28.5,null,null],"temp_c":28.25,"humidity":58,"sensor_ok":true,"relay1":1,"relay2":1,"relay3":1,"relay4":0}
  ```
- **Relay Command (Host PC $\rightarrow$ ESP32)**:
  ```json
  {"type":"relay","channel":1,"state":0}
  ```
- **Firmware Acknowledgment (ESP32 $\rightarrow$ Host PC)**:
  ```json
  {"type":"ack","channel":1,"state":0}
  ```

---

## 6. Software Architecture & Technology Stack

```
SmartClassroom/
├── app.py                     # Streamlit application entry point & navigation
├── config.yaml                # Master configuration (COM ports, thresholds, timers)
├── requirements.txt           # Python dependency specifications
├── run_demo.bat               # One-click Windows launch script
├── core/
│   ├── engine.py              # Background orchestration thread (0.5s execution loop)
│   ├── logic.py               # Occupancy FSM, AC inference, SwitchGuard policy
│   ├── vision.py              # YOLO inference thread, OpenCV drawing, region maps
│   ├── regions.py             # 4-quadrant centroid coordinate classification
│   ├── esp32.py               # Serial driver, JSON framing, port detection, simulator
│   ├── alerts.py              # Alert dispatching, desktop toast, Telegram bot API
│   ├── storage.py             # SQLite persistence engine (log table, events table)
│   ├── analysis.py            # Historical aggregator & energy metrics generator
│   └── ui.py                  # Custom styling, KPI widgets, card components
├── views/
│   ├── dashboard.py           # Real-time monitoring, live stream, relay toggles
│   ├── analysis.py            # Energy consumption charts, waste metrics, trends
│   └── logs.py                # Auditable SQLite log viewer with CSV export
├── firmware/
│   └── esp32_controller/      # Arduino C++ embedded firmware
└── tests/
    └── test_core.py           # 81 automated test cases (100% core logic coverage)
```

### Technology Matrix
- **Edge Artificial Intelligence**: Ultralytics YOLOv8 (COCO pretrained `yolov8n.pt` / `yolov8s-worldv2.pt` or custom classroom weights).
- **Embedded Computing**: Arduino C++ running on Espressif ESP32 Tensilica Xtensa 32-bit core.
- **Frontend / Dashboard**: Streamlit with Altair charting engine.
- **Data Persistence**: SQLite3 with thread-safe connection pooling.
- **Unit Testing**: Pytest (81 automated test cases covering FSM states, timeouts, and thermal mathematics).

---

## 7. Energy Analytics, Tariff Calculation & ROI

The `views/analysis.py` module quantifies the environmental and economic impact of the system in real time:

### Mathematical Formulas
1. **Energy Consumed ($E_{\text{total}}$)**:
   $$E_{\text{total}} = \sum_{j} \left( \frac{P_j \times t_{j, \text{active}}}{1000} \right) \quad [\text{kWh}]$$
   *Where $P_j$ is rated power in Watts (e.g., Light: 40W, Fan: 75W, AC: 2000W).*
2. **Energy Wasted ($E_{\text{wasted}}$)**:
   $$E_{\text{wasted}} = \sum_{j} \left( \frac{P_j \times t_{j, \text{vacant}}}{1000} \right) \quad [\text{kWh}]$$
   *Appliance runtime accumulated while `State == VACANT`.*
3. **Financial Loss**:
   $$\text{Cost} = E_{\text{wasted}} \times \text{Tariff Rate (default ₹8.00 / kWh)}$$

### Projected Campus Savings
For a mid-sized college with **50 classrooms**:
- Average wasted AC/light time: **2 hours per room/day**.
- 2 ACs (4 kW) + 6 Lights (0.24 kW) + 4 Fans (0.3 kW) = **4.54 kW per room**.
- Daily waste per room: $4.54 \text{ kW} \times 2 \text{ h} = 9.08\text{ kWh}$.
- Across 50 rooms: $454\text{ kWh / day} \approx \mathbf{₹3,632 \text{ saved per day}}$.
- **Annual Savings (220 academic days): Over ₹7,99,000 (~$9,600 USD) and 80 Metric Tons of $\text{CO}_2$ emissions eliminated.**

---

## 8. Step-by-Step Live Demo Presentation Script

Follow this script to deliver a smooth, high-scoring live presentation.

### Option A: Running with Full Physical Hardware
1. **Plug in ESP32 & Webcam**.
2. Run `python tools/test_serial.py` in PowerShell to prove to staff that all hardware tests pass (`PASS DHT11`, `PASS Relay 1-4`).
3. Launch the app:
   ```powershell
   streamlit run app.py
   ```
4. Show the top status banner: **ESP32: Connected COMx**, live temperature readings updating every 2 seconds.
5. In the **Appliances** tab, click **Light 1** and **Fan**: show the physical LEDs toggling instantaneously.
6. Turn **Automation ON** in the sidebar. Walk out of the camera view.
7. Point to the screen:
   - State changes from `OCCUPIED` $\rightarrow$ `VACANCY_PENDING`.
   - Timer counts down to vacancy.
   - Relays click **OFF** automatically!

---

### Option B: High-Speed Simulation Demo (No Hardware Needed)
> **Tip for Viva**: Configure fast demo timers in `config.yaml` (`vacancy_confirm_s: 20`, `observation_s: 30`).

1. **Launch App**:
   ```powershell
   streamlit run app.py
   ```
2. **Setup**:
   - In sidebar, toggle **Automation: ON**.
   - Under Camera, choose **Source: Simulated**.
   - Under Controller, select **Port: Simulator**.
3. **Phase 1: Active Classroom**:
   - Set **Simulated people in room: 3**.
   - Point out `OCCUPIED` badge, 4-quadrant person distribution, and active lights/fans.
4. **Phase 2: Vacancy & Auto-Shutdown**:
   - Change **Simulated people in room: 0**.
   - Show the state change to **`VACANCY_PENDING`** with real-time countdown badge.
   - At $t=0$, room switches to **`VACANT`** $\rightarrow$ Light 1, Light 2, and Fan relays immediately switch **OFF**.
   - Show the blue **AUTO_SHUTDOWN** event notification in the log.
5. **Phase 3: The AC Leak Detection**:
   - While still vacant, check **Simulated AC running** (or simulate cooling).
   - Point to the **AC Observation** countdown widget.
   - Temperature drops from $29^\circ\text{C}$ to $27.5^\circ\text{C}$ ($\Delta T = -1.5^\circ\text{C}$).
   - **BAM!** The dashboard flashes a red banner:  
     🚨 **CRITICAL: AC left ON in vacant room! Temperature dropped by 1.5°C.**
   - Show the Windows popup notification and the simulated Telegram alert message!
6. **Phase 4: Analytics & Audit**:
   - Click over to the **Analysis** tab: show energy used, energy wasted, and financial cost.
   - Click over to the **Logs** tab: show the complete auditable history and click **Download events (CSV)**.

---

## 9. Viva & Staff Q&A Defense Guide

### Q1: Why use computer vision when PIR sensors cost only ₹100?
> **Answer**:  
> "PIR (Passive Infrared) sensors only detect relative heat-source motion. In lecture halls and exams, students sit still for 30–60 minutes. A PIR sensor triggers false vacancy, repeatedly switching off lights during class. YOLO object detection identifies human presence regardless of motion, posture, or stillness, eliminating false shutoffs."

### Q2: Why doesn't the system automatically turn off the AC with a relay?
> **Answer**:  
> "Switching off a 230V 2-ton commercial split AC via a standard mechanical relay causes two severe issues:  
> 1. It cuts power abruptly to the scroll compressor, causing refrigerant pressure lock and voiding manufacturer warranty.  
> 2. Tampering with 230V 16A wiring violates institutional electrical safety codes.  
> Instead, our system non-invasively verifies if cooling is occurring via thermal gradient $\Delta T$ and alerts facility staff via Telegram or desktop notifications to adjust the thermostat or switch it off properly."

### Q3: What happens if someone sits under the camera where there is a blind spot?
> **Answer**:  
> "The system includes an intelligent debounce parameter (`vacancy_confirm_frames` and `vacancy_confirm_s`). A brief occlusion or blind spot does not trigger vacancy. Furthermore, our architecture supports multi-camera streams and 4-quadrant spatial mapping."

### Q4: If the camera cable is accidentally disconnected, will the classroom turn OFF?
> **Answer**:  
> "No, our system is strictly **fail-safe**. As verified in unit test `test_detection_failure_never_vacant`, when frame capture fails, the Occupancy FSM holds the last known valid state for 30 seconds before entering `State.ERROR`. In the `ERROR` state, automation is locked, and all loads remain untouched. A system failure never punishes occupants."

### Q5: How do you handle privacy concerns with cameras in classrooms?
> **Answer**:  
> "All video processing is performed **entirely on edge** (local machine RAM). Frames are processed in real-time by YOLO to extract person counts and centroid coordinates, after which the raw image is discarded immediately. No video streams or photos are ever saved to the cloud, preserving student and faculty privacy."

### Q6: How does the system handle noisy temperature readings from cheap DHT11 sensors?
> **Answer**:  
> "DHT11 sensors have an integer resolution of $1^\circ\text{C}$. To prevent single-frame noise from creating false alarms, our `ACInference` class uses a moving-average buffer over the first and last 15% of the observation window ($\tau_{edge}$). In addition, our `ACMonitor` uses ordinary least squares regression over a rolling deque to measure true slope rather than instantaneous spikes."

---

## 10. Quickstart & Deployment Guide

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12 (Tested on Windows 10/11)
- USB Webcam or simulated mode
- (Optional) ESP32 Dev Module flashed with `firmware/esp32_controller/esp32_controller.ino`

### Installation Commands (PowerShell)
```powershell
# 1. Clone / Navigate to project directory
cd "C:\Users\shant\Downloads\idea lab\SmartClassroom\SmartClassroom"

# 2. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify test suite (81 automated unit tests)
python -m pytest

# 5. Launch the application
streamlit run app.py
```

### Configuration Highlights (`config.yaml`)
```yaml
esp32:
  port: "auto"              # Set to "" for simulator mode, or "COM5"
occupancy:
  vacancy_confirm_s: 300    # Set to 20 for quick presentation demo
ac_inference:
  observation_s: 120        # Set to 30 for quick presentation demo
  cooling_threshold_c: 1.0  # Temperature drop threshold
telegram:
  enabled: true             # Enable for live phone push notifications
  bot_token: "YOUR_TOKEN"
  chat_id: "YOUR_CHAT_ID"
```

---

## 🏆 Summary Checklist for Presenters
- [x] Run `pytest` beforehand to prove 81/81 tests pass.
- [x] Open `http://localhost:8501` and verify the three tabs: **Dashboard**, **Analysis**, **Logs**.
- [x] Memorize the 30-second elevator pitch from [Section 1](#1-executive-summary--30-second-pitch).
- [x] Keep [Section 9 (Viva Q&A)](#9-viva--staff-qa-defense-guide) handy during questions!
