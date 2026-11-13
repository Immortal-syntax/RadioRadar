# RadioRadar 📻🎯

> Free, open-source internet-radio advertisement detection platform with shared real-time stream monitoring.

---

## 🌟 Overview

**RadioRadar** is an open-source platform that monitors live internet radio streams to detect commercial advertisements in near real-time. Built to run efficiently within free-tier resource constraints, RadioRadar implements a shared per-station monitoring architecture: multiple users listening to the same station share a single backend ingestion and inference worker, preventing duplicate processing and memory bloat.

### Key Capabilities
- 🌐 **Global Station Catalog by Language**: Discover and filter internet radio streams categorized across global languages (English, Hindi, Spanish, French, German, and more).
- 🎛️ **Multi-Stream Monitoring**: Monitor up to 3 stations simultaneously in a unified dashboard.
- 🔄 **Shared Ingestion Architecture**: A single background worker serves all active listeners of a station, conserving CPU and bandwidth.
- 📊 **Deterministic Monitoring States**:
  - 🔴 **Advertisement detected**
  - 🟢 **No advertisement detected**
  - 🟡 **Analyzing**
  - 🟠 **Uncertain**
  - ⚪ **Unavailable**
- 🎧 **Direct In-Browser Audio Player**: Stream public broadcast audio directly in your browser without proxying or caching copyrighted audio on the backend.

---

## 📚 Architectural & Planning Documentation

Before writing code, RadioRadar has undergone a rigorous architectural planning phase:

- 📐 [**ARCHITECTURE.md**](./ARCHITECTURE.md): Complete system blueprint, data flow diagrams, memory budgeting, FFmpeg ingestion pipeline, feature extraction engine, and WebSocket protocols.
- 🗺️ [**ROADMAP.md**](./ROADMAP.md): Step-by-step development roadmap divided into independently testable milestones (M0 through M7) with MVP scope boundaries.
- ⚠️ [**RISKS.md**](./RISKS.md): Brutally honest technical risk assessment covering Render 512MB RAM constraints, acoustic ML limitations, stream dropouts, dataset scarcity, and legal considerations.

---

## 🛠️ Technology Stack & Feasibility Decisions

| Layer | Selected Technology | Feasibility & Rationale |
|---|---|---|
| **Frontend** | React, Vite, Tailwind CSS, Lucide Icons | Fast, lightweight SPA with low client bundle footprint. |
| **Backend** | Python 3.11, FastAPI, Uvicorn, AsyncIO | High-performance asynchronous API & WebSocket event bus. |
| **Audio Ingestion** | FFmpeg Subprocess (PCM pipe) | Downsamples directly to 16kHz mono `f32le` ($64\text{ KB/s}$), avoiding memory spikes. |
| **Feature Extraction** | Pure NumPy & SciPy.signal | **Replaced Librosa** to prevent 120MB+ cold-start RAM bloat and Numba overhead. |
| **ML Inference** | Scikit-learn (Random Forest) + Hysteresis | Lightweight model ($<15\text{MB}$ RAM) combined with temporal smoothing to eliminate UI flickering. |
| **Database** | SQLite with WAL mode | Zero-dependency, low-memory embedded storage for station directory. |
| **Deployment** | Render Free Tier (Docker) | Engineered to operate strictly within the 512MB RAM ceiling under 3 active streams. |

---

## 🚦 System Architecture at a Glance

```mermaid
graph LR
    RadioStream[Live Radio Stream] -->|FFmpeg Ingestion| Worker[StationMonitorWorker]
    Worker -->|10s Ring Buffer| Features[NumPy Feature Extractor]
    Features -->|MFCCs, RMS, ZCR| Classifier[Random Forest Classifier]
    Classifier -->|Probabilities| Hysteresis[Hysteresis State Machine]
    Hysteresis -->|State Updates| StreamManager[StreamManager]
    StreamManager -->|WebSocket Pub/Sub| Frontend[React Web App]
```

---

## 📜 Development Status

RadioRadar is currently in **Milestone 0 (Planning & Architecture)**. Application implementation will begin in **Milestone 1** following architectural approval.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
