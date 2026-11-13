# RadioRadar 📻🎯

> Free, open-source internet-radio advertisement detection platform with shared real-time stream monitoring.

---

## 🌟 Overview

**RadioRadar** is an open-source platform designed to monitor live internet radio streams and detect commercial advertisements in near real-time. Built to explore resource-efficient audio processing on free-tier cloud infrastructure, RadioRadar implements a shared per-station monitoring architecture: multiple users monitoring the same station share a single backend ingestion and inference worker, eliminating redundant processing and network bandwidth.

### Key Capabilities
- 🌐 **Global Station Catalog by Language**: Discover and filter internet radio streams categorized by language backed by a rebuildable cache. Functional ML ad detection is targeted exclusively at **English** and **Hindi**. Conceptual UI placeholder categories include **Tamil**, **Kannada**, and **Punjabi**.
- 🎛️ **Multi-Stream Monitoring**: Initial operational target of **1 active stream** for the MVP prototype on Render Free Tier (with a design ceiling of up to 3 streams subject to empirical CPU/RAM benchmarking).
- 🔄 **Shared Ingestion Architecture**: A single background worker serves all active listeners of a station, conserving CPU and bandwidth.
- 📊 **Deterministic Monitoring States**:
  - 🔴 **Advertisement detected**
  - 🟢 **No advertisement detected**
  - 🟡 **Analyzing**
  - 🟠 **Uncertain**
  - ⚪ **Unavailable**
- 🎧 **Direct In-Browser Audio Player**: Connect directly from the browser to public station streams, substantially reducing server redistribution exposure while respecting broadcaster endpoints.

---

## 📚 Architectural & Planning Documentation

Before writing code, RadioRadar has undergone a comprehensive architectural planning, dataset research, and feasibility audit:

- 📐 [**ARCHITECTURE.md**](./ARCHITECTURE.md): Complete system blueprint, data flow diagrams, unverified resource estimates, FFmpeg ingestion pipeline, latency decomposition, and WebSocket protocols.
- 🗺️ [**ROADMAP.md**](./ROADMAP.md): Step-by-step development roadmap divided into independently testable milestones (M0 through M7) with empirical measurement completion criteria.
- ⚠️ [**RISKS.md**](./RISKS.md): Brutally realistic technical risk assessment covering Render 512MB RAM / 0.1 CPU constraints, acoustic ML ambiguity, stream dropouts, dataset scarcity, and legal/licensing considerations.
- 🔬 [**DATASET_RESEARCH.md**](./DATASET_RESEARCH.md): Exhaustive empirical survey of 10 audio/speech/broadcast datasets, verification of zero public radio ad datasets, English and Hindi availability analysis, and the **RadioRadar-10h** data collection protocol.
- 🧠 [**ML_EXPERIMENT_PLAN.md**](./ML_EXPERIMENT_PLAN.md): Technical comparison of 8 machine learning approaches, feature extraction specifications, multi-stage hybrid pipeline architecture, station-level split validation protocol, and Render free-tier compute budgeting.

---

## 🛠️ Technology Stack & Feasibility Decisions

| Layer | Selected Technology | Feasibility Rationale & Operational Constraints |
|---|---|---|
| **Frontend** | React, Vite, Tailwind CSS, Lucide Icons | Fast, lightweight SPA with minimal client bundle size and cold-start UX indicators. |
| **Backend** | Python 3.11, FastAPI, Uvicorn, AsyncIO | High-performance asynchronous API & WebSocket event bus (single Uvicorn worker). |
| **Audio Ingestion** | FFmpeg Subprocess (PCM pipe) | Downsamples directly to 16kHz mono `f32le` ($64\text{ KB/s}$); evaluated for 0.1 CPU subprocess overhead. |
| **Feature Extraction** | Pure NumPy & SciPy.signal | **Replaced Librosa** to eliminate 120MB+ cold-start RAM bloat and Numba JIT compilation. |
| **ML Inference** | Scikit-learn (Baseline Model) + Hysteresis | Lightweight initial baseline evaluated against dummy and linear models; multi-frame temporal smoothing. |
| **Database** | SQLite (Rebuildable Cache, WAL mode) | Disposable local cache for station catalog; ephemeral on free-tier containers. |
| **Deployment** | Render Free Tier (Docker) | Target: 512MB RAM, 0.1 CPU, 15m inactivity sleep, ~50-60s cold starts; subject to empirical benchmarks. |

---

## 🚦 System Architecture at a Glance

```mermaid
graph LR
    RadioStream[Live Radio Stream] -->|FFmpeg Ingestion| Worker[StationMonitorWorker]
    Worker -->|10s Ring Buffer| Features[NumPy Feature Extractor]
    Features -->|MFCCs, RMS, ZCR| Classifier[Baseline Classifier]
    Classifier -->|Probabilities| Hysteresis[Hysteresis State Machine]
    Hysteresis -->|State Updates| StreamManager[StreamManager]
    StreamManager -->|WebSocket Pub/Sub| Frontend[React Web App]
```

---

## ⏱️ Latency Budget Breakdown

Detection is not instantaneous; the operational end-to-end latency consists of:
1. **Audio Analysis Window**: 2.50 seconds of contiguous raw PCM audio per frame `[VERIFIED FACT]`.
2. **Prediction Interval**: 0.75 seconds between inference steps (70% frame overlap) `[VERIFIED FACT]`.
3. **Prediction Buffer Span**: 6-prediction buffer covers **6.25 seconds of continuous audio** across $3.75\text{s}$ of elapsed prediction steps `[VERIFIED FACT]`. (Audio coverage exceeds elapsed prediction time because sliding windows overlap by 70%).
4. **State Stabilization Delay**: 2.25 to 3.0 seconds of sustained consensus ($\ge 4$ agreeing frames) to prevent UI flickering `[ENGINEERING ESTIMATE]`.
5. **Actual Detection Latency**: **~7.8 to 11.6+ seconds total** from broadcast transition to UI notification, accounting for stream buffering, ingestion pipes, and network transit `[UNMEASURED HYPOTHESIS until benchmarked]`.

---

## ⚖️ Legal, Copyright & Licensing Notes

- **Reduced Redistribution Exposure**: Audio is not proxied, transcoded, or archived on RadioRadar servers; clients connect directly to public origin streams. However, this does not grant blanket legal immunity. Broadcaster terms of service, hotlinking restrictions, and broadcast reproduction rights (e.g. Section 37 of the Indian Copyright Act, 1957, particularly Section 37(3)(c)) apply. Section 52(1)(a)(i) statutory fair dealing for private/research use cannot be assumed to provide a blanket exemption for automated recording, and this documentation does not provide legal advice or definitive determinations of legality.
- **Dataset Redistribution Prohibition & Feature Invertibility Nuance**: RadioRadar will never publicly redistribute raw recorded broadcast audio files. Furthermore, derived numerical feature matrices cannot be assumed to be definitively non-reconstructible without empirical verification, as phase reconstruction (Griffin-Lim) and neural vocoders (MelGAN/HiFi-GAN) can partially invert mel representations. Withholding raw audio does not eliminate all legal questions regarding broadcast fixation or data governance.
- **Transformative Research**: Transient in-memory audio feature extraction is designed for analytical metadata generation. A formal legal review is required prior to commercial scaling.
- **Deployment Constraints**: Designed for Render free tier (512 MB RAM ceiling, 0.1 vCPU CFS quota, 15-minute inactivity spin-down; Source: `https://render.com/docs/free`).
- **Software License**: This open-source software is licensed under the [MIT License](LICENSE).

---

## 📜 Development Status

RadioRadar is currently in **Milestone 0 (Planning, Feasibility & Architecture Audit)**. Application implementation will begin in **Milestone 1** following architectural approval.
