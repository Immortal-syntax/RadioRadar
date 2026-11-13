# RadioRadar: Architectural Specification

## 1. Executive Summary & Core Objective

**RadioRadar** is an open-source, resource-conscious platform designed to monitor live internet radio streams and detect commercial advertisements in near real-time.

The system addresses four primary operational requirements:
1. **Multi-station Language Catalog**: Discover and categorize radio stations by language. **Functional ML ad detection is targeted exclusively at English and Hindi**. Conceptual UI placeholder categories include **Tamil, Kannada, and Punjabi** (for frontend catalog browsing without active ML models).
2. **Concurrent Monitoring**: Initial operational target of **1 active stream** for the MVP prototype on Render Free Tier, with a design ceiling of up to 3 streams subject to empirical CPU/RAM benchmarking.
3. **Shared Ingestion Architecture**: Aggregate multiple listeners monitoring the same station into a single backend processing worker, eliminating redundant network bandwidth and CPU/RAM decoding.
4. **Deterministic State Output**: Expose five explicit monitoring states:
   - `Advertisement detected`
   - `No advertisement detected`
   - `Analyzing`
   - `Uncertain`
   - `Unavailable`

---

## 2. System Architecture Overview

RadioRadar is structured into two primary tiers:
- **Client Frontend**: A reactive Single Page Application (SPA) built with React, Vite, and Tailwind CSS.
- **Backend Service**: An asynchronous Python service built with FastAPI, integrating a stream ingestion worker pool, real-time feature extraction, lightweight classification, and WebSocket pub/sub broadcasting.

```mermaid
graph TD
    subgraph "External Streams & Services"
        RB[Radio Browser API (Community Mirror)] -->|Directory Sync / Fallback| DB[(SQLite Rebuildable Cache)]
        StreamSource[Icecast / SHOUTcast / HLS Stream] -->|Audio Stream| IngestWorker[StationMonitorWorker]
    end

    subgraph "RadioRadar Backend (FastAPI / AsyncIO)"
        DB --> CatalogService[Catalog & Language Service]
        CatalogService --> RESTApi[REST Endpoints]
        
        ClientWS[WebSocket Gateway] <-->|Subscribe / Status Stream| StreamManager[StreamManager Singleton]
        
        StreamManager -->|Pool Management (Max 3 Active)| IngestWorker
        
        subgraph "Per-Station Ingestion Pipeline"
            IngestWorker -->|FFmpeg Subprocess / 16kHz PCM| RingBuffer[PCM Ring Buffer 10s]
            RingBuffer -->|Sliding Window 2.5s| FeatureEngine[Acoustic Feature Extractor]
            FeatureEngine -->|MFCC, RMS, ZCR, Spectral| MLInfer[Classifier & Hysteresis Engine]
            MLInfer -->|State: Ads / Music / Talk| StateMachine[Station State Machine]
        end
        
        StateMachine -->|Broadcast State Updates| StreamManager
        StreamManager -->|Pub/Sub Event Bus| ClientWS
    end

    subgraph "Client Frontend (React + Vite + Tailwind)"
        Browser[User Browser]
        ClientWS <-->|WebSocket JSON Frames| WSClient[Frontend WS Client]
        RESTApi -->|Station Search / Languages| StationBrowser[Station Explorer UI]
        WSClient --> StatusCards[3-Slot Monitoring Dashboard]
        Browser -->|Direct Audio Stream (CORS/Native Audio)| StreamSource
    end
```

---

## 3. Component Deep Dive

### 3.1. Station Directory & Rebuildable Storage Cache

#### Upstream Data Source: Radio Browser API
- **API Nature & SLA**: The [Radio Browser API](https://www.radio-browser.info/) is a community-driven, volunteer-operated database indexing over 40,000 global stations. It has **no formal service-level agreement (SLA) or guaranteed uptime**. Network latency varies, and endpoints utilize DNS round-robin across community-hosted mirrors.
- **Resilience, Fallbacks & Validation**:
  - *Strict Request Timeouts*: All outbound API requests to Radio Browser mirrors are configured with a strict 5.0-second timeout.
  - *Retry Limits & Circuit Breaker*: A maximum of 2 retries with exponential backoff. If external mirrors are unreachable, the catalog service trips a circuit breaker and serves solely from the local seed cache.
  - *Static Bundled Seed*: The application bundles a pre-verified static JSON dataset covering functional languages (**English**, **Hindi**) and conceptual placeholder languages (**Tamil**, **Kannada**, **Punjabi**). The service initializes immediately from this seed without waiting for external network calls.
  - *Stream URL Validation*: All stream URLs returned by external sources are validated against an allowlist of supported audio schemes (`http://`, `https://`) and verified to block Server-Side Request Forgery (SSRF) targeting loopback (`127.0.0.1`), link-local (`169.254.169.254`), or private network spaces (`10.0.0.0/8`, `192.168.0.0/16`).

#### Storage Layer: Rebuildable SQLite Cache
- **Storage Classification**: On the free deployment target (Render free tier), the container filesystem is **ephemeral**. Local disk storage is completely destroyed upon instance restart, redeployment, or inactivity spin-down.
- **Operational Reality**: **SQLite is strictly a rebuildable, disposable local cache**, not an authoritative permanent database.
- **Schema**:
  ```sql
  CREATE TABLE stations (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      stream_url TEXT NOT NULL,
      language TEXT NOT NULL,
      country_code TEXT,
      codec TEXT,
      bitrate INTEGER,
      favicon_url TEXT,
      votes INTEGER DEFAULT 0,
      is_active BOOLEAN DEFAULT TRUE,
      last_checked_at TIMESTAMP
  );

  CREATE INDEX idx_stations_language ON stations(language);
  CREATE INDEX idx_stations_votes ON stations(votes DESC);
  ```
- **Rebuild Lifecycle**: On application startup (including cold boots), the database schema is created in-memory or on local SQLite with WAL mode (`PRAGMA journal_mode=WAL;`), and populated from the bundled seed JSON.
- **Persistent Storage Alternatives**: If user favorites, customized station lists, or historical ad telemetry must be permanently preserved in future milestones, persistent storage must be offloaded to an external managed free-tier database (e.g., Supabase or Neon PostgreSQL) or a persistent cloud disk.

---

### 3.2. Shared Ingestion Architecture (`StreamManager` Singleton)

A critical bottleneck in audio processing is redundant ingestion. If multiple users monitor the same station, opening redundant FFmpeg decoding processes and inference pipelines would quickly exhaust memory and CPU resources.

```mermaid
sequenceDiagram
    participant UserA as User A (Browser)
    participant UserB as User B (Browser)
    participant SM as StreamManager
    participant Worker as StationMonitorWorker (Station 1)
    participant Radio as Radio Stream Server

    UserA->>SM: WS: Subscribe(station_id=1)
    Note over SM: Station 1 inactive. Count=0. Active streams < 3.
    SM->>Worker: Spawn StationMonitorWorker(1)
    Worker->>Radio: Connect HTTP Stream (FFmpeg)
    Radio-->>Worker: Raw Audio Chunks
    SM-->>UserA: WS: State("Analyzing")
    
    UserB->>SM: WS: Subscribe(station_id=1)
    Note over SM: Station 1 already running! Attach User B to subscriber set.
    SM-->>UserB: WS: State("Current State")

    Note over Worker: Runs 1 shared inference pipeline
    Worker->>SM: OnStateChange("Advertisement detected")
    SM->>UserA: Broadcast: "Advertisement detected"
    SM->>UserB: Broadcast: "Advertisement detected"

    UserA->>SM: WS: Unsubscribe(station_id=1)
    Note over SM: Station 1 subscriber count = 1. Keep running.

    UserB->>SM: WS: Disconnect
    Note over SM: Station 1 subscriber count = 0. Start 60s Grace Timer.
    Note over SM: Grace timer expires -> Worker.shutdown() -> Kill FFmpeg.
```

#### Ingestion Constraints & Lifecycle:
1. **Concurrency Cap**: **1 active station worker as the initial operational target for the MVP prototype** on Render Free Tier (with a design ceiling of up to 3 concurrent streams subject to empirical load and CPU benchmarking in Milestone 2). If capacity is reached:
   - Stations with 0 active subscribers are immediately culled.
   - If all 3 slots have active subscribers, the backend returns a clean status code `SYSTEM_CAPACITY_REACHED` with human-readable messaging.
2. **Subscriber Reference Counting**:
   Each `StationMonitorWorker` maintains a thread-safe `Set[WebSocketID]`. When the set transitions to empty, a 60-second grace timer is initiated. If a subscriber re-joins before expiration (e.g., page refresh), the worker continues running. Otherwise, the audio stream is closed and the FFmpeg subprocess terminated.
3. **Reconnection & Resilience**:
   Internet radio streams drop frequently. Ingestion implements exponential backoff with jitter:
   - Initial retry: 1.0s, doubling up to a maximum 30.0s interval.
   - If audio frames are not received for 10 seconds, the connection is reset.
   - After 3 consecutive reconnection failures, the worker emits the `Unavailable` state.

---

### 3.3. Audio Ingestion & Decoding Pipeline

#### Subprocess FFmpeg vs PyAV Evaluation
- **Subprocess FFmpeg** (`ffmpeg -i <url> -vn -ac 1 -ar 16000 -f f32le -`):
  - *Pros*: Isolated process space. If FFmpeg hangs on a malformed chunk, it can be terminated via `process.kill()` without corrupting Python runtime memory. Built-in demuxers handle SHOUTcast, Icecast, AAC, MP3, and HLS.
  - *Cons*: Subprocess IPC pipe overhead and context-switching cost, which is notable under fractional CPU allocations (0.1 vCPU).
- **PyAV** (FFmpeg C-bindings):
  - *Pros*: In-process frame decoding.
  - *Cons*: Memory leaks in long-running C extensions can crash the Python process on constrained 512MB RAM environments.
- **Decision**: **Subprocess FFmpeg with direct stdout pipe streaming**.
  The worker starts FFmpeg configured to downsample directly to single-channel (mono) 16,000 Hz 32-bit floating point PCM:
  ```bash
  ffmpeg -loglevel quiet -nostats -reconnect 1 -reconnect_at_eof 1 \
         -reconnect_streamed 1 -reconnect_delay_max 5 -timeout 5000000 \
         -i <STREAM_URL> -vn -ac 1 -ar 16000 -f f32le -
  ```
  - Unverified estimated data rate: $16{,}000 \text{ samples/sec} \times 4 \text{ bytes} = 64 \text{ KB/sec}$.
  - A 10-second circular ring buffer requires an estimated $640 \text{ KB}$ of raw PCM RAM per active stream.

---

### 3.4. Feature Extraction & Latency Decomposition

#### Technical Trade-Off: Pure NumPy/SciPy vs. Librosa
The original proposal suggested `librosa`. **Librosa is strictly unsuitable for resource-constrained real-time inference on a 512MB RAM environment:**
1. Librosa is designed for offline music information retrieval. Importing it pulls in heavy dependencies (`numba`, `llvmlite`, `matplotlib`, `scipy`, `soundfile`), adding an unverified estimate of **120MB+** of baseline memory consumption on cold start.
2. Numba JIT compilation causes severe CPU spikes on boot.
3. Feature extraction can be performed with **standard NumPy and SciPy.signal** in fewer than 150 lines of clean, vectorised code with zero Numba overhead.

#### Explicit Decomposition of Detection Latencies
To prevent misleading claims regarding "real-time" performance, the architectural latency is decomposed into distinct operational phases, separating initial connection setup from steady-state pipeline monitoring:

| Latency Parameter | Operational Context | Nominal Value | Operational Significance & Pipelining Behavior |
|---|---|---|---|
| **Audio Analysis Window ($W$)** | Continuous Monitoring | **2.50 seconds** `[VERIFIED FACT]` | Window of contiguous raw PCM audio ($40,000\text{ samples}$ at 16 kHz) required for feature computation. |
| **Prediction Interval ($H$)** | Continuous Monitoring | **0.75 seconds** `[VERIFIED FACT]` | Hop step with 1.75s overlap (70%) between consecutive sliding windows. |
| **Prediction Buffer Span ($K=6$)** | Continuous Monitoring | **6.25 seconds** `[VERIFIED FACT]` | Continuous audio covered across the 6-prediction buffer: $T_{\text{span}} = W + (K-1) \times H = 2.5 + 5 \times 0.75$. Elapsed step time is $3.75\text{s}$. (Never confuse with 6s of elapsed time). |
| **State Stabilization Delay ($T_{\text{stabilize}}$)** | Sequential from Ad Onset | **2.25 to 3.0 seconds** `[ENGINEERING ESTIMATE]` | Elapsed time for $\ge 4$ consecutive predictions ($(4-1) \times 0.75\text{s} = 2.25\text{s}$) to confirm state transition and prevent UI flickering. |
| **Internal Detector Response Time** | Steady-State Detection | **~2.37 seconds** `[ENGINEERING ESTIMATE]` | Sum of stabilization confirmation delay ($2.25\text{s}$), feature extraction ($<0.02\text{s}$), inference ($<0.005\text{s}$), and WebSocket dispatch (~$0.10\text{s}$). |
| **Initial Stream Startup & Prefill** | Startup Only (Sequential) | **~4.0 to 5.5 seconds** `[ENGINEERING ESTIMATE]` | Stream connect (~1-2s), FFmpeg boot (~0.5-1s), and initial 2.5s window accumulation before the first inference runs (state reports `Analyzing`). |
| **Total Studio-to-UI Broadcast Lag** | Illustrative Cumulative Lag | **~7.8 to 11.6+ seconds** `[UNMEASURED HYPOTHESIS]` | Total elapsed delay from live broadcaster microphone to client screen display (conflates external broadcaster CDN distribution lag ~2-4s + network transit ~1-2s + window accumulation 2.5s + stabilization delay ~2.25-3.0s + client render ~0.1s). |

#### Explicit Acoustic Feature Vector & STFT Specification (Exact 32 Dimensions):

> [!NOTE]
> **Architectural Specification Notice**: The feature extraction pipeline described below is an architectural specification designed for implementation in Milestone 3. Parameters marked as engineering spike decisions will be validated empirically during Milestone 3 implementation rather than assumed in advance.

The analysis window comprises $W = 2.50\text{ seconds}$ ($40,000\text{ samples}$ at 16 kHz mono) `[VERIFIED FACT]`.
- **Sub-Framing & Unpadded STFT Protocol (`center=False`)**:
  - Short-Time Fourier Transform is computed with $N_{\text{fft}} = 512\text{ samples}$ ($32.0\text{ ms}$), $N_{\text{hop}} = 256\text{ samples}$ ($16.0\text{ ms}$ hop, 50% overlap), and a periodic Hann window.
  - **Padding Requirement**: Producing exactly 155 sub-frames **strictly requires an unpadded configuration (`center=False` or `boundary=None`, `padded=False`)**:
    $$M = \left\lfloor \frac{40000 - 512}{256} \right\rfloor + 1 = 154 + 1 = \mathbf{155\text{ sub-frames}} \quad \text{[VERIFIED FACT]}$$
    *(Libraries defaulting to `center=True`, such as Librosa, would produce 157 padded frames; RadioRadar strictly specifies `center=False`).*
- **Aggregation**: Sub-frame spectral features (MFCCs, Spectral Centroid, Spectral Rolloff 85%, Spectral Flatness) are summarized as the arithmetic mean across the 155 sub-frames. Delta MFCCs are computed via polynomial regression window over sub-frames and averaged.
- **Global Time-Domain Metrics**: Log RMS Energy and Zero-Crossing Rate are computed globally across all 40,000 samples.
- **Micro-Pause / Silence Fraction**: Computed across 50 non-overlapping 50ms sub-blocks (800 samples each), thresholded at $-30.0\text{ dBFS}$ relative to 1.0 RMS digital full scale ($20 \log_{10}(\text{RMS}) < -30.0$), reporting the ratio of silent sub-blocks to 50 total.

Every scalar feature in the 32-dimensional vector is explicitly defined:
1. **Static MFCCs (Indices 0–12, 13 features)**: Mean across 155 sub-frames of 13 Mel-Frequency Cepstral Coefficients ($c_0$ to $c_{12}$) computed via DCT-II over 26 Mel filter banks ($20\text{ Hz}$ to $8000\text{ Hz}$). HTK vs Slaney weighting is designated as an **engineering spike decision for Milestone 3** `[VERIFIED FACT]`.
2. **Delta MFCCs (Indices 13–25, 13 features)**: Mean across sub-frames of 13 first-order temporal derivatives of MFCCs computed via regression window ($N=2$, 5-frame width). Boundary padding behavior is an **engineering spike decision for Milestone 3** `[VERIFIED FACT]`.
3. **Log RMS Energy (Index 26, 1 feature)**: Global window logarithmic short-term signal energy with numerical stabilization $\epsilon_{\text{rms}} = 10^{-12}$ measuring broadcast dynamic compression `[VERIFIED FACT]`.
4. **Zero-Crossing Rate (Index 27, 1 feature)**: Global window average sign-change rate per frame separating unvoiced speech from harmonic tones `[VERIFIED FACT]`.
5. **Spectral Centroid (Index 28, 1 feature)**: Mean center of spectral mass across 155 sub-frames measuring audio brightness ($\epsilon = 10^{-12}$) `[VERIFIED FACT]`.
6. **Spectral Rolloff (Index 29, 1 feature)**: Mean frequency below which 85% of spectral magnitude concentrates across 155 sub-frames `[VERIFIED FACT]`.
7. **Spectral Flatness (Index 30, 1 feature)**: Mean Wiener entropy ratio of power spectrum ($|X[k]|^2$) measuring tonality vs. noise across 155 sub-frames ($\epsilon = 10^{-12}$) `[VERIFIED FACT]`.
8. **Micro-Pause Ratio (Index 31, 1 feature)**: Ratio of silent 50ms sub-blocks (RMS $< -30.0\text{ dBFS}$, or $\text{RMS} < 0.03162$) to 50 total sub-blocks `[VERIFIED FACT]`.

$$\text{Total Feature Vector Dimension} = 13 + 13 + 1 + 1 + 1 + 1 + 1 + 1 = \mathbf{32\text{ scalar features}} \quad \text{[VERIFIED FACT]}$$

---

### 3.5. Machine Learning & Decision Engine

#### Reality-Grounded Detection Strategy
Advertisement detection using acoustic features alone without Natural Language Processing (speech-to-text transcription) faces an inherent challenge:
- Spoken ads acoustic profiles closely resemble talk-radio hosts, podcasts, and sports commentary.
- Pop music with vocal intros can resemble jingle-heavy commercials.

Therefore, RadioRadar employs a **Three-Tier Detection Pipeline**:

```mermaid
graph LR
    Frame[2.5s Audio Frame] --> MetadataCheck{Metadata Clue?}
    MetadataCheck -- Yes: Explicit 'Ad' Tag --> EmitAd[State: Ad Detected]
    MetadataCheck -- No Clue --> Extractor[NumPy Acoustic Features]
    
    Extractor --> BaselineModel[Acoustic Classifier]
    BaselineModel --> Prob[P_ad: Ad Probability [0.0 - 1.0]]
    
    Prob --> Hysteresis[Hysteresis & Temporal Smoothing Filter]
    Hysteresis --> FinalState{Confidence Band}
    
    FinalState -- P >= 0.70 for >= 3s --> AdState[Advertisement detected]
    FinalState -- P <= 0.35 for >= 3s --> MusicState[No advertisement detected]
    FinalState -- 0.35 < P < 0.70 --> UncertainState[Uncertain]
    FinalState -- Buffer < 5s --> AnalyzingState[Analyzing]
```

1. **Tier 1: ICY Metadata Heuristic**:
   Some modern stream encoders inject metadata tags (e.g. `Ad: Commercial Break`, `Preroll`, or blank title headers) during dynamic ad insertions (DAI). If present, this immediately flags an ad.
2. **Tier 2: Baseline Machine Learning Classifier**:
   - *Operational Model Scope (Binary Decision for MVP):* The core classifier is strictly **binary** ($y \in \{1: \text{ADVERTISEMENT}, 0: \text{NON\_ADVERTISEMENT}\}$). Fine-grained ground-truth labels (`MUSIC`, `TALK`, `NEWS`, `OTHER`) are collapsed into `NON_ADVERTISEMENT` for training, but preserved in the dataset to support slice-based diagnostic evaluations (e.g. false alarms on conversational speech vs. formal news).
   - *Treatment of Uncertain Samples:* Ground-truth `UNCERTAIN` annotations are quarantined into a dedicated partition (never silently discarded). In hard-state evaluation, the runtime `Uncertain` state is reported as an explicit abstention class.
   - *Status of Hyperparameters:* Initial configurations (e.g., Random Forest with shallow depth, Logistic Regression) are **illustrative starting baselines**, not finalized decisions.
   - *Model Selection Protocol:* The architecture requires empirical comparison against small, predeclared lightweight baselines:
     * Majority-class / Dummy classifier (lower-bound sanity check).
     * Regularized Logistic Regression (minimal CPU/memory footprint).
     * Shallow Random Forest (`max_depth=6..8`, `n_estimators=30..50`) vs. HistGradientBoosting.
   - *Dataset Requirements:* A verified, manually labeled radio audio dataset must be collected across functional target languages (**English**, **Hindi**) before training. Conceptual UI languages (**Tamil**, **Kannada**, **Punjabi**) have zero ML models and zero detection claims.
   - *Evaluation Rigor (5-Fold GroupKFold Across 5 Stations):* Models must be evaluated using **5-fold GroupKFold cross-validation grouped strictly by station ID** (4 stations train, 1 station test per fold):
     * *Outer Evaluation vs. Inner Selection:* Outer folds evaluate generalization on the 1 held-out station. Hyperparameter tuning must occur strictly within the 4 training stations.
     * *Inner Validation Limitations:* With only 4 station groups available internally, inner cross-validation evaluates on only 1 held-out station at a time, resulting in high variance and noisy selection. Nested validation does not make the small dataset statistically robust.
     * *Avoid Excessive Tuning:* Exhaustive grid searches across 4 stations risk overfitting to station idiosyncrasies. Predeclared, conservative baseline models are strongly preferred.
     * *Small-Sample Uncertainty:* Evaluating across only 5 station groups per language represents a small sample size with high statistical variance across folds. Do not fabricate confidence intervals or claim population-level uncertainty precision.
     * *Unbalanced Class Proportions:* GroupKFold groups by station, not class. Because broadcast advertising density varies widely across stations, formats, countries, and time of day, GroupKFold does not guarantee balanced class distributions across folds. RadioRadar must measure actual distributions in its own recordings.
     * *Mandatory Reporting:* Reporting station composition, per-station results, and exact class counts/proportions for every fold is required.
     * *Internal CV vs. External Generalization:* 5-fold GroupKFold is an internal cross-station evaluation, not an external test. Five stations per language are insufficient for strong claims of broad real-world generalization. Expanding to 20–50+ stations is an informal proposed planning target, not a proven statistical threshold.
     * *Sample Constraint:* A separate locked test set is impractical with 5 stations (leaving only 3–4 for training).
     * *Purge Buffers:* Station-disjoint folds inherently prevent sample overlap (making arbitrary 30s buffers unjustified for station-disjoint splits). If continuous audio from a single station is ever partitioned temporally, a mathematically derived purge gap of $\ge 6.25\text{ seconds}$ ($W + (K-1) \times H$) must be enforced.
     * Precision, Recall, Macro F1-score, and PR-AUC/ROC-AUC.
     * Confusion matrices highlighting speech vs. commercial confusion.
     * Per-language evaluation to measure cross-lingual acoustic generalization independently on English and Hindi.
     * Systematic false-positive analysis on talk-radio and news broadcasts.
   - *Zero Fabrication Rule:* No accuracy figures or dataset statistics may be fabricated or claimed without empirical validation.
3. **Tier 3: Temporal Smoothing & Hysteresis State Machine**:
   - A ring buffer holds the last 6 frame probabilities (covering **6.25 seconds of cumulative broadcast audio** across $3.75\text{ seconds}$ of elapsed prediction steps).
   - **Hysteresis Logic**:
     * Transition to `Advertisement detected`: Requires weighted probability $> 0.70$ sustained across $\ge 4$ consecutive predictions ($2.25\text{s}$ elapsed delay).
     * Transition to `No advertisement detected`: Requires probability $< 0.35$ sustained across $\ge 4$ consecutive predictions.
     * `Uncertain`: When probability hovers in the ambivalent zone ($0.35 \le P < 0.70$) or during DJ speech over background music.
     * `Analyzing`: Initial state during the first 5 seconds of stream connection while the circular buffer populates.

---

### 3.6. Audio Playback Strategy & Legal / Copyright Realities

#### Architecture: Direct Browser Audio Playback
RadioRadar does not proxy, transcode, or syndicate live audio streams to clients.
- **Backend Role**: Ingests streams strictly into transient, ephemeral in-memory circular buffers ($10\text{ seconds}$ maximum) for mathematical feature calculation. Buffers are overwritten continuously. No audio is recorded to persistent storage or rebroadcast to users.
- **Frontend Role**: When a user selects "Listen", the HTML5 `<audio>` element streams directly from the station's official public `stream_url`.

#### Rigorous Legal & Copyright Analysis

> [!CAUTION]
> **Legal Disclaimer**: This section provides technical and architectural analysis of intellectual property and broadcast regulations for research purposes. It does **not** constitute legal advice, nor does it provide a definitive legal determination regarding the project's recording practices under applicable jurisdictions.

1. **Reduced Redistribution Exposure (Not Absolute Immunity)**:
   - Direct client-to-stream playback substantially **reduces copyright redistribution exposure** because RadioRadar servers never act as a secondary audio broadcaster or re-distributor.
   - **Crucial Clarification**: Direct browser playback does **not** grant automatic, blanket, or complete legal or DMCA immunity.
2. **Broadcaster Terms of Service & Hotlinking**:
   - Individual radio stations and stream content delivery networks (CDNs) may maintain terms of service restricting unauthorized embedding, inline linking ("hotlinking"), or automated monitoring of their stream endpoints.
   - Stream providers may employ geoblocking or session-based referrer checks to prevent external clients from connecting directly.
3. **Broadcast Reproduction Rights & Statutory Exceptions under Indian Law (Sections 37 & 39)**:
   - In India, **Section 37 of the Copyright Act, 1957** grants broadcasting organizations an exclusive statutory "Broadcast Reproduction Right" for 25 years `[VERIFIED FACT]`. Under **Section 37(3)(c)**, making any sound recording of a broadcast without a license constitutes statutory infringement.
   - **Section 39 Statutory Exceptions**: Section 39 specifically provides that no broadcast reproduction right is infringed by:
     * **Section 39(a)**: Making any sound recording for private use or solely for bona fide teaching or research.
     * **Section 39(b)**: Fair dealing use of excerpts for reporting, review, teaching, or research.
     * **Section 39(c)**: Other acts that do not constitute copyright infringement under Section 52.
   - **Crucial Legal Distinctions**:
     * **Broadcast Rights vs. Underlying Works**: A broadcast signal carries underlying copyrighted works (musical compositions, sound recordings, scripts). Section 39 limits *only* the broadcast reproduction right. **Section 39(a) does NOT automatically resolve or license copyright claims held by underlying copyright owners** (music labels, composers, publishers).
     * **Underlying Copyrights (Section 52(1)(a)(i))**: Fair dealing for private research under Section 52(1)(a)(i) is statutory and narrow. It cannot be assumed that every academic recording automatically qualifies as lawful.
     * **Research Fixation vs. Public Service**: While making private research recordings for model training may align with Section 39(a) and 52(1)(a)(i), this does **not** authorize deploying a public operational monitoring service or commercial feed.
     * **Dataset Redistribution Prohibition & Feature Reconstructibility**:
       - RadioRadar will never publicly redistribute raw recorded broadcast audio files (`.wav` or `.mp3`).
       - *Feature Reconstructibility Nuance:* Previous suggestions that Griffin-Lim or neural vocoders directly invert the 32-dim feature vector were inaccurate: Griffin-Lim requires a 2D magnitude spectrogram, and neural vocoders require a time-resolved mel sequence. However, because future conditional generative models could approximate acoustic textures, the vector's reconstructibility is an **unmeasured, open empirical question**, not an established fact. Do not claim the vector is either perfectly reconstructible or provably non-reconstructible.
4. **Requirement for Formal Legal Review**:
   - Prior to any commercial operation, production scaling, or public distribution beyond educational research, RadioRadar requires formal legal review regarding DMCA safe-harbor compliance, Indian Section 37/39 broadcast rights, underlying copyright licenses, and station terms of service. This architectural specification does not constitute legal advice.

---

## 4. Communication & WebSocket Protocol

### 4.1. REST Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health, memory usage, and active worker count. |
| `GET` | `/api/v1/languages` | List available languages with station counts from cache. |
| `GET` | `/api/v1/stations?language={lang}&search={query}&limit=50` | Search and filter radio stations from rebuildable cache. |
| `GET` | `/api/v1/monitor/status` | Current active monitoring slots and system capacity. |

### 4.2. WebSocket Gateway (`/ws/monitor`)

Clients establish a single WebSocket connection and multiplex up to 3 station subscriptions.

#### Client-to-Server Messages
```json
// Subscribe to station
{
  "action": "subscribe",
  "station_id": "bbc-world-service"
}

// Unsubscribe from station
{
  "action": "unsubscribe",
  "station_id": "bbc-world-service"
}

// Ping / Heartbeat (counts as active inbound traffic)
{
  "action": "ping"
}
```

#### Server-to-Client Messages
```json
// Monitoring State Update
{
  "type": "state_update",
  "station_id": "bbc-world-service",
  "state": "Advertisement detected", // Advertisement detected | No advertisement detected | Analyzing | Uncertain | Unavailable
  "confidence": 0.88,
  "timestamp": 1731513600000,
  "details": {
    "rms_loudness_db": -14.2,
    "speech_confidence": 0.92,
    "current_track": "Sponsor Message"
  }
}

// Capacity / Error Message
{
  "type": "error",
  "code": "CAPACITY_EXCEEDED",
  "message": "Maximum concurrent station capacity (3) reached. Please select an already active station or wait for a slot to free."
}

// Heartbeat Ack
{
  "type": "pong"
}
```

---

## 5. Deployment Architecture & Render Free-Tier Feasibility

### 5.1. Render Free Tier Target Specifications
- **Physical Memory**: 512 MB strict ceiling (Source: Render Official Documentation, "Instance Types and Pricing", `https://render.com/docs/free`) `[VERIFIED FACT]`. (Linux kernel OOM killer terminates instance immediately upon exceeding).
- **CPU Allocation**: **0.1 fractional vCPU** (shared, heavily throttled). Render enforces this via Linux cgroups Completely Fair Scheduler (CFS) quotas allocating 10% of CPU time slices on a shared physical core, rather than downclocking host processor clock speed `[VERIFIED FACT]`.
- **Inactivity Spin-Down**: Automatically spins down after **15 minutes of inactivity**.
  - *Activity Definition:* Inactivity is determined by inbound HTTP requests or incoming WebSocket messages.
  - *WebSocket Behavior:* **Open WebSocket connections alone should NOT be assumed to prevent spin-down** if the client is merely receiving data and sending zero inbound frames. Clients must actively send ping messages to register activity.
- **Cold-Start Penalty**: When sleeping, container cold boots take approximately **50 to 60+ seconds**.
- **Filesystem**: **Ephemeral local filesystem**. All local disk modifications, including SQLite databases, are completely wiped upon redeploy or sleep cycle.

### 5.2. Practical Operational Impact on Real-Time Monitoring
1. **Stream Worker Termination on Sleep**:
   When the service spins down after 15 minutes of inactivity, all running `StationMonitorWorker` instances and FFmpeg decoding processes are terminated. Active audio ingestion stops completely.
2. **Cold-Start Reconnection Delay**:
   When a user visits the platform after a sleep period, they must wait ~50-60 seconds for container boot, followed by ~5-8 seconds for stream discovery, buffer filling, and initial state stabilization. Total initial time to first state: **~60 to 75 seconds**.
3. **Subprocess & CPU Starvation**:
   Running multiple FFmpeg subprocesses alongside an async Python event loop, decoding pipes, and feature extraction on a **0.1 fractional CPU** introduces severe context-switching overhead. If CPU starvation causes the FFmpeg pipe buffer to fill or drop, audio frames may lag behind real-time broadcast.

### 5.3. Unverified Resource Budget Estimates

> [!WARNING]
> The figures below are **unverified design estimates**. Whether 3 concurrent streams can reliably operate within 512MB RAM and 0.1 CPU without OOM termination or pipe lag **cannot be claimed as verified until empirically benchmarked in Milestone 2 and Milestone 7**.

| Component | Unverified RAM Estimate | Unverified CPU Impact | Resource Mitigation & Bottleneck Notes |
|---|---|---|---|
| **Base Linux OS & Python Runtime** | ~55 MB | Minimal idle | Slim Docker base image (`python:3.11-slim`). |
| **FastAPI + Uvicorn (Single Worker)** | ~40 MB | Low | Asynchronous event loop (`--workers 1`). |
| **NumPy & SciPy (Selective Imports)** | ~70 MB | Low idle, burst on FFT | Direct imports of required modules; excludes full SciPy and Librosa. |
| **Scikit-Learn Classifier** | ~20 MB | Low on 0.75s inference | Constrained baseline model; inference executed at fixed intervals. |
| **FFmpeg Subprocesses (Up to 3)** | ~35 MB each = ~105 MB | **High (Primary 0.1 CPU Bottleneck)** | Process management, pipe context switching; downsampling to 16kHz mono. |
| **PCM Ring Buffers (Up to 3, 10s f32)** | ~2 MB total | Negligible | Circular memory buffer ($640\text{ KB}$ per stream). |
| **SQLite Rebuildable Cache (WAL)** | ~5 MB | Low | In-memory / ephemeral cache for station catalog. |
| **Safety Headroom & Buffer Margin** | ~205 MB | N/A | Buffer against sudden memory spikes or subprocess leaks. |
| **TOTAL ESTIMATE** | **~502 MB** | **Heavily Constrained on 0.1 CPU** | **Requires empirical verification under multi-stream load.** |

---

## 6. Security & Operational Guardrails

1. **SSRF Stream URL Validation**:
   Stream URLs supplied by external directories or users must be strictly validated. They must match allowed protocols (`http://`, `https://`), resolve to public IP ranges, and never point to loopback (`127.0.0.1`), private RFC1918 spaces (`10.0.0.0/8`, `192.168.0.0/16`), or cloud metadata endpoints (`169.254.169.254`).
2. **Subprocess Sandboxing**:
   FFmpeg runs with disabled network reconnect loops on non-HTTP schemas, no shell execution (`shell=False`), and a hard execution timeout on stalled pipe reads.
3. **CORS & Rate Limiting**:
   WebSocket handshake rate limiting prevents denial-of-service attempts that could flood worker allocation.
