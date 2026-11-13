# RadioRadar: Machine Learning Experiment Plan & Feasibility Analysis

**Project:** RadioRadar  
**Milestone:** 0.3 - Dataset Research and Machine Learning Feasibility  
**Role:** Senior Machine Learning Research Engineer & Audio-Processing Specialist  
**Status:** Completed Research Integrity Audit & Experimental Specification  

---

## 1. Problem Formulation & Operational Bounds

The objective of RadioRadar is to detect when an internet radio broadcast transitions from standard programming (music, news, talk dialogue) into commercial advertising blocks, emitting one of five deterministic monitoring states:
- `Advertisement detected`
- `No advertisement detected`
- `Analyzing`
- `Uncertain`
- `Unavailable`

### Operational Bounds & Epistemic Status:
- **Audio Stream**: Ingested via FFmpeg subprocess, decoded to single-channel (mono) 16,000 Hz 32-bit floating-point PCM ($64\text{ KB/s}$ raw data rate) `[VERIFIED FACT]`.
- **Target Concurrency**: **1 active stream** as the operational target for the MVP prototype on Render Free Tier `[ENGINEERING ESTIMATE]`, with 2–3 streams reserved as an empirical stress-testing benchmark in Milestone 2.
- **Compute Envelope**: **512 MB physical RAM ceiling** and **0.1 fractional vCPU allocation** on the Render Free Web Service tier (Source: Render Official Documentation, "Instance Types and Pricing", `https://render.com/docs/free`) `[VERIFIED FACT]`. Render enforces CPU limits via Linux cgroups Completely Fair Scheduler (CFS) quotas allocating 10% of CPU time slices on a shared core, rather than downclocking physical processor clock speeds.
- **Supported Functional Languages**: **English** and **Hindi** (including code-switched Hinglish). Conceptual placeholder languages (**Tamil**, **Kannada**, **Punjabi**) have **zero active ML models** and **zero detection claims**.

---

## 2. In-Depth Comparison of Machine Learning Approaches

We evaluated eight distinct algorithmic approaches against computational constraints, data requirements, and classification efficacy:

### Approach 1: Rule-Based & Heuristic Detection
- **Mechanism**: Inspects stream metadata (ICY headers) for dynamic ad insertion tags, monitors short-term RMS energy jumps, and measures boundary silences ($>200\text{ms}$).
- **Data Required**: None (hand-tuned thresholds).
- **Can it distinguish ads from ordinary speech?**: **No.** Fails on continuous speech programming (talk radio, news, podcasts).
- **Resource Footprint**: Minimal CPU ($<1\%$ `[ENGINEERING ESTIMATE]`), negligible RAM ($<2\text{ MB}$ `[ENGINEERING ESTIMATE]`), 0 KB model size.
- **Suitability for 0.1 CPU**: **Excellent.**
- **Verdict for Student MVP**: **Essential as Tier 1 auxiliary heuristic**, but insufficient as a standalone detector.

---

### Approach 2: Logistic Regression using Acoustic Features
- **Mechanism**: Evaluates an $L_2$-regularized linear decision boundary over a normalized 32-dimensional acoustic feature vector extracted from a 2.5-second sliding window:
  $$P(\text{ad}) = \sigma(\mathbf{w}^T \mathbf{x} + b)$$
- **Data Required**: Small labeled feature dataset (~500–2,000 annotated windows `[ENGINEERING ESTIMATE]`).
- **Can it distinguish ads from ordinary speech?**: **Poorly.** Linear hyperplanes cannot resolve non-linear acoustic overlaps between animated speech and commercial voiceovers.
- **Resource Footprint**: Minimal CPU ($<2\text{ms}$ inference `[ENGINEERING ESTIMATE]`), $<10\text{ MB}$ RAM `[ENGINEERING ESTIMATE]`, model size $<50\text{ KB}$ `[ENGINEERING ESTIMATE]`.
- **Suitability for 0.1 CPU**: **Ideal.**
- **Verdict for Student MVP**: **Mandatory linear baseline** to establish the lower bound of acoustic classification.

---

### Approach 3: Random Forest Classifier
- **Mechanism**: An ensemble of decorrelated decision trees trained on 32-dimensional acoustic feature vectors. Outputs class probabilities via leaf voting.
- **Data Required**: Moderate labeled dataset (~2,000–10,000 annotated windows `[ENGINEERING ESTIMATE]`).
- **Can it distinguish ads from ordinary speech?**: **Moderately.** Captures non-linear feature interactions (e.g., high RMS + elevated high-frequency rolloff + vocal formants). However, acoustic features alone still produce false positives on animated speech.
- **Resource Footprint**:
  - *Constrained Configuration* (`n_estimators=30–50`, `max_depth=8`, `n_jobs=1`): RAM consumption ~15–25 MB `[ENGINEERING ESTIMATE]`, CPU inference ~5–15ms per window `[ENGINEERING ESTIMATE]`, model disk size $<4\text{ MB}$ `[ENGINEERING ESTIMATE]`.
- **Suitability for 0.1 CPU**: **Feasible if strictly constrained.**
- **Verdict for Student MVP**: **Recommended primary baseline classifier**.

---

### Approach 4: HistGradientBoosting / Gradient Boosted Trees
- **Mechanism**: Sequentially builds shallow regression trees optimizing cross-entropy loss, using histogram binning of numerical features.
- **Data Required**: Moderate labeled dataset.
- **Can it distinguish ads from ordinary speech?**: Similar to Random Forest, with tighter probability calibration near class boundaries.
- **Resource Footprint**: Model size $<1\text{ MB}$ `[ENGINEERING ESTIMATE]`, RAM ~15 MB `[ENGINEERING ESTIMATE]`, inference $<5\text{ms}$ `[ENGINEERING ESTIMATE]`.
- **Suitability for 0.1 CPU**: **Very Good.**
- **Verdict for Student MVP**: **Top candidate model** to evaluate against Random Forest in Milestone 4.

---

### Approach 5: Lightweight Neural Audio Classifiers (1D-CNN / MobileNetV3)
- **Mechanism**: Mel-spectrogram patches fed into convolutional layers.
- **Data Required**: Large labeled audio dataset (50+ hours `[ENGINEERING ESTIMATE]`).
- **Resource Footprint**: PyTorch / ONNX Runtime imports consume ~150MB–250MB RAM `[ENGINEERING ESTIMATE]`; inference takes 50–150ms on 0.1 vCPU `[ENGINEERING ESTIMATE]`.
- **Suitability for 0.1 CPU**: **High Risk of OOM and CPU starvation.**
- **Verdict for Student MVP**: **Postponed to post-MVP / Paid Tier.**

---

### Approach 6: Pretrained Audio Embeddings (YAMNet / VGGish)
- **Mechanism**: Extracts 1024-dimensional embeddings via a pre-trained frozen neural network (e.g. YAMNet on AudioSet), followed by a linear classifier.
- **Resource Footprint**: TF-Lite runtime requires ~80–120 MB RAM `[ENGINEERING ESTIMATE]`; inference requires ~40–80ms per window on throttled CPU `[ENGINEERING ESTIMATE]`.
- **Suitability for 0.1 CPU**: **Marginal.**
- **Verdict for Student MVP**: **Postponed.**

---

### Approach 7: Automatic Speech Recognition (ASR) + NLP Transcript Classification (RadIA Audit)
- **Audit of the RadIA Research Framework (arXiv:2403.03538, 2024)**:
  - *What the Paper Implemented*: The authors implemented an offline two-stage text-centric pipeline. Radio audio was transcribed using OpenAI's **Whisper** model, and the resulting transcribed text was classified into advertisement vs. non-advertisement segments using a fine-tuned **RoBERTa** language model `[RESEARCH-REPORTED RESULT]`.
  - *What the Results Establish*: On pre-segmented and tagged text transcripts, the model achieved an **F1-macro score of 87.76%** (against a human agreement ceiling of 89.33%) `[RESEARCH-REPORTED RESULT]`. This proves that semantic text classification of ASR transcripts is effective for ad detection.
  - *What the Results DO NOT Establish*:
    1. It does **not** establish that ASR is universally mandatory for ad detection. Acoustic classification and audio fingerprinting remain viable and standard in commercial monitoring.
    2. It does **not** evaluate continuous streaming audio in a real-time, low-latency sliding-window production architecture.
    3. It was **not** benchmarked under free-tier hardware constraints (0.1 vCPU / 512MB RAM).
  - *Whisper Resource Analysis by Implementation*:
    Whisper's resource footprint depends strictly on the model size, quantization, runtime implementation, and execution environment `[VERIFIED FACT]`:
    - Full PyTorch `whisper-base` / `small`: Consumes **800MB to 1.5GB+ RAM** `[ENGINEERING ESTIMATE]` (instant OOM crash on Render 512MB).
    - Quantized `whisper.cpp` (`tiny.en`, 4-bit `q4_0`): Operates in **~150–200 MB RAM** `[ENGINEERING ESTIMATE]`.
    - *The 0.1 Fractional vCPU Compute Barrier*: Even if quantized `whisper.cpp` fits in RAM, continuous transcription on a **0.1 fractional vCPU** (a 10% cgroups CFS time quota on a shared core, not an underclocked hardware clock speed) is heavily compute-bound. Transcribing a 5-second audio chunk under a throttled 10% CPU quota is hypothesized to take 15–30+ seconds `[UNMEASURED HYPOTHESIS]`, which would make real-time stream processing unviable without catastrophic backlog and stream lag. Exact throughput remains an unmeasured hypothesis until empirically benchmarked.
- **Verdict for Student MVP**: **Strictly Deferred to Dedicated Paid GPU/CPU Infrastructure.**

---

### Approach 8: Audio Fingerprinting & Repeated-Content Detection
- **Mechanism**: Computes landmark acoustic hashes (Chromaprint / Dejavu peak pairs) from audio streams. Because radio commercials are broadcast repeatedly throughout the day, matching recurring audio hashes identifies commercial spots without semantic understanding.
- **Resource Footprint**: Landmark hashing is lightweight ($<15\text{ms}$ CPU `[ENGINEERING ESTIMATE]`), but storing hash tables requires persistent disk storage.
- **Ephemeral Storage Barrier**: On Render Free Tier, the local container disk is completely wiped upon 15-minute inactivity spin-down or redeployment `[VERIFIED FACT]`. A persistent fingerprint database cannot survive without an external database service (e.g. Supabase / Neon).
- **Verdict for Student MVP**: **Deferred to Phase 2 (Persistent Database Architecture).**

---

## 3. Systematic Approach Comparison Matrix

| Approach | Working Principle | Data Required | Can Distinguish Ad vs Speech? | Estimated RAM | Estimated CPU on 0.1 vCPU | Model Storage | Multilingual (EN/HI) | 0.1 CPU Viable? | Student MVP Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| **1. Rule-Based** | Metadata + RMS threshold | None | No (Fails on talk) | $<2$ MB `[EST]` | $<1\%$ `[EST]` | 0 KB | Agnostic | **Yes** | **Include as Tier 1** |
| **2. Logistic Regression** | Linear hyperplane on MFCC/RMS | Small (~1k samples) | Poorly | $<10$ MB `[EST]` | $<2$ ms `[EST]` | $<50$ KB | Good (Acoustic) | **Yes** | **Include as Baseline** |
| **3. Random Forest** | Constrained tree ensemble | Moderate (~5k samples) | Moderately | ~20 MB `[EST]` | ~10 ms `[EST]` | $<4$ MB | Good (Acoustic) | **Yes** | **Recommended Primary** |
| **4. HistGradientBoosting**| Histogram boosted trees | Moderate (~5k samples) | Moderately | ~15 MB `[EST]` | ~5 ms `[EST]` | $<1$ MB | Good (Acoustic) | **Yes** | **Primary Candidate** |
| **5. Lightweight CNN** | 2D Mel-Spectrogram ConvNet | Large (50+ hours) | Good | ~180 MB `[EST]`| ~80 ms `[EST]` | ~15 MB | Moderate | **High Risk** | **Postpone** |
| **6. Audio Embeddings** | Frozen YAMNet / TF-Lite | Small fine-tuning | Good | ~120 MB `[EST]`| ~60 ms `[EST]` | ~25 MB | Moderate | **Marginal** | **Postpone** |
| **7. ASR + NLP** | Whisper transcribe + RoBERTa | Enormous (ASR) | **Best (Semantic)** | 200MB–1.5GB `[EST]`| $>2000$ ms `[HYPOTHESIS]`| $>200$ MB | Language-dependent | **No (Compute bound)**| **Strictly Defer** |
| **8. Fingerprinting** | Repeated landmark hashing | Continuous stream | Yes (if repeated) | ~60 MB `[EST]` | ~15 ms `[EST]` | Needs DB | **Universal** | **Needs Disk** | **Defer to Phase 2** |

---

## 4. Acoustic Feature Vector Audit (Exact 32 Dimensions)

> [!NOTE]
> **Architectural Specification Notice**: The feature extraction pipeline described below is an architectural specification designed for implementation and benchmarking in Milestone 3. It does not represent already executed or benchmarked code.

### 4.1. STFT Sub-Framing & Feature Aggregation Protocol
The analysis window consists of $W = 2.50\text{ seconds}$ of contiguous mono audio at 16,000 Hz, totaling $N = 40,000\text{ samples}$ `[VERIFIED FACT]`.
- **Sub-Frame STFT Parameters**:
  - FFT size: $N_{\text{fft}} = 512\text{ samples}$ ($32.0\text{ ms}$ duration at 16 kHz).
  - Hop size: $N_{\text{hop}} = 256\text{ samples}$ ($16.0\text{ ms}$ hop step, 50% overlap).
  - Window function: Periodic Hann window: $w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{N_{\text{fft}}}\right)$.
  - Total sub-frames per 2.5s window: $M = \lfloor (40000 - 512)/256 \rfloor + 1 = \mathbf{155\text{ sub-frames}}$ `[VERIFIED FACT]`.
- **Spectral Feature Aggregation (Mean Across Sub-Frames)**:
  For all sub-frame-derived spectral features (MFCCs, Spectral Centroid, Spectral Rolloff, Spectral Flatness), the 2.5-second summary scalar feature is computed as the **arithmetic mean across the 155 sub-frames**.
- **Global Time-Domain Calculations**:
  Log RMS Energy and Zero-Crossing Rate are computed globally across all 40,000 samples of the 2.5s window.
- **Micro-Pause / Silence Fraction Specification**:
  The 2.5-second window ($40,000\text{ samples}$) is partitioned into exactly **50 non-overlapping 50ms sub-blocks** ($800\text{ samples}$ each at 16 kHz). A sub-block is classified as silent if its RMS energy is below **$-30.0\text{ dBFS}$** relative to a $1.0\text{ RMS}$ digital full scale ($20 \log_{10}(\text{RMS}) < -30.0$). The feature value is the ratio of silent sub-blocks to the 50 total sub-blocks ($\text{silent\_subblocks} / 50$).

### 4.2. Explicit 32-Dimensional Feature Table

| Feature Index | Feature Name | Description, Computation & Mathematical Definition | Physical Justification in Audio |
|---|---|---|---|
| **Index 0 to 12** | **Static MFCCs ($c_0$ to $c_{12}$)** | Mean across 155 sub-frames of 13 Mel-Frequency Cepstral Coefficients computed via Discrete Cosine Transform over a 26-band static Mel filter bank `[VERIFIED FACT]`. | Models spectral envelope and vocal tract formant resonances; differentiates instruments from human voices. |
| **Index 13 to 25** | **Delta MFCCs ($\Delta c_0$ to $\Delta c_{12}$)** | Mean across sub-frames of 13 first-order temporal derivatives of MFCCs computed via regression window: $\Delta c_k(t) = \frac{\sum_{n=1}^N n (c_k(t+n) - c_k(t-n))}{2 \sum_{n=1}^N n^2}$ `[VERIFIED FACT]`. | Measures dynamic rate of change in timbre and rapid phoneme transitions. |
| **Index 26** | **Log RMS Energy** | Global window calculation across all 40,000 samples: $\log_{10}\left(\sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x[n]^2} + \epsilon\right)$ `[VERIFIED FACT]`. | Measures absolute short-term loudness and broadcast audio dynamic compression. |
| **Index 27** | **Zero-Crossing Rate (ZCR)** | Global window calculation across all 40,000 samples: $\frac{1}{2(N-1)} \sum_{n=1}^{N-1} |\text{sgn}(x[n]) - \text{sgn}(x[n-1])|$ `[VERIFIED FACT]`. | Differentiates unvoiced fricative speech and percussion from tonal music. |
| **Index 28** | **Spectral Centroid** | Mean center of spectral mass across 155 sub-frames: $f_c = \frac{\sum_{k} f_k \|X[k]\|}{\sum_{k} \|X[k]\|}$ `[VERIFIED FACT]`. | Measures perceptual "brightness" of audio (bright commercial jingles vs. low-frequency speech). |
| **Index 29** | **Spectral Rolloff (85%)** | Mean across 155 sub-frames of frequency $f_r$ below which 85% of spectral magnitude resides: $\sum_{k=0}^{r} \|X[k]\| = 0.85 \sum_{k} \|X[k]\|$ `[VERIFIED FACT]`. | Distinguishes wideband music production from band-limited speech. |
| **Index 30** | **Spectral Flatness** | Mean across 155 sub-frames of Wiener entropy ratio (geometric mean to arithmetic mean of power spectrum): $\frac{\exp\left(\frac{1}{K}\sum \ln \|X[k]\|^2\right)}{\frac{1}{K}\sum \|X[k]\|^2}$ `[VERIFIED FACT]`. | Measures noisiness vs. harmonic tonality (noise/speech vs. harmonic chords). |
| **Index 31** | **Micro-Pause Ratio** | Ratio of silent 50ms sub-blocks (RMS $< -30.0\text{ dBFS}$) to 50 total sub-blocks across the 2.5s window `[VERIFIED FACT]`. | Detects boundary silences characteristic of radio ad breaks. |

$$\text{Total Feature Vector Dimension} = 13 + 13 + 1 + 1 + 1 + 1 + 1 + 1 = \mathbf{32\text{ scalar features}} \quad \text{[VERIFIED FACT]}$$

---

## 5. Detection Latency & Prediction Buffer Mathematics

To prevent confusion, the timing parameters of the sliding window and hysteresis state machine are defined with exact mathematical precision:

```mermaid
gantt
    title Sliding Window Timeline & 6-Prediction Coverage (Total: 6.25s)
    dateFormat X
    axisFormat %s s
    section Audio Windows
    Pred 1 [0.00s to 2.50s] :0, 25
    Pred 2 [0.75s to 3.25s] :7, 32
    Pred 3 [1.50s to 4.00s] :15, 40
    Pred 4 [2.25s to 4.75s] :22, 47
    Pred 5 [3.00s to 5.50s] :30, 55
    Pred 6 [3.75s to 6.25s] :37, 62
```

### Exact Latency Definitions:
1. **Audio Analysis Window ($W$)**: **2.50 seconds** `[VERIFIED FACT]`. The duration of contiguous raw PCM audio extracted from the circular buffer for one feature extraction step.
2. **Prediction Interval / Hop Step ($H$)**: **0.75 seconds** `[VERIFIED FACT]`. The cadence at which the sliding window advances forward in time (overlapping the previous window by $1.75\text{s}$, or 70% overlap).
3. **Prediction Buffer Size ($K$)**: **6 consecutive predictions** stored in a circular ring buffer.
4. **Buffer Time Span Mathematics**:
   - The wall-clock elapsed time between the 1st prediction and the 6th prediction is:
     $$(K - 1) \times H = (6 - 1) \times 0.75\text{s} = \mathbf{3.75\text{ seconds}} \quad \text{[VERIFIED FACT]}$$
   - The **total continuous audio time span covered** from the start of Prediction 1's audio window to the end of Prediction 6's audio window is:
     $$T_{\text{span}} = W + (K - 1) \times H = 2.50\text{s} + (6 - 1) \times 0.75\text{s} = \mathbf{6.25\text{ seconds}} \quad \text{[VERIFIED FACT]}$$
   - **Crucial Distinction**: A 6-prediction buffer represents **3.75s of elapsed prediction steps** (after the initial 2.5s window is filled) and covers **6.25s of cumulative broadcast audio**. Because sliding windows overlap by 70% ($1.75\text{s}$), successive predictions do not add time sequentially. Adding window durations sequentially ($6 \times 2.5\text{s} = 15\text{s}$) is physically false. It must **never** be described as "6 seconds of elapsed time" or "4.5 seconds of audio".
5. **State Stabilization Delay ($T_{\text{stabilize}}$)**:
   - To prevent rapid UI flickering, a state transition requires a sustained consensus of $\ge 4$ consecutive predictions agreeing on a new state `[ENGINEERING ESTIMATE]`.
   - The elapsed stabilization confirmation time is $(4 - 1) \times 0.75\text{s} = \mathbf{2.25\text{ seconds}}$ `[VERIFIED FACT]`. (If requiring 5 consecutive predictions, elapsed delay is $(5 - 1) \times 0.75\text{s} = 3.00\text{ seconds}$).
6. **End-to-End Measured Detection Latency**:
   The true end-to-end delay from when an ad begins playing in the radio broadcast until the client UI updates is the sum of:
   - Stream encoder & CDN broadcast buffering: ~2.0 to 4.0s `[ENGINEERING ESTIMATE]`
   - Network transit & FFmpeg socket buffer: ~1.0 to 2.0s `[ENGINEERING ESTIMATE]`
   - Window accumulation: 2.50s `[VERIFIED FACT]`
   - Stabilization confirmation delay: 2.25 to 3.00s `[ENGINEERING ESTIMATE]`
   - WebSocket dispatch & DOM render: ~0.05 to 0.15s `[ENGINEERING ESTIMATE]`
   - **Total End-to-End Latency**: **~7.8 to 11.6+ seconds** `[UNMEASURED HYPOTHESIS until benchmarked]`.

---

## 6. Experimental Validation & Evaluation Plan

### 6.1. Rigorous Data Partitioning Protocol (GroupKFold & Station Independence)

#### 1. Five-Fold GroupKFold with Five Stations Per Language
With 5 English stations and 5 Hindi stations (10 stations total in RadioRadar-10h):
- **Fold Structure**: Evaluated independently for English and Hindi using **5-fold GroupKFold**, where data is grouped strictly by `station_id` `[VERIFIED FACT]`.
- **Per-Fold Mechanics**: In each fold, **4 stations are used for training** and **1 held-out station is used for testing**. Rotating across 5 folds ensures that every station is evaluated exactly once as an out-of-sample test station.
- **Small-Sample Uncertainty**: Evaluating on only 5 station groups per language represents a small sample size. Results will exhibit high statistical variance across folds, which must be transparently reported as confidence intervals or per-fold distributions.
- **Unbalanced Class Distributions Across Folds**: GroupKFold groups samples strictly by station, **not by target class**. Broadcast formats vary widely in commercial density (e.g. commercial pop music radio may contain 20–25% ads, talk radio ~10–15% ads, and public radio virtually 0% ads). Consequently, **GroupKFold does NOT guarantee balanced class distributions across folds**.
- **Mandatory Fold Reporting**: For every fold, the evaluation suite must report:
  1. The specific station composition (which station is in test, which 4 in train).
  2. The exact class counts and proportions (number and percentage of `ADVERTISEMENT` vs. `NON_ADVERTISEMENT` windows in train and test).
- **Hyperparameter Selection (Inner Nested Cross-Validation)**: Hyperparameter tuning (e.g. tree depth, estimator count, regularization) must **never** be tuned against the held-out test fold. Hyperparameters must be selected strictly through inner cross-validation on the 4 training stations (or an internal tuning split within the training stations).
- **Cross-Validation vs. External Generalization**: 5-fold GroupKFold is an **internal cross-station validation protocol**, not an independent external benchmark. It estimates how well the model transfers across the 5 sampled stations. **Robust claims of generalized real-world performance cannot be made from only 5 stations per language.** A well-powered generalization claim would require a diverse evaluation corpus of **20–50+ independent radio stations** across varying formats, bitrates, and regions.
- **Impracticability of a Separate Locked Test Set**: With only 5 stations per language, reserving a separate locked station-level test set (e.g. 2 stations) would leave only 3 stations for training and validation, starving the model of training diversity. Hence, 5-fold GroupKFold provides the most informative cross-station assessment feasible under a 5-station constraint.

#### 2. Temporal Leakage Prevention & Rationale on Purge Buffers
- **Station-Disjoint Folds Inherently Eliminate Sample Overlap**: When splits are partitioned strictly by station (where all recordings from a given station are 100% in train or 100% in test), **no audio samples from that station can ever cross split boundaries**. In this station-disjoint setting, enforcing an arbitrary 30-second purge buffer between recording blocks is **scientifically unjustified**, as station isolation already guarantees zero sample overlap.
- **Temporal Leakage Mechanism (Within-Station Splits)**: Temporal leakage occurs only when continuous broadcast audio from a single station is partitioned across splits (e.g. during internal train/validation tuning within the 4 training stations). Because sliding windows of duration $W = 2.50\text{s}$ advance by hop $H = 0.75\text{s}$, adjacent windows share $W - H = 1.75\text{ seconds}$ of raw audio.
- **Mathematically Derived Purge Gap**: If continuous audio from a single station is ever partitioned across splits, the minimum required purge interval must separate the splits by at least the temporal reach of the feature extraction window and the 6-prediction smoothing buffer:
  $$\text{Purge Gap} \ge T_{\text{span}} = W + (K - 1) \times H = 2.50\text{s} + (6 - 1) \times 0.75\text{s} = \mathbf{6.25\text{ seconds}}$$
  Enforcing a purge gap of $\ge 6.25\text{ seconds}$ mathematically guarantees that no raw audio sample present in any window of the training sequence can overlap with any window in the evaluation sequence.

### 6.2. Baseline vs. Candidate Suite
1. **Baseline 1 (Majority Class / Dummy)**: Always predicts `No advertisement detected`.
2. **Baseline 2 (Heuristic Threshold)**: Flags an ad if RMS energy $> -12\text{ dBFS}$ and micro-pause count $> 3$ in a 5-second window.
3. **Baseline 3 (Regularized Logistic Regression)**: Standardized 32-dim features with $L_2$ penalty.
4. **Candidate Model A (Constrained Random Forest)**: `n_estimators=40`, `max_depth=8`, `min_samples_split=10`.
5. **Candidate Model B (HistGradientBoosting)**: `max_iter=50`, `max_depth=6`, `learning_rate=0.08`.

### 6.3. Mandatory Evaluation Metrics
- **Precision, Recall, and Macro F1-score** (accuracy alone is rejected due to broadcast class imbalance where ads represent ~15–25% of commercial airtime) `[VERIFIED FACT]`.
- **Confusion Matrix**: Highlighting Speech-to-Ad false alarms and Music-to-Ad misses.
- **False Positive Rate on Continuous Speech**: Evaluated specifically on pure talk/news streams (*Shrutilipi* Hindi news, NPR English news). Target: $\text{FPR} < 0.15$ `[UNMEASURED HYPOTHESIS]`.
- **Per-Language Reporting**: Evaluated independently on the English test split and the Hindi/Hinglish test split.

---

## 7. Render Free-Tier Feasibility: Concurrency Recommendation

### 7.1. Execution Costs on 0.1 vCPU
- **Compute Allocation Reality**: Render's 0.1 vCPU allocation represents a fractional CFS time quota (allocating 10% of CPU time slices on a shared physical core under Linux cgroups, as specified in the Render Official Documentation, `https://render.com/docs/free`), not an underclocked hardware clock speed `[VERIFIED FACT]`.
- **FFmpeg Subprocess**: Decoding 16kHz mono PCM consumes an estimated 20% to 40% of the 0.1 vCPU slice per active stream `[ENGINEERING ESTIMATE]`.
- **NumPy Feature Extraction**: 32-feature extraction consumes an estimated 15% to 20% of the 0.1 vCPU slice per active stream `[ENGINEERING ESTIMATE]`.
- **Running 3 Concurrent Streams**: Summing 3 FFmpeg processes (60–120% CPU) + 3 feature extractors (45–60% CPU) on a 0.1 vCPU slice guarantees CPU throttling, pipe buffer underruns, and severe audio stream lag `[ENGINEERING ESTIMATE]`.

### 7.2. Corrected Concurrency Target
- **MVP Operational Target**: **1 active monitored stream** on Render Free Tier `[ENGINEERING ESTIMATE]`.
- **Milestone 2 Stress Test**: Empirically profile CPU and RAM under 1, 2, and 3 active streams using `psutil`. Concurrency will only be expanded if empirical measurements prove CPU load remains $<75\%$.

---

## 8. Summary of Research Integrity Corrections

1. **Render Free-Tier CPU Allocation**: Removed conversion of 0.1 vCPU to an assumed 200–300 MHz clock speed; documented that 0.1 CPU is a fractional cgroups CFS time quota (10% slice), cited the official Render documentation (`https://render.com/docs/free`), and reclassified Whisper throughput claims (5s audio in 15–30+s) as `[UNMEASURED HYPOTHESIS]`.
2. **Feature Extraction Specification**: Defined exact STFT parameters (512-sample FFT, 256-sample hop, Hann window, 155 sub-frames), mean aggregation for spectral features, global 40,000-sample calculations for RMS and ZCR, and 50 non-overlapping 50ms sub-blocks with $-30.0\text{ dBFS}$ threshold for Micro-Pause Ratio. Tagged as an architectural specification for Milestone 3.
3. **Latency Verification**: Reaffirmed mathematical definitions ($W=2.50\text{s}$, $H=0.75\text{s}$, 1st-to-4th pred = $2.25\text{s}$, 1st-to-6th pred = $3.75\text{s}$, 6-prediction buffer audio span $T_{\text{span}} = 6.25\text{ seconds}$, end-to-end latency ~7.8–11.6+s `[UNMEASURED HYPOTHESIS]`). Explained why audio coverage differs from elapsed prediction time.
4. **GroupKFold Methodology**: Clarified 5-fold GroupKFold with 5 stations per language (4 train, 1 test per fold), acknowledged small-sample statistical uncertainty, documented that grouping by station does not balance class distribution across folds, required reporting class counts and station composition per fold, specified inner-loop hyperparameter tuning, clarified that CV is an internal cross-station evaluation (not an external test), noted that 20–50+ stations would be required for robust generalization, and explained why a separate locked test set is impractical with 5 stations.
5. **Purge Buffer Rigor**: Explained that station-disjoint folds inherently prevent sample overlap (making an arbitrary 30s purge buffer unjustified for station-disjoint splits), and mathematically derived the $\ge 6.25\text{s}$ minimum purge gap required whenever continuous audio from a single station is split temporally.
6. **Numerical Claims & MVP Policy**: Tagged all claims with their exact epistemic status (`[VERIFIED FACT]`, `[RESEARCH-REPORTED RESULT]`, `[ENGINEERING ESTIMATE]`, `[UNMEASURED HYPOTHESIS]`), maintaining English and Hindi as functional targets, Tamil/Kannada/Punjabi as conceptual UI categories, and 1 active stream as the initial operational target.
