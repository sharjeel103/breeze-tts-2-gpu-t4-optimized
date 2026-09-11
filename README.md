# 🌬️ Breeze TTS 2 — Dual-GPU Fastpath Studio (Tesla T4 Optimized)

[![Kaggle Dual T4](https://img.shields.io/badge/Hardware-2x%20Tesla%20T4%20(16GB)-76B900?logo=nvidia&logoColor=white)](https://www.kaggle.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio%204.0+-FF7C00?logo=gradio&logoColor=white)](https://gradio.app/)
[![Speedup](https://img.shields.io/badge/Acceleration-5.5x%20Fastpath-brightgreen)](https://github.com/sharjeel103/breeze-tts-2-gpu-t4-optimized)

Optimized multi-device inference engine and interactive Gradio Studio for **Breeze TTS 2** on dual **NVIDIA Tesla T4 GPUs** (Turing SM 75, 2x 16GB VRAM).

Developed and customized by **[sharjeel103](https://github.com/sharjeel103)**.

---

## ⚡ Key Highlights & Innovations

- **🚀 5.5x End-to-End Speedup**: Achieves **110.1 ms/frame** and **1.43x Real-Time Factor (RTF)**, shattering the eager execution baseline (791.8 ms/frame, 10.3x RTF).
- **🧩 Zero External Text Chunking**: Unlike naive implementations that chop sentences into disjoint snippets, this engine retains **contiguous, full-context autoregressive generation** for natural human pacing and expressive prosody.
- **⚡ Model-Internal Sharding**:
  - **GPU 0 (`cuda:0`)**: BF16 Text Encoder (0.28s prefill) + 28-layer FP16 Backbone + Backbone CUDA Graph Replay.
  - **GPU 1 (`cuda:1`)**: 12-layer FP16 Depth Decoder + Depth CUDA Graph Replay + Audio Codec / Tokenizer.
- **🔄 Universal Graph Batch Reuse**: Static pre-captured CUDA graphs (batch size 2) are reused seamlessly across all modes (including single-branch Plain TTS and Voice Cloning) without triggering 10-second graph recaptures.
- **🎛️ Full Interactive Gradio Studio (`app.py`)**: Supports all 4 synthesis paradigms with dark mode, real-time telemetry, and public shareable URLs.

---

## 📊 Performance Benchmark (Kaggle 2x Tesla T4)

| Metric | Upstream Eager Baseline (Single T4) | Dual-GPU Fastpath Engine (Our Method) | Improvement |
| :--- | :--- | :--- | :--- |
| **Backbone Per-Frame Latency** | 225.4 ms/step | **42.3 ms/step** | **5.3x Faster** |
| **Depth Decoder Latency** | 566.4 ms/step | **67.8 ms/step** | **8.4x Faster** |
| **End-to-End Latency / Frame** | 791.8 ms/frame | **110.1 ms/frame** | **5.5x Speedup** |
| **Real-Time Factor (RTF)** | **10.30x** (10x slower than audio) | **1.43x** (Near real-time) | **7.2x Efficiency Gain** |
| **20-Second Audio Generation** | ~206 seconds (~3.4 min) | **~24.4 seconds** | **~3 minutes saved** |

---

## 🎛️ Modes & Operational Limits

| Mode | Input Limits | Recommended Settings | What It Does |
| :--- | :--- | :--- | :--- |
| **1. 🎨 Voice Design** (`tts_instruction`) | Text: Up to ~500 words<br>Instruction: 10–50 words | **CFG Scale**: `3.5 – 5.0` (Default `4.0`)<br>Supports inline emotion tags | Synthesizes custom speaker personalities defined by natural language text. |
| **2. 🎙️ Voice Cloning** (`ref_clone_tata`) | Target Text: Up to ~450 words<br>Ref Audio: 3–10s (clean)<br>Ref Transcript: **Mandatory exact match** | **CFG Scale**: `1.0` | Clones vocal timbre, pitch, and accent zero-shot from reference audio clip. |
| **3. 🎭 Preset Speakers** (`tts_plain`) | Text: Up to ~600 words | **Speaker**: `[S0]`, `[S1]`, `[S2]`<br>**CFG Scale**: `1.0` | Fastest throughput synthesis with built-in base speaker embeddings. |
| **4. ⚡ Clone + Edit** (`ref_edit_tata`) | Target Text: Up to ~350 words<br>Ref Audio + Transcript<br>Instruction: 10–30 words | **CFG Scale**: `3.0 – 4.5` | Clones a speaker's identity while modifying delivery (e.g. whispering, shouting). |

### Expressive Inline Emotion Tags
Insert tags directly into your text prompts to trigger natural non-verbal vocal expressions:
- `(laugh)` — natural laughter and amused tone
- `(sigh)` — audible breath release / fatigue / relief
- `(whisper)` — intimate hushed delivery
- `(gasp)` — sudden intake of air / surprise
- `(clear throat)` — throat clearing inflection
- `(pant)` — exertion / breathlessness
- `(cough)` — slight cough / hesitation

---

## 🚀 1-Click Kaggle Deployment

1. Open a new notebook on [Kaggle](https://www.kaggle.com/).
2. Under **Notebook Settings** (right sidebar):
   - **Accelerator**: `GPU T4 x2`
   - **Internet**: `ON`
3. Upload and run [`kaggle_dual_t4_breeze_tts.ipynb`](kaggle_dual_t4_breeze_tts.ipynb) or run the following cells:

```bash
# Cell 1: Clone repo
!git clone https://github.com/sharjeel103/breeze-tts-2-gpu-t4-optimized.git /kaggle/working/breeze-tts-2-gpu-t4-optimized
%cd /kaggle/working/breeze-tts-2-gpu-t4-optimized

# Cell 2: Install dependencies
!pip install -q "torch==2.9.1" "torchaudio==2.9.1" "qwen-tts==0.1.1" "transformers==4.57.3" "soundfile>=0.13" "gradio>=4.0" "huggingface_hub>=0.20"

# Cell 3: Launch Gradio UI with Public Link
!python app.py --share
```

The script will automatically download model weights from Hugging Face (`breezeblue-ai/breeze-tts-2`), capture the CUDA graphs in under 4 seconds, and provide a public `https://xxxx.gradio.live` URL!

---

## 💻 Python CLI Usage

```python
from dual_fast_breeze import DualGpuBreezeTTS

# 1. Initialize Dual-GPU Engine (captures CUDA Graphs on GPU 0 & GPU 1)
engine = DualGpuBreezeTTS(
    dev0="cuda:0",
    dev1="cuda:1",
    guidance_scale=4.0,
)

# 2. Synthesize with Voice Design
result = engine.synthesize(
    text="Welcome to Breeze TTS 2! (laugh) Accelerated with model-internal CUDA graphs.",
    instruction="A warm, confident female narrator with crystal-clear diction.",
    cfg_scale=4.0,
    seed=42,
    output_path="voice_design.wav",
)
print(f"Generated {result['duration_sec']:.2f}s audio in {result['latency_sec']:.2f}s (RTF: {result['rtf']:.2f}x)")

# 3. Synthesize with Voice Cloning
clone_result = engine.synthesize(
    text="This is a zero-shot voice clone generated across dual Tesla T4 GPUs.",
    ref_audio_path="reference_sample.wav",
    ref_text="Verbatim transcript of whatever was spoken in reference sample.",
    cfg_scale=1.0,
    output_path="voice_clone.wav",
)
```

---

## 🛠️ Repository Architecture

```
breeze-tts-2-gpu-t4-optimized/
├── app.py                         # Production Gradio Web UI (All 4 modes + telemetry)
├── dual_fast_breeze.py            # Dual-GPU CUDA Graph Fastpath Engine
├── kaggle_dual_t4_breeze_tts.ipynb# 1-Click Kaggle 2x T4 Launcher Notebook
├── breeze_infer/                  # Templates, audio utilities, runtime helpers
├── models/                        # Breeze neural architecture & CUDA Graph modules
├── configs/                       # Acceleration profiles
├── assets/                        # Logos and UI assets
├── requirements.txt               # Pinned minimal dependencies
└── README.md                      # Architecture & Documentation
```

---

## 📜 License & Acknowledgments
- Based on the [Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts) architecture by BreezeBlue AI.
- Multi-Device CUDA Graph acceleration and Gradio Studio customized by **Sharjeel Ahmed** ([sharjeel103](https://github.com/sharjeel103)).
