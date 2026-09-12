"""
Breeze TTS 2 — Dual-GPU Fastpath Web UI (Gradio)
Accelerated on Dual Tesla T4 GPUs (Turing SM 75) via Model-Internal CUDA Graphs.
Features:
  1. Voice Design (Instruction-guided TTS + Emotive inline tags + Dynamic CFG)
  2. Voice Cloning (Zero-shot reference audio + verbatim transcript)
  3. Preset Speakers (Plain TTS with speaker IDs)
  4. Voice Clone + Edit (Reference audio identity + style modification)
"""
import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np

# Ensure local repo is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dual_fast_breeze import DualGpuBreezeTTS

# Global engine singleton
GLOBAL_ENGINE: Optional[DualGpuBreezeTTS] = None


def get_engine(model_dir: Optional[str] = None) -> DualGpuBreezeTTS:
    global GLOBAL_ENGINE
    if GLOBAL_ENGINE is None:
        print("Initializing DualGpuBreezeTTS engine...")
        GLOBAL_ENGINE = DualGpuBreezeTTS(model_dir=model_dir)
    return GLOBAL_ENGINE


def format_diagnostics(metrics: dict) -> str:
    """Format inference telemetry into a clean markdown badge card."""
    return f"""
<div style="background-color: #1e1e2e; border: 1px solid #313244; border-radius: 8px; padding: 12px; margin-top: 10px; color: #cdd6f4; font-family: monospace;">
  <div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-weight: bold; color: #89b4fa;">
    <span>⚡ Synthesis Telemetry [{metrics.get('template', 'N/A')}]</span>
    <span>RTF: {metrics.get('rtf', 0.0):.2f}x</span>
  </div>
  <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; font-size: 0.9em;">
    <div>🕒 Audio Duration: <b>{metrics.get('duration_sec', 0.0):.2f} s</b></div>
    <div>⏱️ Total Latency: <b>{metrics.get('latency_sec', 0.0):.2f} s</b></div>
    <div>🚀 Per-Frame Speed: <b>{metrics.get('step_ms', 0.0):.1f} ms/frame</b></div>
    <div>📊 Codec Frames: <b>{metrics.get('frames', 0)} frames ({metrics.get('fps', 0.0):.1f} fps)</b></div>
  </div>
</div>
"""


# ------------------------------------------------------------------------------
# Generation Handlers
# ------------------------------------------------------------------------------
def generate_voice_design(
    text: str,
    instruction: str,
    cfg_scale: float,
    seed: int,
    max_tokens: int,
    randomize_seed: bool,
) -> Tuple[Optional[str], str]:
    if not text or not text.strip():
        return None, "⚠️ Please provide text to synthesize."

    if randomize_seed or seed == -1:
        seed = random.randint(0, 2147483647)

    engine = get_engine()
    try:
        res = engine.synthesize(
            text=text,
            instruction=instruction,
            cfg_scale=float(cfg_scale),
            seed=int(seed),
            max_new_tokens=int(max_tokens),
        )
        return res['audio_path'], format_diagnostics(res)
    except Exception as exc:
        return None, f"❌ Error during generation: {str(exc)}"


def generate_voice_clone(
    text: str,
    ref_audio: Optional[str],
    ref_text: str,
    seed: int,
    max_tokens: int,
    randomize_seed: bool,
) -> Tuple[Optional[str], str]:
    if not text or not text.strip():
        return None, "⚠️ Please provide target text to synthesize."
    if not ref_audio:
        return None, "⚠️ Please upload or record reference audio."
    if not ref_text or not ref_text.strip():
        return None, "⚠️ Reference transcript is mandatory for alignment. Please provide the exact transcript spoken in the reference audio."

    if randomize_seed or seed == -1:
        seed = random.randint(0, 2147483647)

    engine = get_engine()
    try:
        res = engine.synthesize(
            text=text,
            ref_audio_path=ref_audio,
            ref_text=ref_text,
            cfg_scale=1.0,
            seed=int(seed),
            max_new_tokens=int(max_tokens),
        )
        return res['audio_path'], format_diagnostics(res)
    except Exception as exc:
        return None, f"❌ Error during voice cloning: {str(exc)}"


def generate_preset_tts(
    text: str,
    speaker: str,
    seed: int,
    max_tokens: int,
    randomize_seed: bool,
) -> Tuple[Optional[str], str]:
    if not text or not text.strip():
        return None, "⚠️ Please provide text to synthesize."

    if randomize_seed or seed == -1:
        seed = random.randint(0, 2147483647)

    engine = get_engine()
    try:
        res = engine.synthesize(
            text=text,
            speaker=speaker,
            cfg_scale=1.0,
            seed=int(seed),
            max_new_tokens=int(max_tokens),
        )
        return res['audio_path'], format_diagnostics(res)
    except Exception as exc:
        return None, f"❌ Error during generation: {str(exc)}"


def generate_clone_edit(
    text: str,
    ref_audio: Optional[str],
    ref_text: str,
    instruction: str,
    cfg_scale: float,
    seed: int,
    max_tokens: int,
    randomize_seed: bool,
) -> Tuple[Optional[str], str]:
    if not text or not text.strip():
        return None, "⚠️ Please provide target text to synthesize."
    if not ref_audio:
        return None, "⚠️ Please provide reference audio."
    if not ref_text or not ref_text.strip():
        return None, "⚠️ Reference transcript is mandatory."
    if not instruction or not instruction.strip():
        return None, "⚠️ Please provide a style modification instruction."

    if randomize_seed or seed == -1:
        seed = random.randint(0, 2147483647)

    engine = get_engine()
    try:
        res = engine.synthesize(
            text=text,
            ref_audio_path=ref_audio,
            ref_text=ref_text,
            instruction=instruction,
            cfg_scale=float(cfg_scale),
            seed=int(seed),
            max_new_tokens=int(max_tokens),
        )
        return res['audio_path'], format_diagnostics(res)
    except Exception as exc:
        return None, f"❌ Error during clone edit: {str(exc)}"


# ------------------------------------------------------------------------------
# UI Layout Definition
# ------------------------------------------------------------------------------
def create_ui() -> gr.Blocks:
    custom_css = """
    .main-container { max-width: 1100px; margin: 0 auto; }
    .tag-btn { margin-right: 4px; margin-bottom: 4px; }
    .status-badge { font-weight: 600; padding: 4px 10px; border-radius: 12px; }
    """

    with gr.Blocks(title="Breeze TTS 2 Studio | Dual-GPU T4 Accelerated", css=custom_css, theme=gr.themes.Soft()) as demo:
        with gr.Column(elem_classes=["main-container"]):
            gr.Markdown(
                """
                # 🌬️ Breeze TTS 2 — Dual-GPU Fastpath Studio
                ### ⚡ Hardware Acceleration: 2x NVIDIA Tesla T4 | Model-Internal Sharding | CUDA Graphs Native Replay
                *Zero external text chunking: True full-context contiguous synthesis with natural human prosody.*
                """
            )

            with gr.Tabs():
                # --------------------------------------------------------------
                # Tab 1: Voice Design
                # --------------------------------------------------------------
                with gr.TabItem("🎨 Voice Design (Instruction TTS)", id="tab_design"):
                    gr.Markdown("Describe the speaker's vocal traits in plain English, and include inline emotional cues directly in the text.")
                    with gr.Row():
                        with gr.Column(scale=3):
                            text_vd = gr.Textbox(
                                label="Text to Synthesize",
                                placeholder="Hello there! (laugh) I am thrilled to demonstrate the power of Breeze TTS 2 running across dual GPUs!",
                                lines=4,
                                value="Welcome to Breeze TTS 2 running on dual Tesla T4 GPUs! (laugh) This voice was crafted dynamically using natural language instruction guidance.",
                            )

                            gr.Markdown("**Quick Expressive Tags:**")
                            with gr.Row():
                                btn_laugh = gr.Button("+(laugh)", size="sm", elem_classes=["tag-btn"])
                                btn_sigh = gr.Button("+(sigh)", size="sm", elem_classes=["tag-btn"])
                                btn_whisper = gr.Button("+(whisper)", size="sm", elem_classes=["tag-btn"])
                                btn_gasp = gr.Button("+(gasp)", size="sm", elem_classes=["tag-btn"])
                                btn_clear = gr.Button("+(clear throat)", size="sm", elem_classes=["tag-btn"])

                            def append_tag(current_text: str, tag: str) -> str:
                                return f"{current_text} {tag} "

                            btn_laugh.click(fn=lambda t: append_tag(t, "(laugh)"), inputs=[text_vd], outputs=[text_vd])
                            btn_sigh.click(fn=lambda t: append_tag(t, "(sigh)"), inputs=[text_vd], outputs=[text_vd])
                            btn_whisper.click(fn=lambda t: append_tag(t, "(whisper)"), inputs=[text_vd], outputs=[text_vd])
                            btn_gasp.click(fn=lambda t: append_tag(t, "(gasp)"), inputs=[text_vd], outputs=[text_vd])
                            btn_clear.click(fn=lambda t: append_tag(t, "(clear throat)"), inputs=[text_vd], outputs=[text_vd])

                            instruction_vd = gr.Textbox(
                                label="Voice Instruction Prompt",
                                placeholder="A warm, confident, and articulate narrator with crisp pronunciation and natural pacing.",
                                lines=2,
                                value="A confident and warm narrator with crystal-clear pronunciation and friendly cadence.",
                            )

                            gr.Markdown("**Style Quick Presets:**")
                            with gr.Row():
                                p_host = gr.Button("🎙️ Podcast Host", size="sm")
                                p_meditate = gr.Button("🧘 Meditation Guide", size="sm")
                                p_news = gr.Button("📻 News Anchor", size="sm")
                                p_whisper = gr.Button("🤫 Secret Storyteller", size="sm")

                            p_host.click(fn=lambda: "An enthusiastic, energetic, and engaging podcast host speaking with expressive pacing and cheerful warmth.", outputs=[instruction_vd])
                            p_meditate.click(fn=lambda: "A very calm, gentle, and soothing mindfulness instructor speaking slowly with deep, comforting resonance.", outputs=[instruction_vd])
                            p_news.click(fn=lambda: "A formal, professional, and authoritative broadcast journalist with crisp diction and neutral inflection.", outputs=[instruction_vd])
                            p_whisper.click(fn=lambda: "A secretive, hushed narrator whispering with suspenseful, delicate breath control and mystery.", outputs=[instruction_vd])

                        with gr.Column(scale=2):
                            cfg_vd = gr.Slider(
                                minimum=1.0, maximum=7.0, value=4.0, step=0.1,
                                label="CFG Guidance Scale (Default: 4.0)"
                            )
                            with gr.Row():
                                seed_vd = gr.Number(label="Seed", value=42, precision=0)
                                rand_vd = gr.Checkbox(label="🎲 Randomize", value=False)
                            max_tok_vd = gr.Slider(minimum=100, maximum=1500, value=1000, step=50, label="Max New Tokens (80ms/token)")

                            btn_gen_vd = gr.Button("🚀 Generate Audio", variant="primary", size="lg")

                    audio_out_vd = gr.Audio(label="Synthesized Audio Output (24 kHz PCM WAV)", type="filepath")
                    diag_out_vd = gr.HTML()

                    btn_gen_vd.click(
                        fn=generate_voice_design,
                        inputs=[text_vd, instruction_vd, cfg_vd, seed_vd, max_tok_vd, rand_vd],
                        outputs=[audio_out_vd, diag_out_vd],
                    )

                # --------------------------------------------------------------
                # Tab 2: Voice Cloning
                # --------------------------------------------------------------
                with gr.TabItem("🎙️ Voice Cloning (Zero-Shot)", id="tab_clone"):
                    gr.Markdown("Provide a reference audio clip (3–10s recommended) and its **exact matching transcript** to clone the vocal timbre.")
                    with gr.Row():
                        with gr.Column(scale=3):
                            text_vc = gr.Textbox(
                                label="Target Text to Synthesize",
                                placeholder="Enter the text you want the cloned voice to speak...",
                                lines=4,
                                value="Voice cloning with Breeze TTS delivers zero-shot acoustic reproduction of pitch, timbre, and accent directly from reference audio.",
                            )
                            ref_audio_vc = gr.Audio(label="Reference Audio (Upload or Record)", type="filepath")
                            ref_text_vc = gr.Textbox(
                                label="Reference Verbatim Transcript (MANDATORY)",
                                placeholder="Type the EXACT words spoken in the reference audio clip...",
                                lines=2,
                            )

                        with gr.Column(scale=2):
                            with gr.Row():
                                seed_vc = gr.Number(label="Seed", value=42, precision=0)
                                rand_vc = gr.Checkbox(label="🎲 Randomize", value=False)
                            max_tok_vc = gr.Slider(minimum=100, maximum=1500, value=1000, step=50, label="Max New Tokens")
                            btn_gen_vc = gr.Button("✨ Clone & Synthesize", variant="primary", size="lg")

                    audio_out_vc = gr.Audio(label="Synthesized Cloned Audio (24 kHz PCM WAV)", type="filepath")
                    diag_out_vc = gr.HTML()

                    btn_gen_vc.click(
                        fn=generate_voice_clone,
                        inputs=[text_vc, ref_audio_vc, ref_text_vc, seed_vc, max_tok_vc, rand_vc],
                        outputs=[audio_out_vc, diag_out_vc],
                    )

                # --------------------------------------------------------------
                # Tab 3: Preset Speakers
                # --------------------------------------------------------------
                with gr.TabItem("🎭 Preset Speakers (Plain TTS)", id="tab_preset"):
                    gr.Markdown("Synthesize natural speech using predefined speaker identity embeddings without CFG overhead.")
                    with gr.Row():
                        with gr.Column(scale=3):
                            text_pt = gr.Textbox(
                                label="Text to Synthesize",
                                lines=4,
                                value="Natural language synthesis using pre-trained speaker embeddings offers the lowest latency and highest throughput.",
                            )
                            speaker_pt = gr.Dropdown(
                                choices=["S0", "S1", "S2", "S3"],
                                value="S0",
                                label="Preset Speaker ID",
                            )

                        with gr.Column(scale=2):
                            with gr.Row():
                                seed_pt = gr.Number(label="Seed", value=42, precision=0)
                                rand_pt = gr.Checkbox(label="🎲 Randomize", value=False)
                            max_tok_pt = gr.Slider(minimum=100, maximum=1500, value=1000, step=50, label="Max New Tokens")
                            btn_gen_pt = gr.Button("⚡ Generate Plain TTS", variant="primary", size="lg")

                    audio_out_pt = gr.Audio(label="Synthesized Audio (24 kHz PCM WAV)", type="filepath")
                    diag_out_pt = gr.HTML()

                    btn_gen_pt.click(
                        fn=generate_preset_tts,
                        inputs=[text_pt, speaker_pt, seed_pt, max_tok_pt, rand_pt],
                        outputs=[audio_out_pt, diag_out_pt],
                    )

                # --------------------------------------------------------------
                # Tab 4: Clone + Edit
                # --------------------------------------------------------------
                with gr.TabItem("⚡ Clone + Edit (Hybrid)", id="tab_edit"):
                    gr.Markdown("Combine a cloned reference voice identity with an instruction to alter emotion, delivery, or style.")
                    with gr.Row():
                        with gr.Column(scale=3):
                            text_ce = gr.Textbox(
                                label="Target Text",
                                lines=3,
                                value="I can scarcely believe we achieved full dual GPU acceleration without breaking any context bounds!",
                            )
                            ref_audio_ce = gr.Audio(label="Reference Audio", type="filepath")
                            ref_text_ce = gr.Textbox(label="Reference Verbatim Transcript", lines=2)
                            instruction_ce = gr.Textbox(
                                label="Style Modification Instruction",
                                placeholder="e.g., Whisper excitedly with breathless anticipation...",
                                value="Speak with breathless excitement and a joyful chuckle.",
                                lines=2,
                            )

                        with gr.Column(scale=2):
                            cfg_ce = gr.Slider(minimum=1.0, maximum=7.0, value=3.5, step=0.1, label="CFG Guidance Scale")
                            with gr.Row():
                                seed_ce = gr.Number(label="Seed", value=42, precision=0)
                                rand_ce = gr.Checkbox(label="🎲 Randomize", value=False)
                            max_tok_ce = gr.Slider(minimum=100, maximum=1500, value=1000, step=50, label="Max New Tokens")
                            btn_gen_ce = gr.Button("✨ Synthesize Cloned & Edited", variant="primary", size="lg")

                    audio_out_ce = gr.Audio(label="Synthesized Audio (24 kHz PCM WAV)", type="filepath")
                    diag_out_ce = gr.HTML()

                    btn_gen_ce.click(
                        fn=generate_clone_edit,
                        inputs=[text_ce, ref_audio_ce, ref_text_ce, instruction_ce, cfg_ce, seed_ce, max_tok_ce, rand_ce],
                        outputs=[audio_out_ce, diag_out_ce],
                    )

        gr.Markdown(
            """
            ---
            *Breeze TTS 2 Dual-GPU Optimized Edition • Developed for [sharjeel103](https://github.com/sharjeel103) • Native Tesla T4 FP16 CUDA Graphs Engine*
            """
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Breeze TTS 2 Dual-GPU Gradio Web UI")
    parser.add_argument("--model-dir", type=str, default=None, help="Path to breeze-tts-2 weights directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
    parser.add_argument("--share", action="store_true", help="Generate public gradio.live shareable link")
    args = parser.parse_args()

    # Pre-initialize engine on startup
    print("Pre-initializing Dual-GPU Breeze TTS engine before starting server...")
    get_engine(model_dir=args.model_dir)

    demo = create_ui()
    print(f"Launching Gradio UI on {args.host}:{args.port} (share={args.share})...")
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=["/kaggle/working", "/kaggle/temp", "/tmp", str(REPO_ROOT), str(Path.cwd())],
    )


if __name__ == "__main__":
    main()
