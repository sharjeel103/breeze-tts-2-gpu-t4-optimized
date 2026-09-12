"""
Model-Internal Dual-GPU Pipeline for Breeze TTS 2 with CUDA Graphs.
GPU 0 (Tesla T4): Text Encoder + Backbone + Backbone CUDA Graph (Native FP16)
GPU 1 (Tesla T4): Depth Decoder + Audio Tokenizer/Codec + Depth Decoder CUDA Graph (Native FP16)

Zero external text chunking: Full contiguous prompt inference with maximum natural prosody.
Supports:
  1. Voice Design (Natural language voice instruction + emotion tags + CFG 3.5-5.0)
  2. Voice Cloning (Zero-shot reference audio + verbatim transcript)
  3. Preset Speakers (Plain TTS with speaker IDs S0, S1, etc.)
  4. Voice Clone + Edit (Reference audio + transcript + modification instruction)
"""
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch

# Ensure local repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer, StaticCache
from models.breeze import BreezeForConditionalGeneration
from qwen_tts import Qwen3TTSTokenizer
from breeze_infer.runtime import update_generation_config_for_breeze, set_all_seeds
from breeze_infer.templates import get_template, prepare_inputs, select_template_name
from models.cudagraph.backbone_graph import BackboneGraph
from models.cudagraph.depth_decoder_graph import DepthDecoderGraph
from models.cudagraph.sampling import sample_logits
from models.fast_streaming import (
    is_backbone_eos_token,
    should_decode_codec_frame,
    _left_pad_tensor,
)


# ==============================================================================
# Multi-Device CUDA Graph Capture Patches
# ==============================================================================
def fixed_bg_capture(self, prefill_len=100, num_warmup=3):
    with torch.cuda.device(self.device):
        torch.cuda.set_device(self.device)
        self._init_cache_layers()
        self._build_initial_attention_mask()
        self.cache_position[0] = prefill_len
        self.position_ids.fill_(prefill_len)
        self._pad_lens.zero_()
        self._set_attention_mask(prefill_len)

        for _ in range(num_warmup):
            self._decode_step()
        torch.cuda.synchronize(device=self.device)

        self.graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            self._decode_step()
            torch.cuda.synchronize(device=self.device)
            with torch.cuda.graph(self.graph, stream=s):
                self._decode_step()

        torch.cuda.current_stream(self.device).wait_stream(s)
        torch.cuda.synchronize(device=self.device)
        self.captured = True
        print(f"Backbone CUDA graph captured on {self.device}!")

BackboneGraph.capture = fixed_bg_capture


@torch.inference_mode()
def fixed_capture_single_bucket(self, bsz: int, num_warmup: int):
    with torch.cuda.device(self.device):
        torch.cuda.set_device(self.device)
        self.batch_size = bsz
        self.half = self._real_batch_size(bsz)
        self.static_cache = StaticCache(
            config=self.config, max_cache_len=self.max_seq, batch_size=bsz
        )
        self._alloc_buffers()
        self._init_cache_layers()
        self._build_attention_masks()

        for _ in range(num_warmup):
            self.static_cache.reset()
            self._full_loop()
        torch.cuda.synchronize(device=self.device)

        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(s):
            self.graph = torch.cuda.CUDAGraph()
            self.static_cache.reset()
            self._full_loop()
            torch.cuda.synchronize(device=self.device)

            self.static_cache.reset()
            with torch.cuda.graph(self.graph, stream=s):
                self._full_loop()

        torch.cuda.current_stream(self.device).wait_stream(s)
        torch.cuda.synchronize(device=self.device)

        self._bucket_graphs[bsz] = self._snapshot_state()
        print(f"  Depth decoder CUDA graph captured for bucket_size={bsz} on {self.device}")

DepthDecoderGraph._capture_single_bucket = fixed_capture_single_bucket

# Patch _project_segments to safely cast segment hidden states to proj.weight.dtype
orig_proj = BreezeForConditionalGeneration._project_segments
def safe_project_segments(self, segment_lengths, seg_hidden_states, seg_layer_hidden_states, is_separate):
    casted_hs = [hs.to(self.text_encoder_proj.weight.dtype) for hs in seg_hidden_states]
    return orig_proj(self, segment_lengths, casted_hs, seg_layer_hidden_states, is_separate)

BreezeForConditionalGeneration._project_segments = safe_project_segments


def resolve_model_dir(model_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve model directory from argument, local cache, or Hugging Face download."""
    if model_dir is not None and Path(model_dir).exists():
        return Path(model_dir)

    # Standard Kaggle or local locations
    candidates = [
        Path('/kaggle/working/breeze-tts-2'),
        REPO_ROOT / 'breeze-tts-2',
        Path.cwd() / 'breeze-tts-2',
    ]
    for cand in candidates:
        if cand.exists() and (cand / 'config.json').exists():
            return cand

    print("Model weights not found locally. Downloading from Hugging Face: BreezeBlue/Breeze-TTS-2 ...")
    from huggingface_hub import snapshot_download
    target_dir = Path('/kaggle/working/breeze-tts-2') if Path('/kaggle/working').exists() else (REPO_ROOT / 'breeze-tts-2')
    snapshot_download('BreezeBlue/Breeze-TTS-2', local_dir=str(target_dir))
    return target_dir


# ==============================================================================
# Dual-GPU Pipeline Implementation
# ==============================================================================
class DualGpuBreezeTTS:
    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        dev0: str = 'cuda:0',
        dev1: str = 'cuda:1',
        dtype: torch.dtype = torch.float16,
        guidance_scale: float = 4.0,
    ):
        self.dev0 = torch.device(dev0)
        self.dev1 = torch.device(dev1)
        self.dtype = dtype
        self.guidance_scale = guidance_scale
        self.model_dir = resolve_model_dir(model_dir)

        print(f"=== Initializing Dual-GPU Breeze TTS (Model-Internal Pipeline) ===")
        print(f"Model Directory: {self.model_dir}")
        print(f"GPU 0: {torch.cuda.get_device_name(self.dev0)} (Text Encoder + Backbone + Backbone CUDA Graph)")
        print(f"GPU 1: {torch.cuda.get_device_name(self.dev1)} (Depth Decoder + Depth CUDA Graph + Codec)")
        print(f"Precision: {self.dtype} (Native Tesla T4 Tensor Cores + BF16 Text Encoder)")

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), fix_mistral_regex=False)

        # 1. Load model in native FP16
        print("Loading model weights in FP16...")
        self.model = BreezeForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            dtype=self.dtype,
            attn_implementation='eager',
        )
        update_generation_config_for_breeze(self.model)

        # 2. Untie shared embeddings before device placement to avoid cross-device pointer conflicts
        print("Untying shared embeddings across GPUs...")
        self.model.backbone_model.embed_tokens.embed_audio_tokens.weight = torch.nn.Parameter(
            self.model.backbone_model.embed_tokens.embed_audio_tokens.weight.detach().clone()
        )
        self.model.depth_decoder.model.embed_tokens.weight = torch.nn.Parameter(
            self.model.depth_decoder.model.embed_tokens.weight.detach().clone()
        )

        # 3. Place GPU 0 components
        print(f"Placing GPU 0 components on {self.dev0} in {self.dtype}...")
        self.model.backbone_model.to(self.dev0, dtype=self.dtype).eval()
        self.model.embed_text_tokens.to(self.dev0, dtype=self.dtype).eval()
        self.model.lm_head = self.model.lm_head.to(self.dev0, dtype=torch.float32).eval()
        if self.model.text_encoder is not None:
            self.model.text_encoder.to(self.dev0, dtype=torch.bfloat16).eval()
        if self.model.text_encoder_proj is not None:
            self.model.text_encoder_proj.to(self.dev0, dtype=self.dtype).eval()

        # 4. Place GPU 1 components
        print(f"Placing GPU 1 components on {self.dev1} in {self.dtype}...")
        self.model.depth_decoder.to(self.dev1, dtype=self.dtype).eval()
        self.model.codec_model.to(self.dev1).eval()

        self.audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(self.model_dir / 'audio_tokenizer'), device_map=str(self.dev1)
        )

        self.sample_rate = int(self.model.config.codec_config.sampling_rate)
        self.num_codebooks = int(self.model.config.num_codebooks)
        self.codec_codebook_size = int(self.model.config.codec_config.codebook_size)
        self._reserved_codec_token_ids = tuple(
            range(self.codec_codebook_size, int(self.model.config.vocab_size))
        )

        # 4. Capture Backbone CUDA Graph on dev0
        print(f"Capturing Backbone CUDA Graph on {self.dev0}...")
        self.backbone_graph = BackboneGraph(
            backbone_model=self.model.backbone_model,
            lm_head=self.model.lm_head,
            embed_tokens=self.model.backbone_model.embed_tokens,
            config=self.model.config,
            device=str(self.dev0),
            dtype=self.dtype,
            max_seq_len=1024,
            guidance_scale=self.guidance_scale,
            batch_size=2,
        )
        self.backbone_graph.capture()

        # 5. Capture Depth Decoder CUDA Graph on dev1
        print(f"Capturing Depth Decoder CUDA Graph on {self.dev1}...")
        depth_gen = self.model.depth_decoder.generation_config
        self.depth_graph = DepthDecoderGraph(
            depth_decoder=self.model.depth_decoder,
            config=self.model.config.depth_decoder_config,
            device=str(self.dev1),
            dtype=self.dtype,
            guidance_scale=self.guidance_scale,
            num_codebooks=self.num_codebooks,
            codec_codebook_size=self.codec_codebook_size,
            fast=False,
            batch_size=2,
            bucket_sizes=[1, 2],
            temperature=float(getattr(depth_gen, 'temperature', 0.9)),
            top_k=int(getattr(depth_gen, 'top_k', 50)),
            top_p=float(getattr(depth_gen, 'top_p', 1.0)),
            do_sample=bool(getattr(depth_gen, 'do_sample', True)),
        )
        self.depth_graph.capture()

        print("\n=== Dual-GPU Pipeline fully initialized and ready for high-throughput inference! ===\n")

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        instruction: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        speaker: str = "S0",
        output_path: Optional[str] = None,
        cfg_scale: float = 4.0,
        seed: int = 42,
        max_new_tokens: int = 1500,
    ) -> Dict[str, Any]:
        """
        Universal multi-mode TTS synthesis with dual-GPU CUDA Graph acceleration.
        Supports:
          - Voice Design: text + instruction (cfg_scale=3.5-5.0)
          - Voice Cloning: text + ref_audio_path + ref_text (cfg_scale=1.0)
          - Preset Voice: text + speaker ID (cfg_scale=1.0)
          - Clone + Edit: text + ref_audio_path + ref_text + instruction (cfg_scale=3.0-4.5)
        """
        t0 = time.time()
        set_all_seeds(seed)

        if not output_path:
            out_dir = Path.cwd() / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(out_dir / f"output_{int(time.time()*1000)}.wav")

        # Build request dict
        request: Dict[str, Any] = {
            'id': f'req_{int(time.time()*1000)}',
            'text': text.strip(),
            'speaker': speaker.strip() if speaker else "S0",
        }

        has_instruction = bool(instruction and instruction.strip())
        if has_instruction:
            request['instruction'] = instruction.strip()

        has_ref_audio = bool(ref_audio_path and str(ref_audio_path).strip() and Path(ref_audio_path).exists())
        has_ref_text = bool(ref_text and ref_text.strip())

        if has_ref_audio != has_ref_text:
            raise ValueError("Voice Cloning requires both reference audio and verbatim reference transcript.")

        if has_ref_audio:
            request['ref_audio_path'] = str(ref_audio_path)
            request['ref_text'] = ref_text.strip()

        template_name = select_template_name(request)
        template = get_template(template_name)

        # CFG is active for instruction or edit templates when cfg_scale != 1.0
        use_cfg = template.build_negative_segments is not None and cfg_scale != 1.0
        prep_cfg_scale = cfg_scale if use_cfg else 1.0

        # Update guidance scale buffers
        effective_cfg = cfg_scale if use_cfg else 1.0
        self.guidance_scale = effective_cfg
        self.backbone_graph.guidance_scale.fill_(effective_cfg)
        self.depth_graph.set_guidance_scale(effective_cfg)

        inputs = prepare_inputs(
            self.tokenizer,
            self.audio_tokenizer,
            self.model,
            [request],
            template,
            guidance_scale=prep_cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )

        t_prep = time.time() - t0

        # Step 1: Text & Prompt Prefill on dev0
        t_prefill_start = time.time()
        with torch.cuda.device(self.dev0):
            if 'cfg_negative_prompt_ids' in inputs:
                # Dual-branch CFG mode (batch_size=2)
                cond_ids = inputs['input_ids'].to(self.dev0)
                uncond_ids = inputs['cfg_negative_prompt_ids'].to(self.dev0)
                max_len = max(cond_ids.shape[1], uncond_ids.shape[1])

                input_ids = torch.cat(
                    [
                        _left_pad_tensor(cond_ids, max_len, 0),
                        _left_pad_tensor(uncond_ids, max_len, 0),
                    ],
                    dim=0,
                )
                attention_mask = torch.cat(
                    [
                        _left_pad_tensor(inputs['attention_mask'].to(self.dev0), max_len, 0),
                        _left_pad_tensor(
                            inputs['cfg_negative_prompt_attention_mask'].to(self.dev0), max_len, 0
                        ),
                    ],
                    dim=0,
                )
                text_ids_mask = torch.cat(
                    [
                        _left_pad_tensor(inputs['text_ids_mask'].to(self.dev0), max_len, False),
                        _left_pad_tensor(inputs['cfg_negative_text_ids_mask'].to(self.dev0), max_len, False),
                    ],
                    dim=0,
                )
                text_ids_len = torch.cat(
                    [inputs['text_ids_len'].to(self.dev0), inputs['cfg_negative_text_ids_len'].to(self.dev0)], dim=0
                )
                cond_values = inputs.get('input_values')
                uncond_values = inputs.get('cfg_negative_input_values')
                if cond_values is not None and uncond_values is not None:
                    input_values = torch.cat([cond_values.to(self.dev0), uncond_values.to(self.dev0)], dim=0)
                elif cond_values is not None:
                    input_values = cond_values.to(self.dev0).repeat(2, 1, 1)
                else:
                    input_values = None
            else:
                # Single-branch mode (Plain TTS or pure Voice Clone)
                # We duplicate cond to form batch=2 so the captured CUDA Graph executes without recapture!
                cond_ids = inputs['input_ids'].to(self.dev0)
                input_ids = cond_ids.repeat(2, 1)
                attention_mask = inputs['attention_mask'].to(self.dev0).repeat(2, 1)
                text_ids_mask = inputs['text_ids_mask'].to(self.dev0).repeat(2, 1)
                text_ids_len = inputs['text_ids_len'].to(self.dev0).repeat(2)
                if inputs.get('input_values') is not None:
                    input_values = inputs['input_values'].to(self.dev0).repeat(2, 1, 1)
                else:
                    input_values = None

            merged = self.model._merge_input_ids_with_input_values(
                input_ids=input_ids,
                attention_mask=attention_mask,
                text_ids_mask=text_ids_mask,
                text_ids_len=text_ids_len,
                input_values=input_values,
            )
            branch_embeds = merged['inputs_embeds'].contiguous()
            branch_mask = attention_mask.contiguous()

            position_ids = branch_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(branch_mask == 0, 1)

            backbone_out = self.model.backbone_model(
                inputs_embeds=branch_embeds,
                attention_mask=branch_mask,
                position_ids=position_ids,
                use_cache=True,
            )
            hidden = backbone_out.last_hidden_state
            logits = self.model.lm_head(hidden[:, -1, :].float()).float()

            prefill_len = self.backbone_graph.prefill_kv(backbone_out.past_key_values)
            self.backbone_graph.set_generation_state(branch_mask)

            # Initial first-codebook token
            cond_logits = logits[:1]
            uncond_logits = logits[1:]
            logits = uncond_logits + effective_cfg * (cond_logits - uncond_logits)

            backbone_params = {
                'temperature': float(self.model.generation_config.temperature),
                'top_k': int(self.model.generation_config.top_k),
                'top_p': float(self.model.generation_config.top_p),
                'do_sample': bool(self.model.generation_config.do_sample),
            }
            token = sample_logits(logits, suppress_tokens=self._reserved_codec_token_ids, **backbone_params).view(1)

        depth_params = {
            'temperature': float(self.model.depth_decoder.generation_config.temperature),
            'top_k': int(self.model.depth_decoder.generation_config.top_k),
            'top_p': float(self.model.depth_decoder.generation_config.top_p),
            'do_sample': bool(self.model.depth_decoder.generation_config.do_sample),
        }

        t_prefill = time.time() - t_prefill_start
        print(f"Mode: [{template_name}] | Prefill done in {t_prefill:.3f}s (prefill_len={prefill_len}). Starting Dual-GPU loop...")

        # Step 2: Autoregressive Loop across GPU 0 & GPU 1
        t_loop_start = time.time()
        collected_frames = []

        for step_idx in range(max_new_tokens):
            if is_backbone_eos_token(token, self.model.config):
                break
            if prefill_len + step_idx >= 1024 - 1:
                break

            token_batch = token.repeat(2)
            depth_hidden = hidden[:, -1, :]

            # PCIe Transfer: 8 KB hidden state + 16 bytes token to GPU 1
            hidden_dev1 = depth_hidden.to(self.dev1, non_blocking=True)
            token_batch_dev1 = token_batch.to(self.dev1, non_blocking=True)

            # --- GPU 1: Depth Decoder CUDA Graph Replay ---
            with torch.cuda.device(self.dev1):
                depth_tokens = self.depth_graph.run(
                    hidden_dev1,
                    token_batch_dev1,
                    guidance_scale=effective_cfg,
                    **depth_params,
                )

                # Assemble full 16-codebook frame [16]
                frame = torch.cat([token_batch_dev1[:1].view(1), depth_tokens[0]], dim=0)
                if should_decode_codec_frame(frame, self.model.config):
                    collected_frames.append(frame.detach())

            # PCIe Transfer: 32 bytes frame back to GPU 0
            frame_dev0 = frame.to(self.dev0, non_blocking=True).view(1, 1, -1).repeat(2, 1, 1)

            # --- GPU 0: Backbone CUDA Graph Replay ---
            with torch.cuda.device(self.dev0):
                hidden, logits = self.backbone_graph.run(frame_dev0, step_idx=step_idx)
                token = sample_logits(
                    logits.float(),
                    suppress_tokens=self._reserved_codec_token_ids,
                    **backbone_params,
                ).view(1)

        t_loop = time.time() - t_loop_start
        steps_run = len(collected_frames)
        step_ms = (t_loop / max(steps_run, 1)) * 1000.0
        fps = steps_run / max(t_loop, 0.001)

        # Step 3: Fast Codec Audio Decoding on dev1
        t_decode_start = time.time()
        if collected_frames:
            with torch.cuda.device(self.dev1):
                # Stack all frames: [num_frames, 16]
                audio_codes_batch = torch.stack(collected_frames, dim=0).to(self.dev1)
                codec_output = self.audio_tokenizer.decode({'audio_codes': [audio_codes_batch]})

                if hasattr(codec_output, 'audio_values'):
                    audio_tensor = codec_output.audio_values
                else:
                    audio_tensor = codec_output

                while isinstance(audio_tensor, (list, tuple)):
                    audio_tensor = audio_tensor[0]

                if isinstance(audio_tensor, torch.Tensor):
                    audio_np = audio_tensor.squeeze().detach().cpu().float().numpy()
                elif isinstance(audio_tensor, np.ndarray):
                    audio_np = audio_tensor.squeeze()
                else:
                    audio_np = np.zeros(1, dtype=np.float32)
        else:
            audio_np = np.zeros(1, dtype=np.float32)

        t_decode = time.time() - t_decode_start
        dur = len(audio_np) / self.sample_rate
        t_total = time.time() - t0
        rtf = t_total / max(dur, 0.001)

        # Write output audio file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio_np, samplerate=self.sample_rate, subtype='PCM_16')

        print("\n" + "=" * 60)
        print(f"✨ DUAL-GPU FASTPATH SYNTHESIS COMPLETE [{template_name}] ✨")
        print(f"Audio Duration   : {dur:.2f} seconds ({len(audio_np)} samples @ {self.sample_rate} Hz)")
        print(f"Total Latency    : {t_total:.2f} seconds")
        print(f"Breakdown        : Prep={t_prep:.3f}s | Prefill={t_prefill:.3f}s | Loop={t_loop:.2f}s ({step_ms:.1f}ms/step) | Codec={t_decode:.3f}s")
        print(f"Real-Time Factor : {rtf:.2f}x (< 1.0x is faster than real time)")
        print(f"Saved Audio File : {output_path}")
        print("=" * 60 + "\n")

        return {
            'audio_path': output_path,
            'audio_np': audio_np,
            'sample_rate': self.sample_rate,
            'duration_sec': dur,
            'latency_sec': t_total,
            'rtf': rtf,
            'step_ms': step_ms,
            'frames': steps_run,
            'fps': fps,
            'template': template_name,
        }


if __name__ == '__main__':
    t_init_start = time.time()
    engine = DualGpuBreezeTTS(
        dev0='cuda:0',
        dev1='cuda:1',
        dtype=torch.float16,
        guidance_scale=4.0,
    )
    print(f"Dual-GPU Engine initialized in {time.time() - t_init_start:.2f}s\n")

    prompt_text = "Welcome to Breeze TTS 2 running with model-internal CUDA graph parallelism across dual Tesla T4 GPUs on Kaggle! This audio was generated without any external text chunking, delivering high-throughput performance with natural expressive cadence."
    instruction_prompt = "A warm, clear, and confident voice with natural pacing."

    res = engine.synthesize(
        text=prompt_text,
        instruction=instruction_prompt,
        output_path='dual_fastpath_benchmark.wav',
        cfg_scale=4.0,
        seed=42,
    )
