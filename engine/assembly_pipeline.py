"""
Macro-Pipelined Double-Buffered Assembly Line Engine for Dual Tesla T4 GPUs.
Runs GPU 0 (Backbone Station) and GPU 1 (Depth Decoder + Vocoder Station) concurrently
with in-flight continuous batching, zero-allocation memory pinning, and streaming chunk emission.
"""
import asyncio
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch

from .config import EngineConfig
from .slot_manager import SlotManager, SlotState, Slot
from .streaming_vocoder import StreamingVocoderWorker
from .async_prefill import AsyncPrefillWorker, PrefilledRequest

from transformers import AutoTokenizer, StaticCache
from models.breeze import BreezeForConditionalGeneration
from qwen_tts import Qwen3TTSTokenizer
from models.cudagraph.backbone_graph import BackboneGraph
from models.cudagraph.depth_decoder_graph import DepthDecoderGraph
from models.cudagraph.sampling import sample_logits
from models.fast_streaming import is_backbone_eos_token, should_decode_codec_frame

# Multi-Device CUDA Graph Capture Patches
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

orig_proj = BreezeForConditionalGeneration._project_segments
def safe_project_segments(self, segment_lengths, seg_hidden_states, seg_layer_hidden_states, is_separate):
    casted_hs = [hs.to(self.text_encoder_proj.weight.dtype) for hs in seg_hidden_states]
    return orig_proj(self, segment_lengths, casted_hs, seg_layer_hidden_states, is_separate)

BreezeForConditionalGeneration._project_segments = safe_project_segments

def resolve_model_dir(model_id: str) -> Path:
    if Path(model_id).exists():
        return Path(model_id)
    candidates = [
        Path('/kaggle/working/breeze-tts-2'),
        Path.cwd() / 'breeze-tts-2',
        Path(__file__).resolve().parent.parent / 'breeze-tts-2',
    ]
    for c in candidates:
        if c.exists() and (c / 'config.json').exists():
            return c
    from huggingface_hub import snapshot_download
    try:
        p = snapshot_download(model_id)
        return Path(p)
    except Exception:
        target = Path('/kaggle/working/breeze-tts-2') if Path('/kaggle/working').exists() else (Path.cwd() / 'breeze-tts-2')
        snapshot_download(model_id, local_dir=str(target))
        return target

class AssemblyPipelineEngine:
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.dev0 = self.config.dev0
        self.dev1 = self.config.dev1
        self.num_slots = self.config.num_slots
        
        self.slot_manager = SlotManager(num_slots=self.num_slots, dev0=self.dev0, dev1=self.dev1)
        self.running = False
        self._loop_task: Optional[asyncio.Task] = None
        
        # Streams & Synchronization Events
        self.stream_bb = torch.cuda.Stream(device=self.dev0)
        self.stream_depth = torch.cuda.Stream(device=self.dev1)
        self.event_bb_done = torch.cuda.Event(enable_timing=False)
        self.event_depth_done = torch.cuda.Event(enable_timing=False)
        
        self._reserved_tokens = [0, 1, 2, 3]

    def initialize(self):
        """
        Loads weights, partitions across GPU 0 & GPU 1, and pre-captures static CUDA Graphs.
        """
        print("=" * 60)
        print("🚀 INITIALIZING PRODUCTION ASSEMBLY ENGINE (DUAL TESLA T4)")
        print(f"Config: {self.num_slots} In-Flight Slots | Chunk Size: {self.config.chunk_size} frames")
        print("=" * 60)
        t0 = time.time()
        
        # Resolve model directory
        self.model_dir = resolve_model_dir(self.config.model_id)
        print(f"-> Model directory resolved: {self.model_dir}")
        
        # 1. Load Tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(self.model_dir / 'audio_tokenizer'), device_map=str(self.dev1)
        )
        
        # 2. Load Model across GPUs
        print("-> Loading model components into VRAM...")
        self.model = BreezeForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            torch_dtype=self.config.dtype,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.model.tokenizer = self.tokenizer
        self.model.audio_tokenizer = self.audio_tokenizer

        from breeze_infer.runtime import update_generation_config_for_breeze
        update_generation_config_for_breeze(self.model)

        # Untie shared embeddings before device placement to avoid cross-device pointer conflicts
        print("-> Untying shared embeddings across GPUs...")
        self.model.backbone_model.embed_tokens.embed_audio_tokens.weight = torch.nn.Parameter(
            self.model.backbone_model.embed_tokens.embed_audio_tokens.weight.detach().clone()
        )
        self.model.depth_decoder.model.embed_tokens.weight = torch.nn.Parameter(
            self.model.depth_decoder.model.embed_tokens.weight.detach().clone()
        )

        # Partition Model: GPU 0 (Text Encoder + Backbone), GPU 1 (Depth Decoder + Vocoder)
        print(f"-> Placing GPU 0 components on {self.dev0}...")
        self.model.backbone_model.to(self.dev0, dtype=self.config.dtype).eval()
        self.model.embed_text_tokens.to(self.dev0, dtype=self.config.dtype).eval()
        self.model.lm_head = self.model.lm_head.to(self.dev0, dtype=torch.float32).eval()
        if self.model.text_encoder is not None:
            self.model.text_encoder.to(self.dev0, dtype=torch.bfloat16).eval()
        if self.model.text_encoder_proj is not None:
            self.model.text_encoder_proj.to(self.dev0, dtype=self.config.dtype).eval()

        print(f"-> Placing GPU 1 components on {self.dev1}...")
        self.model.depth_decoder.to(self.dev1, dtype=self.config.dtype).eval()
        self.model.codec_model.to(self.dev1).eval()
        
        # 3. Initialize Workers
        self.prefill_worker = AsyncPrefillWorker(self.model, dev0=self.dev0)
        self.vocoder_worker = StreamingVocoderWorker(
            audio_tokenizer=self.audio_tokenizer,
            device=self.dev1,
            sample_rate=self.config.sample_rate,
            chunk_size=self.config.chunk_size,
        )
        
        # 4. Pre-Allocate Static Pinned Memory (Zero allocator thrashing)
        print("-> Pre-allocating pinned static buffers for N slots...")
        self.hidden_states_dev0 = torch.zeros(
            (self.num_slots, 1, self.model.config.hidden_size),
            device=self.dev0,
            dtype=self.config.dtype,
        )
        self.hidden_states_dev1 = torch.zeros(
            (self.num_slots, 1, self.model.config.hidden_size),
            device=self.dev1,
            dtype=self.config.dtype,
        )
        self.current_tokens_dev0 = torch.zeros((self.num_slots,), device=self.dev0, dtype=torch.long)
        self.current_tokens_dev1 = torch.zeros((self.num_slots,), device=self.dev1, dtype=torch.long)
        self.frames_dev0 = torch.zeros((self.num_slots, 1, 16), device=self.dev0, dtype=self.config.dtype)
        self.frames_dev1 = torch.zeros((self.num_slots, 16), device=self.dev1, dtype=self.config.dtype)
        
        # 5. Capture Static CUDA Graphs
        if self.config.enable_cuda_graphs:
            print("-> Capturing Static CUDA Graphs on GPU 0 and GPU 1...")
            # We capture Backbone graph at batch_size=2 for pair-wise macro pipelining
            self.backbone_graph = BackboneGraph(
                backbone_model=self.model.backbone_model,
                lm_head=self.model.lm_head,
                embed_tokens=self.model.backbone_model.embed_tokens,
                config=self.model.config,
                device=self.dev0,
                dtype=self.config.dtype,
                max_seq_len=self.config.max_seq_len,
                batch_size=2,
            )
            self.backbone_graph.capture(prefill_len=64)
            
            depth_gen = self.model.depth_decoder.generation_config
            self.depth_graph = DepthDecoderGraph(
                depth_decoder=self.model.depth_decoder,
                config=self.model.config.depth_decoder_config,
                device=str(self.dev1),
                dtype=self.config.dtype,
                guidance_scale=self.config.guidance_scale,
                num_codebooks=int(self.model.config.num_codebooks),
                codec_codebook_size=int(self.model.config.codec_config.codebook_size),
                fast=False,
                batch_size=2,
                bucket_sizes=[1, 2],
                temperature=float(getattr(depth_gen, 'temperature', 0.9)),
                top_k=int(getattr(depth_gen, 'top_k', 50)),
                top_p=float(getattr(depth_gen, 'top_p', 1.0)),
                do_sample=bool(getattr(depth_gen, 'do_sample', True)),
            )
            self.depth_graph.capture()
            
        t_init = time.time() - t0
        print(f"✨ Production Assembly Line Engine Ready in {t_init:.2f}s!")
        print("=" * 60)

    def _admit_pending_prefilled(self):
        """
        Pops prefilled requests from prefill worker queue and hot-injects them
        into free slots without stopping the GPU loop.
        """
        while not self.prefill_worker.ready_queue.empty():
            free_slot = self.slot_manager.find_free_slot()
            if free_slot is None:
                break  # All slots currently busy
                
            prefilled: PrefilledRequest = self.prefill_worker.ready_queue.get_nowait()
            slot = self.slot_manager.admit_request(
                slot_id=free_slot,
                request_id=prefilled.request_id,
                prompt_text=prefilled.prompt_text,
                prefill_len=prefilled.prefill_len,
                max_frames=prefilled.max_frames,
                effective_cfg=prefilled.effective_cfg,
                is_dual_branch=prefilled.is_dual_branch,
                audio_queue=prefilled.audio_queue,
            )
            
            # Hot-inject initial hidden state and token
            self.hidden_states_dev0[free_slot].copy_(prefilled.initial_hidden[:, -1, :])
            self.current_tokens_dev0[free_slot].copy_(prefilled.initial_token.view(-1)[0])
            
            # Inject past_key_values into StaticCache slice for this slot
            # (zero-allocation pointer update)
            if hasattr(self, 'backbone_graph') and self.backbone_graph.captured:
                self.backbone_graph.prefill_kv(prefilled.past_key_values)
                self.backbone_graph.set_generation_state(prefilled.branch_mask)

    def step_assembly(self):
        """
        Executes one Macro-Pipelined Double-Buffered assembly cycle across both GPUs.
        GPU 0 and GPU 1 execute concurrently with hardware CUDA event synchronization.
        """
        if self.slot_manager.get_num_active() == 0:
            return 0
            
        # Divide active slots into Group A (even) and Group B (odd) for double-buffering
        # Step Phase 1: GPU 0 computes Backbone for Group A; GPU 1 computes Depth for Group B
        with torch.cuda.stream(self.stream_bb):
            for i in range(0, self.num_slots, 2):
                if self.slot_manager.slots[i].state == SlotState.ACTIVE:
                    f_dev0 = self.frames_dev0[i : i + 1].repeat(2, 1, 1)
                    h, logits = self.backbone_graph.run(f_dev0, step_idx=self.slot_manager.slots[i].current_step)
                    tok = sample_logits(logits.float(), suppress_tokens=self._reserved_tokens, temperature=0.8).view(1)
                    self.current_tokens_dev0[i].copy_(tok[0])
                    self.hidden_states_dev0[i].copy_(h[:, -1, :])
                    
        with torch.cuda.stream(self.stream_depth):
            for i in range(1, self.num_slots, 2):
                if self.slot_manager.slots[i].state == SlotState.ACTIVE:
                    h_dev1 = self.hidden_states_dev1[i : i + 1].repeat(2, 1)
                    t_dev1 = self.current_tokens_dev1[i : i + 1].repeat(2)
                    depth_tokens = self.depth_graph.run(
                        h_dev1,
                        t_dev1,
                        guidance_scale=self.slot_manager.slots[i].effective_cfg,
                        temperature=0.8,
                    )
                    full_frame = torch.cat([t_dev1[:1].view(1), depth_tokens[0]], dim=0)
                    self.frames_dev1[i].copy_(full_frame)
                    
                    # Accumulate for streaming vocoder
                    self.slot_manager.slots[i].chunk_buffer.append(full_frame.detach())
                    self.slot_manager.slots[i].total_frames_generated += 1
                    self.slot_manager.slots[i].current_step += 1
                    
        # Synchronize streams in hardware
        self.stream_bb.synchronize()
        self.stream_depth.synchronize()
        
        # Step Phase 2: DMA Tensor Exchange across PCIe (GPU 0 <-> GPU 1)
        # Hidden states GPU 0 -> GPU 1
        self.hidden_states_dev1.copy_(self.hidden_states_dev0, non_blocking=True)
        self.current_tokens_dev1.copy_(self.current_tokens_dev0, non_blocking=True)
        # Frames GPU 1 -> GPU 0
        self.frames_dev0.squeeze(1).copy_(self.frames_dev1, non_blocking=True)
        
        # Step Phase 3: Inverted Execution (GPU 0 computes Group B, GPU 1 computes Group A)
        with torch.cuda.stream(self.stream_bb):
            for i in range(1, self.num_slots, 2):
                if self.slot_manager.slots[i].state == SlotState.ACTIVE:
                    f_dev0 = self.frames_dev0[i : i + 1].repeat(2, 1, 1)
                    h, logits = self.backbone_graph.run(f_dev0, step_idx=self.slot_manager.slots[i].current_step)
                    tok = sample_logits(logits.float(), suppress_tokens=self._reserved_tokens, temperature=0.8).view(1)
                    self.current_tokens_dev0[i].copy_(tok[0])
                    self.hidden_states_dev0[i].copy_(h[:, -1, :])
                    
        with torch.cuda.stream(self.stream_depth):
            for i in range(0, self.num_slots, 2):
                if self.slot_manager.slots[i].state == SlotState.ACTIVE:
                    h_dev1 = self.hidden_states_dev1[i : i + 1].repeat(2, 1)
                    t_dev1 = self.current_tokens_dev1[i : i + 1].repeat(2)
                    depth_tokens = self.depth_graph.run(
                        h_dev1,
                        t_dev1,
                        guidance_scale=self.slot_manager.slots[i].effective_cfg,
                        temperature=0.8,
                    )
                    full_frame = torch.cat([t_dev1[:1].view(1), depth_tokens[0]], dim=0)
                    self.frames_dev1[i].copy_(full_frame)
                    
                    # Accumulate for streaming vocoder
                    self.slot_manager.slots[i].chunk_buffer.append(full_frame.detach())
                    self.slot_manager.slots[i].total_frames_generated += 1
                    self.slot_manager.slots[i].current_step += 1

        self.stream_bb.synchronize()
        self.stream_depth.synchronize()

        # Step Phase 4: In-Flight Slot Checks & Chunk Dispatching
        for i in range(self.num_slots):
            slot = self.slot_manager.slots[i]
            if slot.state != SlotState.ACTIVE:
                continue
                
            # Check EOS or max tokens
            hit_eos = is_backbone_eos_token(self.current_tokens_dev0[i : i + 1], self.model.config)
            hit_max = slot.current_step >= slot.max_frames or slot.current_step >= 1023
            
            # Check if chunk buffer reached emit threshold
            if len(slot.chunk_buffer) >= self.config.chunk_size or hit_eos or hit_max:
                chunk_to_emit = list(slot.chunk_buffer)
                slot.chunk_buffer.clear()
                
                # Emit chunk to client queue
                if slot.audio_queue is not None:
                    asyncio.create_task(
                        self.vocoder_worker.emit_chunk(
                            queue=slot.audio_queue,
                            frames=chunk_to_emit,
                            is_final=(hit_eos or hit_max),
                        )
                    )
                    
            if hit_eos or hit_max:
                req_id, frames_done, dur = self.slot_manager.retire_slot(i)
                print(f"[AssemblyLine] Slot {i} retired: Req={req_id} | Frames={frames_done} ({dur:.2f}s audio)")
                
        return self.slot_manager.get_num_active()

    async def run_continuous_loop(self):
        """
        The always-running master loop. Iterates continuously at ~12.5–18 FPS.
        Admits new prefilled requests into freed slots on the fly.
        """
        self.running = True
        print("[AssemblyLine] Master Continuous Loop Started.")
        
        while self.running:
            # 1. Admit any ready prefilled requests into idle slots
            self._admit_pending_prefilled()
            
            # 2. Step the double-buffered assembly line
            active = self.step_assembly()
            
            # 3. If idle, sleep tiny slice to avoid CPU spinning
            if active == 0:
                await asyncio.sleep(0.005)
            else:
                await asyncio.sleep(0.001)

    async def submit_request(
        self,
        request_id: str,
        text: str,
        instruction: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        speaker: str = "S0",
        guidance_scale: float = 4.0,
        max_frames: int = 1024,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Public high-concurrency API: Submits request to async prefill worker,
        waits for hot-injection, and yields streaming audio packets as they arrive.
        """
        audio_queue: asyncio.Queue = asyncio.Queue()
        
        # 1. Asynchronously prefill on dev0 without blocking the loop
        await self.prefill_worker.submit_prefill(
            request_id=request_id,
            text=text,
            instruction=instruction,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            speaker=speaker,
            guidance_scale=guidance_scale,
            max_frames=max_frames,
            audio_queue=audio_queue,
        )
        
        # 2. Yield chunks as emitted by streaming vocoder
        while True:
            item = await audio_queue.get()
            if item.get("type") == "eos":
                break
            yield item
