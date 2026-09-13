"""
Configuration for the Production Assembly Line Engine.
"""
from dataclasses import dataclass, field
from typing import Optional
import torch

@dataclass
class EngineConfig:
    # Slot & Concurrency Settings
    num_slots: int = 4              # Number of in-flight active slots (4 concurrent streams)
    batch_size_per_station: int = 2 # In-flight batch size per station (2 on dev0, 2 on dev1)
    max_seq_len: int = 2560         # Max acoustic frames (2560 * 0.08s = 204.8s for 500-word prompts)
    chunk_size: int = 8             # Frames per streaming vocoder chunk (8 * 80ms = 640ms)
    sample_rate: int = 24000        # Audio sample rate in Hz
    
    # Device Allocations
    dev0: str = "cuda:0"            # GPU 0: Text Encoder + Backbone + Station A Depth
    dev1: str = "cuda:1"            # GPU 1: Station B Depth + Neural Vocoder
    vocoder_dev: str = "cuda:1"     # Neural vocoder on GPU 1 (balances VRAM & removes GPU 0 contention)
    enable_dual_depth: bool = True  # Station A on dev0, Station B on dev1
    
    # Model Weights & Precision
    model_id: str = "BreezeBlue/Breeze-TTS-2"
    dtype: torch.dtype = torch.float16
    
    # Sampling Defaults
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    guidance_scale: float = 1.0     # 1.0 for sub-1.0x Real-Time Fastpath; 4.0 for Guided Voice Design
    
    # Assembly Pipeline Tuning
    enable_macro_pipeline: bool = True  # Interleaved Batch A / Batch B execution
    enable_cuda_graphs: bool = True     # Static CUDA Graph replay
    lock_free_queues: bool = True       # AsyncIO thread-safe communication
