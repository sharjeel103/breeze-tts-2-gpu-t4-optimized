"""
Configuration for the Production Assembly Line Engine.
"""
from dataclasses import dataclass, field
from typing import Optional
import torch

@dataclass
class EngineConfig:
    # Slot & Concurrency Settings
    num_slots: int = 4              # Number of in-flight active slots (4 or 8)
    max_seq_len: int = 1024         # Max acoustic frames (1024 * 0.08s = 81.92s)
    chunk_size: int = 5             # Frames per streaming vocoder chunk (5 * 80ms = 400ms)
    sample_rate: int = 24000        # Audio sample rate in Hz
    
    # Device Allocations
    dev0: str = "cuda:0"            # GPU 0: Text Encoder + Backbone
    dev1: str = "cuda:1"            # GPU 1: Depth Decoder + Neural Vocoder
    
    # Model Weights & Precision
    model_id: str = "BreezeBlue/Breeze-TTS-2"
    dtype: torch.dtype = torch.float16
    
    # Sampling Defaults
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    guidance_scale: float = 4.0     # Default for Voice Design / Edit
    
    # Assembly Pipeline Tuning
    enable_macro_pipeline: bool = True  # Interleaved Batch A / Batch B execution
    enable_cuda_graphs: bool = True     # Static CUDA Graph replay
    lock_free_queues: bool = True       # AsyncIO thread-safe communication
