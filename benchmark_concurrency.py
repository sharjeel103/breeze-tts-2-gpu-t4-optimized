"""
Automated Multi-Tenant Concurrency & TTFA Benchmark for Breeze TTS 2 Assembly Engine.
Simulates concurrent users with heterogeneous prompt lengths (short, medium, long)
to verify continuous in-flight batching, zero-waste early exit, and sub-350ms TTFA.
"""
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch

from engine.assembly_pipeline import AssemblyPipelineEngine
from engine.config import EngineConfig

BENCHMARK_PROMPTS = [
    {
        "id": "Client_1_Short",
        "text": "Order confirmed, thank you.",
        "instruction": "Speak cleanly and politely.",
        "expected_frames": 25,  # ~2 seconds
    },
    {
        "id": "Client_2_Medium",
        "text": "Welcome back to the studio. Today we will discuss modern artificial intelligence, deep neural vocoders, and real-time speech architectures.",
        "instruction": "Energetic podcast host tone.",
        "expected_frames": 85,  # ~6.8 seconds
    },
    {
        "id": "Client_3_Short",
        "text": "System update completed successfully.",
        "instruction": "Calm female assistant.",
        "expected_frames": 30,  # ~2.4 seconds
    },
    {
        "id": "Client_4_Long",
        "text": "Autonomous agent systems require meticulous engineering across GPU memory hierarchy, kernel fusion, and continuous in-flight batching to achieve optimal throughput without suffering from driver context-switching penalties or memory allocator locks.",
        "instruction": "Authoritative documentary voice.",
        "expected_frames": 160, # ~12.8 seconds
    },
]

async def run_client_worker(
    engine: AssemblyPipelineEngine,
    client_meta: Dict[str, Any],
    results: Dict[str, Any],
):
    req_id = client_meta["id"]
    text = client_meta["text"]
    instruction = client_meta["instruction"]
    
    t_start = time.time()
    t_first_chunk = None
    total_bytes = 0
    chunks_received = 0
    
    print(f"[{req_id}] -> Request Submitted ('{text[:30]}...')")
    
    async for chunk in engine.submit_request(
        request_id=req_id,
        text=text,
        instruction=instruction,
        guidance_scale=4.0,
    ):
        if chunks_received == 0:
            t_first_chunk = time.time()
            ttfa_ms = (t_first_chunk - t_start) * 1000.0
            print(f"[{req_id}] ⚡ FIRST AUDIO CHUNK RECEIVED (TTFA: {ttfa_ms:.1f} ms)!")
            
        data = chunk["data"]
        total_bytes += len(data)
        chunks_received += 1
        
    t_end = time.time()
    total_time = t_end - t_start
    
    # Calculate audio duration from PCM bytes (24000 Hz, 16-bit Mono = 48,000 bytes/sec)
    audio_dur = max(total_bytes - 44, 0) / 48000.0
    rtf = total_time / max(audio_dur, 0.001)
    
    results[req_id] = {
        "ttfa_ms": (t_first_chunk - t_start) * 1000.0 if t_first_chunk else 0.0,
        "total_time_s": total_time,
        "audio_dur_s": audio_dur,
        "chunks": chunks_received,
        "rtf": rtf,
    }
    print(f"[{req_id}] ✅ Finished: Audio={audio_dur:.2f}s | Latency={total_time:.2f}s | RTF={rtf:.2f}x")

async def main():
    print("=" * 65)
    print("🎯 STARTING DUAL-GPU CONCURRENCY & BATCHING BENCHMARK")
    print("=" * 65)
    
    cfg = EngineConfig(num_slots=4, chunk_size=5)
    engine = AssemblyPipelineEngine(cfg)
    engine.initialize()
    
    # Start continuous loop in background
    loop_task = asyncio.create_task(engine.run_continuous_loop())
    
    # Wait 1 sec for initialization stabilization
    await asyncio.sleep(1.0)
    
    # Launch all 4 concurrent clients simultaneously!
    t_global_start = time.time()
    results = {}
    
    client_tasks = [
        run_client_worker(engine, client_meta, results)
        for client_meta in BENCHMARK_PROMPTS
    ]
    
    gather_task = asyncio.create_task(asyncio.gather(*client_tasks))
    done, pending = await asyncio.wait([gather_task, loop_task], return_when=asyncio.FIRST_COMPLETED)
    if loop_task in done and loop_task.exception():
        raise loop_task.exception()
    await gather_task
    
    t_global_end = time.time()
    total_wall_time = t_global_end - t_global_start
    
    # Aggregate Metrics
    total_audio_generated = sum(r["audio_dur_s"] for r in results.values())
    aggregate_fps = total_audio_generated / 0.08 / max(total_wall_time, 0.001)
    system_throughput = total_audio_generated / max(total_wall_time, 0.001)
    
    print("\n" + "=" * 65)
    print("📊 MULTI-TENANT CONCURRENCY BENCHMARK RESULTS")
    print("=" * 65)
    print(f"{'Client ID':<22} | {'TTFA (ms)':<10} | {'Audio (s)':<10} | {'Latency (s)':<12} | {'RTF':<6}")
    print("-" * 65)
    for req_id, r in results.items():
        print(f"{req_id:<22} | {r['ttfa_ms']:<10.1f} | {r['audio_dur_s']:<10.2f} | {r['total_time_s']:<12.2f} | {r['rtf']:<6.2f}x")
        
    print("-" * 65)
    print(f"Total Wall-Clock Time     : {total_wall_time:.2f} seconds")
    print(f"Total Audio Synthesized   : {total_audio_generated:.2f} seconds across 4 users")
    print(f"Aggregate System Throughput: {system_throughput:.2f}x Real-Time ({aggregate_fps:.1f} audio frames/sec)")
    print(f"Average TTFA Across Users : {np.mean([r['ttfa_ms'] for r in results.values()]):.1f} ms")
    print("=" * 65)
    
    engine.running = False
    loop_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
