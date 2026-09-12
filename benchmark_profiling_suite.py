"""
Comprehensive Multi-Condition Concurrency & Dual-GPU Telemetry Benchmark.
Evaluates:
  Condition 1: N=1 (Single Short & Single Long Request)
  Condition 2: N=2 (Dual Concurrent Requests - Macro Pipeline Activation)
  Condition 3: N=3 (Dual Active + 1 Queued - Seamless Admission Handoff)
  Condition 4: N=4 (Heterogeneous Mix - Zero Head-of-Line Blocking)
  Condition 5: N=10 (Scale Burst Stress Test - Queue Backlog & Continuous Recycling)

Captures 200ms low-overhead hardware telemetry (GPUProfiler) and per-client QoS metrics.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from engine.assembly_pipeline import AssemblyPipelineEngine
from engine.config import EngineConfig
from engine.gpu_profiler import GPUProfiler

PROMPT_POOL = [
    {
        "id": "Req_01_Short",
        "text": "Order confirmed, thank you.",
        "instruction": "Speak cleanly and politely.",
    },
    {
        "id": "Req_02_Medium",
        "text": "Welcome back to the studio. Today we will discuss modern artificial intelligence, deep neural vocoders, and real-time speech architectures.",
        "instruction": "Energetic podcast host tone.",
    },
    {
        "id": "Req_03_Short",
        "text": "System update completed successfully.",
        "instruction": "Calm female assistant.",
    },
    {
        "id": "Req_04_Long",
        "text": "Autonomous agent systems require meticulous engineering across GPU memory hierarchy, kernel fusion, and continuous in-flight batching to achieve optimal throughput without suffering from driver context-switching penalties or memory allocator locks.",
        "instruction": "Authoritative documentary voice.",
    },
    {
        "id": "Req_05_Short",
        "text": "Good morning! How may I assist you today?",
        "instruction": "Friendly customer service.",
    },
    {
        "id": "Req_06_Medium",
        "text": "Neural text to speech models have advanced rapidly over the past few years, enabling expressive prosody and zero-shot voice cloning.",
        "instruction": "Informative narrator.",
    },
    {
        "id": "Req_07_Short",
        "text": "Your flight has been scheduled on time.",
        "instruction": "Airport announcement chime.",
    },
    {
        "id": "Req_08_Long",
        "text": "Distributed model execution across multi-GPU environments requires balancing compute density, PCIe interconnect bandwidth, and tensor placement to maintain high GPU occupancy without pipeline starvation.",
        "instruction": "Technical professor lecture.",
    },
    {
        "id": "Req_09_Medium",
        "text": "Thank you for joining our live technology conference, streaming worldwide to engineers and researchers.",
        "instruction": "Conference keynote speaker.",
    },
    {
        "id": "Req_10_Short",
        "text": "All operations finalized. Have a great day!",
        "instruction": "Pleasant closing tone.",
    },
]

async def run_client_task(
    engine: AssemblyPipelineEngine,
    client_meta: Dict[str, Any],
    results: Dict[str, Any],
    global_t0: float,
):
    req_id = client_meta["id"]
    text = client_meta["text"]
    instruction = client_meta.get("instruction", "Speak naturally.")
    
    t_submit = time.time()
    t_first_chunk = None
    total_bytes = 0
    chunks_received = 0
    
    print(f"  [{req_id}] -> Submitted at T+{(t_submit - global_t0):.2f}s ('{text[:32]}...')")
    
    async for chunk in engine.submit_request(
        request_id=req_id,
        text=text,
        instruction=instruction,
        guidance_scale=4.0,
    ):
        if chunks_received == 0:
            t_first_chunk = time.time()
            ttfa_ms = (t_first_chunk - t_submit) * 1000.0
            print(f"  [{req_id}] ⚡ TTFA: {ttfa_ms:.1f}ms (T+{(t_first_chunk - global_t0):.2f}s)")
            
        data = chunk["data"]
        total_bytes += len(data)
        chunks_received += 1
        
    t_end = time.time()
    total_latency = t_end - t_submit
    # 24000 Hz, 16-bit mono = 48000 bytes/sec
    audio_dur = max(total_bytes - 44, 0) / 48000.0
    rtf = total_latency / max(audio_dur, 0.001)
    
    results[req_id] = {
        "text": text,
        "submit_t": round(t_submit - global_t0, 3),
        "ttfa_ms": round((t_first_chunk - t_submit) * 1000.0, 1) if t_first_chunk else 0.0,
        "complete_t": round(t_end - global_t0, 3),
        "total_latency_s": round(total_latency, 3),
        "audio_dur_s": round(audio_dur, 2),
        "chunks": chunks_received,
        "rtf": round(rtf, 2),
    }
    print(f"  [{req_id}] ✅ Complete: Audio={audio_dur:.2f}s | Latency={total_latency:.2f}s | RTF={rtf:.2f}x (Exit at T+{(t_end - global_t0):.2f}s)")

async def run_condition_benchmark(
    engine: AssemblyPipelineEngine,
    condition_name: str,
    prompts: List[Dict[str, Any]],
    csv_path: str,
) -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print(f"🔬 RUNNING BENCHMARK: {condition_name} ({len(prompts)} concurrent requests)")
    print("=" * 75)
    
    client_results: Dict[str, Any] = {}
    profiler = GPUProfiler(log_csv_path=csv_path, interval_ms=200)
    
    # Start profiler
    profiler.start()
    t_start = time.time()
    
    # Launch concurrent client coroutines
    tasks = [
        run_client_task(engine, p, client_results, t_start)
        for p in prompts
    ]
    await asyncio.gather(*tasks)
    
    t_end = time.time()
    hardware_telemetry = profiler.stop()
    
    wall_time = t_end - t_start
    total_audio = sum(r["audio_dur_s"] for r in client_results.values())
    throughput_mult = total_audio / max(wall_time, 0.001)
    agg_fps = (total_audio / 0.08) / max(wall_time, 0.001)
    
    # Print Per-Request QoS Table
    print("\n" + "-" * 75)
    print(f"📊 {condition_name} - Per-Client User Experience")
    print("-" * 75)
    print(f"{'Client ID':<18} | {'Audio (s)':<10} | {'TTFA (ms)':<10} | {'Latency (s)':<12} | {'RTF':<6} | {'Exit (s)':<8}")
    print("-" * 75)
    for cid, r in client_results.items():
        print(f"{cid:<18} | {r['audio_dur_s']:<10.2f} | {r['ttfa_ms']:<10.1f} | {r['total_latency_s']:<12.2f} | {r['rtf']:<6.2f}x | T+{r['complete_t']:<6.2f}s")
        
    print("-" * 75)
    print(f"Wall-Clock Duration    : {wall_time:.2f}s")
    print(f"Total Audio Synthesized: {total_audio:.2f}s")
    print(f"Cluster Throughput     : {throughput_mult:.2f}x Real-Time ({agg_fps:.1f} frames/sec)")
    
    # Print Hardware Stats Summary
    if hardware_telemetry.get("status") == "SUCCESS":
        print("\n⚡ Hardware Telemetry Summary (200ms Native Daemon):")
        for gname, gstats in hardware_telemetry.get("gpus", {}).items():
            cu = gstats["core_utilization"]
            mu = gstats["memory_bus_utilization"]
            pw = gstats["power_watts"]
            vr = gstats["vram_mb"]
            print(f"  [{gname.upper()}] Core Util: Mean={cu['mean_pct']}% | Max={cu['max_pct']}% | P90={cu['p90_pct']}% | Saturated(>=90%)={cu['pct_time_saturated_above_90']}% | Idle(<20%)={cu['pct_time_idle_below_20']}%")
            print(f"             Memory Bus: Mean={mu['mean_pct']}% | Max={mu['max_pct']}% | Power: Mean={pw['mean_w']}W / 70W | VRAM: Peak={vr['peak_allocated_mb']}MB (Free: {vr['remaining_headroom_mb']}MB)")
        
        comb = hardware_telemetry.get("cluster_combined_power", {})
        print(f"  [CLUSTER] Total Power: Mean={comb.get('mean_total_watts', 0)}W | Max={comb.get('max_total_watts', 0)}W / {comb.get('cluster_tdp_envelope_w', 140)}W TDP")
    print("=" * 75)
    
    # Rest 2 seconds between conditions to let GPU cool down and queues settle
    await asyncio.sleep(2.0)
    
    return {
        "condition": condition_name,
        "num_requests": len(prompts),
        "wall_time_s": round(wall_time, 2),
        "total_audio_s": round(total_audio, 2),
        "throughput_multiplier": round(throughput_mult, 2),
        "aggregate_fps": round(agg_fps, 1),
        "client_results": client_results,
        "hardware_telemetry": hardware_telemetry,
    }

async def main():
    print("*" * 75)
    print("🚀 BREEZE TTS 2: PRODUCTION DUAL-GPU TELEMETRY & CONCURRENCY BENCHMARK")
    print("*" * 75)
    
    cfg = EngineConfig(num_slots=4, chunk_size=5)
    engine = AssemblyPipelineEngine(cfg)
    engine.initialize()
    
    # Launch continuous assembly line background loop
    loop_task = asyncio.create_task(engine.run_continuous_loop())
    await asyncio.sleep(1.0)
    
    all_benchmarks: Dict[str, Any] = {}
    
    try:
        # -------------------------------------------------------------
        # Condition 1A: N=1 Single Short Request
        # -------------------------------------------------------------
        res_1a = await run_condition_benchmark(
            engine,
            condition_name="Condition_1A_Single_Short",
            prompts=[PROMPT_POOL[0]],
            csv_path="/tmp/gpu_profile_c1a.csv",
        )
        all_benchmarks["Condition_1A"] = res_1a

        # -------------------------------------------------------------
        # Condition 1B: N=1 Single Long Request (~15s audio)
        # -------------------------------------------------------------
        res_1b = await run_condition_benchmark(
            engine,
            condition_name="Condition_1B_Single_Long",
            prompts=[PROMPT_POOL[3]],
            csv_path="/tmp/gpu_profile_c1b.csv",
        )
        all_benchmarks["Condition_1B"] = res_1b

        # -------------------------------------------------------------
        # Condition 2: N=2 Dual Concurrent Requests (Station A || Station B)
        # -------------------------------------------------------------
        res_2 = await run_condition_benchmark(
            engine,
            condition_name="Condition_2_Dual_Concurrent",
            prompts=[PROMPT_POOL[1], PROMPT_POOL[5]],
            csv_path="/tmp/gpu_profile_c2.csv",
        )
        all_benchmarks["Condition_2"] = res_2

        # -------------------------------------------------------------
        # Condition 3: N=3 (2 Active in Stations + 1 Queued)
        # -------------------------------------------------------------
        res_3 = await run_condition_benchmark(
            engine,
            condition_name="Condition_3_Three_Concurrent",
            prompts=[PROMPT_POOL[1], PROMPT_POOL[2], PROMPT_POOL[5]],
            csv_path="/tmp/gpu_profile_c3.csv",
        )
        all_benchmarks["Condition_3"] = res_3

        # -------------------------------------------------------------
        # Condition 4: N=4 Heterogeneous (Short, Medium, Long)
        # -------------------------------------------------------------
        res_4 = await run_condition_benchmark(
            engine,
            condition_name="Condition_4_Four_Heterogeneous",
            prompts=[PROMPT_POOL[0], PROMPT_POOL[1], PROMPT_POOL[2], PROMPT_POOL[3]],
            csv_path="/tmp/gpu_profile_c4.csv",
        )
        all_benchmarks["Condition_4"] = res_4

        # -------------------------------------------------------------
        # Condition 5: N=10 Scale Burst Stress Test (10 Concurrent Requests)
        # -------------------------------------------------------------
        res_5 = await run_condition_benchmark(
            engine,
            condition_name="Condition_5_Ten_Request_Burst",
            prompts=PROMPT_POOL,
            csv_path="/tmp/gpu_profile_c5.csv",
        )
        all_benchmarks["Condition_5"] = res_5

    finally:
        engine.running = False
        loop_task.cancel()

    # Save comprehensive results
    output_path = Path("/kaggle/working/benchmark_results_telemetry.json") if Path("/kaggle/working").exists() else Path("benchmark_results_telemetry.json")
    with open(output_path, "w") as f:
        json.dump(all_benchmarks, f, indent=2)
        
    print("\n" + "*" * 75)
    print(f"🎉 ALL BENCHMARK CONDITIONS COMPLETE! Results saved to: {output_path}")
    print("*" * 75)

if __name__ == "__main__":
    asyncio.run(main())
