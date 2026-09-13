"""
Comprehensive Multi-Condition Concurrency & Dual-GPU Telemetry Benchmark.
Includes:
  Condition 1: N=2 Dual Concurrent (Station A on GPU 0 || Station B on GPU 1)
  Condition 2: N=15 Scale Burst Stress Test (with 500-word ultra-long prompt)

Measures 200ms low-overhead hardware telemetry and provides exact row-by-row decomposed RTF:
  - Queue Wait Time (s)
  - Active Compute Time (s)
  - Audio Duration (s)
  - Active Compute RTF (Pure Hardware Speed)
  - Turnaround RTF (End-to-End Client Experience)
  - Post-Admission TTFA (ms)
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

ESSAY_500_WORDS = (
    "Artificial intelligence has undergone a profound transformation over the past decade, "
    "evolving from rudimentary rule-based systems into sophisticated neural architectures "
    "capable of synthesizing natural speech, understanding multilingual text, and reasoning "
    "through intricate scientific problems. At the foundation of this revolution lies deep learning, "
    "powered by massive parallel computation on graphics processing units and specialized tensor accelerators. "
    "Modern neural speech synthesis represents one of the most compelling achievements in this domain. "
    "Historically, text to speech relied on concatenative methods, stitching together pre-recorded "
    "acoustic fragments from human speakers, or statistical parametric models that produced robotic "
    "and muffled voices. Today, generative models utilize autoregressive semantic language models "
    "paired with non-autoregressive depth transformers and neural vocoders to produce lifelike "
    "voices with expressive prosody, emotional nuance, and zero-shot speaker cloning. Achieving this "
    "level of quality requires unprecedented engineering across the entire computing stack. Engineers "
    "must optimize memory bandwidth, tensor core occupancy, static cache allocation, and kernel launch "
    "latency to ensure real-time interaction. When serving millions of concurrent users, techniques such "
    "as continuous in-flight batching, asynchronous prefilling, and decoupled streaming architectures "
    "become indispensable. As machine learning hardware advances toward higher memory bandwidth and "
    "specialized tensor instructions, the boundary between biological and synthetic voice continues to "
    "dissolve, opening new horizons for communication, accessibility, education, and creative human "
    "expression across the globe. Looking forward, distributed computing paradigms must address the "
    "physical realities of interconnect latency and thermal design limits. As neural parameters expand "
    "into the billions, distributing layers across multiple physical devices demands rigorous balancing "
    "of compute density and bus throughput. By orchestrating deterministic memory strides, CUDA graph "
    "replays, and zero-allocation execution loops, modern inference engines unlock the full physical "
    "potential of silicon architectures. This convergence of software precision and hardware design "
    "ensures that intelligent conversational agents can respond instantaneously with natural warmth and "
    "clarity, redefining how humanity interacts with computational intelligence forever. Furthermore, "
    "the emergence of multi-modal foundation models demonstrates that voice is an essential communicative "
    "bridge carrying intent, sentiment, urgency, and subtle empathetic cues across dynamic environments. "
    "The future of interactive voice technology promises seamless, ubiquitous, and deeply personalized "
    "acoustic experiences for everyone."
)

PROMPT_POOL_40 = [
    # Batch 1 (01 - 10): Customer Service & Announcements
    {"id": "Req_01", "text": "Your order has been confirmed and is currently being packed for express delivery this afternoon.", "instruction": "Clean and polite."},
    {"id": "Req_02", "text": "The system update completed successfully without any errors, and all security patches have been safely installed.", "instruction": "Calm female assistant."},
    {"id": "Req_03", "text": "Good morning! Please let me know how I can assist you with your schedule and appointments today.", "instruction": "Friendly customer service."},
    {"id": "Req_04", "text": "Flight seven twelve to San Francisco is now boarding at terminal two, gate twenty-four.", "instruction": "Airport announcement."},
    {"id": "Req_05", "text": "The project planning meeting has been rescheduled to Thursday at three in the afternoon, conference room four.", "instruction": "Professional assistant."},
    {"id": "Req_06", "text": "Your payment was processed successfully, and an itemized digital receipt has been sent to your primary email.", "instruction": "Polite confirmation."},
    {"id": "Req_07", "text": "The air quality index is currently moderate across the valley with light northwesterly winds throughout the morning.", "instruction": "Radio broadcaster."},
    {"id": "Req_08", "text": "All cluster network services are running normally with optimal bandwidth and zero packet loss detected.", "instruction": "Technical narrator."},
    {"id": "Req_09", "text": "Please remember to submit your weekly engineering progress report before five this evening for team review.", "instruction": "Courteous reminder."},
    {"id": "Req_10", "text": "Your ride has arrived outside the main hotel lobby. The silver vehicle license plate is five alpha seven.", "instruction": "Navigation prompt."},

    # Batch 2 (11 - 20): Assistant & Smart Home
    {"id": "Req_11", "text": "Today will be mostly clear and sunny with mild afternoon temperatures reaching seventy-four degrees across the city.", "instruction": "Weather reporter."},
    {"id": "Req_12", "text": "The conference keynote begins in ten minutes in the primary auditorium on floor three, open to all attendees.", "instruction": "Event coordinator."},
    {"id": "Req_13", "text": "Your prescription order is ready for pickup at the neighborhood pharmacy counter on Maple Avenue.", "instruction": "Healthcare assistant."},
    {"id": "Req_14", "text": "The express commuter train to central station will depart from platform four in exactly six minutes.", "instruction": "Transit announcement."},
    {"id": "Req_15", "text": "A new firmware update is available for your smart display. Please ensure a stable Wi-Fi connection to proceed.", "instruction": "Helpful smart home assistant."},
    {"id": "Req_16", "text": "Your checking account balance has been updated following the recent automated monthly savings transfer.", "instruction": "Banking alert."},
    {"id": "Req_17", "text": "The university library will be closing in fifteen minutes. Please bring all borrowed materials to the front circulation desk.", "instruction": "Campus announcement."},
    {"id": "Req_18", "text": "Traffic on the interstate highway is moving smoothly with an estimated total travel time of twenty-two minutes.", "instruction": "Navigation assistant."},
    {"id": "Req_19", "text": "Welcome to the national science center. Guided audio tours commence every hour on the hour at the main rotunda.", "instruction": "Tour guide."},
    {"id": "Req_20", "text": "Your table reservation for four guests at Bistro Bella has been confirmed for eight tonight on the patio.", "instruction": "Concierge tone."},

    # Batch 3 (21 - 30): Alerts, Logistics & Operational
    {"id": "Req_21", "text": "The morning courier package has been safely delivered to the front reception desk for your immediate collection.", "instruction": "Office concierge."},
    {"id": "Req_22", "text": "Routine server infrastructure maintenance is scheduled for tonight at midnight and will last approximately one hour.", "instruction": "System administrator."},
    {"id": "Req_23", "text": "Temperatures will drop noticeably tonight under clear starry skies with a gentle autumn breeze from the north.", "instruction": "Calm narrator."},
    {"id": "Req_24", "text": "Your international flight check-in is complete, and your digital boarding passes have been synchronized to your phone.", "instruction": "Airline assistant."},
    {"id": "Req_25", "text": "The live technical webinar on distributed computing architectures will begin promptly at noon Eastern Standard Time.", "instruction": "Webinar host."},
    {"id": "Req_26", "text": "Security notification: a new login was detected from a personal laptop in Chicago, Illinois. Please verify your identity.", "instruction": "Security alert."},
    {"id": "Req_27", "text": "The passenger elevator on the north wing is currently undergoing maintenance and will reopen at two this afternoon.", "instruction": "Building announcement."},
    {"id": "Req_28", "text": "Your premium software subscription has been renewed successfully, unlocking continuous priority access to all cloud tools.", "instruction": "Customer care."},
    {"id": "Req_29", "text": "Passengers traveling to terminal B should proceed to shuttle stop three for immediate baggage transfer.", "instruction": "Transit audio."},
    {"id": "Req_30", "text": "The downtown business shuttle departs every fifteen minutes from the central transit plaza near the historic clock tower.", "instruction": "City transit guide."},

    # Batch 4 (31 - 40): Daily Summaries & Notifications
    {"id": "Req_31", "text": "A temporary authorization code has been dispatched to your mobile phone number via secure text messaging.", "instruction": "Verification voice."},
    {"id": "Req_32", "text": "The resident fitness facility will remain open until eleven tonight for all registered hotel and club members.", "instruction": "Hospitality host."},
    {"id": "Req_33", "text": "Local traffic monitors report minor road construction delays near the east river crossing during evening peak hours.", "instruction": "Traffic broadcast."},
    {"id": "Req_34", "text": "Your analytical quarterly summary report has finished generating and is now available for download on the management portal.", "instruction": "Business assistant."},
    {"id": "Req_35", "text": "The interactive workshop on modern deep learning frameworks begins at ten sharp in computer laboratory C.", "instruction": "Instructor voice."},
    {"id": "Req_36", "text": "Thank you for visiting our technology showroom today. Please take your complimentary catalog and have a wonderful day.", "instruction": "Warm goodbye."},
    {"id": "Req_37", "text": "Tomorrow's weather forecast calls for brief morning showers followed by pleasant sunshine and light southerly breezes.", "instruction": "Weather anchor."},
    {"id": "Req_38", "text": "Your consultation appointment with Doctor Reynolds has been confirmed for Tuesday morning at ten thirty.", "instruction": "Medical receptionist."},
    {"id": "Req_39", "text": "The production cluster deployment completed without incident, and all containerized microservices report healthy operational status.", "instruction": "DevOps assistant."},
    {"id": "Req_40", "text": "All pending banking transactions have cleared, and your comprehensive monthly financial statement is now available to view.", "instruction": "Financial adviser."},
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
    t_admit = None
    t_first_chunk = None
    station_admitted = None
    total_bytes = 0
    chunks_received = 0
    
    print(f"  [{req_id}] -> Submitted at T+{(t_submit - global_t0):.2f}s ('{text[:30]}...')")
    
    async for item in engine.submit_request(
        request_id=req_id,
        text=text,
        instruction=instruction,
        guidance_scale=engine.config.guidance_scale,
        max_frames=engine.config.max_seq_len,
    ):
        if item.get("type") == "admitted":
            t_admit = item["t_admit"]
            station_admitted = item.get("station", "?")
            wait_s = t_admit - t_submit
            print(f"  [{req_id}] 🎟️ Admitted to Station {station_admitted} at T+{(t_admit - global_t0):.2f}s (Queue wait: {wait_s:.2f}s)")
            continue
            
        if item.get("type") == "audio_chunk":
            if chunks_received == 0:
                t_first_chunk = time.time()
                total_ttfa_ms = (t_first_chunk - t_submit) * 1000.0
                post_admit_ttfa_ms = ((t_first_chunk - t_admit) * 1000.0) if t_admit else total_ttfa_ms
                print(f"  [{req_id}] ⚡ First Chunk: Total TTFA={total_ttfa_ms:.1f}ms | Post-Admit TTFA={post_admit_ttfa_ms:.1f}ms (T+{(t_first_chunk - global_t0):.2f}s)")
                
            data = item["data"]
            total_bytes += len(data)
            chunks_received += 1
        
    t_end = time.time()
    total_latency = t_end - t_submit
    t_admit = t_admit or t_submit
    active_compute_s = t_end - t_admit
    queue_wait_s = t_admit - t_submit
    
    # 24000 Hz, 16-bit mono = 48000 bytes/sec
    audio_dur = max(total_bytes - 44, 0) / 48000.0
    active_rtf = active_compute_s / max(audio_dur, 0.001)
    turnaround_rtf = total_latency / max(audio_dur, 0.001)
    post_admit_ttfa = ((t_first_chunk - t_admit) * 1000.0) if (t_first_chunk and t_admit) else 0.0
    total_ttfa = ((t_first_chunk - t_submit) * 1000.0) if t_first_chunk else 0.0
    
    results[req_id] = {
        "text_snippet": text[:40],
        "station": station_admitted,
        "queue_wait_s": round(queue_wait_s, 3),
        "active_compute_s": round(active_compute_s, 3),
        "total_latency_s": round(total_latency, 3),
        "audio_dur_s": round(audio_dur, 2),
        "post_admit_ttfa_ms": round(post_admit_ttfa, 1),
        "total_ttfa_ms": round(total_ttfa, 1),
        "active_rtf": round(active_rtf, 2),
        "turnaround_rtf": round(turnaround_rtf, 2),
        "exit_t": round(t_end - global_t0, 2),
    }
    print(f"  [{req_id}] ✅ Complete: Audio={audio_dur:.2f}s | ActiveRTF={active_rtf:.2f}x | QueueWait={queue_wait_s:.2f}s | TurnaroundRTF={turnaround_rtf:.2f}x (Exit T+{(t_end - global_t0):.2f}s)")

async def run_condition_benchmark(
    engine: AssemblyPipelineEngine,
    condition_name: str,
    prompts: List[Dict[str, Any]],
    csv_path: str,
) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print(f"🔬 BENCHMARK: {condition_name} ({len(prompts)} concurrent requests)")
    print("=" * 90)
    
    client_results: Dict[str, Any] = {}
    profiler = GPUProfiler(log_csv_path=csv_path, interval_ms=200)
    
    profiler.start()
    t_start = time.time()
    
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
    
    # Print Row-by-Row Decomposed Table
    print("\n" + "-" * 95)
    print(f"📊 {condition_name} - Row-by-Row Complete QoS Decomposition")
    print("-" * 95)
    header = f"{'Client ID':<16} | {'Station':<7} | {'Audio(s)':<8} | {'Wait(s)':<7} | {'Active(s)':<9} | {'PostTTFA':<9} | {'ActiveRTF':<9} | {'TotalRTF':<8} | {'Exit'}"
    print(header)
    print("-" * 95)
    for cid, r in client_results.items():
        row = (
            f"{cid:<16} | {str(r['station']):<7} | {r['audio_dur_s']:<8.2f} | "
            f"{r['queue_wait_s']:<7.2f} | {r['active_compute_s']:<9.2f} | "
            f"{r['post_admit_ttfa_ms']:<7.1f}ms | {r['active_rtf']:<7.2f}x | "
            f"{r['turnaround_rtf']:<6.2f}x | T+{r['exit_t']:.2f}s"
        )
        print(row)
        
    print("-" * 95)
    print(f"Wall-Clock Duration     : {wall_time:.2f}s")
    print(f"Total Audio Synthesized : {total_audio:.2f}s across {len(prompts)} requests")
    print(f"Cluster Throughput      : {throughput_mult:.2f}x Real-Time ({agg_fps:.1f} frames/sec)")
    
    if hardware_telemetry.get("status") == "SUCCESS":
        print("\n⚡ Dual-GPU Telemetry (200ms Native Daemon):")
        for gname, gstats in hardware_telemetry.get("gpus", {}).items():
            cu = gstats["core_utilization"]
            mu = gstats["memory_bus_utilization"]
            pw = gstats["power_watts"]
            vr = gstats["vram_mb"]
            print(f"  [{gname.upper()}] Core Util: Mean={cu['mean_pct']}% | Max={cu['max_pct']}% | P90={cu['p90_pct']}% | Saturated(>=90%)={cu['pct_time_saturated_above_90']}% | Idle(<20%)={cu['pct_time_idle_below_20']}%")
            print(f"             Memory Bus: Mean={mu['mean_pct']}% | Power: Mean={pw['mean_w']}W / 70W | VRAM: Peak={vr['peak_allocated_mb']}MB (Free: {vr['remaining_headroom_mb']}MB)")
        
        comb = hardware_telemetry.get("cluster_combined_power", {})
        print(f"  [CLUSTER] Total Power: Mean={comb.get('mean_total_watts', 0)}W | Max={comb.get('max_total_watts', 0)}W / {comb.get('cluster_tdp_envelope_w', 140)}W TDP")
    print("=" * 95)
    
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
    print("*" * 90)
    print("🚀 BREEZE TTS 2: SYMMETRICAL DUAL-ENGINE CLUSTER & 15-REQUEST HARD LIMIT BENCHMARK")
    print("*" * 90)
    
    cfg = EngineConfig(max_seq_len=2560, chunk_size=8, guidance_scale=1.0)
    engine = AssemblyPipelineEngine(cfg)
    engine.initialize()
    
    loop_task = asyncio.create_task(engine.run_continuous_loop())
    await asyncio.sleep(1.0)
    
    all_benchmarks: Dict[str, Any] = {}
    
    try:
        # -------------------------------------------------------------
        # Condition 1: N=2 Dual Concurrent Symmetrical Verification
        # -------------------------------------------------------------
        res_dual = await run_condition_benchmark(
            engine,
            condition_name="Condition_1_Symmetrical_Dual",
            prompts=[PROMPT_POOL_40[0], PROMPT_POOL_40[1]],  # 2 balanced prompts
            csv_path="/tmp/gpu_profile_symmetrical_dual.csv",
        )
        all_benchmarks["Condition_1_Symmetrical_Dual"] = res_dual

        # -------------------------------------------------------------
        # Condition 2: N=40 Balanced Scale Burst Stress Test
        # -------------------------------------------------------------
        res_40 = await run_condition_benchmark(
            engine,
            condition_name="Condition_2_Forty_Request_Burst",
            prompts=PROMPT_POOL_40,  # All 40 balanced requests
            csv_path="/tmp/gpu_profile_40_burst.csv",
        )
        all_benchmarks["Condition_2_Forty_Request_Burst"] = res_40

    finally:
        engine.running = False
        loop_task.cancel()

    output_path = Path("/kaggle/working/benchmark_results_telemetry.json") if Path("/kaggle/working").exists() else Path("benchmark_results_telemetry.json")
    with open(output_path, "w") as f:
        json.dump(all_benchmarks, f, indent=2)
        
    print("\n" + "*" * 90)
    print(f"🎉 COMPREHENSIVE BENCHMARK COMPLETE! Results saved to: {output_path}")
    print("*" * 90)

if __name__ == "__main__":
    asyncio.run(main())
