"""
40-Request In-Flight Dual-Batching Benchmark Suite for Breeze TTS 2.
Evaluates 4 concurrent slots across Dual Tesla T4 GPUs (batch_size=2 on GPU 0, batch_size=2 on GPU 1).
Measures:
  1. Per-request QoS (Queue Wait, Active Compute, Post-Admit TTFA, Active RTF, Turnaround RTF).
  2. Cluster Throughput (Total Audio Synthesized / Total Wall-Clock Time).
  3. Continuous 200ms Dual-GPU Telemetry (Core Util, Memory Bus, Power Draw, Peak VRAM).
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.assembly_pipeline import AssemblyPipelineEngine
from engine.config import EngineConfig
from engine.gpu_profiler import GPUProfiler


async def run_client_worker(
    engine: AssemblyPipelineEngine,
    client_meta: Dict[str, Any],
    t_global_start: float,
    results: Dict[str, Any],
):
    req_id = client_meta["id"]
    text = client_meta["text"]
    instruction = client_meta.get("instruction", "Speak clearly.")
    speaker = client_meta.get("speaker", "S0")

    t_submit = time.time()
    t_admit = None
    t_first_chunk = None
    station_admitted = None
    slot_admitted = None
    total_bytes = 0
    chunks_received = 0

    try:
        async for item in engine.submit_request(
            request_id=req_id,
            text=text,
            instruction=instruction,
            speaker=speaker,
            guidance_scale=1.0,  # Fastpath independent generation
        ):
            msg_type = item.get("type")
            if msg_type == "admitted":
                t_admit = item.get("t_admit", time.time())
                station_admitted = item.get("station", "?")
                slot_admitted = item.get("slot", 0)
            elif msg_type == "audio_chunk":
                if chunks_received == 0:
                    t_first_chunk = time.time()
                data = item.get("data", b"")
                total_bytes += len(data)
                chunks_received += 1
    except Exception as e:
        print(f"[{req_id}] ERROR in client worker: {e}", flush=True)

    t_end = time.time()
    t_admit_clean = t_admit or t_submit
    t_first_clean = t_first_chunk or t_end

    audio_dur = max(total_bytes - 44, 0) / 48000.0  # 24kHz 16-bit Mono = 48,000 bytes/sec
    queue_wait = max(t_admit_clean - t_submit, 0.0)
    active_compute = max(t_end - t_admit_clean, 0.001)
    post_admit_ttfa_ms = max(t_first_clean - t_admit_clean, 0.0) * 1000.0
    active_rtf = active_compute / max(audio_dur, 0.001)
    turnaround_rtf = (t_end - t_submit) / max(audio_dur, 0.001)
    exit_wall = t_end - t_global_start

    results[req_id] = {
        "id": req_id,
        "category": client_meta.get("category", "General"),
        "text": text,
        "station": station_admitted,
        "slot": slot_admitted,
        "audio_duration_seconds": round(audio_dur, 2),
        "queue_wait_seconds": round(queue_wait, 2),
        "active_compute_seconds": round(active_compute, 2),
        "post_admit_ttfa_ms": round(post_admit_ttfa_ms, 1),
        "active_rtf": round(active_rtf, 2),
        "turnaround_rtf": round(turnaround_rtf, 2),
        "exit_wall_time": f"T+{exit_wall:.2f}s",
        "chunks": chunks_received,
    }

    print(
        f"[{req_id}] Stn {station_admitted}-S{slot_admitted} Done: "
        f"Audio={audio_dur:.2f}s | Wait={queue_wait:.2f}s | Active={active_compute:.2f}s | "
        f"TTFA={post_admit_ttfa_ms:.1f}ms | RTF={active_rtf:.2f}x | Wall=T+{exit_wall:.2f}s",
        flush=True,
    )


async def main():
    print("=" * 80)
    print("⚡ 40-REQUEST IN-FLIGHT DUAL-BATCHING BURST BENCHMARK (DUAL TESLA T4)")
    print("⚡ Architecture: Symmetrical Dual-Engine with batch_size=2 per station (4 slots)")
    print("=" * 80)

    # 1. Load Requests
    req_file = REPO_ROOT / "benchmark_requests_40.json"
    if not req_file.exists():
        raise FileNotFoundError(f"Missing {req_file}")
    with open(req_file, "r") as f:
        meta_data = json.load(f)
    requests_pool = meta_data.get("requests", [])
    print(f"-> Loaded {len(requests_pool)} requests from {req_file.name}")

    # 2. Configure & Initialize Engine
    cfg = EngineConfig(
        num_slots=4,
        batch_size_per_station=2,
        chunk_size=8,
        guidance_scale=1.0,
    )
    engine = AssemblyPipelineEngine(cfg)
    engine.initialize()

    # 3. Start Continuous Assembly Loop
    loop_task = asyncio.create_task(engine.run_continuous_loop())
    await asyncio.sleep(1.0)  # Allow loop to stabilize

    # 4. Start Continuous 200ms Telemetry Profiler
    profiler = GPUProfiler(log_csv_path="/tmp/gpu_telemetry_dual_batch.csv", interval_ms=200)
    profiler.start()

    # 5. Launch all 40 requests simultaneously at T=0.00s!
    print(f"\n🚀 Launching all {len(requests_pool)} requests simultaneously at T=0.00s...")
    t_global_start = time.time()
    results = {}

    tasks = [
        run_client_worker(engine, req_meta, t_global_start, results)
        for req_meta in requests_pool
    ]

    gather_fut = asyncio.ensure_future(asyncio.gather(*tasks))
    done, pending = await asyncio.wait([gather_fut, loop_task], return_when=asyncio.FIRST_COMPLETED)
    if loop_task in done and loop_task.exception():
        raise loop_task.exception()
    await gather_fut

    t_global_end = time.time()
    total_wall_time = t_global_end - t_global_start

    # Stop Profiler
    telemetry_stats = profiler.stop()

    # Aggregate Statistics
    total_audio_synthesized = sum(r["audio_duration_seconds"] for r in results.values())
    cluster_throughput_rtf = total_audio_synthesized / max(total_wall_time, 0.001)
    frames_per_sec = (total_audio_synthesized / 0.08) / max(total_wall_time, 0.001)
    mean_active_rtf = float(np.mean([r["active_rtf"] for r in results.values()]))
    mean_ttfa_ms = float(np.mean([r["post_admit_ttfa_ms"] for r in results.values()]))
    mean_wait_s = float(np.mean([r["queue_wait_seconds"] for r in results.values()]))
    max_wait_s = float(np.max([r["queue_wait_seconds"] for r in results.values()]))

    # Print Table
    print("\n" + "=" * 110)
    print("📊 40-REQUEST IN-FLIGHT DUAL-BATCHING QoS BREAKDOWN")
    print("=" * 110)
    print(
        f"{'Client ID':<9} | {'Stn-Slot':<8} | {'Audio(s)':<8} | {'Wait(s)':<8} | "
        f"{'Active(s)':<9} | {'TTFA(ms)':<9} | {'Active RTF':<10} | {'Turn RTF':<9} | {'Exit Wall':<9}"
    )
    print("-" * 110)
    for req_meta in requests_pool:
        rid = req_meta["id"]
        r = results.get(rid, {})
        stn_slot = f"{r.get('station', '?')}-S{r.get('slot', 0)}"
        print(
            f"{rid:<9} | {stn_slot:<8} | {r.get('audio_duration_seconds', 0):<8.2f} | "
            f"{r.get('queue_wait_seconds', 0):<8.2f} | {r.get('active_compute_seconds', 0):<9.2f} | "
            f"{r.get('post_admit_ttfa_ms', 0):<9.1f} | {r.get('active_rtf', 0):<10.2f}x | "
            f"{r.get('turnaround_rtf', 0):<9.2f}x | {r.get('exit_wall_time', ''):<9}"
        )

    print("=" * 110)
    print(f"✨ Total Audio Synthesized   : {total_audio_synthesized:.2f} seconds (~{total_audio_synthesized/60:.1f} minutes)")
    print(f"✨ Cluster Wall-Clock Time   : {total_wall_time:.2f} seconds")
    print(f"✨ Cluster Throughput Multiplier: {cluster_throughput_rtf:.2f}x Real-Time ({frames_per_sec:.1f} frames/sec)")
    print(f"✨ Mean Active Compute RTF   : {mean_active_rtf:.2f}x")
    print(f"✨ Mean Post-Admit TTFA      : {mean_ttfa_ms:.1f} ms")
    print(f"✨ Mean Queue Wait Time      : {mean_wait_s:.2f} s (Peak: {max_wait_s:.2f} s)")
    print("=" * 110)

    # Print Telemetry
    if telemetry_stats:
        print("\n📈 DUAL-GPU HARDWARE TELEMETRY (200ms SAMPLING):")
        for dev, s in telemetry_stats.items():
            if isinstance(s, dict):
                print(
                    f"  [{dev}] Core Util: Mean={s.get('gpu_util_mean', 0):.2f}% | "
                    f"Peak={s.get('gpu_util_max', 0):.1f}% | Saturated(>=90%)={s.get('time_saturated_90_pct', 0):.1f}% | "
                    f"Power Mean={s.get('power_w_mean', 0):.2f}W | Peak VRAM={s.get('vram_mb_peak', 0):.1f}MB"
                )

    # Save Output
    output_payload = {
        "benchmark": "40_request_dual_batching_burst_stress_test",
        "architecture": "Design 5 with In-Flight Dual-Batching (batch_size=2 per station, 4 slots)",
        "hardware": "Dual NVIDIA Tesla T4 (2x 16GB, sm_75)",
        "total_requests": len(requests_pool),
        "total_audio_synthesized_seconds": round(total_audio_synthesized, 2),
        "total_wall_clock_time_seconds": round(total_wall_time, 2),
        "cluster_throughput_rtf": round(cluster_throughput_rtf, 2),
        "frames_per_second": round(frames_per_sec, 1),
        "mean_active_rtf": round(mean_active_rtf, 2),
        "mean_post_admit_ttfa_ms": round(mean_ttfa_ms, 1),
        "mean_queue_wait_seconds": round(mean_wait_s, 2),
        "max_queue_wait_seconds": round(max_wait_s, 2),
        "telemetry": telemetry_stats,
        "results": results,
    }

    out_file = Path("/kaggle/working/benchmark_dual_batching_results.json")
    try:
        with open(out_file, "w") as f:
            json.dump(output_payload, f, indent=2)
        print(f"\n💾 Results successfully saved to {out_file}!")
    except Exception as e:
        local_out = REPO_ROOT / "benchmark_dual_batching_results.json"
        with open(local_out, "w") as f:
            json.dump(output_payload, f, indent=2)
        print(f"\n💾 Results saved locally to {local_out} ({e})")

    engine.running = False
    loop_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
