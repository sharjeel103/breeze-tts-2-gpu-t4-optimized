"""
Continuous Dual-GPU Telemetry Profiler (200ms Low-Overhead Sampling)
Tracks core utilization, memory bus activity, power draw, and VRAM across all GPUs.
"""

import subprocess
import time
import os
import json
import numpy as np
from typing import Dict, Any, Optional

class GPUProfiler:
    """
    Context manager and controller for high-frequency (200ms) NVIDIA GPU profiling.
    Uses native asynchronous nvidia-smi background daemon to guarantee zero Python GIL
    overhead or interference with running inference workloads.
    """
    def __init__(self, log_csv_path: str = "/tmp/gpu_profile.csv", interval_ms: int = 200):
        self.log_csv_path = log_csv_path
        self.interval_ms = interval_ms
        self.proc: Optional[subprocess.Popen] = None
        self.t_start: float = 0.0
        self.t_end: float = 0.0

    def start(self):
        """Launches the background nvidia-smi telemetry daemon."""
        if os.path.exists(self.log_csv_path):
            try:
                os.remove(self.log_csv_path)
            except OSError:
                pass

        # Query: timestamp, gpu index, core util %, memory bus util %, power draw (W), memory.used
        monitor_cmd = (
            f"nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw,memory.used "
            f"--format=csv,nounits,noheader -lms {self.interval_ms} > {self.log_csv_path}"
        )
        self.proc = subprocess.Popen(monitor_cmd, shell=True, executable="/bin/bash")
        self.t_start = time.time()
        time.sleep(0.3)  # Allow daemon to establish file handle

    def stop(self) -> Dict[str, Any]:
        """Terminates the profiler and returns the consolidated statistical summary."""
        self.t_end = time.time()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        time.sleep(0.3)  # Flush file buffer

        return self.analyze()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def analyze(self) -> Dict[str, Any]:
        """Parses raw CSV logs into statistical distributions per GPU."""
        wall_time = self.t_end - self.t_start if self.t_end > self.t_start else 0.0

        if not os.path.exists(self.log_csv_path):
            return {"status": "NO_LOG_FILE_FOUND"}

        # Organize records by GPU index: {0: {'util': [], 'mem': [], 'pwr': [], 'vram': []}, ...}
        gpu_data: Dict[int, Dict[str, list]] = {}

        with open(self.log_csv_path, "r") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    try:
                        # parts: [timestamp, index, util.gpu, util.mem, power.draw, memory.used]
                        idx = int(parts[1])
                        g_util = float(parts[2])
                        m_util = float(parts[3])
                        pwr = float(parts[4])
                        vram = float(parts[5])

                        if idx not in gpu_data:
                            gpu_data[idx] = {"util": [], "mem": [], "pwr": [], "vram": []}

                        gpu_data[idx]["util"].append(g_util)
                        gpu_data[idx]["mem"].append(m_util)
                        gpu_data[idx]["pwr"].append(pwr)
                        gpu_data[idx]["vram"].append(vram)
                    except ValueError:
                        continue

        if not gpu_data:
            return {"status": "EMPTY_DATA"}

        summary = {
            "status": "SUCCESS",
            "profiling_duration_seconds": round(wall_time, 2),
            "sampling_interval_ms": self.interval_ms,
            "total_samples_per_gpu": {idx: len(d["util"]) for idx, d in gpu_data.items()},
            "gpus": {}
        }

        total_mean_pwr = 0.0
        total_max_pwr = 0.0

        for idx, d in sorted(gpu_data.items()):
            u_arr = np.array(d["util"]) if d["util"] else np.array([0.0])
            m_arr = np.array(d["mem"]) if d["mem"] else np.array([0.0])
            p_arr = np.array(d["pwr"]) if d["pwr"] else np.array([0.0])
            v_arr = np.array(d["vram"]) if d["vram"] else np.array([0.0])

            pct_above_90 = float(np.mean(u_arr >= 90.0) * 100.0)
            pct_50_to_89 = float(np.mean((u_arr >= 50.0) & (u_arr < 90.0)) * 100.0)
            pct_below_20 = float(np.mean(u_arr < 20.0) * 100.0)

            mean_pwr = float(np.mean(p_arr))
            max_pwr = float(np.max(p_arr))
            total_mean_pwr += mean_pwr
            total_max_pwr += max_pwr

            # Generate a 10-bucket timeline of the run
            n_samples = len(u_arr)
            bucket_size = max(1, n_samples // 10)
            timeline = []
            for b in range(10):
                s_idx = b * bucket_size
                e_idx = min(n_samples, (b + 1) * bucket_size) if b < 9 else n_samples
                if s_idx < e_idx:
                    chunk = u_arr[s_idx:e_idx]
                    timeline.append({
                        "time_window": f"{int((s_idx / n_samples) * wall_time)}s - {int((e_idx / n_samples) * wall_time)}s",
                        "avg_util_pct": round(float(np.mean(chunk)), 1),
                        "peak_util_pct": round(float(np.max(chunk)), 1)
                    })

            summary["gpus"][f"gpu_{idx}"] = {
                "core_utilization": {
                    "mean_pct": round(float(np.mean(u_arr)), 2),
                    "median_pct": round(float(np.median(u_arr)), 2),
                    "min_pct": round(float(np.min(u_arr)), 2),
                    "max_pct": round(float(np.max(u_arr)), 2),
                    "p90_pct": round(float(np.percentile(u_arr, 90)), 2),
                    "p95_pct": round(float(np.percentile(u_arr, 95)), 2),
                    "pct_time_saturated_above_90": round(pct_above_90, 2),
                    "pct_time_active_50_to_89": round(pct_50_to_89, 2),
                    "pct_time_idle_below_20": round(pct_below_20, 2)
                },
                "memory_bus_utilization": {
                    "mean_pct": round(float(np.mean(m_arr)), 2),
                    "max_pct": round(float(np.max(m_arr)), 2)
                },
                "power_watts": {
                    "mean_w": round(mean_pwr, 2),
                    "max_w": round(max_pwr, 2),
                    "t4_tdp_envelope_w": 70.0
                },
                "vram_mb": {
                    "peak_allocated_mb": round(float(np.max(v_arr)), 1),
                    "card_total_mb": 15360.0,
                    "remaining_headroom_mb": round(15360.0 - float(np.max(v_arr)), 1)
                },
                "timeline_10_segments": timeline
            }

        summary["cluster_combined_power"] = {
            "mean_total_watts": round(total_mean_pwr, 2),
            "max_total_watts": round(total_max_pwr, 2),
            "cluster_tdp_envelope_w": 70.0 * len(gpu_data)
        }

        return summary
