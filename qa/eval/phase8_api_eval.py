import sys
import os
import io
import time
import json
import numpy as np
from PIL import Image
from starlette.testclient import TestClient

PROJECT_ROOT = os.path.abspath(".")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "backend"))

from app.main import app
from app.config import settings
import app.components.face_detector as fd_module

TEST_IMAGE_PATH = os.path.join(PROJECT_ROOT, "qa", "data", "frontal_set", "frontal_00_pins_Adriana Lima_Adriana Lima0_0.jpg")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "qa", "reports")

def run_performance_benchmarks(n_runs=15):
    print("=" * 70)
    print("PHASE 8: API LATENCY & PERFORMANCE BENCHMARKS")
    print("=" * 70)

    settings.mediapipe_model_path = os.path.abspath("backend/data/face_landmarker.task")
    settings.celebrity_db_path = os.path.abspath("backend/data/celebrity_embeddings.pkl")
    fd_module.face_detector = None

    client = TestClient(app)

    with open(TEST_IMAGE_PATH, "rb") as f:
        img_bytes = f.read()

    # Warm-up run
    print("Warming up API pipeline...")
    client.post(
        "/api/analyze?include_celebrity_match=false",
        files={"file": ("warmup.jpg", img_bytes, "image/jpeg")}
    )

    # 1. P0 Scoring Path (landmarks + golden ratio scoring, no celebrity matcher)
    print(f"\nBenchmarking P0 Scoring Path (N={n_runs} runs)...")
    scoring_latencies = []
    scoring_component_times = []

    for i in range(n_runs):
        t0 = time.perf_counter()
        resp = client.post(
            "/api/analyze?include_celebrity_match=false&include_overlay_data=true",
            files={"file": (f"bench_{i}.jpg", img_bytes, "image/jpeg")}
        )
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        assert resp.status_code == 200, f"Run {i} failed: {resp.text}"
        data = resp.json()
        proc_time = data.get("processing_time_ms", elapsed_ms)
        scoring_latencies.append(elapsed_ms)
        scoring_component_times.append(proc_time)

    # 2. Full Path (landmarks + scoring + celebrity matching)
    print(f"Benchmarking Full Path with Celebrity Matching (N={n_runs} runs)...")
    full_latencies = []
    full_component_times = []

    for i in range(n_runs):
        t0 = time.perf_counter()
        resp = client.post(
            "/api/analyze?include_celebrity_match=true&include_overlay_data=true",
            files={"file": (f"bench_full_{i}.jpg", img_bytes, "image/jpeg")}
        )
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        assert resp.status_code == 200, f"Full Run {i} failed: {resp.text}"
        data = resp.json()
        proc_time = data.get("processing_time_ms", elapsed_ms)
        full_latencies.append(elapsed_ms)
        full_component_times.append(proc_time)

    # Calculate statistics
    scoring_p50 = float(np.percentile(scoring_latencies, 50))
    scoring_p95 = float(np.percentile(scoring_latencies, 95))
    scoring_mean = float(np.mean(scoring_latencies))
    scoring_min = float(np.min(scoring_latencies))
    scoring_max = float(np.max(scoring_latencies))

    full_p50 = float(np.percentile(full_latencies, 50))
    full_p95 = float(np.percentile(full_latencies, 95))
    full_mean = float(np.mean(full_latencies))
    full_min = float(np.min(full_latencies))
    full_max = float(np.max(full_latencies))

    celebrity_overhead_p50 = full_p50 - scoring_p50
    celebrity_overhead_p95 = full_p95 - scoring_p95

    print("\nBenchmark Summary (PRD Target: P0 Path < 3000 ms):")
    print(f"  P0 Scoring Path (No Celebrity):")
    print(f"    P50:  {scoring_p50:.1f} ms")
    print(f"    P95:  {scoring_p95:.1f} ms (Target < 3000 ms: {'MET' if scoring_p95 < 3000 else 'MISSED'})")
    print(f"    Mean: {scoring_mean:.1f} ms (Min: {scoring_min:.1f} ms, Max: {scoring_max:.1f} ms)")

    print(f"\n  Full Path (With Celebrity Matcher):")
    print(f"    P50:  {full_p50:.1f} ms")
    print(f"    P95:  {full_p95:.1f} ms")
    print(f"    Mean: {full_mean:.1f} ms (Min: {full_min:.1f} ms, Max: {full_max:.1f} ms)")

    print(f"\n  Celebrity Matching Incremental Overhead:")
    print(f"    P50 Overhead: +{celebrity_overhead_p50:.1f} ms")
    print(f"    P95 Overhead: +{celebrity_overhead_p95:.1f} ms")

    bench_results = {
        "n_runs": n_runs,
        "scoring_path_ms": {
            "p50": round(scoring_p50, 1),
            "p95": round(scoring_p95, 1),
            "mean": round(scoring_mean, 1),
            "min": round(scoring_min, 1),
            "max": round(scoring_max, 1),
            "prd_target_ms": 3000,
            "target_met": scoring_p95 < 3000
        },
        "full_path_ms": {
            "p50": round(full_p50, 1),
            "p95": round(full_p95, 1),
            "mean": round(full_mean, 1),
            "min": round(full_min, 1),
            "max": round(full_max, 1)
        },
        "celebrity_overhead_ms": {
            "p50": round(celebrity_overhead_p50, 1),
            "p95": round(celebrity_overhead_p95, 1)
        }
    }

    out_path = os.path.join(REPORTS_DIR, "08_api_benchmarks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(bench_results, f, indent=2)
    print(f"\nSaved benchmark metrics to {out_path}")

if __name__ == "__main__":
    run_performance_benchmarks()
