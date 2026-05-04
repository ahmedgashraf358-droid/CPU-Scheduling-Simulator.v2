
#App.py
"""
Flask API for CPU Scheduling ML Decision System
================================================
Endpoints:
  POST /api/predict   → ML model prediction (best algorithm + probabilities)
  POST /api/schedule  → Run selected algorithm → gantt + metrics
"""

import heapq
import os
from collections import deque

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

# ─────────────────────────────────────────────
# Load ML artefacts (once at startup)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

clf     = joblib.load(os.path.join(BASE_DIR, "scheduler_classifier.pkl"))
le      = joblib.load(os.path.join(BASE_DIR, "scheduler_label_encoder.pkl"))
FEATURE_COLUMNS = joblib.load(os.path.join(BASE_DIR, "classifier_feature_columns.pkl"))

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
ALGORITHM_NAMES    = ["FCFS", "SJF_NP", "SRTF", "PRIORITY", "RR"]
SHORT_JOB_THRESHOLD = 5
GANTT_MERGE_TOL    = 1e-9
COST_WEIGHTS       = {"waiting": 0.5, "turnaround": 0.3, "response": 0.2}

# ─────────────────────────────────────────────
# Feature Engineering  (must match Main.py)
# ─────────────────────────────────────────────
def workload_features(processes: list) -> dict:
    arrivals   = np.array([p["arrival_time"] for p in processes], dtype=float)
    bursts     = np.array([p["burst_time"]   for p in processes], dtype=float)
    priorities = np.array([p.get("priority", 1) for p in processes], dtype=float)

    n            = len(processes)
    arrival_span = float(arrivals.max() - arrivals.min())
    burst_sum    = float(bursts.sum())
    burst_range  = float(bursts.max() - bursts.min())

    arrival_clustering_score = float(
        1.0 - arrivals.std(ddof=0) / (arrival_span + 1.0)
    )
    burst_variability_index = float(
        bursts.std(ddof=0) / (burst_range + 1.0)
    )

    return {
        "n_processes":                    n,
        "arrival_mean":                   float(arrivals.mean()),
        "arrival_std":                    float(arrivals.std(ddof=0)),
        "arrival_range":                  arrival_span,
        "burst_mean":                     float(bursts.mean()),
        "burst_std":                      float(bursts.std(ddof=0)),
        "burst_min":                      float(bursts.min()),
        "burst_max":                      float(bursts.max()),
        "burst_sum":                      burst_sum,
        "burst_coefficient_of_variation": float(
            bursts.std(ddof=0) / (bursts.mean() + 1e-9)
        ),
        "workload_density":               float(burst_sum / (arrival_span + 1.0)),
        "short_job_ratio":                float(np.sum(bursts < SHORT_JOB_THRESHOLD) / n),
        "arrival_clustering_score":       arrival_clustering_score,
        "burst_variability_index":        burst_variability_index,
        "priority_mean":                  float(priorities.mean()),
        "priority_std":                   float(priorities.std(ddof=0)),
        "priority_min":                   float(priorities.min()),
        "priority_max":                   float(priorities.max()),
        "priority_span":                  float(priorities.max() - priorities.min()),
    }


# ─────────────────────────────────────────────
# Gantt helpers
# ─────────────────────────────────────────────
def _append(gantt, pid, start, end):
    if (gantt and gantt[-1]["id"] == pid
            and abs(gantt[-1]["end"] - start) <= GANTT_MERGE_TOL):
        gantt[-1]["end"] = end
    else:
        gantt.append({"id": pid, "start": start, "end": end})


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def calculate_metrics(processes, gantt):
    completion, first_start = {}, {}
    for b in gantt:
        pid = b["id"]
        if pid not in first_start:
            first_start[pid] = b["start"]
        completion[pid] = b["end"]

    wts, tats, rts = [], [], []
    detail = []
    for p in processes:
        pid    = p["id"]
        at     = p["arrival_time"]
        bt     = p["burst_time"]
        ct     = completion[pid]
        tat    = ct - at
        wt     = tat - bt
        rt     = first_start[pid] - at
        wts.append(wt); tats.append(tat); rts.append(rt)
        detail.append({"id": pid, "at": at, "bt": bt,
                        "ct": ct, "tat": tat, "wt": wt, "rt": rt})

    return {
        "avg_waiting":    round(float(np.mean(wts)),  4),
        "avg_turnaround": round(float(np.mean(tats)), 4),
        "avg_response":   round(float(np.mean(rts)),  4),
        "detail":         detail,
    }


def compute_cost(m):
    return (COST_WEIGHTS["waiting"]    * m["avg_waiting"]
          + COST_WEIGHTS["turnaround"] * m["avg_turnaround"]
          + COST_WEIGHTS["response"]   * m["avg_response"])


# ─────────────────────────────────────────────
# Scheduling Algorithms
# ─────────────────────────────────────────────
def fcfs(processes):
    procs = sorted(processes, key=lambda p: (p["arrival_time"], p["id"]))
    t, gantt = 0.0, []
    for p in procs:
        if t < p["arrival_time"]:
            t = float(p["arrival_time"])
        _append(gantt, p["id"], t, t + p["burst_time"])
        t += p["burst_time"]
    return gantt, calculate_metrics(procs, gantt)


def sjf_np(processes):
    procs = sorted(processes, key=lambda p: (p["arrival_time"], p["id"]))
    heap, done, gantt = [], set(), []
    t, idx = 0.0, 0
    while len(done) < len(processes):
        while idx < len(procs) and procs[idx]["arrival_time"] <= t:
            p = procs[idx]
            heapq.heappush(heap, (p["burst_time"], p["arrival_time"], p["id"], p))
            idx += 1
        if not heap:
            t = float(procs[idx]["arrival_time"]); continue
        _, _, _, chosen = heapq.heappop(heap)
        _append(gantt, chosen["id"], t, t + chosen["burst_time"])
        t += chosen["burst_time"]; done.add(chosen["id"])
    return gantt, calculate_metrics(list(processes), gantt)


def srtf(processes):
    procs      = sorted(processes, key=lambda p: (p["arrival_time"], p["id"]))
    remaining  = {p["id"]: float(p["burst_time"]) for p in procs}
    proc_by_id = {p["id"]: p for p in procs}
    heap, in_heap, gantt = [], set(), []
    t, i = 0.0, 0

    def enqueue(pid):
        p = proc_by_id[pid]
        heapq.heappush(heap, (remaining[pid], p["arrival_time"], pid))
        in_heap.add(pid)

    while i < len(procs) and procs[i]["arrival_time"] == 0:
        enqueue(procs[i]["id"]); i += 1

    while i < len(procs) or heap:
        if not heap:
            t = float(procs[i]["arrival_time"])
            while i < len(procs) and procs[i]["arrival_time"] <= t:
                enqueue(procs[i]["id"]); i += 1
        while heap and heap[0][2] not in in_heap:
            heapq.heappop(heap)
        if not heap: continue
        _, _, pid = heapq.heappop(heap); in_heap.discard(pid)
        next_arr  = float(procs[i]["arrival_time"]) if i < len(procs) else float("inf")
        run_until = min(t + remaining[pid], next_arr)
        _append(gantt, pid, t, run_until)
        remaining[pid] -= (run_until - t); t = run_until
        while i < len(procs) and procs[i]["arrival_time"] <= t:
            enqueue(procs[i]["id"]); i += 1
        if remaining[pid] > GANTT_MERGE_TOL:
            enqueue(pid)

    return gantt, calculate_metrics(procs, gantt)


def priority_np(processes, mode="static"):
    procs = sorted(processes, key=lambda p: (p["arrival_time"], p["id"]))
    heap, done, gantt = [], set(), []
    t, idx = 0.0, 0
    while len(done) < len(processes):
        while idx < len(procs) and procs[idx]["arrival_time"] <= t:
            p = procs[idx]
            heapq.heappush(heap, (p.get("priority", 1), p["burst_time"],
                                   p["arrival_time"], p["id"], p))
            idx += 1
        if not heap:
            t = float(procs[idx]["arrival_time"]); continue
        _, _, _, _, chosen = heapq.heappop(heap)
        _append(gantt, chosen["id"], t, t + chosen["burst_time"])
        t += chosen["burst_time"]; done.add(chosen["id"])
    return gantt, calculate_metrics(list(processes), gantt)


def round_robin(processes, quantum=2):
    procs     = sorted(processes, key=lambda p: (p["arrival_time"], p["id"]))
    remaining = {p["id"]: float(p["burst_time"]) for p in procs}
    t, i, queue, gantt = 0.0, 0, deque(), []
    while i < len(procs) or queue:
        if not queue:
            t = max(t, float(procs[i]["arrival_time"]))
            while i < len(procs) and procs[i]["arrival_time"] <= t:
                queue.append(procs[i]); i += 1
        p   = queue.popleft(); pid = p["id"]
        run = min(float(quantum), remaining[pid])
        _append(gantt, pid, t, t + run)
        remaining[pid] -= run; t += run
        while i < len(procs) and procs[i]["arrival_time"] <= t:
            queue.append(procs[i]); i += 1
        if remaining[pid] > GANTT_MERGE_TOL:
            queue.append(p)
    return gantt, calculate_metrics(procs, gantt)


# ─────────────────────────────────────────────
# Parameter resolver
# ─────────────────────────────────────────────
def resolve_parameters(algo, processes):
    bursts   = np.array([p["burst_time"]   for p in processes], dtype=float)
    arrivals = np.array([p["arrival_time"] for p in processes], dtype=float)
    if algo == "RR":
        bm = float(bursts.mean())
        return {"quantum": 2 if bm <= 6 else (4 if bm <= 10 else 6)}
    if algo == "PRIORITY":
        return {"priority_mode": "dynamic" if float(arrivals.std(ddof=0)) > 5 else "static"}
    return {}


def run_algorithm(algo, processes, params=None):
    params = params or {}
    dispatch = {
        "FCFS":     lambda: fcfs(processes),
        "SJF_NP":   lambda: sjf_np(processes),
        "SRTF":     lambda: srtf(processes),
        "PRIORITY": lambda: priority_np(processes, mode=params.get("priority_mode", "static")),
        "RR":       lambda: round_robin(processes, quantum=params.get("quantum", 2)),
    }
    if algo not in dispatch:
        raise ValueError(f"Unknown algorithm: {algo}")
    return dispatch[algo]()


# ─────────────────────────────────────────────
# Serve frontend
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return app.send_static_file("index.html")


# ─────────────────────────────────────────────
# API: Predict best algorithm
# ─────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Body:  { "processes": [ {arrival_time, burst_time, priority?} , ...] }
    Returns: { probabilities: [...], best: {algorithm, probability} }
    """
    try:
        data      = request.get_json(force=True)
        processes = data.get("processes", [])
        if not processes:
            return jsonify({"error": "No processes provided"}), 400

        # Assign sequential IDs if missing
        for i, p in enumerate(processes):
            p.setdefault("id", i + 1)
            p.setdefault("priority", 1)

        import pandas as pd
        feats  = workload_features(processes)
        X_pred = pd.DataFrame([feats])[FEATURE_COLUMNS]
        proba  = clf.predict_proba(X_pred)[0]

        prob_table = sorted(
            [{"algorithm": algo, "probability": round(float(p), 4)}
             for algo, p in zip(le.classes_, proba)],
            key=lambda x: x["probability"], reverse=True
        )

        best   = prob_table[0]
        params = resolve_parameters(best["algorithm"], processes)

        return jsonify({
            "probabilities": prob_table,
            "best":          best,
            "parameters":    params,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# API: Run scheduling algorithm
# ─────────────────────────────────────────────
@app.route("/api/schedule", methods=["POST"])
def schedule():
    """
    Body:  {
             "algorithm": "FCFS"|"SJF_NP"|"SRTF"|"PRIORITY"|"RR",
             "processes": [...],
             "params":    { quantum?: int, priority_mode?: str }
           }
    Returns: { gantt: [...], metrics: {...}, detail: [...] }
    """
    try:
        data      = request.get_json(force=True)
        algo      = data.get("algorithm", "FCFS")
        processes = data.get("processes", [])
        params    = data.get("params", {})

        if not processes:
            return jsonify({"error": "No processes provided"}), 400

        for i, p in enumerate(processes):
            p.setdefault("id", i + 1)
            p.setdefault("priority", 1)

        gantt, metrics = run_algorithm(algo, processes, params)

        # Compute cost & compare all algorithms
        cost = compute_cost(metrics)
        all_costs = {}
        for a in ALGORITHM_NAMES:
            try:
                p2 = resolve_parameters(a, processes)
                _, m2 = run_algorithm(a, processes, p2)
                all_costs[a] = round(compute_cost(m2), 4)
            except Exception:
                pass

        actual_best = min(all_costs, key=all_costs.get)

        return jsonify({
            "gantt":       gantt,
            "metrics":     metrics,
            "cost":        round(cost, 4),
            "all_costs":   all_costs,
            "actual_best": actual_best,
            "match":       actual_best == algo,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  CPU Scheduling ML Server  →  http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True, port=5000)