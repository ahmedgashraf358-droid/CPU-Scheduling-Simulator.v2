#Main.py

"""
ML-based CPU Scheduling Decision System
========================================
Matches the Enhanced Prompt exactly:
  - Dataset Generator (Simulation Engine)
  - ML Classification Model  →  probability distribution over algorithms
  - Decision Engine          →  argmax(probabilities)

Cost function  : Cost = 0.5*WT + 0.3*TAT + 0.2*RT   (lower = better)
Best algorithm : argmin(cost)  →  classification label
Output         : probability per algorithm + argmax decision
"""


import heapq
import random
from collections import deque

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder


# =========================================================
# 0) Global Config
# =========================================================
ALGORITHM_NAMES = ["FCFS", "SJF_NP", "SRTF", "PRIORITY", "RR"]

COST_WEIGHTS = {
    "waiting": 0.5,
    "turnaround": 0.3,
    "response": 0.2,
}

# Short-job threshold: burst < this value is considered a "short job"
SHORT_JOB_THRESHOLD = 5

GANTT_MERGE_TOL = 1e-9

MODEL_FILENAME        = "scheduler_classifier.pkl"
ENCODER_FILENAME      = "scheduler_label_encoder.pkl"
FEATURES_FILENAME     = "classifier_feature_columns.pkl"


# =========================================================
# 1) Process Generator
# =========================================================
def generate_processes(
    arrival_pattern="scattered",
    burst_pattern="medium",
    n=None,
    arrival_max=20,
    burst_max=20,
    priority_max=10,
    seed=None,
):
    rng = random.Random(seed)

    if n is None:
        n = rng.randint(5, 12)

    # --- Arrival times ---
    if arrival_pattern == "scattered":
        arrivals = [rng.randint(0, arrival_max) for _ in range(n)]
    elif arrival_pattern == "clustered":
        c1 = rng.randint(0, max(1, arrival_max // 2))
        c2 = rng.randint(max(1, arrival_max // 2), arrival_max)
        arrivals = [
            rng.choice([c1, c2, max(0, c1 - 1), min(arrival_max, c2 + 1)])
            for _ in range(n)
        ]
    elif arrival_pattern == "tight":
        pivot = rng.randint(0, arrival_max)
        arrivals = [
            max(0, min(arrival_max, pivot + rng.randint(-1, 1)))
            for _ in range(n)
        ]
    elif arrival_pattern == "ordered":
        arrivals = sorted(rng.randint(0, arrival_max) for _ in range(n))
    else:
        arrivals = [rng.randint(0, arrival_max) for _ in range(n)]

    # --- Burst times ---
    if burst_pattern == "short":
        bursts = [rng.randint(1, 4) for _ in range(n)]
    elif burst_pattern == "medium":
        bursts = [rng.randint(5, 10) for _ in range(n)]
    elif burst_pattern == "long":
        low = min(11, burst_max)
        bursts = [rng.randint(low, burst_max) for _ in range(n)]
    elif burst_pattern == "mixed":
        bursts = [rng.randint(1, burst_max) for _ in range(n)]
    else:
        bursts = [rng.randint(1, burst_max) for _ in range(n)]

    priorities = [rng.randint(1, priority_max) for _ in range(n)]

    processes = [
        {
            "id": i + 1,
            "arrival_time": int(arrivals[i]),
            "burst_time":   int(bursts[i]),
            "priority":     int(priorities[i]),
        }
        for i in range(n)
    ]
    processes.sort(key=lambda p: (p["arrival_time"], p["burst_time"], p["id"]))
    return processes


# =========================================================
# 2) Gantt Utils
# =========================================================
def _append_gantt(gantt_chart, pid, start, end):
    """Append/merge Gantt segment on-the-fly."""
    if (
        gantt_chart
        and gantt_chart[-1]["id"] == pid
        and abs(gantt_chart[-1]["end"] - start) <= GANTT_MERGE_TOL
    ):
        gantt_chart[-1]["end"] = end
    else:
        gantt_chart.append({"id": pid, "start": start, "end": end})


def print_gantt_chart(gantt_chart):
    if not gantt_chart:
        print("Gantt chart is empty.")
        return
    print("\nGantt Chart:")
    for seg in gantt_chart:
        print(f"  P{seg['id']}  [{seg['start']:.2f} → {seg['end']:.2f}]")


# =========================================================
# 3) Metrics & Cost
# =========================================================
def calculate_metrics(processes, gantt_chart):
    completion_time = {}
    first_start = {}

    for block in gantt_chart:
        pid = block["id"]
        if pid not in first_start:
            first_start[pid] = block["start"]
        completion_time[pid] = block["end"]

    waiting_times, turnaround_times, response_times = [], [], []

    for p in processes:
        pid    = p["id"]
        arrival = p["arrival_time"]
        burst   = p["burst_time"]

        if pid not in first_start or pid not in completion_time:
            raise RuntimeError(f"Missing schedule data for process {pid}")

        tat = completion_time[pid] - arrival
        wt  = tat - burst
        rt  = first_start[pid]   - arrival

        waiting_times.append(wt)
        turnaround_times.append(tat)
        response_times.append(rt)

    return {
        "avg_waiting":    float(np.mean(waiting_times)),
        "avg_turnaround": float(np.mean(turnaround_times)),
        "avg_response":   float(np.mean(response_times)),
    }


def compute_cost(metrics: dict) -> float:
    """
    Cost = 0.5 * WT + 0.3 * TAT + 0.2 * RT
    Lower cost = better algorithm.
    Label = argmin(cost) across all algorithms.
    """
    return (
        COST_WEIGHTS["waiting"]    * metrics["avg_waiting"]
        + COST_WEIGHTS["turnaround"] * metrics["avg_turnaround"]
        + COST_WEIGHTS["response"]   * metrics["avg_response"]
    )


# =========================================================
# 4) Scheduling Algorithms
# =========================================================

# ── FCFS ──────────────────────────────────────────────────
def fcfs_algorithm(processes):
    procs = sorted(processes, key=lambda p: (p["arrival_time"], p["id"]))
    current_time = 0.0
    gantt_chart  = []

    for p in procs:
        if current_time < p["arrival_time"]:
            current_time = float(p["arrival_time"])
        end = current_time + p["burst_time"]
        _append_gantt(gantt_chart, p["id"], current_time, end)
        current_time = end

    return gantt_chart, calculate_metrics(procs, gantt_chart)


# ── SJF Non-Preemptive ────────────────────────────────────
def sjf_non_preemptive_algorithm(processes):
    sorted_procs = sorted(processes, key=lambda p: (p["arrival_time"], p["id"]))
    heap         = []
    done         = set()
    current_time = 0.0
    gantt_chart  = []
    idx          = 0

    while len(done) < len(processes):
        while idx < len(sorted_procs) and sorted_procs[idx]["arrival_time"] <= current_time:
            p = sorted_procs[idx]
            heapq.heappush(heap, (p["burst_time"], p["arrival_time"], p["id"], p))
            idx += 1

        if not heap:
            current_time = float(sorted_procs[idx]["arrival_time"])
            continue

        _, _, pid, chosen = heapq.heappop(heap)
        end = current_time + chosen["burst_time"]
        _append_gantt(gantt_chart, pid, current_time, end)
        current_time = end
        done.add(pid)

    return gantt_chart, calculate_metrics(list(processes), gantt_chart)


# ── SRTF (Preemptive SJF) ─────────────────────────────────
def sjf_preemptive_algorithm(processes):
    """
    Event-driven SRTF: runs the current shortest-remaining process until
    the next arrival or its own completion, then re-evaluates.
    in_heap tracks which pid's heap entry is current (lazy deletion).
    """
    procs       = sorted(processes, key=lambda p: (p["arrival_time"], p["id"]))
    n           = len(procs)
    remaining   = {p["id"]: float(p["burst_time"]) for p in procs}
    proc_by_id  = {p["id"]: p for p in procs}

    heap:    list = []
    in_heap: set  = set()
    gantt_chart   = []
    current_time  = 0.0
    i             = 0

    def enqueue(pid: int) -> None:
        p = proc_by_id[pid]
        heapq.heappush(heap, (remaining[pid], p["arrival_time"], pid))
        in_heap.add(pid)

    while i < n and procs[i]["arrival_time"] == 0:
        enqueue(procs[i]["id"])
        i += 1

    while i < n or heap:
        if not heap:
            current_time = float(procs[i]["arrival_time"])
            while i < n and procs[i]["arrival_time"] <= current_time:
                enqueue(procs[i]["id"])
                i += 1

        # discard stale heap entries
        while heap and heap[0][2] not in in_heap:
            heapq.heappop(heap)
        if not heap:
            continue

        _, _, pid = heapq.heappop(heap)
        in_heap.discard(pid)

        next_arrival = float(procs[i]["arrival_time"]) if i < n else float("inf")
        run_until    = min(current_time + remaining[pid], next_arrival)

        _append_gantt(gantt_chart, pid, current_time, run_until)
        remaining[pid] -= (run_until - current_time)
        current_time    = run_until

        while i < n and procs[i]["arrival_time"] <= current_time:
            enqueue(procs[i]["id"])
            i += 1

        if remaining[pid] > GANTT_MERGE_TOL:
            enqueue(pid)

    return gantt_chart, calculate_metrics(procs, gantt_chart)


# ── Priority Non-Preemptive ───────────────────────────────
def priority_non_preemptive_algorithm(processes, mode="static"):
    sorted_procs = sorted(processes, key=lambda p: (p["arrival_time"], p["id"]))
    heap         = []
    done         = set()
    current_time = 0.0
    gantt_chart  = []
    idx          = 0

    while len(done) < len(processes):
        while idx < len(sorted_procs) and sorted_procs[idx]["arrival_time"] <= current_time:
            p = sorted_procs[idx]
            heapq.heappush(heap, (p["priority"], p["burst_time"], p["arrival_time"], p["id"], p))
            idx += 1

        if not heap:
            current_time = float(sorted_procs[idx]["arrival_time"])
            continue

        if mode == "dynamic":
            candidates = list(heap)
            heap.clear()
            best = min(
                candidates,
                key=lambda x: (
                    x[4]["priority"] - 0.1 * (current_time - x[4]["arrival_time"]),
                    x[4]["burst_time"],
                    x[4]["arrival_time"],
                    x[4]["id"],
                ),
            )
            chosen = best[4]
            for c in candidates:
                if c[3] != chosen["id"]:
                    heapq.heappush(heap, c)
        else:
            _, _, _, _, chosen = heapq.heappop(heap)

        end = current_time + chosen["burst_time"]
        _append_gantt(gantt_chart, chosen["id"], current_time, end)
        current_time = end
        done.add(chosen["id"])

    return gantt_chart, calculate_metrics(list(processes), gantt_chart)


# ── Round Robin ───────────────────────────────────────────
def round_robin_algorithm(processes, quantum=2):
    procs        = sorted(processes, key=lambda p: (p["arrival_time"], p["id"]))
    remaining    = {p["id"]: float(p["burst_time"]) for p in procs}
    current_time = 0.0
    i            = 0
    queue        = deque()
    gantt_chart  = []

    while i < len(procs) or queue:
        if not queue:
            current_time = max(current_time, float(procs[i]["arrival_time"]))
            while i < len(procs) and procs[i]["arrival_time"] <= current_time:
                queue.append(procs[i])
                i += 1

        p   = queue.popleft()
        pid = p["id"]

        run_time = min(float(quantum), remaining[pid])
        end      = current_time + run_time
        _append_gantt(gantt_chart, pid, current_time, end)
        remaining[pid] -= run_time
        current_time    = end

        while i < len(procs) and procs[i]["arrival_time"] <= current_time:
            queue.append(procs[i])
            i += 1

        if remaining[pid] > GANTT_MERGE_TOL:
            queue.append(p)

    return gantt_chart, calculate_metrics(procs, gantt_chart)


# ── Dispatch ──────────────────────────────────────────────
_ALGO_DISPATCH = {
    "FCFS":     lambda procs, params: fcfs_algorithm(procs),
    "SJF_NP":   lambda procs, params: sjf_non_preemptive_algorithm(procs),
    "SRTF":     lambda procs, params: sjf_preemptive_algorithm(procs),
    "PRIORITY": lambda procs, params: priority_non_preemptive_algorithm(
        procs, mode=params.get("priority_mode", "static")
    ),
    "RR":       lambda procs, params: round_robin_algorithm(
        procs, quantum=params.get("quantum", 2)
    ),
}


def run_algorithm(algo_name, processes, params=None):
    params = params or {}
    try:
        return _ALGO_DISPATCH[algo_name](processes, params)
    except KeyError:
        raise ValueError(f"Unknown algorithm: {algo_name}")


# =========================================================
# 5) Feature Engineering  (matches prompt exactly)
# =========================================================
def workload_features(processes) -> dict:
    """
    Extracts ALL features listed in the prompt:

    Process-level statistics:
        n_processes, arrival_mean, arrival_std, arrival_range,
        burst_mean, burst_std, burst_min, burst_max, burst_sum,
        burst_coefficient_of_variation

    Derived workload features:
        workload_density     = total_burst / (arrival_span + 1)
        short_job_ratio      = fraction of processes with burst < SHORT_JOB_THRESHOLD
        arrival_clustering_score  = 1 - (arrival_std / (arrival_range + 1))
                                    high → arrivals tightly clustered
        burst_variability_index   = burst_std / (burst_max - burst_min + 1)

    Optional / priority statistics:
        priority_mean, priority_std, priority_min, priority_max, priority_span
    """
    arrivals   = np.array([p["arrival_time"] for p in processes], dtype=float)
    bursts     = np.array([p["burst_time"]   for p in processes], dtype=float)
    priorities = np.array([p["priority"]     for p in processes], dtype=float)

    n            = len(processes)
    arrival_span = float(arrivals.max() - arrivals.min())
    burst_sum    = float(bursts.sum())
    burst_range  = float(bursts.max() - bursts.min())

    # arrival_clustering_score: close to 1 → tightly clustered arrivals
    arrival_clustering_score = float(
        1.0 - arrivals.std(ddof=0) / (arrival_span + 1.0)
    )

    # burst_variability_index: high → burst times vary a lot relative to range
    burst_variability_index = float(
        bursts.std(ddof=0) / (burst_range + 1.0)
    )

    return {
        # ── Process-level statistics ──────────────────────
        "n_processes":                  n,
        "arrival_mean":                 float(arrivals.mean()),
        "arrival_std":                  float(arrivals.std(ddof=0)),
        "arrival_range":                arrival_span,          # prompt: arrival_range
        "burst_mean":                   float(bursts.mean()),
        "burst_std":                    float(bursts.std(ddof=0)),
        "burst_min":                    float(bursts.min()),
        "burst_max":                    float(bursts.max()),
        "burst_sum":                    burst_sum,
        "burst_coefficient_of_variation": float(            # prompt: burst_cv
            bursts.std(ddof=0) / (bursts.mean() + 1e-9)
        ),
        # ── Derived workload features ─────────────────────
        "workload_density":             float(burst_sum / (arrival_span + 1.0)),
        "short_job_ratio":              float(
            np.sum(bursts < SHORT_JOB_THRESHOLD) / n
        ),
        "arrival_clustering_score":     arrival_clustering_score,
        "burst_variability_index":      burst_variability_index,
        # ── Priority statistics ───────────────────────────
        "priority_mean":                float(priorities.mean()),
        "priority_std":                 float(priorities.std(ddof=0)),
        "priority_min":                 float(priorities.min()),
        "priority_max":                 float(priorities.max()),
        "priority_span":                float(priorities.max() - priorities.min()),
    }


# Derive feature column list automatically — never goes out of sync
_DUMMY = [{"arrival_time": 0, "burst_time": 1, "priority": 1, "id": 1}]
WORKLOAD_FEATURE_COLUMNS = list(workload_features(_DUMMY).keys())


def algo_row_features(base_features: dict, algo_name: str) -> dict:
    """Add algorithm one-hot encoding to the base feature dict."""
    row = base_features.copy()
    for name in ALGORITHM_NAMES:
        row[f"algo_{name}"] = 1 if name == algo_name else 0
    row["is_preemptive"] = 1 if algo_name in ("SRTF", "RR") else 0
    row["rr_quantum"]    = 2 if algo_name == "RR"           else 0
    return row


# =========================================================
# 6) Rule-Based Parameter Resolver
# =========================================================
def resolve_parameters(algo_name: str, processes: list) -> dict:
    bursts   = np.array([p["burst_time"]   for p in processes], dtype=float)
    arrivals = np.array([p["arrival_time"] for p in processes], dtype=float)

    if algo_name == "RR":
        burst_mean = float(bursts.mean())
        quantum    = 2 if burst_mean <= 6 else (4 if burst_mean <= 10 else 6)
        return {"quantum": quantum}

    if algo_name == "PRIORITY":
        arrival_std  = float(arrivals.std(ddof=0))
        return {"priority_mode": "dynamic" if arrival_std > 5 else "static"}

    return {}


# =========================================================
# 7) Dataset Builder
# =========================================================
def _build_single_workload(workload_id, arrival_pattern, burst_pattern, seed):
    processes    = generate_processes(
        arrival_pattern=arrival_pattern,
        burst_pattern=burst_pattern,
        seed=seed,
    )
    base_features = workload_features(processes)

    algo_rows = []
    cost_map  = {}

    for algo_name in ALGORITHM_NAMES:
        _, metrics = run_algorithm(algo_name, processes)
        cost       = compute_cost(metrics)
        cost_map[algo_name] = cost

        algo_rows.append({
            "workload_id":    workload_id,
            "algorithm":      algo_name,
            "arrival_pattern": arrival_pattern,
            "burst_pattern":  burst_pattern,
            **algo_row_features(base_features, algo_name),
            "avg_waiting":    metrics["avg_waiting"],
            "avg_turnaround": metrics["avg_turnaround"],
            "avg_response":   metrics["avg_response"],
            "cost":           cost,
        })

    # ── Classification label: algorithm with lowest cost ──
    best_algorithm = min(cost_map, key=cost_map.get)

    workload_row = {
        "workload_id":    workload_id,
        "arrival_pattern": arrival_pattern,
        "burst_pattern":  burst_pattern,
        **base_features,
        "best_algorithm": best_algorithm,       # ← classification target
    }

    return algo_rows, workload_row


def build_datasets(num_workloads: int = 1500, seed: int = 42):
    """
    Generates diverse workloads (scattered/clustered/tight/ordered/mixed)
    to avoid dataset bias as required by the prompt.
    """
    rng = random.Random(seed)

    # Balanced scenario pool — every pattern type is equally represented
    scenario_pool = [
        ("scattered", "short"),
        ("scattered", "medium"),
        ("scattered", "long"),
        ("scattered", "mixed"),
        ("clustered", "short"),
        ("clustered", "medium"),
        ("clustered", "mixed"),
        ("tight",     "short"),
        ("tight",     "medium"),
        ("ordered",   "short"),
        ("ordered",   "medium"),
        ("mixed",     "mixed"),
    ]

    tasks = [
        (wid, *rng.choice(scenario_pool), rng.randint(0, 10**9))
        for wid in range(num_workloads)
    ]

    results = joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_build_single_workload)(wid, ap, bp, s)
        for wid, ap, bp, s in tasks
    )

    all_algo_rows     = []
    all_workload_rows = []
    for algo_rows, workload_row in results:
        all_algo_rows.extend(algo_rows)
        all_workload_rows.append(workload_row)

    return pd.DataFrame(all_algo_rows), pd.DataFrame(all_workload_rows)


def save_csv(df: pd.DataFrame, filename: str):
    df.to_csv(filename, index=False)


# =========================================================
# 8) Classification Model  (matches prompt: argmax probabilities)
# =========================================================
def train_classification_model(workload_df: pd.DataFrame):
    """
    Trains a RandomForestClassifier on workload features.

    Input  : workload characteristics
    Output : probability distribution over {FCFS, SJF_NP, SRTF, PRIORITY, RR}
    Decision: Best Algorithm = argmax(probabilities)

    This matches the prompt's Section 5 exactly.
    """
    feature_columns = WORKLOAD_FEATURE_COLUMNS      # only workload features
    X      = workload_df[feature_columns]
    y_raw  = workload_df["best_algorithm"]
    groups = workload_df["workload_id"]

    # Encode string labels → integers
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx],       y[test_idx]

    clf = RandomForestClassifier(
        n_estimators   = 400,
        random_state   = 42,
        n_jobs         = -1,
        min_samples_leaf = 2,
        class_weight   = "balanced",       # guard against dataset bias (prompt §A)
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n=== Classification Model Evaluation ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print("\nDetailed Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0,
    ))

    # ── Dataset distribution (bias check, prompt §A) ──────
    print("=== Dataset Label Distribution ===")
    dist = workload_df["best_algorithm"].value_counts()
    total = dist.sum()
    for algo, cnt in dist.items():
        print(f"  {algo:<10} {cnt:>5}  ({100*cnt/total:.1f}%)")

    return clf, le, feature_columns


# =========================================================
# 9) Decision Engine  —  probability distribution + argmax
# =========================================================
def predict_best_algorithm(processes: list, clf, le: LabelEncoder, feature_columns: list):
    """
    Prediction Phase (prompt §6):
      1. Extract workload features
      2. Model outputs probability distribution over all algorithms
      3. Best Algorithm = argmax(probabilities)

    Returns:
        prob_table  : list of {algorithm, probability} sorted descending
        best        : {algorithm, probability} — the argmax decision
    """
    base   = workload_features(processes)
    X_pred = pd.DataFrame([base])[feature_columns]

    # probability distribution — shape (1, n_classes)
    proba  = clf.predict_proba(X_pred)[0]

    # Map class indices back to algorithm names
    prob_table = [
        {"algorithm": algo, "probability": float(p)}
        for algo, p in zip(le.classes_, proba)
    ]
    prob_table.sort(key=lambda x: x["probability"], reverse=True)

    best = prob_table[0]   # argmax
    return prob_table, best


# =========================================================
# 10) Main
# =========================================================
if __name__ == "__main__":

    print("=" * 55)
    print("   ML-based CPU Scheduling Decision System")
    print("=" * 55)

    # ── Build Dataset ─────────────────────────────────────
    print("\n[1/4] Building simulation dataset …")
    algo_df, workload_df = build_datasets(num_workloads=1500, seed=42)

    save_csv(algo_df,    "algorithm_cost_dataset.csv")
    save_csv(workload_df,"workload_labels.csv")
    del algo_df          # free memory

    # ── Train Classifier ──────────────────────────────────
    print("\n[2/4] Training classification model …")
    clf, le, feature_columns = train_classification_model(workload_df)
    del workload_df

    # ── Save Model ────────────────────────────────────────
    joblib.dump(clf,             MODEL_FILENAME)
    joblib.dump(le,              ENCODER_FILENAME)
    joblib.dump(feature_columns, FEATURES_FILENAME)

    print(f"\n[3/4] Model saved.")
    print(f"  → {MODEL_FILENAME}")
    print(f"  → {ENCODER_FILENAME}")
    print(f"  → {FEATURES_FILENAME}")

    # ── Prediction Phase ──────────────────────────────────
    print("\n[4/4] Prediction on test workload …")
    test_processes = generate_processes(
        arrival_pattern="scattered",
        burst_pattern="medium",
        seed=7,
    )

    print("\n=== Test Workload ===")
    print(f"  {'ID':<4} {'Arrival':>8} {'Burst':>7} {'Priority':>9}")
    print("  " + "-" * 32)
    for p in test_processes:
        print(f"  P{p['id']:<3} {p['arrival_time']:>8} {p['burst_time']:>7} {p['priority']:>9}")

    # Probability distribution (prompt §5)
    prob_table, best = predict_best_algorithm(test_processes, clf, le, feature_columns)

    print("\n=== Model Output — Probability Distribution ===")
    for row in prob_table:
        bar   = "█" * int(row["probability"] * 30)
        arrow = "  ← SELECTED (argmax)" if row["algorithm"] == best["algorithm"] else ""
        print(f"  {row['algorithm']:<10}  {row['probability']:.4f}  {bar}{arrow}")

    selected_algorithm = best["algorithm"]
    params             = resolve_parameters(selected_algorithm, test_processes)

    print(f"\n  Best Algorithm  : {selected_algorithm}")
    print(f"  Confidence      : {best['probability']*100:.1f}%")
    print(f"  Parameters      : {params if params else 'default'}")

    # ── Run Selected Algorithm ────────────────────────────
    gantt_chart, metrics = run_algorithm(selected_algorithm, test_processes, params)

    print("\n=== Selected Algorithm Performance ===")
    cost = compute_cost(metrics)
    print(f"  Avg Waiting Time    : {metrics['avg_waiting']:.4f}")
    print(f"  Avg Turnaround Time : {metrics['avg_turnaround']:.4f}")
    print(f"  Avg Response Time   : {metrics['avg_response']:.4f}")
    print(f"  Weighted Cost       : {cost:.4f}  (lower = better)")

    print_gantt_chart(gantt_chart)

    # ── Verification: actual best vs predicted ─────────────
    print("\n=== Verification — Actual vs Predicted ===")
    actual_costs = {}
    for algo in ALGORITHM_NAMES:
        p = resolve_parameters(algo, test_processes)
        _, m = run_algorithm(algo, test_processes, p)
        actual_costs[algo] = compute_cost(m)

    actual_best = min(actual_costs, key=actual_costs.get)
    print(f"  Actual best algorithm  : {actual_best}")
    print(f"  Predicted algorithm    : {selected_algorithm}")
    match = "✓ CORRECT" if actual_best == selected_algorithm else "✗ MISMATCH"
    print(f"  Result                 : {match}")
    print("\n  All algorithm costs:")
    for algo, c in sorted(actual_costs.items(), key=lambda x: x[1]):
        marker = " ← actual best" if algo == actual_best else ""
        print(f"    {algo:<10}  cost = {c:.4f}{marker}")