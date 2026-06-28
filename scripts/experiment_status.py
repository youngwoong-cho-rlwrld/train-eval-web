#!/usr/bin/env python3
"""Experiment train/eval status tables (PoC1/PoC2/PoC3, discrete_MI, AH/BS ablations).

Reproduces the analysis format established 2026-06-11/12:
per experiment, one markdown table with
    Variant | Train job | Train state | Train resources | Eval job | Eval state | Eval resources

Sources
-------
1. GET {backend}/api/jobs?start=<START>   full job history, all clusters.
   (!) The default window of /api/jobs is 24h — passing start= is essential,
   otherwise older train/eval jobs silently disappear.
2. GET {backend}/api/results              authoritative eval->train pairing:
   each result carries checkpoint_job_name; pair via its YYYYMMDD_HHMMSS
   timestamp (checkpoint_job_id can be stale for repeated runs).
3. sacct on skt + kakao (AllocTRES + ReqTRES) for slurm CPU/GPU/mem.
   skt slurm binaries need a login shell (bash -lc). Memory falls back to
   ReqTRES when AllocTRES omits it (skt), so skt figures are *requested* mem.
4. mlxp pod resources derived from the job detail GPU count using the fixed
   preset map in backend/app/mlxp_submit.py.

Variant normalization (naming generations)
-------------------------------------------
- physixel_poc1_* / heuristic_* / physixel_poc3_* / discrete_MI_* : as-is.
- train_physixel_multitask_pt<K>_ps<S> + eval_poc1_* / eval_physixel_multitask_*_ah16_pt<K>_ps<S>
  -> "physixel_poc1_pt<K>_ps<S> (old)"  (PoC1 original generation, skt 05-22).
- physixel_multitask_3tasks_480_ah<N> (no pt suffix) -> "ah<N> (old)"
  (aborted first AH generation; *_skt_eval / *ckpt* sanity evals excluded).
- action_horizon_ablation_ah<N> -> "ah<N>"; batch_size_ablation_bs<N> -> "bs<N>".

Usage
-----
    python3 scripts/experiment_status.py [--start 2026-05-15] \
        [--backend http://localhost:8000] [--workdir /tmp/experiment-status]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import urllib.request

MLXP_RES = {
    1: "1 GPU, 14 CPU, 220Gi",
    2: "2 GPU, 28 CPU, 440Gi",
    4: "4 GPU, 56 CPU, 880Gi",
    8: "8 GPU, 100 CPU, 1500Gi",
}
SACCT_FMT = "JobID,JobName%150,State%30,NodeList,AllocTRES%110,ReqTRES%110,End"
STATE_RANK = {"COMPLETED": 0, "RUNNING": 1, "PENDING": 2, "TIMEOUT": 3, "FAILED": 4, "CANCELLED": 5}
BAD_EVAL_STATES = ("TIMEOUT", "FAILED")


def fetch_json(url: str, path: str) -> dict:
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)
    return json.loads(data)


def sacct_dump(host: str, start: str, path: str, login_shell: bool) -> None:
    cmd = f'sacct -X -u "$USER" -S {start} -P -n -o {SACCT_FMT}'
    if login_shell:
        cmd = f"bash -lc '{cmd}'"
    out = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=10", host, cmd],
        capture_output=True, text=True, timeout=120,
    ).stdout
    with open(path, "w") as f:
        f.write(out)


def tres_field(s: str, key: str) -> str:
    m = re.search(rf"(?:^|,){key}=([^,]+)", s or "")
    return m.group(1) if m else ""


def load_slurm_resources(res: dict, cluster: str, path: str) -> None:
    for line in open(path):
        p = line.rstrip("\n").split("|")
        if len(p) < 7:
            continue
        jid, _name, _state, _node, alloc, req, _end = p[:7]
        cpu = tres_field(alloc, "cpu") or tres_field(req, "cpu")
        gpu = tres_field(alloc, "gres/gpu") or tres_field(req, "gres/gpu")
        mem = tres_field(alloc, "mem") or tres_field(req, "mem")
        if mem.endswith("M") and mem[:-1].isdigit():
            mem = f"{round(int(mem[:-1]) / 1024)}Gi"
        res[(cluster, jid)] = f"{gpu or '?'} GPU, {cpu or '?'} CPU, {mem or '?'}"


def fetch_mlxp_details(backend: str, ids: list[str], cache_dir: str, res: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)

    def fetch_one(jid: str) -> None:
        path = os.path.join(cache_dir, f"mlxp_{jid}.json")
        if not os.path.exists(path):
            try:
                fetch_json(f"{backend}/api/jobs/mlxp/{jid}", path)
            except Exception:
                return
        try:
            d = json.load(open(path))
            res[("mlxp", jid)] = MLXP_RES.get(int(d.get("GPUs") or 0), "?")
        except Exception:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(fetch_one, ids))


def ts_of(name: str) -> str:
    m = re.search(r"(\d{8}_\d{6})", (name or "").replace("-", "_"))
    return m.group(1) if m else ""


def normalize(variant: str, job_name: str) -> tuple[str | None, str | None]:
    v, n = variant or "", job_name or ""
    pt_ps = re.search(r"pt(\d+)_ps(\d+)", n)
    if re.match(r"^physixel_poc4", v):
        return "PoC4", v
    if re.match(r"^physixel_poc3", v):
        return "PoC3", v
    if re.match(r"^physixel_poc1", v):
        return "PoC1", v
    if re.match(r"^discrete_MI", v):
        return "discrete_MI", v
    if re.match(r"^heuristic_", v):
        return "PoC2", v
    if re.match(r"^batch_size_ablation", v):
        m = re.search(r"bs\d+", v)
        return ("BS", m.group(0)) if m else (None, None)
    if re.match(r"^action_horizon_ablation", v):
        m = re.search(r"_ah(\d+)", v)
        return ("AH", "ah" + m.group(1)) if m else (None, None)
    if v.startswith("physixel_multitask") and pt_ps:
        return "PoC1old", f"physixel_poc1_pt{pt_ps.group(1)}_ps{pt_ps.group(2)} (old)"
    if v.startswith("physixel_multitask") and re.search(r"_ah(\d+)", v):
        if "skt_eval" in v or "ckpt" in v:
            return None, None
        m = re.search(r"_ah(\d+)", v)
        return ("AH", "ah" + m.group(1) + " (old)") if m else (None, None)
    return None, None


def collect(jobs: list, results: list, res: dict):
    trains: dict = {}
    evals: dict = {}
    for j in jobs:
        exp, cv = normalize(j.get("variant"), j.get("job_name"))
        if not exp:
            continue
        rec = dict(
            cluster=j["cluster"], id=str(j["job_id"]), name=j["job_name"] or "",
            state=j["state"], ts=ts_of(j["job_name"]),
            res=res.get((j["cluster"], str(j["job_id"])), "?"),
        )
        bucket = trains if j["phase"] == "train" else evals
        bucket.setdefault((exp, cv), []).append(rec)
    pair = {}
    for r in results:
        ck = r.get("checkpoint_job_name") or ""
        if ck:
            pair[(r["cluster"], str(r["job_id"]))] = ts_of(ck)
    return trains, evals, pair


def chrono(recs: list) -> list:
    return sorted(recs, key=lambda r: (r["ts"], r["id"]))


def pick_best(recs: list) -> dict | None:
    """Latest attempt; within the latest name-timestamp group prefer the best state."""
    if not recs:
        return None
    ordered = chrono(recs)
    last_ts = ordered[-1]["ts"]
    group = [r for r in ordered if r["ts"] == last_ts]
    return sorted(group, key=lambda r: (STATE_RANK.get(r["state"], 9), -int(re.sub(r"\D", "", r["id"]) or 0)))[0]


def pick_train(recs: list) -> dict | None:
    if not recs:
        return None
    return sorted(recs, key=lambda r: (STATE_RANK.get(r["state"], 9), r["ts"]))[0] if any(
        r["state"] in ("COMPLETED", "RUNNING") for r in recs
    ) else chrono(recs)[-1]


def attempt_no(chosen: dict, recs: list) -> int:
    ordered = chrono(recs)
    return ordered.index(chosen) + 1 if chosen in ordered else len(ordered)


def fmt_state(state: str, n_attempts: int) -> str:
    s = f"**{state}**" if state in BAD_EVAL_STATES else state
    if n_attempts > 1:
        s += f" ({n_attempts}{'nd' if n_attempts == 2 else 'rd' if n_attempts == 3 else 'th'} try)"
    return s


def job_cell(rec: dict | None) -> str:
    return f"{rec['id']} ({rec['cluster']})" if rec else "—"


def variant_row(cv: str, tl: list, el: list, pair: dict) -> str:
    t = pick_train(tl)
    # paired evals for the chosen train, else any eval for the variant
    paired = [e for e in el if t and t["ts"] and pair.get((e["cluster"], e["id"])) == t["ts"]]
    pool = el  # attempts counted across the variant
    e = pick_best(pool) if pool else None
    # prefer the paired record when it is the same attempt timestamp group
    if paired and e and pick_best(paired)["ts"] == e["ts"]:
        e = pick_best(paired)
    if t is None:
        trow = ["— (no train record)", "—", "—"]
    else:
        trow = [job_cell(t), t["state"], t["res"]]
    if e is None:
        erow = ["—", "—", "—"]
    else:
        erow = [job_cell(e), fmt_state(e["state"], attempt_no(e, pool)), e["res"]]
    return "| " + " | ".join([cv] + trow + erow) + " |"


HEADER = (
    "| Variant | Train job | Train state | Train resources | Eval job | Eval state | Eval resources |\n"
    "|---|---|---|---|---|---|---|"
)


def emit_simple(title: str, exp: str, trains: dict, evals: dict, pair: dict, order=None) -> None:
    keys = sorted({k[1] for k in list(trains) + list(evals) if k[0] == exp})
    if order:
        keys = sorted(keys, key=order)
    if not keys:
        return
    print(f"\n## {title}\n\n{HEADER}")
    foot = []
    for cv in keys:
        tl, el = trains.get((exp, cv), []), evals.get((exp, cv), [])
        print(variant_row(cv, tl, el, pair))
        extra_t = [t for t in tl if t is not pick_train(tl) and t["state"] not in ("COMPLETED", "RUNNING")]
        if extra_t:
            foot.append(f"{cv}: earlier train attempts " + ", ".join(f"{t['id']} {t['state']}" for t in extra_t))
    for line in foot:
        print(f"\n> {line}")


def emit_runs(title: str, exp: str, trains: dict, evals: dict, pair: dict) -> None:
    """One row per train run (for repeat-run experiments: AH, BS)."""
    keys = sorted({k[1] for k in list(trains) + list(evals) if k[0] == exp},
                  key=lambda v: (("(old)" in v), len(v), v))
    if not keys:
        return
    print(f"\n## {title}\n\n{HEADER}")
    all_notes = []
    for cv in keys:
        tl, el = chrono(trains.get((exp, cv), [])), evals.get((exp, cv), [])
        tl = [t for t in tl if "smoke" not in t["name"]]
        matched_ids = set()
        runs = [t for t in tl if t["state"] in ("COMPLETED", "RUNNING")] or tl[-1:]
        for i, t in enumerate(runs):
            label = cv if len(runs) == 1 else f"{cv} (run {i + 1})"
            paired = [e for e in el if t["ts"] and pair.get((e["cluster"], e["id"])) == t["ts"]]
            matched_ids |= {e["id"] for e in paired}
            if paired:
                ecell = ", ".join(job_cell(e) for e in chrono(paired))
                estate = ", ".join(e["state"] for e in chrono(paired))
                eres = " / ".join(sorted({e["res"] for e in paired}))
            else:
                ecell = estate = eres = "—"
            print("| " + " | ".join([label, job_cell(t), t["state"], t["res"], ecell, estate, eres]) + " |")
        aborted = [t for t in tl if t not in runs]
        notes = []
        leftovers = [e for e in el if e["id"] not in matched_ids]
        if aborted:
            notes.append(f"{len(aborted)} aborted train attempts ({', '.join(t['id'] + ' ' + t['state'] for t in aborted[:8])}{'…' if len(aborted) > 8 else ''})")
        if leftovers:
            done = [e for e in leftovers if e["state"] == "COMPLETED"]
            other = [e for e in leftovers if e["state"] != "COMPLETED"]
            if done:
                notes.append(f"unpaired completed evals: {', '.join(e['id'] for e in done)}")
            if other:
                notes.append(f"{len(other)} failed/timeout/cancelled eval attempts")
        if notes:
            all_notes.append(f"> {cv}: " + "; ".join(notes))
    for line in all_notes:
        print(f"\n{line}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2026-05-15")
    ap.add_argument("--backend", default="http://localhost:8000")
    ap.add_argument("--workdir", default="/tmp/experiment-status")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    jobs = fetch_json(f"{args.backend}/api/jobs?start={args.start}", f"{args.workdir}/jobs_full.json")["jobs"]
    results = fetch_json(f"{args.backend}/api/results", f"{args.workdir}/results.json")["variants"]
    res: dict = {}
    sacct_dump("skt", args.start, f"{args.workdir}/skt.psv", login_shell=True)
    sacct_dump("kakao-login-1", args.start, f"{args.workdir}/kakao.psv", login_shell=False)
    load_slurm_resources(res, "skt", f"{args.workdir}/skt.psv")
    load_slurm_resources(res, "kakao", f"{args.workdir}/kakao.psv")
    mlxp_ids = [str(j["job_id"]) for j in jobs if j["cluster"] == "mlxp"]
    fetch_mlxp_details(args.backend, mlxp_ids, f"{args.workdir}/jobdetails", res)

    trains, evals, pair = collect(jobs, results, res)

    def poc1_order(v: str):
        m = re.search(r"pt(\d+)(?:_ps(\d+))?", v)
        return (0 if "baseline" in v else 1, int(m.group(1)) if m else 0, int(m.group(2) or 0) if m else 0)

    print(f"# Experiment status (window start {args.start})")
    emit_simple("PoC1 — current generation", "PoC1", trains, evals, pair, order=poc1_order)
    emit_simple("PoC1 — original generation (skt, old naming)", "PoC1old", trains, evals, pair, order=poc1_order)
    emit_simple("discrete_MI (PoC1-style)", "discrete_MI", trains, evals, pair, order=poc1_order)
    emit_simple("PoC2 (heuristic)", "PoC2", trains, evals, pair, order=poc1_order)
    emit_simple("PoC3", "PoC3", trains, evals, pair)
    emit_simple("PoC4 (part-specific state encoder)", "PoC4", trains, evals, pair)
    emit_runs("Action-horizon ablation", "AH", trains, evals, pair)
    emit_runs("Batch-size ablation", "BS", trains, evals, pair)

    # Open items
    print("\n## Open items\n")
    for exp in ("PoC1", "PoC1old", "discrete_MI", "PoC2", "PoC3", "PoC4", "AH", "BS"):
        for (e, cv), tl in sorted(trains.items()):
            if e != exp:
                continue
            el = evals.get((e, cv), [])
            done_train = [t for t in tl if t["state"] == "COMPLETED"]
            if done_train and not el:
                print(f"- {cv}: trained but never evaluated")
            elif el:
                best = pick_best(el)
                if best and best["state"] in BAD_EVAL_STATES:
                    print(f"- {cv}: latest eval {best['id']} {best['state']} — needs retry/resume")


if __name__ == "__main__":
    main()
