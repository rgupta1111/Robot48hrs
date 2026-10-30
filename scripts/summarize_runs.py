"""Summarise the run logs in data/runs: success rate, failure classes, where runs stop, durations, and the last few runs.

    scripts/run.sh scripts/summarize_runs.py [--dir data/runs] [--last 8]
"""
import argparse
import collections
import glob
import json
import os
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="data/runs")
ap.add_argument("--last", type=int, default=8)
a = ap.parse_args()
runs = sorted((json.load(open(f)) for f in glob.glob(os.path.join(a.dir, "*.json"))), key=lambda d: d["started_at"])
if not runs:
    raise SystemExit("no run logs yet")
ok = [r for r in runs if r["success"]]
print(f"{len(runs)} runs, {len(ok)} successful ({100 * len(ok) / len(runs):.0f}%); "
      f"median duration {statistics.median(r['duration_s'] for r in runs):.0f} s")
print("\nfailure classes:")
for k, n in collections.Counter(r["failure_class"] for r in runs if not r["success"]).most_common():
    print(f"  {k:16} {n}")
print("\nlast stage reached by failed runs:")
for k, n in collections.Counter(r["stage_reached"] for r in runs if not r["success"]).most_common():
    print(f"  {k:16} {n}")
print(f"\nlast {a.last} runs:")
for r in runs[-a.last:]:
    print(f"  {r['run_id']}  {'OK  ' if r['success'] else 'FAIL'}  {r['duration_s']:5.0f} s  {r['stage_reached'] or '-':11} {r['failure_class'] or ''}  {(r['failure_reason'] or '')[:70]}")
