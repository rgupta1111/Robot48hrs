"""Send saved run logs (data/runs/*.json) to MongoDB. The URI comes from the MONGODB_URI environment variable (never from a file here).

    MONGODB_URI='mongodb+srv://...' scripts/run.sh scripts/push_runs_mongo.py [--db robot48] [--coll pick_runs] [--dir data/runs]
Safe to repeat: each run is upserted by run_id.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mimic.runlog import push_mongo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="data/runs")
ap.add_argument("--db", default="robot48")
ap.add_argument("--coll", default="pick_runs")
a = ap.parse_args()
uri = os.environ.get("MONGODB_URI")
if not uri:
    raise SystemExit("set MONGODB_URI")
files = sorted(glob.glob(os.path.join(a.dir, "*.json")))
ok = sum(push_mongo(json.load(open(f)), uri, a.db, a.coll) for f in files)
print(f"sent {ok}/{len(files)} run logs to {a.db}.{a.coll}")
