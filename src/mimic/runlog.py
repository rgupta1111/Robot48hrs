"""One JSON document per pick run: stage timings, every message, the outcome and a failure class. Written to data/runs/<run_id>.json and,
if a MongoDB URI is given, inserted into a collection (upsert on run_id, so re-sending is safe). No credentials are ever stored in it."""
import json
import os
import time
import uuid

FAILURES = [("grasp failed", "grasp_failed"), ("retry declined", "operator_stop"), ("lost", "lost_object"), ("could not find", "not_found"), ("no ", "not_found"), ("not reachable", "unreachable"),
            ("unreachable", "unreachable"), ("beyond the arm's reach", "unreachable"), ("joint 6", "wrist_limit"),
            ("cycle limit", "no_convergence"), ("stopped from the page", "operator_stop"), ("moved?", "object_moved"),
            ("cannot size", "no_width"), ("memory looks wrong", "bad_memory"), ("collision", "collision"), ("error", "error")]


def classify(reason):
    r = (reason or "").lower()
    return next((c for k, c in FAILURES if k in r), "other")


class RunLog:
    def __init__(self, args, out_dir="data/runs"):
        self.doc = {"run_id": time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6], "started_at": time.time(), "object": args.object,
                    "mode": "live" if args.sink == "viam" else "dry", "machine": args.viam_address.split(".")[0],
                    "params": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))
                               and not any(s in k.lower() for s in ("key", "token", "secret", "password", "uri"))},
                    "stages": [], "events": [], "metrics": {}}
        self.out_dir, self._stage = out_dir, None

    def event(self, msg):
        self.doc["events"].append({"t": round(time.time() - self.doc["started_at"], 2), "msg": str(msg)[:300]})

    def stage(self, state, text=""):
        if state == self._stage:
            return
        now = round(time.time() - self.doc["started_at"], 2)
        if self.doc["stages"]:
            self.doc["stages"][-1]["end_s"] = now
        self.doc["stages"].append({"stage": state, "start_s": now, "note": str(text)[:120]})
        self._stage = state

    def metric(self, key, value):
        self.doc["metrics"][key] = value

    def finish(self, exc=None):
        """exc: the exception that ended the run (None = it ran to the end). Success means the object was lifted (handed over or put down)."""
        d, now = self.doc, time.time()
        d["ended_at"], d["duration_s"] = now, round(now - d["started_at"], 2)
        if d["stages"]:
            d["stages"][-1]["end_s"] = round(d["duration_s"], 2)
        reached = [s["stage"] for s in d["stages"]]
        d["stage_reached"] = reached[-1] if reached else None
        reason = None if exc is None else (str(exc) or type(exc).__name__)
        clean = exc is None or (isinstance(exc, SystemExit) and exc.code in (0, None))
        d["success"] = bool(clean and "lifted" in reached)
        d["failure_reason"] = None if d["success"] else reason or "ended before the object was lifted"
        d["failure_class"] = None if d["success"] else classify(d["failure_reason"])
        return d

    def save(self):
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, self.doc["run_id"] + ".json")
        json.dump(self.doc, open(path, "w"), indent=2, default=str)
        return path


def push_mongo(doc, uri, db="robot48", coll="pick_runs", timeout_ms=4000):
    """Upsert one run document into MongoDB. Returns True on success; never raises (a logging failure must not hurt a run)."""
    try:
        from pymongo import MongoClient
        c = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        c[db][coll].replace_one({"run_id": doc["run_id"]}, doc, upsert=True)
        c.close()
        return True
    except Exception as e:
        print("mongo push failed (the JSON file is kept):", type(e).__name__, flush=True)
        return False
