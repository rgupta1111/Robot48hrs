import json
import types

from mimic.runlog import RunLog, classify


def args():
    return types.SimpleNamespace(object="cup", sink="viam", viam_address="sfhack16-main.x.local.viam.cloud:8080", step_mm=100, mongo_uri="secret", api_key="k")


def test_success_and_no_secrets(tmp_path):
    r = RunLog(args(), str(tmp_path))
    r.stage("approaching"), r.stage("lifted"), r.event("lifted.")
    d = r.finish(None)
    assert d["success"] and d["stage_reached"] == "lifted" and d["failure_class"] is None
    assert "mongo_uri" not in d["params"] and "api_key" not in d["params"] and d["machine"] == "sfhack16-main"
    assert json.load(open(r.save()))["run_id"] == d["run_id"]


def test_failure_is_classified():
    r = RunLog(args())
    r.stage("approaching")
    d = r.finish(SystemExit("did not reach the object within the cycle limit"))
    assert not d["success"] and d["failure_class"] == "no_convergence" and d["stage_reached"] == "approaching"
    assert classify("stopped from the page") == "operator_stop"
