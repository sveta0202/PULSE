import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BASE = "https://stackmorelayers.ai/api/v1"
COMPETITION = 810
PHASE1 = 838
PHASE2 = 839


def token():
    return (Path.home() / ".sml_token").read_text().strip()


def get(path):
    req = urllib.request.Request(f"{BASE}/{path}", headers={"Authorization": f"Bearer {token()}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def request(method, path, body=None, headers=None, raw=None):
    h = {"Authorization": f"Bearer {token()}", "Idempotency-Key": uuid.uuid4().hex}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    if raw is not None:
        data = raw
        h["Content-Type"] = "application/octet-stream"
    h.update(headers or {})
    req = urllib.request.Request(f"{BASE}/{path}", data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path} -> {e.code}: {e.read().decode()[:1000]}")


def submit(path, phase, description, allow_final=False):
    if phase != PHASE1 and not allow_final:
        raise SystemExit("final phase submission requires --i-am-sure-final")
    f = Path(path)
    sess = request("POST", f"competitions/{COMPETITION}/submission-uploads/", {
        "phase_id": phase, "file_name": f.name, "file_size": f.stat().st_size,
        "content_type": "text/csv", "submission_kind": "prediction_file", "name": f.stem, "description": description})
    hdr = {"X-SML-Upload-Session": sess["upload_session_id"]}
    request("PUT", f"competitions/{COMPETITION}/submission-uploads/content/", raw=f.read_bytes(), headers=hdr)
    res = request("POST", f"competitions/{COMPETITION}/submission-uploads/complete/", {"name": f.stem, "description": description}, headers=hdr)
    print(json.dumps(res, ensure_ascii=False, indent=1)[:2000])


def submissions():
    for s in get("submissions/")["results"]:
        print(json.dumps(s, ensure_ascii=False)[:600])


def leaderboard():
    for ph in get(f"competitions/{COMPETITION}/leaderboard/")["phases"]:
        print(ph["phase_name"])
        for e in ph["entries"]:
            print(f"  {e['rank']:>3} {e['username']:<30} {e['score']:.5f}")


def status():
    print(json.dumps(get("auth/whoami/")["scopes"]))
    part = get(f"competitions/{COMPETITION}/participation/")
    for w in part["participation"]["windows"]:
        print(w["phase_id"], w["starts_at"], w["deadline_at"], "max", w["max_submissions"], "can_submit", w["can_submit"])
    print("my submissions:", get("submissions/")["meta"]["total"])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "submit":
        phase = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else PHASE1
        submit(sys.argv[2], phase, "airi-cu pipeline", allow_final="--i-am-sure-final" in sys.argv)
    else:
        {"leaderboard": leaderboard, "status": status, "submissions": submissions}[cmd]()
