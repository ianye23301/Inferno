import json, sys
from datetime import datetime


def log_event(event: str, **fields):
    payload = {"ts": datetime.utcnow().isoformat()+"Z", "event": event, **fields}
    sys.stdout.write(json.dumps(payload)+"\n")
    sys.stdout.flush()

