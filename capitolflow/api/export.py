"""Dump every dashboard view to static JSON so the site can be hosted anywhere
(GitHub Pages, S3, a USB stick) with no server running."""
from __future__ import annotations
import json, shutil
from datetime import datetime, timezone
from pathlib import Path

from ..db import json_safe
from . import views

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def export_site(con, out_dir: str = "site") -> dict:
    out = Path(out_dir)
    api = out / "api"
    api.mkdir(parents=True, exist_ok=True)

    payloads = {
        "summary": views.summary(con),
        "tickers": views.tickers(con, limit=40),
        "members": views.members_view(con, limit=150),
        "timeseries": views.timeseries(con, freq="M"),
        "delays": views.delays(con),
        "clusters": views.clusters(con, min_members=3, window_days=21, limit=30),
        "lobbying": views.lobbying(con),
        "events": views.events(con, limit=100),
        "predictions": views.predictions(con),
    }
    payloads["meta"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "Public disclosure data. Trade dates are self-reported by filers.",
    }
    written = {}
    for name, data in payloads.items():
        p = api / f"{name}.json"
        p.write_text(json.dumps(json_safe(data), separators=(",", ":"),
                                default=str, allow_nan=False))
        written[name] = p.stat().st_size

    for f in WEB_DIR.glob("*"):
        if f.is_file():
            shutil.copy2(f, out / f.name)
    return {"out_dir": str(out.resolve()), "files": written}
