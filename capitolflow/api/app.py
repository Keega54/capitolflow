"""FastAPI app serving the same JSON the static export writes, plus the dashboard."""
from __future__ import annotations
from pathlib import Path

from .. import db
from ..db import json_safe
from . import views

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def create_app(db_path: str | None = None):
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="CapitolFlow", version="1.0",
                  description="Congressional & executive-branch trading analysis")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"],
                       allow_headers=["*"])

    def conn():
        return db.connect(db_path)

    def wrap(fn, *a, **kw):
        con = conn()
        try:
            return JSONResponse(json_safe(fn(con, *a, **kw)))
        finally:
            con.close()

    @app.get("/api/summary.json")
    def _summary():
        return wrap(views.summary)

    @app.get("/api/tickers.json")
    def _tickers(limit: int = 40, start: str | None = None, end: str | None = None,
                 chamber: str | None = None, party: str | None = None):
        return wrap(views.tickers, limit=limit, start=start, end=end,
                    chamber=chamber, party=party)

    @app.get("/api/members.json")
    def _members(limit: int = 150, chamber: str | None = None, party: str | None = None):
        return wrap(views.members_view, limit=limit, chamber=chamber, party=party)

    @app.get("/api/timeseries.json")
    def _ts(freq: str = "M", ticker: str | None = None):
        return wrap(views.timeseries, freq=freq, ticker=ticker)

    @app.get("/api/delays.json")
    def _delays():
        return wrap(views.delays)

    @app.get("/api/clusters.json")
    def _clusters(window_days: int = 14, min_members: int = 4, limit: int = 30):
        return wrap(views.clusters, window_days=window_days,
                    min_members=min_members, limit=limit)

    @app.get("/api/lobbying.json")
    def _lobbying():
        return wrap(views.lobbying)

    @app.get("/api/predictions.json")
    def _predictions():
        return wrap(views.predictions)

    @app.get("/api/scoreboard.json")
    def _scoreboard():
        return wrap(views.scoreboard)

    @app.get("/api/events.json")
    def _events(limit: int = 100):
        return wrap(views.events, limit=limit)

    @app.get("/api/member/{member_id}.json")
    def _member(member_id: str):
        con = conn()
        try:
            d = views.member_detail(con, member_id)
            if not d["member"]:
                raise HTTPException(404, "unknown member")
            return JSONResponse(json_safe(d))
        finally:
            con.close()

    @app.get("/")
    def _index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/{fname}")
    def _static(fname: str):
        p = WEB_DIR / fname
        if p.is_file():
            return FileResponse(p)
        raise HTTPException(404)

    return app
