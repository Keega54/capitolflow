"""Command line entry point:  python -m capitolflow <command>"""
from __future__ import annotations
import argparse, json, logging, sys

from . import db, pipeline
from .config import SETTINGS


def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="capitolflow",
                                 description="Congressional & executive trading analysis")
    ap.add_argument("--db", default=None, help="path to the SQLite database")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create the database")
    sub.add_parser("reference", help="sync member roster, committees, ticker map")

    p = sub.add_parser("ingest", help="fetch new disclosures")
    p.add_argument("--full", action="store_true", help="backfill all years, not just recent")
    p.add_argument("--years", type=int, nargs="*")

    p = sub.add_parser("lobbying", help="fetch lobbying disclosures")
    p.add_argument("--years", type=int, nargs="*")

    p = sub.add_parser("prices", help="fetch price history")
    p.add_argument("--max-tickers", type=int, default=None)

    sub.add_parser("analytics", help="compute returns, member scores, timing, event studies")
    sub.add_parser("context", help="fetch earnings, current-events themes and sector labels")
    sub.add_parser("timing", help="report how much of the edge survives the disclosure lag")

    p = sub.add_parser("backtest", help="fit factor weights and test them against a null")
    p.add_argument("--splits", type=int, default=6)
    p.add_argument("--null", type=int, default=120, help="shuffled-label draws per fold")

    p = sub.add_parser("predict", help="produce the ranked short- and long-term lists")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--scoreboard", action="store_true",
                   help="score previously-stored predictions against what happened")

    p = sub.add_parser("model", help="train the forward-return model")
    p.add_argument("--leakage-check", action="store_true",
                   help="also train with trade-date timing to size the disclosure lag advantage")

    p = sub.add_parser("refresh", help="run every stage in order (use this in a scheduler)")
    p.add_argument("--full", action="store_true")
    p.add_argument("--with-model", action="store_true",
                   help="also refit weights and regenerate predictions")

    p = sub.add_parser("export", help="write dashboard JSON")
    p.add_argument("--out", default="site")

    p = sub.add_parser("serve", help="run the API + dashboard locally")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    sub.add_parser("health", help="row counts and freshness")

    p = sub.add_parser("synthetic", help="populate a demo database with synthetic data")
    p.add_argument("--seed", type=int, default=7)

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    path = a.db or SETTINGS.db_path

    if a.cmd == "serve":
        import uvicorn
        from .api.app import create_app
        uvicorn.run(create_app(str(path)), host=a.host, port=a.port)
        return 0

    if a.cmd == "synthetic":
        sys.path.insert(0, str(SETTINGS.data_dir.parent))
        from tests.make_synthetic import build
        build(str(path), a.seed)
        return 0

    with db.session(path) as con:
        if a.cmd == "init":
            _print({"ok": True, "db": str(path)})
        elif a.cmd == "reference":
            _print(pipeline.sync_reference(con))
        elif a.cmd == "ingest":
            _print(pipeline.ingest_disclosures(con, years=a.years, incremental=not a.full))
        elif a.cmd == "lobbying":
            _print(pipeline.ingest_lobbying(con, years=a.years))
        elif a.cmd == "prices":
            _print(pipeline.sync_prices(con, max_tickers=a.max_tickers))
        elif a.cmd == "analytics":
            _print(pipeline.compute_analytics(con))
        elif a.cmd == "context":
            _print(pipeline.ingest_context(con))
        elif a.cmd == "timing":
            from .analytics import timing
            t = timing.decompose(con)
            timing.store(con, t)
            _print({"summary": timing.summary(t), "decay": timing.fit_decay(con)})
        elif a.cmd == "backtest":
            _print(pipeline.run_backtest(con, n_splits=a.splits, n_null=a.null))
        elif a.cmd == "predict":
            if a.scoreboard:
                from .analytics.rank import evaluate_past_predictions
                from .analytics.features import LONG_HORIZON, SHORT_HORIZON
                out = {}
                for h in (SHORT_HORIZON, LONG_HORIZON):
                    df = evaluate_past_predictions(con, h)
                    out[h] = ({"n": 0} if df is None or df.empty else
                              {"n": len(df),
                               "mean_realised_excess": float(df["realised_excess"].mean()),
                               "hit_rate": float((df["realised_excess"] > 0).mean())})
                _print(out)
            else:
                _print(pipeline.make_predictions(con, top_n=a.top))
        elif a.cmd == "model":
            if a.leakage_check:
                from .analytics.model import leakage_check
                r = leakage_check(con)
                _print({k: v for k, v in r.items() if k not in ("honest", "oracle")})
            else:
                _print(pipeline.train_model(con))
        elif a.cmd == "refresh":
            _print(pipeline.refresh(con, full=a.full, with_model=a.with_model))
        elif a.cmd == "export":
            from .api.export import export_site
            _print(export_site(con, a.out))
        elif a.cmd == "health":
            _print(pipeline.health(con))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
