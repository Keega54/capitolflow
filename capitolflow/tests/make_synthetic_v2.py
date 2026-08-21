"""Synthetic world with a KNOWN, planted signal structure, used to prove the
backtest can both find real signal and reject fake signal.

Planted truths:
  * politician "skilled" trades genuinely predict forward returns, but the edge
    is split: PRE_SHARE of it lands before the filing becomes public
  * earnings surprise has a real, modest forward effect
  * the `conflict` theme genuinely moves defense-sector names
  * lobbying spend is deliberately UNCORRELATED with returns — it is the placebo
    factor, and a correct backtest must give it a near-zero weight

If the fitted weights load onto the planted factors and not the placebo, the
harness works. That is the whole point of this file.
"""
from __future__ import annotations
import argparse, math, random, sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capitolflow import db
from capitolflow.util.amounts import geo_mid

# A wider universe than a toy: with overlapping forward returns the effective
# sample size is driven by the number of tickers, not the number of rows, so a
# 16-name universe cannot distinguish a real factor from a lucky one.
DEFENSE = ["LMT", "RTX", "BA", "NOC", "GD", "LHX", "HII", "TXT", "LDOS", "KTOS"]
TECH = ["AAPL", "MSFT", "NVDA", "GOOG", "META", "AMZN", "AVGO", "AMD", "INTC",
        "CRM", "ORCL", "ADBE", "QCOM", "MU", "TXN"]
OTHER = ["JPM", "XOM", "PFE", "UNH", "CAT", "TSLA", "BAC", "WFC", "CVX", "COP",
         "MRK", "LLY", "ABBV", "JNJ", "HD", "MCD", "KO", "PEP", "WMT", "DE"]
TICKERS = DEFENSE + TECH + OTHER + ["SPY"]
SECTORS = ({t: "Aerospace & Defense" for t in DEFENSE} |
           {t: "Technology" for t in TECH} |
           {t: "Financial Services" for t in OTHER})
BRACKETS = [(1001, 15000), (15001, 50000), (50001, 100000), (100001, 250000),
            (250001, 500000), (500001, 1000000)]

PRE_SHARE = 0.45          # fraction of politician edge realised before disclosure
POL_EFFECT = 0.055        # forward excess return attributable to a strong pol signal
EARN_EFFECT = 0.035
CONFLICT_EFFECT = 0.045


# Child tables first: deleting a parent row while children reference it trips
# the foreign-key constraint the schema deliberately enforces.
_WIPE_ORDER = [
    "trade_returns", "trade_timing", "event_studies", "member_scores",
    "committee_memberships", "lobbying_activities", "predictions",
    "factor_weights", "backtest_results", "transactions", "filings",
    "member_aliases", "committees", "members", "prices", "securities",
    "earnings", "event_index", "ticker_sectors", "lobbying_filings",
    "policy_events", "ingest_runs", "kv",
]


def _wipe(con) -> None:
    have = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for tbl in _WIPE_ORDER:
        if tbl in have:
            con.execute(f"DELETE FROM {tbl}")


def build(path: str, seed: int = 11, days: int = 2200, n_members: int = 70):
    rng = random.Random(seed)
    start = date.today() - timedelta(days=days + 260)

    # ---- latent drivers, defined first so prices can be built to contain them
    conflict = [0.0]
    for _ in range(days):
        conflict.append(max(0.0, conflict[-1] * 0.985 + rng.gauss(0, 0.25) +
                            (2.5 if rng.random() < 0.004 else 0.0)))

    earn_days = {t: sorted(rng.sample(range(80, days - 200), max(1, days // 91)))
                 for t in TICKERS if t != "SPY"}
    earn_surp = {t: {d: rng.gauss(0, 0.35) for d in ds} for t, ds in earn_days.items()}

    pol_days = {t: sorted(rng.sample(range(70, days - 250), 22)) for t in TICKERS if t != "SPY"}
    pol_sign = {t: {d: rng.choice([1, -1]) for d in ds} for t, ds in pol_days.items()}

    # ---- prices: random walk PLUS the planted effects
    mkt = [0.0]
    for _ in range(days):
        mkt.append(mkt[-1] + rng.gauss(0.0003, 0.0085))
    series = {}
    for t in TICKERS:
        beta = 1.0 if t == "SPY" else rng.uniform(0.7, 1.4)
        lvl = math.log(rng.uniform(40, 260))
        idio, drift = [0.0], [0.0] * (days + 2)
        if t != "SPY":
            for d, s in pol_sign[t].items():
                # politician edge spread over the following ~120 days
                for k in range(d, min(d + 120, days + 1)):
                    drift[k] += s * POL_EFFECT / 120
            for d, sp in earn_surp[t].items():
                for k in range(d, min(d + 63, days + 1)):
                    drift[k] += sp * EARN_EFFECT / 63
            if t in DEFENSE:
                for k in range(1, days + 1):
                    drift[k] += (conflict[k] - 1.0) * CONFLICT_EFFECT / 120
        for i in range(days):
            idio.append(idio[-1] + rng.gauss(0.0, 0.011 if t != "SPY" else 0.0) + drift[i])
        series[t] = [math.exp(lvl + beta * mkt[i] + idio[i]) for i in range(days + 1)]

    with db.session(path) as con:
        _wipe(con)

        # prices
        prows = []
        for t, px in series.items():
            for i, p in enumerate(px):
                d = start + timedelta(days=i)
                if d.weekday() >= 5:
                    continue
                prows.append({"ticker": t, "date": d.isoformat(), "open": p, "high": p * 1.01,
                              "low": p * .99, "close": p, "adj_close": p, "volume": 1e6})
        db.upsert_many(con, "prices", prows, mode="REPLACE")
        db.upsert_many(con, "ticker_sectors",
                       [{"ticker": t, "sector": s, "industry": s, "source": "synthetic"}
                        for t, s in SECTORS.items()], mode="REPLACE")

        # earnings
        erows = []
        for t, ds in earn_days.items():
            for d in ds:
                rd = start + timedelta(days=d)
                sp = earn_surp[t][d]
                erows.append({"ticker": t, "report_date": rd.isoformat(),
                              "fiscal_period": rd.isoformat(), "eps_actual": 1 + sp,
                              "eps_estimate": 1.0, "surprise_pct": sp,
                              "revenue_actual": None, "revenue_estimate": None,
                              "rev_surprise_pct": None, "time_of_day": "amc",
                              "source": "synthetic"})
        db.upsert_many(con, "earnings", erows, mode="REPLACE")

        # events
        evrows = []
        for i in range(days + 1):
            d = (start + timedelta(days=i)).isoformat()
            evrows.append({"theme": "conflict", "date": d, "intensity": conflict[i],
                           "z_score": None, "source": "synthetic"})
            for th in ("tariffs", "energy", "health", "ai", "monetary"):
                evrows.append({"theme": th, "date": d, "intensity": abs(rng.gauss(5, 1.5)),
                               "z_score": None, "source": "synthetic"})
        db.upsert_many(con, "event_index", evrows, mode="REPLACE")
        from capitolflow.sources.events import compute_zscores
        compute_zscores(con)

        # members
        arch = (["skilled"] * 14) + (["unskilled"] * 12) + (["noise"] * (n_members - 26))
        rng.shuffle(arch)
        members = []
        for k, a in enumerate(arch):
            ch = "house" if k % 3 else "senate"
            members.append({"member_id": f"{'H' if ch=='house' else 'S'}{k:06d}",
                            "bioguide_id": f"X{k:06d}",
                            "full_name": f"{a.title()} Member {k}", "first_name": a.title(),
                            "last_name": f"Member{k}", "chamber": ch,
                            "party": ["Democrat", "Republican", "Independent"][k % 3],
                            "state": ["CA", "TX", "NY", "FL", "OH"][k % 5],
                            "district": str(k % 20 + 1),
                            "role_title": "Rep." if ch == "house" else "Sen.",
                            "term_start": "2019-01-03", "term_end": "2029-01-03", "active": 1})
        db.upsert_many(con, "members", members, mode="REPLACE")
        arch_by = {m["member_id"]: a for m, a in zip(members, arch)}

        # trades: skilled members trade WITH the planted pol_sign, near its onset
        filings, txns = [], []
        for t, ds in pol_days.items():
            for d in ds:
                s = pol_sign[t][d]
                for _ in range(rng.randint(2, 6)):
                    m = rng.choice(members)
                    a = arch_by[m["member_id"]]
                    if a == "skilled":
                        direction = s if rng.random() < 0.85 else -s
                    elif a == "unskilled":
                        direction = -s if rng.random() < 0.85 else s
                    else:
                        direction = rng.choice([1, -1])
                    # trade lands slightly before the drift begins
                    ti = max(1, d - rng.randint(0, 10))
                    tdate = start + timedelta(days=ti)
                    # lag chosen so PRE_SHARE of the 120-day drift precedes the filing
                    lag = int(120 * PRE_SHARE * rng.uniform(0.6, 1.4))
                    lag = max(6, min(lag, 110))
                    fdate = tdate + timedelta(days=lag)
                    if (fdate - start).days >= days - 130:
                        continue
                    lo, hi = rng.choice(BRACKETS)
                    fid = f"syn2:{m['member_id']}:{t}:{ti}:{rng.randint(0,10**6)}"
                    filings.append({"filing_id": fid, "source": "house", "doc_id": fid,
                                    "member_id": m["member_id"], "filer_name_raw": m["full_name"],
                                    "filing_type": "ptr", "filing_year": fdate.year,
                                    "filed_date": fdate.isoformat(), "url": None,
                                    "doc_format": "synthetic", "parse_status": "ok",
                                    "parse_note": "synthetic"})
                    txns.append({
                        "txn_id": db.txn_id(fid, t, tdate.isoformat(),
                                            "buy" if direction > 0 else "sell", lo, "self"),
                        "filing_id": fid, "member_id": m["member_id"],
                        "transaction_date": tdate.isoformat(),
                        "notification_date": fdate.isoformat(), "filed_date": fdate.isoformat(),
                        "filing_delay_days": lag, "owner": "self",
                        "asset_name_raw": f"{t} Inc. ({t}) [ST]", "ticker": t,
                        "ticker_confidence": 0.99, "asset_type": "stock",
                        "txn_type": "buy" if direction > 0 else "sell", "direction": direction,
                        "amount_low": lo, "amount_high": hi, "amount_est": geo_mid(lo, hi),
                        "comment": None, "cap_gains_over_200": None, "source": "synthetic",
                        "raw": "{}"})
        db.upsert_many(con, "filings", filings, mode="IGNORE")
        db.upsert_many(con, "transactions", txns, mode="IGNORE")

        # lobbying — the PLACEBO: deliberately unrelated to returns
        lob = []
        for yr in range(start.year, date.today().year + 1):
            for q, (a, b) in enumerate([("01-01", "03-31"), ("04-01", "06-30"),
                                        ("07-01", "09-30"), ("10-01", "12-31")], 1):
                for t in TICKERS:
                    if t == "SPY":
                        continue
                    lob.append({"filing_uuid": f"syn2-{t}-{yr}Q{q}", "filing_year": yr,
                                "filing_period": f"Q{q}", "filing_type": "LD-2",
                                "dt_posted": f"{yr}-{b}", "period_start": f"{yr}-{a}",
                                "period_end": f"{yr}-{b}",
                                "registrant_name": f"{t} GA LLC", "client_name": f"{t} Inc.",
                                "client_id": t, "amount": rng.uniform(1e5, 8e6),
                                "ticker": t, "ticker_confidence": 0.95})
        db.upsert_many(con, "lobbying_filings", lob, mode="REPLACE")

        db.set_kv(con, "synthetic_truth_v2", {
            "archetypes": arch_by, "pre_share": PRE_SHARE,
            "planted_factors": ["politician_flow", "politician_conviction",
                                "politician_freshness", "earnings", "events"],
            "placebo_factors": ["lobbying"],
        })
        print(f"synthetic v2 -> {path}")
        for tbl in ("members", "filings", "transactions", "prices", "earnings",
                    "event_index", "lobbying_filings", "ticker_sectors"):
            print(f"  {tbl:20} {con.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]:>7}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/tmp/syn2.db")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    build(a.db, a.seed)
