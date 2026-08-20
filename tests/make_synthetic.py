"""Build a synthetic database with KNOWN ground truth so the analytics stack can
be verified offline. Three member archetypes:
  * skilled   : buys are timed just before a real +drift, sells before a -drift
  * unskilled : trades are timed with the opposite sign
  * noise     : random timing
If the pipeline is correct, member_scores must rank skilled > noise > unskilled,
and the shrinkage must pull low-trade-count members toward the middle.
"""
from __future__ import annotations
import argparse, json, math, random, sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from capitolflow import db
from capitolflow.util.amounts import geo_mid
from capitolflow.util.dates import delay_days

TICKERS = ["AAPL", "MSFT", "NVDA", "LMT", "RTX", "JPM", "XOM", "PFE", "TSLA", "GOOG",
           "AMZN", "META", "BA", "CAT", "UNH", "SPY"]
BRACKETS = [(1001, 15000), (15001, 50000), (50001, 100000), (100001, 250000),
            (250001, 500000), (500001, 1000000), (1000001, 5000000)]


def gen_prices(rng, start: date, days: int):
    """Geometric random walk per ticker plus a market factor, so a benchmark exists."""
    mkt = [0.0]
    for _ in range(days):
        mkt.append(mkt[-1] + rng.gauss(0.0003, 0.009))
    series = {}
    for t in TICKERS:
        beta = 1.0 if t == "SPY" else rng.uniform(0.6, 1.6)
        idio = [0.0]
        for _ in range(days):
            idio.append(idio[-1] + rng.gauss(0.0, 0.012 if t != "SPY" else 0.0))
        lvl = math.log(rng.uniform(30, 300))
        px = []
        for i in range(days + 1):
            px.append(math.exp(lvl + beta * mkt[i] + idio[i]))
        series[t] = px
    return series


def forward_excess(series, t, i, h, bench="SPY"):
    j = min(i + h, len(series[t]) - 1)
    a = series[t][j] / series[t][i] - 1
    b = series[bench][j] / series[bench][i] - 1
    return a - b


def build(path: str, seed: int = 7, n_members: int = 60, days: int = 900):
    rng = random.Random(seed)
    start = date.today() - timedelta(days=days + 400)
    series = gen_prices(rng, start, days)

    with db.session(path) as con:
        for t in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            if t[0] not in ("sqlite_sequence",):
                con.execute(f"DELETE FROM {t[0]}")

        # ---- prices
        rows = []
        for t, px in series.items():
            for i, p in enumerate(px):
                d = (start + timedelta(days=i))
                if d.weekday() >= 5:
                    continue
                rows.append({"ticker": t, "date": d.isoformat(), "open": p, "high": p * 1.01,
                             "low": p * 0.99, "close": p, "adj_close": p, "volume": 1e6})
        db.upsert_many(con, "prices", rows, mode="REPLACE")

        # ---- members
        archetypes = (["skilled"] * 12) + (["unskilled"] * 12) + (["noise"] * (n_members - 24))
        rng.shuffle(archetypes)
        members = []
        for k, arch in enumerate(archetypes):
            chamber = "house" if k % 3 else "senate"
            mid = f"{'H' if chamber=='house' else 'S'}{k:06d}"
            members.append({
                "member_id": mid, "bioguide_id": mid,
                "full_name": f"{arch.title()} Member {k}", "first_name": arch.title(),
                "last_name": f"Member{k}", "chamber": chamber,
                "party": ["Democrat", "Republican", "Independent"][k % 3],
                "state": ["CA", "TX", "NY", "FL", "OH"][k % 5], "district": str(k % 20 + 1),
                "role_title": "Rep." if chamber == "house" else "Sen.",
                "term_start": "2019-01-03", "term_end": "2027-01-03", "active": 1,
            })
        db.upsert_many(con, "members", members, mode="REPLACE")
        arch_by_id = {m["member_id"]: a for m, a in zip(members, archetypes)}

        # ---- trades
        truth = {}
        filings, txns = [], []
        for m in members:
            mid = m["member_id"]
            arch = arch_by_id[mid]
            n = rng.choice([3, 4, 6, 10, 18, 30, 55])       # deliberately uneven sample sizes
            truth[mid] = {"archetype": arch, "n": n}
            for j in range(n):
                i = rng.randrange(60, days - 200)
                t = rng.choice([x for x in TICKERS if x != "SPY"])
                fx = forward_excess(series, t, i, 90)
                if arch == "skilled":
                    direction = 1 if fx > 0 else -1
                    if rng.random() < 0.25:                  # skill is not perfect
                        direction *= -1
                elif arch == "unskilled":
                    direction = -1 if fx > 0 else 1
                    if rng.random() < 0.25:
                        direction *= -1
                else:
                    direction = rng.choice([1, -1])
                lo, hi = rng.choice(BRACKETS)
                tdate = start + timedelta(days=i)
                delay = rng.choice([12, 20, 28, 33, 39, 44, 52, 71, 120])
                fdate = tdate + timedelta(days=delay)
                fid = f"synthetic:{mid}:{j}"
                filings.append({
                    "filing_id": fid, "source": "house" if m["chamber"] == "house" else "senate",
                    "doc_id": f"{mid}-{j}", "member_id": mid, "filer_name_raw": m["full_name"],
                    "filing_type": "ptr", "filing_year": fdate.year,
                    "filed_date": fdate.isoformat(),
                    "url": f"https://example.invalid/{fid}", "doc_format": "synthetic",
                    "parse_status": "ok", "parse_note": "synthetic",
                })
                txns.append({
                    "txn_id": db.txn_id(fid, t, tdate.isoformat(),
                                        "buy" if direction > 0 else "sell", lo, "self"),
                    "filing_id": fid, "member_id": mid,
                    "transaction_date": tdate.isoformat(),
                    "notification_date": fdate.isoformat(), "filed_date": fdate.isoformat(),
                    "filing_delay_days": delay, "owner": "self",
                    "asset_name_raw": f"{t} Inc. ({t}) [ST]", "ticker": t,
                    "ticker_confidence": 0.99, "asset_type": "stock",
                    "txn_type": "buy" if direction > 0 else "sell", "direction": direction,
                    "amount_low": lo, "amount_high": hi, "amount_est": geo_mid(lo, hi),
                    "comment": None, "cap_gains_over_200": None, "source": "synthetic",
                    "raw": "{}",
                })
        db.upsert_many(con, "filings", filings, mode="REPLACE")
        db.upsert_many(con, "transactions", txns, mode="REPLACE")

        # ---- lobbying: three tickers get heavy spend, the rest little
        heavy = ["LMT", "RTX", "PFE"]
        lob, acts = [], []
        for yr in (date.today().year - 2, date.today().year - 1, date.today().year):
            for q, (a, b) in enumerate([("01-01", "03-31"), ("04-01", "06-30"),
                                        ("07-01", "09-30"), ("10-01", "12-31")], 1):
                for t in TICKERS:
                    if t == "SPY":
                        continue
                    amt = rng.uniform(2e6, 9e6) if t in heavy else rng.uniform(2e4, 4e5)
                    uid = f"syn-{t}-{yr}Q{q}"
                    lob.append({"filing_uuid": uid, "filing_year": yr,
                                "filing_period": f"Q{q}", "filing_type": "LD-2",
                                "dt_posted": f"{yr}-{b}", "period_start": f"{yr}-{a}",
                                "period_end": f"{yr}-{b}",
                                "registrant_name": f"{t} Government Affairs LLC",
                                "client_name": f"{t} Inc.", "client_id": t,
                                "amount": amt, "ticker": t, "ticker_confidence": 0.95})
                    acts.append({"filing_uuid": uid,
                                 "issue_code": "DEF" if t in ("LMT", "RTX", "BA") else "TAX",
                                 "description": "Appropriations and authorization matters",
                                 "entity": "U.S. House of Representatives"})
        db.upsert_many(con, "lobbying_filings", lob, mode="REPLACE")
        db.upsert_many(con, "lobbying_activities", acts, mode="IGNORE")

        db.set_kv(con, "synthetic_truth", truth)
        print(f"synthetic db -> {path}")
        for tbl in ("members", "filings", "transactions", "prices", "lobbying_filings"):
            print(f"  {tbl:18} {con.execute(f'SELECT COUNT(*) FROM {tbl}').fetchone()[0]:>7}")
    return truth


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/tmp/synthetic.db")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    build(a.db, a.seed)
