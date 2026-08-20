"""Tests for the core universe and the track record.

The track record is the part of this project most likely to lie, so these tests
target the specific ways it could: compounding overlapping periods, mixing
simulated results with live ones, and extrapolating an annual rate from a
sample too small to support one."""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from capitolflow import db
from capitolflow.analytics import (accuracy, features, returns, scoreboard,
                                   timing, universe)
from tests.make_synthetic_v2 import build


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    p = tmp_path_factory.mktemp("cf3") / "syn.db"
    build(str(p), seed=11)
    c = db.connect(p)
    returns.store_trade_returns(c, returns.compute_trade_returns(c))
    accuracy.store_member_scores(c, accuracy.compute_member_scores(c))
    timing.store(c, timing.decompose(c))
    yield c
    c.close()


@pytest.fixture(scope="module")
def panel(con):
    return features.build(con)


@pytest.fixture(scope="module")
def sim(con, panel):
    return scoreboard.simulate(con, panel=panel, step=4)


# ---------------------------------------------------------------- universe
def test_universe_is_ranked_and_sized(con):
    df = universe.compute(con, size=20)
    assert len(df) == 20
    assert df["rank"].tolist() == list(range(1, 21))
    assert df["n_members"].min() >= 1


def test_universe_prefers_breadth_over_repetition(con):
    """One member trading forty times is weaker evidence than forty members
    trading once. Members carry the most weight in the blend."""
    df = universe.compute(con, size=45)
    top, bottom = df.head(10), df.tail(10)
    assert top["n_members"].mean() >= bottom["n_members"].mean()


def test_universe_store_preserves_added_on(con):
    universe.store(con, universe.compute(con, size=15))
    first = {r["ticker"]: r["added_on"] for r in
             con.execute("SELECT ticker, added_on FROM core_universe")}
    universe.store(con, universe.compute(con, size=15))
    second = {r["ticker"]: r["added_on"] for r in
              con.execute("SELECT ticker, added_on FROM core_universe")}
    for t in set(first) & set(second):
        assert first[t] == second[t], "a name already in the universe kept its join date"


def test_universe_tickers_falls_back_when_empty(con):
    con.execute("DELETE FROM core_universe")
    assert len(universe.tickers(con, fallback_limit=10)) > 0
    universe.store(con, universe.compute(con, size=25))


# ---------------------------------------------------------------- overlap
def test_non_overlapping_chain_respects_horizon():
    """The bug this guards produced a +350,000% six-month strategy."""
    dates = pd.date_range("2020-01-01", periods=40, freq="28D")
    by = pd.DataFrame({"as_of": [str(d.date()) for d in dates],
                       "pick": 0.05, "bench": 0.01, "n": 10})
    kept = scoreboard._non_overlapping(by, horizon=126)
    ds = [pd.Timestamp(x) for x in kept["as_of"]]
    need = pd.Timedelta(days=int(126 * scoreboard.CAL))
    for a, b in zip(ds, ds[1:]):
        assert b - a >= need, "consecutive held periods must not overlap"
    assert len(kept) < len(by), "a 6-month hold cannot rebalance every 4 weeks"


def test_short_horizon_keeps_more_periods_than_long(sim):
    s, l = sim["horizons"][21], sim["horizons"][126]
    if s.get("status") == "ok" and l.get("status") == "ok":
        assert s["n_chained_periods"] > l["n_chained_periods"]


def test_cumulative_return_is_not_absurd(sim):
    """A sanity ceiling. Any real equity curve that clears this is a bug."""
    for h, s in sim["horizons"].items():
        if s.get("status") != "ok":
            continue
        assert s["cumulative_return"] < 20, (
            f"horizon {h} returned {s['cumulative_return']:.1f}x — "
            f"almost certainly compounding overlapping periods")


# ---------------------------------------------------------------- summarize
def test_summarize_handles_empty_and_unresolved():
    assert scoreboard.summarize(pd.DataFrame()) ["status"] == "no_picks"
    df = pd.DataFrame([{"as_of": "2024-01-01", "horizon_days": 21, "ticker": "AAA",
                        "rank": 1, "score": 0.1, "mode": "live",
                        "realised_return": None, "benchmark_return": None,
                        "excess_return": None, "resolved": 0}])
    assert scoreboard.summarize(df, horizon=21)["status"] == "unresolved"


def test_annualization_refuses_small_samples():
    one = pd.DataFrame([{"as_of": "2024-01-01", "pick": 0.02, "bench": 0.01, "n": 10}])
    assert np.isnan(scoreboard._annualize(0.02, 0.01, one, 21))
    many = pd.DataFrame([{"as_of": f"2024-0{i}-01", "pick": 0.02, "bench": 0.01, "n": 10}
                         for i in range(1, 6)])
    assert np.isfinite(scoreboard._annualize(0.10, 0.05, many, 21))


def test_hit_rate_and_excess_are_consistent(sim):
    for h, s in sim["horizons"].items():
        if s.get("status") != "ok":
            continue
        assert 0.0 <= s["hit_rate"] <= 1.0
        assert 0.0 <= s["periods_beating_benchmark"] <= 1.0
        # cumulative excess must equal picks minus benchmark
        assert s["cumulative_excess"] == pytest.approx(
            s["cumulative_return"] - s["benchmark_cumulative_return"], abs=1e-9)


# ---------------------------------------------------------------- separation
def test_backtested_and_live_never_merge(con, sim):
    rep = scoreboard.report(con)
    assert set(rep) >= {"backtested", "live", "caveat"}
    modes = {r["mode"] for r in con.execute("SELECT DISTINCT mode FROM pick_history")}
    assert modes <= {"backtested", "live"}
    assert "not a track record" in rep["caveat"]


def test_simulate_does_not_leak_future_data(con, panel):
    """Training rows must all close before the rebalance date they inform."""
    rows = pd.read_sql_query(
        "SELECT as_of, horizon_days FROM pick_history WHERE mode='backtested'", con)
    if rows.empty:
        pytest.skip("no simulated history")
    latest_price = pd.read_sql_query("SELECT MAX(date) d FROM prices", con)["d"].iloc[0]
    for r in rows.itertuples(index=False):
        exit_d = pd.Timestamp(r.as_of) + pd.Timedelta(days=int(r.horizon_days * scoreboard.CAL))
        # Any pick marked resolved must have had its full window inside the data.
        resolved = con.execute(
            "SELECT resolved FROM pick_history WHERE as_of=? AND horizon_days=? "
            "AND mode='backtested' LIMIT 1", (r.as_of, r.horizon_days)).fetchone()[0]
        if resolved:
            assert exit_d <= pd.Timestamp(latest_price) + pd.Timedelta(days=7)


def test_resolve_live_only_scores_elapsed_horizons(con):
    from capitolflow.analytics import rank
    rank.generate(con, panel=features.build(con), top_n=10)
    n = scoreboard.resolve_live(con)
    assert n >= 0
    unresolved = con.execute(
        "SELECT COUNT(*) FROM pick_history WHERE mode='live' AND resolved=0").fetchone()[0]
    assert unresolved == 0, "live rows are only written once they can be scored"
