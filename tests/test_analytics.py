"""End-to-end analytics tests against a synthetic dataset with known ground truth.

The generator plants three archetypes (skilled / noise / unskilled). If the
returns, shrinkage and event-study code are correct, they must recover that
ordering. If a refactor breaks the math, these fail loudly.
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capitolflow import db
from capitolflow.analytics import accuracy, aggregates, eventstudy, lobbying_join, returns
from tests.make_synthetic import build


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    p = tmp_path_factory.mktemp("cf") / "syn.db"
    build(str(p), seed=7)
    c = db.connect(p)
    r = returns.compute_trade_returns(c)
    returns.store_trade_returns(c, r)
    s = accuracy.compute_member_scores(c)
    accuracy.store_member_scores(c, s)
    yield c
    c.close()


@pytest.fixture(scope="module")
def truth(con):
    return db.get_kv(con, "synthetic_truth")


def test_returns_are_computed_for_every_horizon(con):
    rows = con.execute("SELECT horizon_days, COUNT(*) n FROM trade_returns "
                       "GROUP BY horizon_days").fetchall()
    horizons = {r["horizon_days"] for r in rows}
    assert {30, 90, 180}.issubset(horizons)
    assert all(r["n"] > 100 for r in rows)


def test_incomplete_horizons_are_skipped_not_imputed(con):
    """A 365-day horizon cannot exist for a trade made 100 days ago."""
    n365 = con.execute("SELECT COUNT(*) FROM trade_returns WHERE horizon_days=365").fetchone()[0]
    n30 = con.execute("SELECT COUNT(*) FROM trade_returns WHERE horizon_days=30").fetchone()[0]
    assert n365 < n30


def test_excess_return_is_signed_by_direction(con):
    """A sale ahead of a decline must score positive."""
    row = con.execute("""
        SELECT r.excess_return, r.asset_return, r.bench_return, t.direction
        FROM trade_returns r JOIN transactions t USING (txn_id)
        WHERE r.horizon_days=90 AND t.direction=-1 LIMIT 50""").fetchall()
    for r in row:
        expected = (r["asset_return"] - r["bench_return"]) * -1
        assert abs(r["excess_return"] - expected) < 1e-9


def test_archetype_ordering_recovered(con, truth):
    df = accuracy.compute_member_scores(con)
    df = df[df.horizon_days == 90].copy()
    df["arch"] = df["member_id"].map(lambda m: truth[m]["archetype"])
    means = df.groupby("arch")["shrunk_excess"].mean()
    assert means["skilled"] > means["noise"] > means["unskilled"], means.to_dict()


def test_hit_rate_ordering_recovered(con, truth):
    df = accuracy.compute_member_scores(con)
    df = df[df.horizon_days == 90].copy()
    df["arch"] = df["member_id"].map(lambda m: truth[m]["archetype"])
    hr = df.groupby("arch")["hit_rate"].mean()
    assert hr["skilled"] > 0.55 and hr["unskilled"] < 0.45


def test_shrinkage_grows_with_sample_size(con):
    """Members with few trades must be pulled harder toward the average."""
    import numpy as np
    df = accuracy.compute_member_scores(con)
    df = df[df.horizon_days == 90].copy()
    mu = np.average(df["mean_excess"], weights=df["n_scored"])
    df["B"] = ((df.shrunk_excess - mu).abs() / (df.mean_excess - mu).abs()).clip(0, 1)
    small = df[df.n_scored <= 5]["B"].mean()
    large = df[df.n_scored >= 30]["B"].mean()
    assert small < large, f"small-sample B={small:.3f} should be below large-sample B={large:.3f}"


def test_weights_are_bounded(con):
    rows = con.execute("SELECT MIN(weight) a, MAX(weight) b FROM member_scores").fetchone()
    assert 0.2 <= rows["a"] and rows["b"] <= 2.0


def test_ticker_leaderboard_shape(con):
    df = aggregates.ticker_leaderboard(con, limit=10)
    assert len(df) == 10
    assert (df["n_members"] >= 1).all()
    assert (df["gross_volume"] > 0).all()
    # buyers + sellers can exceed n_members (a person may do both), never be below the max
    assert (df["n_members"] >= df[["n_members_buying", "n_members_selling"]].max(axis=1)).all()


def test_net_flow_is_bounded_by_gross(con):
    df = aggregates.ticker_leaderboard(con, limit=20)
    assert (df["net_flow"].abs() <= df["gross_volume"] + 1e-6).all()


def test_aggregates_key_on_trade_date_not_filing_date(con):
    """The whole premise: the monthly series must move with trade dates."""
    ts = aggregates.flow_timeseries(con, freq="M")
    first_trade = con.execute("SELECT MIN(transaction_date) FROM transactions").fetchone()[0]
    assert ts["period"].min() == first_trade[:7]


def test_cluster_detector_finds_planted_density(con):
    df = aggregates.cluster_detector(con, window_days=21, min_members=3)
    assert len(df) > 0
    assert (df["n_members"] >= 3).all()
    assert set(df["side"]) <= {"buy", "sell"}


def test_event_study_separates_archetypes(con, truth):
    df = eventstudy.study_trades(con, store=False)
    assert len(df) > 100
    df["arch"] = df["member_id"].map(lambda m: truth[m]["archetype"])
    s = eventstudy.summarize(df, by="arch").set_index("arch")
    assert s.loc["skilled", "mean_car"] > s.loc["unskilled", "mean_car"]


def test_event_study_betas_are_sane(con):
    df = eventstudy.study_trades(con, store=False)
    assert 0.5 < df["beta"].mean() < 1.8
    assert df["r2"].between(0, 1).all()


def test_lobbying_overlay_joins(con):
    ov = lobbying_join.overlay(con)
    assert not ov.empty
    assert {"ticker", "quarter", "lobby_spend", "gross_volume"} <= set(ov.columns)
    assert (ov["lobby_spend"] >= 0).all()


def test_lobbying_lag_shifts_quarters(con):
    a = lobbying_join.overlay(con, lag_quarters=0)
    b = lobbying_join.overlay(con, lag_quarters=1)
    assert not a["lobby_spend"].equals(b["lobby_spend"])
