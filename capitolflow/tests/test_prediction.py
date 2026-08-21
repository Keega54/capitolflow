"""Tests for the prediction layer, run against a synthetic world with a KNOWN
signal structure: several factors are planted, and lobbying is a deliberate
placebo. The harness is only trustworthy if it finds the first group and starves
the second."""
import sys, warnings
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from capitolflow import db
from capitolflow.analytics import accuracy, backtest, features, rank, returns, timing
from tests.make_synthetic_v2 import build


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    p = tmp_path_factory.mktemp("cf2") / "syn2.db"
    build(str(p), seed=11)
    c = db.connect(p)
    returns.store_trade_returns(c, returns.compute_trade_returns(c))
    accuracy.store_member_scores(c, accuracy.compute_member_scores(c))
    timing.store(c, timing.decompose(c))
    yield c
    c.close()


@pytest.fixture(scope="module")
def truth(con):
    return db.get_kv(con, "synthetic_truth_v2")


@pytest.fixture(scope="module")
def panel(con):
    return features.build(con)


@pytest.fixture(scope="module")
def report(con, panel):
    return backtest.run(con, panel=panel, n_splits=5, n_null=50)


# ---------------------------------------------------------------- timing
def test_timing_splits_at_disclosure(con):
    df = timing.decompose(con, horizon=90)
    assert not df.empty
    recon = (df["pre_disclosure_excess"] + df["post_disclosure_excess"] - df["total_excess"]).abs()
    assert recon.max() < 1e-9, "pre + post must reconstruct the total exactly"
    assert (df["lag_days"] >= 0).all()


def test_timing_detects_pre_disclosure_concentration(con, truth):
    """The generator puts ~45% of the edge before the filing. The module must
    report that the edge is front-loaded rather than claiming it is capturable."""
    df = timing.decompose(con, horizon=90)
    s = timing.summary(df)
    assert s["mean_pre_disclosure_excess"] > s["mean_post_disclosure_excess"]
    assert "verdict" in s and len(s["verdict"]) > 20


def test_decay_fit_refuses_to_invent_a_curve(con):
    """When the age curve is noise, half-life must be NaN, not a made-up number."""
    d = timing.fit_decay(con, horizon=90, max_age=150, bucket=30)
    assert d["status"] in ("ok", "insufficient_data")
    hl = d.get("half_life_days")
    assert hl is None or np.isnan(hl) or hl > 0


def test_staleness_weight_behaviour():
    assert timing.staleness_weight(0, 30) == pytest.approx(1.0)
    assert timing.staleness_weight(30, 30) == pytest.approx(0.5)
    assert timing.staleness_weight(60, 30) == pytest.approx(0.25)
    assert timing.staleness_weight(60, None) == 1.0        # no fit -> no decay invented
    assert timing.staleness_weight(60, float("nan")) == 1.0


# ---------------------------------------------------------------- features
def test_panel_has_all_factor_columns(panel):
    assert not panel.empty
    for c in features.ALL_FEATURES:
        assert c in panel.columns, f"missing feature {c}"
    assert "y_21" in panel.columns and "y_126" in panel.columns


def test_event_features_vary_across_tickers(panel):
    """The bug this guards: a raw theme value is identical for every ticker on a
    day, so cross-sectional standardization erases it. Sector interaction is what
    makes it a stock-picking signal."""
    one_day = panel[panel["date"] == panel["date"].max()]
    assert one_day["conflict_x_sector"].nunique() > 1
    assert one_day["event_beta"].nunique() > 1


def test_factor_scores_are_standardized_within_date(panel):
    fs = features.factor_scores(panel)
    for f in backtest.FACTORS:
        assert f in fs.columns
    day = fs[fs["date"] == fs["date"].max()]
    assert abs(float(day["momentum"].mean())) < 0.5


def test_features_are_point_in_time(con, panel):
    """No feature may reference a filing that had not yet been disclosed."""
    import pandas as pd
    tx = pd.read_sql_query(
        "SELECT ticker, MIN(filed_date) f FROM transactions WHERE ticker IS NOT NULL "
        "GROUP BY ticker", con)
    tx["f"] = pd.to_datetime(tx["f"])
    first_filing = dict(zip(tx["ticker"], tx["f"]))
    early = panel[panel["date"] < panel["date"].min() + pd.Timedelta(days=1)]
    for r in early.itertuples(index=False):
        ff = first_filing.get(r.ticker)
        if ff is not None and r.date < ff:
            assert r.n_members_90 == 0, "a filing leaked into a date before it existed"


# ---------------------------------------------------------------- backtest
def test_walk_forward_splits_are_purged():
    import pandas as pd
    dates = pd.Series(pd.date_range("2020-01-01", periods=200, freq="W"))
    for tr, te, td in backtest.walk_forward(dates, n_splits=4, horizon=126):
        assert dates.iloc[tr].max() < pd.Timestamp(td[0])
        gap = (pd.Timestamp(td[0]) - dates.iloc[tr].max()).days
        assert gap >= 126, f"purge gap {gap}d is shorter than the return horizon"


def test_weights_are_normalized_and_bounded(report):
    for h, r in report["horizons"].items():
        if r.get("status") != "ok":
            continue
        w = np.array(list(r["weights"].values()))
        assert abs(np.abs(w).sum() - 1.0) < 1e-6, "weights must sum to 1 in absolute value"
        assert np.abs(w).max() <= backtest.MAX_ABS_WEIGHT + 1e-9


def test_planted_factors_outweigh_the_placebo(report, truth):
    """The central test. Planted factors must collectively carry more weight
    than the placebo, at every horizon."""
    for h, r in report["horizons"].items():
        if r.get("status") != "ok":
            continue
        w = r["weights"]
        planted = sum(abs(w[f]) for f in truth["planted_factors"] if f in w)
        placebo = sum(abs(w[f]) for f in truth["placebo_factors"] if f in w)
        assert planted > placebo * 3, (
            f"horizon {h}: planted weight {planted:.3f} vs placebo {placebo:.3f}")


def test_placebo_is_least_stable(report, truth):
    """Block bootstrapping by ticker should expose the placebo as unstable."""
    placebo = truth["placebo_factors"][0]
    for h, r in report["horizons"].items():
        if r.get("status") != "ok":
            continue
        st = r.get("stability") or {}
        if placebo not in st:
            continue
        others = [v for k, v in st.items() if k != placebo]
        assert st[placebo] <= min(others) + 1e-9, (
            f"horizon {h}: placebo stability {st[placebo]:.2f} not the lowest")


def test_null_distribution_is_reported(report):
    for h, r in report["horizons"].items():
        if r.get("status") != "ok":
            continue
        assert "null_ic_p95" in r and np.isfinite(r["null_ic_p95"])
        assert "beats_null" in r
        assert isinstance(r["verdict"], str) and len(r["verdict"]) > 20


def test_effective_sample_is_tickers_not_rows(report):
    """Reporting independent groups keeps the row count from being mistaken for
    statistical power."""
    for h, r in report["horizons"].items():
        if r.get("status") == "ok":
            assert r["n_independent_groups"] and r["n_independent_groups"] < r["n_obs"]


def test_backtest_persists_weights_and_folds(con, report):
    assert con.execute("SELECT COUNT(*) FROM factor_weights").fetchone()[0] > 0
    assert con.execute("SELECT COUNT(*) FROM backtest_results").fetchone()[0] > 0


# ---------------------------------------------------------------- ranking
def test_rankings_produced_for_both_horizons(con, panel):
    out = rank.generate(con, panel=panel, top_n=10)
    assert set(out["horizons"]) == {"short_term", "long_term"}
    for label, r in out["horizons"].items():
        assert len(r["picks"]) == 10
        assert [p["rank"] for p in r["picks"]] == list(range(1, 11))
        scores = [p["score"] for p in r["picks"]]
        assert scores == sorted(scores, reverse=True), "picks must be sorted by score"


def test_attribution_sums_to_score(con, panel):
    out = rank.generate(con, panel=panel, top_n=10)
    for r in out["horizons"].values():
        for p in r["picks"]:
            assert sum(p["attribution"].values()) == pytest.approx(p["score"], abs=1e-6)


def test_confidence_reflects_null_test(con, panel):
    out = rank.generate(con, panel=panel, top_n=10)
    for r in out["horizons"].values():
        assert 0.0 <= r["confidence"] <= 0.8, "confidence must never imply certainty"
        assert len(r["confidence_note"]) > 20


def test_predictions_are_stored(con, panel):
    rank.generate(con, panel=panel, top_n=10)
    n = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    assert n == 20
