"""Join lobbying spend to trading activity on (ticker, quarter).

The claim this supports is narrow and worth stating plainly: it shows whether
members traded a company around the time that company was spending heavily to
lobby Congress, and whether the specific committees those members sit on match
the agencies the company lobbied. It is a co-occurrence measure. It does not
establish that anyone acted on anything.
"""
from __future__ import annotations
import pandas as pd


def _quarter_expr(col: str) -> str:
    return (f"strftime('%Y', {col}) || 'Q' || "
            f"((CAST(strftime('%m', {col}) AS INTEGER) + 2) / 3)")


def lobbying_by_ticker_quarter(con) -> pd.DataFrame:
    return pd.read_sql_query(f"""
        SELECT ticker,
               {_quarter_expr('period_start')} AS quarter,
               SUM(COALESCE(amount,0)) AS lobby_spend,
               COUNT(*) AS n_filings,
               COUNT(DISTINCT registrant_name) AS n_registrants
        FROM lobbying_filings
        WHERE ticker IS NOT NULL AND ticker_confidence >= 0.6 AND period_start IS NOT NULL
        GROUP BY ticker, quarter""", con)


def trades_by_ticker_quarter(con, min_conf: float = 0.6) -> pd.DataFrame:
    return pd.read_sql_query(f"""
        SELECT t.ticker,
               {_quarter_expr('t.transaction_date')} AS quarter,
               COUNT(*) AS n_trades,
               COUNT(DISTINCT t.member_id) AS n_members,
               SUM(COALESCE(t.amount_est,0)) AS gross_volume,
               SUM(COALESCE(t.amount_est,0)*t.direction) AS net_flow,
               SUM(COALESCE(t.amount_est,0)*t.direction*COALESCE(s.weight,1.0)) AS weighted_net_flow
        FROM transactions t
        LEFT JOIN member_scores s ON s.member_id=t.member_id AND s.horizon_days=90
        WHERE t.ticker IS NOT NULL AND t.ticker_confidence >= ? AND t.transaction_date IS NOT NULL
        GROUP BY t.ticker, quarter""", con, params=[min_conf])


def overlay(con, *, lag_quarters: int = 0) -> pd.DataFrame:
    """Merge the two panels. lag_quarters>0 aligns trades against lobbying that
    happened N quarters EARLIER (i.e. did spend precede the trading?)."""
    lob = lobbying_by_ticker_quarter(con)
    trd = trades_by_ticker_quarter(con)
    if lob.empty or trd.empty:
        return pd.DataFrame()

    def shift(q: str, n: int) -> str:
        y, qq = q.split("Q")
        v = int(y) * 4 + int(qq) - 1 + n
        return f"{v // 4}Q{v % 4 + 1}"

    if lag_quarters:
        lob = lob.assign(quarter=lob["quarter"].map(lambda q: shift(q, lag_quarters)))
    df = trd.merge(lob, on=["ticker", "quarter"], how="left")
    df["lobby_spend"] = df["lobby_spend"].fillna(0.0)
    df["lobby_spend_musd"] = df["lobby_spend"] / 1e6
    return df.sort_values(["quarter", "gross_volume"], ascending=[False, False])


def correlation_report(con, lags=(0, 1, 2)) -> pd.DataFrame:
    """Spearman correlation between lobbying spend and trading interest, by lag.

    Spearman rather than Pearson because both variables are heavily right-skewed
    and a handful of mega-spenders would otherwise drive the whole coefficient.
    """
    from scipy import stats
    out = []
    for lag in lags:
        df = overlay(con, lag_quarters=lag)
        if df.empty or len(df) < 10:
            continue
        for metric in ("n_members", "gross_volume", "weighted_net_flow"):
            x, y = df["lobby_spend"], df[metric].abs()
            mask = x.notna() & y.notna()
            if mask.sum() < 10:
                continue
            rho, p = stats.spearmanr(x[mask], y[mask])
            out.append({"lag_quarters": lag, "metric": metric, "spearman_rho": round(float(rho), 4),
                        "p_value": float(p), "n": int(mask.sum())})
    return pd.DataFrame(out)


def committee_alignment(con, limit: int = 50) -> pd.DataFrame:
    """Members who traded a company while sitting on a committee whose subject
    matter matches the issue codes that company lobbied on."""
    ISSUE_TO_COMMITTEE = {
        "DEF": ["HSAS", "SSAS", "HSAP", "SSAP"], "HCR": ["HSIF", "SSHR", "HSWM"],
        "MMM": ["HSIF", "SSCM"], "TAX": ["HSWM", "SSFI"], "BAN": ["HSBA", "SSBK"],
        "ENG": ["HSIF", "SSEG"], "TEC": ["HSIF", "SSCM", "HSSY"], "TRD": ["HSWM", "SSFI"],
        "AGR": ["HSAG", "SSAF"], "TRA": ["HSPW", "SSCM"], "ENV": ["HSII", "SSEV"],
    }
    rows = pd.read_sql_query("""
        SELECT t.member_id, m.full_name, m.chamber, t.ticker, t.transaction_date,
               t.direction, COALESCE(t.amount_est,0) AS amount_est
        FROM transactions t JOIN members m ON m.member_id=t.member_id
        WHERE t.ticker IS NOT NULL AND t.member_id IS NOT NULL AND t.ticker_confidence>=0.6""", con)
    if rows.empty:
        return rows
    issues = pd.read_sql_query("""
        SELECT lf.ticker, la.issue_code, SUM(COALESCE(lf.amount,0)) AS spend
        FROM lobbying_filings lf JOIN lobbying_activities la USING (filing_uuid)
        WHERE lf.ticker IS NOT NULL GROUP BY lf.ticker, la.issue_code""", con)
    memb = pd.read_sql_query("SELECT member_id, committee_id FROM committee_memberships", con)
    if issues.empty or memb.empty:
        return pd.DataFrame()

    comm_by_member = memb.groupby("member_id")["committee_id"].apply(set).to_dict()
    out = []
    for tic, grp in issues.groupby("ticker"):
        targets = set()
        for _, ir in grp.iterrows():
            targets |= set(ISSUE_TO_COMMITTEE.get(str(ir["issue_code"]).upper(), []))
        if not targets:
            continue
        spend = float(grp["spend"].sum())
        sub = rows[rows["ticker"] == tic]
        for mid, mg in sub.groupby("member_id"):
            hit = comm_by_member.get(mid, set()) & targets
            if hit:
                out.append({"member_id": mid, "full_name": mg["full_name"].iloc[0],
                            "chamber": mg["chamber"].iloc[0], "ticker": tic,
                            "matched_committees": ",".join(sorted(hit)),
                            "n_trades": int(len(mg)), "volume": float(mg["amount_est"].sum()),
                            "lobby_spend": spend})
    if not out:
        return pd.DataFrame()
    return (pd.DataFrame(out).sort_values(["lobby_spend", "volume"], ascending=False)
            .head(limit).reset_index(drop=True))
