# app.py — Corporate Finance Dashboard (Enhanced)
# rchrostowski / elorev

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import datetime as dt
import pytz
from typing import Optional, Dict, List, Tuple

st.set_page_config(page_title="Corporate Finance Dashboard", layout="wide")

# =============================
# Helpers: formatting & styling
# =============================
def human_format_number(x):
    if pd.isna(x): return ""
    try: x = float(x)
    except: return str(x)
    ax = abs(x)
    if ax >= 1_000_000_000: return f"{x/1_000_000_000:.2f} B"
    if ax >= 1_000_000:     return f"{x/1_000_000:.2f} M"
    if ax >= 1_000:         return f"{x:,.0f}"
    return f"{x:.2f}"

def styled_money_table(df: pd.DataFrame) -> str:
    if df is None or df.empty: return "<i>No data</i>"
    d = df.copy()
    for c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.style.format(human_format_number).to_html()

def styled_ratio_table(df: pd.DataFrame) -> str:
    if df is None or df.empty: return "<i>No data</i>"
    d = df.copy()
    for c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
    # no background_gradient here to avoid matplotlib dependency on Streamlit Cloud
    return d.style.format("{:.2%}").to_html()

def find_label(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty: return None
    # try exact first, then substring
    for c in candidates:
        for i in df.index:
            if str(i) == c: return i
    for c in candidates:
        for i in df.index:
            if c.lower() in str(i).lower(): return i
    return None

# =============================
# Cached Data Loaders
# =============================
@st.cache_data(show_spinner=False)
def get_fx_rate_to_usd(currency: Optional[str]) -> float:
    if not currency: return 1.0
    cur = currency.upper()
    if cur == "USD": return 1.0
    try:
        df = yf.Ticker(f"{cur}USD=X").history(period="7d")
        if not df.empty: return float(df["Close"].iloc[-1])
    except: pass
    try:
        df = yf.Ticker(f"USD{cur}=X").history(period="7d")
        if not df.empty and df["Close"].iloc[-1] != 0:
            return 1.0 / float(df["Close"].iloc[-1])
    except: pass
    return 1.0

@st.cache_data(show_spinner=False)
def fetch_data(tickers: List[str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for t in tickers:
        tkr = yf.Ticker(t)
        try: info = tkr.info or {}
        except: info = {}
        out[t] = {
            "info": info,
            "financials": getattr(tkr, "financials", pd.DataFrame()),
            "balance": getattr(tkr, "balance_sheet", pd.DataFrame()),
            "cashflow": getattr(tkr, "cashflow", pd.DataFrame()),
            "hist": tkr.history(period="2y")
        }
    return out

def to_usd(series: pd.Series, fx: float) -> pd.Series:
    try: return pd.to_numeric(series, errors="coerce") * fx
    except: return series

# =============================
# Beta & WACC logic
# =============================
@st.cache_data(show_spinner=False)
def estimate_beta(ticker: str, market="^GSPC", years=2) -> Optional[float]:
    try:
        end = dt.datetime.now()
        start = end - dt.timedelta(days=int(365*years))
        f = yf.Ticker(ticker).history(start=start, end=end)["Close"].pct_change().dropna()
        m = yf.Ticker(market).history(start=start, end=end)["Close"].pct_change().dropna()
        df = pd.concat([f, m], axis=1).dropna()
        df.columns = ["firm", "mkt"]
        if df.empty: return None
        X = sm.add_constant(df["mkt"])
        res = sm.OLS(df["firm"], X).fit()
        return float(res.params.get("mkt", np.nan))
    except:
        return None

def infer_tax_rate(fin_df: pd.DataFrame) -> Optional[float]:
    try:
        tax_lbl = find_label(fin_df, ["Income Tax Expense", "Tax Provision"])
        pbt_lbl = find_label(fin_df, ["Pretax Income", "Earnings Before Tax", "Income Before Tax"])
        if tax_lbl and pbt_lbl:
            tax = pd.to_numeric(fin_df.loc[tax_lbl], errors="coerce")
            pbt = pd.to_numeric(fin_df.loc[pbt_lbl], errors="coerce").replace(0, np.nan)
            r = (tax / pbt).dropna()
            if not r.empty:
                x = float(r.mean())
                return max(0.0, min(0.35, x))
    except: pass
    return None

def infer_cost_of_debt(fin_df: pd.DataFrame, bs_df: pd.DataFrame) -> Optional[float]:
    try:
        int_lbl = find_label(fin_df, ["Interest Expense"])
        debt_lbl = find_label(bs_df, ["Total Debt", "Long Term Debt", "Total Liab"])
        if not int_lbl or not debt_lbl: return None
        interest = abs(pd.to_numeric(fin_df.loc[int_lbl], errors="coerce"))
        debt = abs(pd.to_numeric(bs_df.loc[debt_lbl], errors="coerce"))
        debt = debt.reindex(interest.index, fill_value=np.nan)
        r = (interest / debt).dropna()
        if not r.empty:
            x = float(r.mean())
            return max(0.0, min(0.15, x))
    except: pass
    return None

def compute_wacc(ticker: str, data: Dict[str, dict], rf: float, mrp: float) -> Dict[str, Optional[float]]:
    fin = data[ticker]["financials"]
    bs  = data[ticker]["balance"]
    info = data[ticker]["info"] or {}

    beta = estimate_beta(ticker)
    ke = None if beta is None else rf + beta * mrp
    kd = infer_cost_of_debt(fin, bs)
    tax = infer_tax_rate(fin) or 0.21

    mcap = info.get("marketCap")
    if mcap is None:
        so = info.get("sharesOutstanding")
        px = None
        try: px = data[ticker]["hist"]["Close"].iloc[-1]
        except: pass
        if so and px: mcap = so * px

    debt_val = None
    dlabel = find_label(bs, ["Total Debt", "Long Term Debt"])
    if dlabel is not None:
        try: debt_val = float(pd.to_numeric(bs.loc[dlabel], errors="coerce").dropna().iloc[0])
        except: debt_val = None

    if None in (ke, kd, mcap, debt_val) or (mcap + debt_val) == 0:
        return {"beta": beta, "ke": ke, "kd": kd, "tax": tax, "E": mcap, "D": debt_val, "wacc": None}

    V = mcap + debt_val
    we, wd = mcap / V, debt_val / V
    wacc = we * ke + wd * kd * (1 - tax)
    return {"beta": beta, "ke": ke, "kd": kd, "tax": tax, "E": mcap, "D": debt_val, "wacc": wacc}

# =============================
# Event Study
# =============================
NYC = pytz.timezone("America/New_York")

@st.cache_data(show_spinner=False)
def event_study_series(ticker: str, date_str: str, market="^GSPC", est_days=200, win_days=5) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.DatetimeIndex]]:
    try:
        ev_dt = NYC.localize(dt.datetime.strptime(date_str, "%Y-%m-%d"))
        est_end = ev_dt - dt.timedelta(days=win_days + 1)
        est_start = est_end - dt.timedelta(days=est_days)
        win_start = ev_dt - dt.timedelta(days=win_days)
        win_end = ev_dt + dt.timedelta(days=win_days)

        f = yf.Ticker(ticker).history(start=est_start, end=win_end)
        m = yf.Ticker(market).history(start=est_start, end=win_end)
        if f.empty or m.empty or "Close" not in f or "Close" not in m:
            return None, None, None

        fr = f["Close"].pct_change().dropna()
        mr = m["Close"].pct_change().dropna()
        df = pd.concat([fr, mr], axis=1).dropna()
        df.columns = ["firm", "mkt"]

        est = df.loc[(df.index >= est_start) & (df.index <= est_end)]
        win = df.loc[(df.index >= win_start) & (df.index <= win_end)]
        if est.empty or win.empty: return None, None, None

        X = sm.add_constant(est["mkt"])
        model = sm.OLS(est["firm"], X).fit()
        alpha, beta = float(model.params.get("const", 0.0)), float(model.params.get("mkt", 0.0))

        norm = alpha + beta * win["mkt"]
        abnormal = win["firm"] - norm
        car = abnormal.cumsum()
        return abnormal, car, win.index
    except:
        return None, None, None

# Updated events (June + Sept included)
EVENTS: Dict[str, str] = {
    "2024-06-03": "Lockheed Martin: major F-35 contract (June 2024)",
    "2024-09-03": "Lockheed Martin: strategic NASA partnership (Sept 2024)",
    "2024-10-22": "Lockheed Martin Q3 2024 Earnings (beat; raised guidance)",
    "2024-10-23": "Boeing Q3 2024 Earnings (miss; production delays)",
    "2024-10-24": "Raytheon & Northrop Q3 2024 Results (mixed)",
    "2024-07-25": "BAE Systems HY2024 (backlog & guidance up)"
}

# =============================
# Sidebar
# =============================
st.sidebar.title("Controls")
main_ticker = st.sidebar.text_input("Main Ticker", "LMT").upper().strip()
comps_raw = st.sidebar.text_input("Competitors (comma-separated)", "BA, RTX, NOC, GD, BAESY")
competitors = [t.strip().upper() for t in comps_raw.split(",") if t.strip()]
tickers = [main_ticker] + competitors

section = st.sidebar.selectbox(
    "Section",
    ["Financial Overview", "Key Ratios & WACC", "Event Studies", "Competitor Comparison"]
)

st.sidebar.markdown("---")
st.sidebar.caption("WACC assumptions (you can override):")
rf = st.sidebar.number_input("Risk-free rate (decimal)", value=0.04, step=0.005, format="%.3f")
mrp = st.sidebar.number_input("Market risk premium (decimal)", value=0.05, step=0.005, format="%.3f")

with st.spinner("Loading market & fundamentals..."):
    data = fetch_data(tickers)
    fx_map = {t: get_fx_rate_to_usd((data[t]["info"] or {}).get("currency", "USD")) for t in tickers}

# =============================
# Financial Overview (enhanced)
# =============================
if section == "Financial Overview":
    st.title("Financial Overview")
    st.write("Price trend, 1Y cumulative returns, quick firm stats & profitability snapshot. All amounts converted to USD where possible.")

    # --- 1) Price trend (2Y)
    fig_price = go.Figure()
    for t in tickers:
        h = data[t]["hist"]
        if h is None or h.empty or "Close" not in h: continue
        series = h["Close"].dropna()
        fig_price.add_trace(go.Scatter(x=series.index, y=series.values, name=t, mode="lines"))
    fig_price.update_layout(title="Price (2Y)", template="plotly_white")
    st.plotly_chart(fig_price, use_container_width=True, key="price_2y")

    # --- 2) Cumulative returns (1Y)
    fig_cum = go.Figure()
    for t in tickers:
        h = data[t]["hist"]
        if h is None or h.empty or "Close" not in h: continue
        pr = h["Close"].dropna()
        pr = pr.loc[pr.index >= (pr.index.max() - pd.Timedelta(days=365))]
        if pr.empty: continue
        rets = pr.pct_change().fillna(0.0)
        cum = (1 + rets).cumprod() - 1
        fig_cum.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name=t))
    fig_cum.update_layout(title="Cumulative Returns (Last 1Y)", template="plotly_white", yaxis_tickformat=".0%")
    st.plotly_chart(fig_cum, use_container_width=True, key="cum_1y")

    # --- 3) Quick stats table: last price, 1m/3m/6m/1y returns, market cap, beta (regressed)
    rows = []
    for t in tickers:
        info = data[t]["info"] or {}
        h = data[t]["hist"]
        if h is None or h.empty or "Close" not in h: continue
        px = h["Close"].dropna()
        if px.empty: continue
        last = px.iloc[-1]
        def period_return(days: int):
            ref_idx = px.index.max() - pd.Timedelta(days=days)
            sub = px.loc[px.index >= ref_idx]
            if sub.empty: return np.nan
            return sub.pct_change().add(1).prod() - 1
        r1m = period_return(30)
        r3m = period_return(90)
        r6m = period_return(180)
        r1y = period_return(365)
        mcap = info.get("marketCap")
        beta = estimate_beta(t)
        rows.append({
            "Ticker": t,
            "Last Price": last,
            "1M": r1m, "3M": r3m, "6M": r6m, "1Y": r1y,
            "Market Cap (USD)": (mcap * fx_map[t]) if mcap is not None else np.nan,
            "Beta (2Y vs S&P)": beta
        })
    if rows:
        qdf = pd.DataFrame(rows).set_index("Ticker")
        fmt = qdf.copy()
        fmt["Last Price"] = fmt["Last Price"].apply(lambda v: "" if pd.isna(v) else f"{v:,.2f}")
        for c in ["1M","3M","6M","1Y"]:
            fmt[c] = fmt[c].apply(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        fmt["Market Cap (USD)"] = fmt["Market Cap (USD)"].apply(human_format_number)
        fmt["Beta (2Y vs S&P)"] = fmt["Beta (2Y vs S&P)"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        st.subheader("Quick Stats")
        st.dataframe(fmt, use_container_width=True)

    # --- 4) Profitability snapshot (latest reported margins)
    st.subheader("Profitability Snapshot (Latest reported)")
    snap_rows = []
    for t in tickers:
        fin = data[t]["financials"]
        if fin is None or fin.empty: continue
        rev = find_label(fin, ["Total Revenue", "Revenue"])
        gp  = find_label(fin, ["Gross Profit"])
        op  = find_label(fin, ["Operating Income", "Operating Income or Loss"])
        ni  = find_label(fin, ["Net Income", "NetIncome"])
        latest_col = fin.columns[0] if len(fin.columns) > 0 else None
        if not latest_col: continue
        row = {"Ticker": t}
        try:
            if rev and gp:
                row["Gross Margin"] = float(fin.loc[gp, latest_col]) / float(fin.loc[rev, latest_col])
        except: row["Gross Margin"] = np.nan
        try:
            if rev and op:
                row["Operating Margin"] = float(fin.loc[op, latest_col]) / float(fin.loc[rev, latest_col])
        except: row["Operating Margin"] = np.nan
        try:
            if rev and ni:
                row["Net Margin"] = float(fin.loc[ni, latest_col]) / float(fin.loc[rev, latest_col])
        except: row["Net Margin"] = np.nan
        snap_rows.append(row)
    if snap_rows:
        sdf = pd.DataFrame(snap_rows).set_index("Ticker")
        sdf_fmt = sdf.applymap(lambda v: "" if pd.isna(v) else f"{v:.2%}")
        st.dataframe(sdf_fmt, use_container_width=True)

# =============================
# Key Ratios & WACC (explained)
# =============================
elif section == "Key Ratios & WACC":
    st.title("Key Ratios & WACC")

    # Ratios (averages across reported periods)
    ratio_dict: Dict[str, dict] = {}
    for t in tickers:
        fin, bs = data[t]["financials"], data[t]["balance"]
        if fin is None or fin.empty: continue
        rev = find_label(fin, ["Total Revenue", "Revenue"])
        gp  = find_label(fin, ["Gross Profit"])
        op  = find_label(fin, ["Operating Income", "Operating Income or Loss"])
        ni  = find_label(fin, ["Net Income", "NetIncome"])
        eq  = None if bs is None or bs.empty else find_label(bs, ["Total Stockholder Equity", "Total Equity", "Stockholders Equity"])
        debt= None if bs is None or bs.empty else find_label(bs, ["Total Debt", "Total Liab", "Long Term Debt"])
        row = {}
        try:
            if rev and gp: row["Gross Margin"] = (pd.to_numeric(fin.loc[gp], errors="coerce") /
                                                  pd.to_numeric(fin.loc[rev], errors="coerce")).mean()
        except: pass
        try:
            if rev and op: row["Operating Margin"] = (pd.to_numeric(fin.loc[op], errors="coerce") /
                                                      pd.to_numeric(fin.loc[rev], errors="coerce")).mean()
        except: pass
        try:
            if rev and ni: row["Net Margin"] = (pd.to_numeric(fin.loc[ni], errors="coerce") /
                                                pd.to_numeric(fin.loc[rev], errors="coerce")).mean()
        except: pass
        try:
            if eq and ni: row["ROE"] = (pd.to_numeric(fin.loc[ni], errors="coerce") /
                                        pd.to_numeric(bs.loc[eq], errors="coerce")).mean()
        except: pass
        try:
            if eq and debt: row["D/E"] = (pd.to_numeric(bs.loc[debt], errors="coerce") /
                                          pd.to_numeric(bs.loc[eq], errors="coerce")).mean()
        except: pass
        if row: ratio_dict[t] = row

    if ratio_dict:
        ratio_df = pd.DataFrame(ratio_dict).T
        st.subheader("Key Ratios (avg across reported periods)")
        st.write(styled_ratio_table(ratio_df), unsafe_allow_html=True)

    # WACC panel with explanation
    st.subheader("WACC Estimation & Components")
    wacc_rows: Dict[str, Dict[str, Optional[float]]] = {}
    for t in tickers:
        wacc_rows[t] = compute_wacc(t, data, rf=rf, mrp=mrp)
    wacc_df = pd.DataFrame(wacc_rows).T

    # Pretty table
    pretty = pd.DataFrame(index=wacc_df.index)
    pretty["Beta"] = wacc_df["beta"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    pretty["Cost of Equity (Ke)"] = wacc_df["ke"].apply(lambda v: "" if pd.isna(v) else f"{v:.2%}")
    pretty["Cost of Debt (Kd)"] = wacc_df["kd"].apply(lambda v: "" if pd.isna(v) else f"{v:.2%}")
    pretty["Tax Rate"] = wacc_df["tax"].apply(lambda v: "" if pd.isna(v) else f"{v:.2%}")
    pretty["Equity (USD)"] = wacc_df["E"].apply(human_format_number)
    pretty["Debt (USD)"] = wacc_df["D"].apply(human_format_number)
    pretty["WACC"] = wacc_df["wacc"].apply(lambda v: "" if pd.isna(v) else f"{v:.2%}")
    st.dataframe(pretty, use_container_width=True)

    # Bar chart of WACC
    fig_wacc = go.Figure()
    for t in tickers:
        val = wacc_rows[t]["wacc"]
        if val is None or pd.isna(val): continue
        fig_wacc.add_trace(go.Bar(x=[t], y=[val], name=t))
    fig_wacc.update_layout(title="WACC by Firm", template="plotly_white", yaxis_tickformat=".1%")
    st.plotly_chart(fig_wacc, use_container_width=True, key="wacc_bar")

    with st.expander("How WACC is calculated (talking points)"):
        st.markdown(
            """
**WACC = w_E · K_e + w_D · K_d · (1 − Tax)**

- **Cost of equity (K_e)** via CAPM: `K_e = R_f + β × MRP`  
  - `R_f` = risk-free rate (user input)  
  - `β` = regression of stock vs S&P 500 (last ~2y daily returns)  
  - `MRP` = market risk premium (user input)
- **Cost of debt (K_d)** ≈ Interest Expense / Total Debt (from statements)
- **Capital weights**:  
  - `w_E = Equity / (Equity + Debt)` where Equity ≈ market cap  
  - `w_D = Debt / (Equity + Debt)`
- **Tax shield**: after-tax cost of debt uses `· (1 − Tax)`, with tax inferred from tax expense / pretax income (fallback 21%).

**Interpretation:** projects must earn returns above WACC to create value. Lower WACC → easier to clear the hurdle; higher WACC → tougher hurdle.
"""
        )

# =============================
# Event Studies (combined charts)
# =============================
elif section == "Event Studies":
    st.title("Event Studies — Abnormal Return & CAR")
    ev_key = st.selectbox("Choose event", list(EVENTS.keys()), format_func=lambda k: EVENTS[k])
    st.caption("Market model using S&P500 (^GSPC), estimation window ≈200d, event window ±5d.")

    # Per-firm panels
    for t in tickers:
        abn, car, idx = event_study_series(t, ev_key)
        st.subheader(f"{t} — {EVENTS[ev_key]}")
        if abn is None:
            st.warning(f"Insufficient data for {t} around {ev_key}.")
            st.markdown("---")
            continue
        # quick read
        try:
            net = float(car.iloc[-1] - car.iloc[0])
            direction = "↑ positive" if net > 0 else ("↓ negative" if net < 0 else "flat")
            st.caption(f"Quick read: CAR over window is **{direction}** ({net:.2%}).")
        except: pass

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx, y=abn.values, mode="lines+markers", name="Abnormal Return",
                                 hovertemplate="%{x}<br>%{y:.2%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=idx, y=car.values, mode="lines+markers", name="CAR",
                                 hovertemplate="%{x}<br>%{y:.2%}<extra></extra>"))
        fig.update_layout(template="plotly_white", title=f"{t} — Abnormal & CAR", yaxis_tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True, key=f"{t}_{ev_key}_panel")
        st.markdown("---")

    # Cross-company combined charts (all companies together)
    st.header("Cross-Company Comparison (Selected Event)")
    comp_ab, comp_car, ix = pd.DataFrame(), pd.DataFrame(), None
    for t in tickers:
        abn, car, idx = event_study_series(t, ev_key)
        if abn is not None:
            comp_ab[t] = abn
            comp_car[t] = car
            ix = idx

    if comp_ab.empty:
        st.info("No comparable cross-company data available for this event.")
    else:
        # Combined Abnormal
        fig_ab = go.Figure()
        for c in comp_ab.columns:
            fig_ab.add_trace(go.Scatter(x=ix, y=comp_ab[c], mode="lines+markers", name=c,
                                        hovertemplate="%{x}<br>%{y:.2%}<extra></extra>"))
        fig_ab.update_layout(title="Abnormal Returns (daily) — All Firms", template="plotly_white",
                             yaxis_tickformat=".2%", hovermode="x unified")
        st.plotly_chart(fig_ab, use_container_width=True, key=f"ab_comp_{ev_key}")

        # Combined CAR
        fig_car = go.Figure()
        for c in comp_car.columns:
            fig_car.add_trace(go.Scatter(x=ix, y=comp_car[c], mode="lines+markers", name=c,
                                         hovertemplate="%{x}<br>%{y:.2%}<extra></extra>"))
        fig_car.update_layout(title="Cumulative Abnormal Returns (CAR) — All Firms", template="plotly_white",
                              yaxis_tickformat=".2%", hovermode="x unified")
        st.plotly_chart(fig_car, use_container_width=True, key=f"car_comp_{ev_key}")

# =============================
# Competitor Comparison
# =============================
elif section == "Competitor Comparison":
    st.title("Competitor Comparison (USD)")
    metric = st.selectbox("Metric", ["Total Revenue", "Gross Profit", "Net Income"])
    cand = {
        "Total Revenue": ["Total Revenue", "Revenue", "TotalRevenue"],
        "Gross Profit": ["Gross Profit", "GrossProfit"],
        "Net Income": ["Net Income", "NetIncome"]
    }[metric]
    frames: Dict[str, pd.Series] = {}
    notes = {}
    for t in tickers:
        fin = data[t]["financials"]
        if fin is None or fin.empty: continue
        lbl = find_label(fin, cand)
        if not lbl: continue
        fx = fx_map[t]
        s = to_usd(fin.loc[lbl], fx)
        frames[t] = s
        cur = (data[t]["info"] or {}).get("currency", "USD")
        if cur and cur.upper() != "USD":
            notes[t] = f"Converted from {cur} @ {fx:.4f}"

    if not frames:
        st.info("No data available for selected metric.")
    else:
        df = pd.DataFrame(frames)
        st.subheader("Formatted Table")
        st.write(styled_money_table(df), unsafe_allow_html=True)

        fig = go.Figure()
        for c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines+markers", name=c,
                                     hovertemplate="%{x}<br>%{y:,.0f} USD<extra></extra>"))
        fig.update_layout(title=f"{metric} Over Time (USD)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True, key=f"{metric}_comp")

        if notes:
            st.caption("Currency conversion notes:")
            for k, v in notes.items(): st.write(f"- {k}: {v}")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built for Corporate Finance module — enhanced overview, full WACC, and cross-company event comparisons.")
