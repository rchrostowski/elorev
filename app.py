# app.py — Corporate Finance Dashboard (Event Study–forward + WACC + Cum Returns)
# rchrostowski / elorev

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import datetime as dt
import pytz

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
    return d.style.format("{:.2%}").background_gradient(cmap="RdYlGn").to_html()

def find_label(df: pd.DataFrame, candidates: list) -> str | None:
    if df is None or df.empty: return None
    for c in candidates:
        # exact first
        for i in df.index:
            if str(i) == c: return i
        # then substring
        for i in df.index:
            if c.lower() in str(i).lower(): return i
    return None

# =============================
# Caching: FX & firm data
# =============================
@st.cache_data(show_spinner=False)
def get_fx_rate_to_usd(currency: str | None) -> float:
    if not currency: return 1.0
    cur = currency.upper()
    if cur == "USD": return 1.0
    # Try CURUSD=X
    try:
        df = yf.Ticker(f"{cur}USD=X").history(period="7d")
        if not df.empty: return float(df["Close"].iloc[-1])
    except: pass
    # Try USDcur=X (invert)
    try:
        df = yf.Ticker(f"USD{cur}=X").history(period="7d")
        if not df.empty and df["Close"].iloc[-1] != 0:
            return 1.0 / float(df["Close"].iloc[-1])
    except: pass
    return 1.0

@st.cache_data(show_spinner=False)
def fetch_firm_blobs(tickers: list[str]) -> dict:
    out = {}
    for t in tickers:
        tkr = yf.Ticker(t)
        try: info = tkr.info or {}
        except: info = {}
        fin = getattr(tkr, "financials", pd.DataFrame())
        bs  = getattr(tkr, "balance_sheet", pd.DataFrame())
        cf  = getattr(tkr, "cashflow", pd.DataFrame())
        px  = tkr.history(period="2y")  # for beta/cum returns
        out[t] = {
            "info": info,
            "financials": fin,
            "balance_sheet": bs,
            "cashflow": cf,
            "hist": px
        }
    return out

# USD convert a row/series
def to_usd(series: pd.Series, fx: float) -> pd.Series:
    try: return pd.to_numeric(series, errors="coerce") * fx
    except: return series

# =============================
# Beta / WACC (with fallbacks)
# =============================
@st.cache_data(show_spinner=False)
def estimate_equity_beta(ticker: str, market="^GSPC", lookback_years=2) -> float | None:
    try:
        end = dt.datetime.now()
        start = end - dt.timedelta(days=int(365*lookback_years))
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

def safe_ratio(numer: float | None, denom: float | None) -> float | None:
    try:
        if numer is None or denom in (None, 0, np.nan): return None
        return float(numer)/float(denom)
    except:
        return None

def pull_first(series_like) -> float | None:
    try:
        if series_like is None: return None
        if isinstance(series_like, (pd.Series, pd.DataFrame)):
            if isinstance(series_like, pd.Series):
                return float(series_like.dropna().iloc[0])
            return float(series_like.stack().dropna().iloc[0])
        return float(series_like)
    except:
        return None

def infer_tax_rate(fin_df: pd.DataFrame) -> float | None:
    try:
        tax_lbl = find_label(fin_df, ["Income Tax Expense", "Tax Provision"])
        pbt_lbl = find_label(fin_df, ["Pretax Income", "Earnings Before Tax", "Income Before Tax"])
        if tax_lbl and pbt_lbl:
            tax = pd.to_numeric(fin_df.loc[tax_lbl], errors="coerce")
            pbt = pd.to_numeric(fin_df.loc[pbt_lbl], errors="coerce").replace(0, np.nan)
            val = (tax / pbt).dropna()
            if not val.empty:
                # clamp to sensible [0, 0.35]
                x = float(val.mean())
                return max(0.0, min(0.35, x))
    except:
        pass
    return None

def infer_cost_of_debt(fin_df: pd.DataFrame, bs_df: pd.DataFrame) -> float | None:
    try:
        int_lbl = find_label(fin_df, ["Interest Expense"])
        debt_lbl = find_label(bs_df, ["Total Debt", "Total Liab", "Long Term Debt"])
        if not int_lbl or not debt_lbl: return None
        interest = pd.to_numeric(fin_df.loc[int_lbl], errors="coerce").abs()  # cost
        debt = pd.to_numeric(bs_df.loc[debt_lbl], errors="coerce").abs()
        # align by column date where available
        debt = debt.reindex(interest.index, fill_value=np.nan)
        ratio = (interest / debt).dropna()
        if not ratio.empty:
            # clamp to sensible [0, 0.15]
            x = float(ratio.mean())
            return max(0.0, min(0.15, x))
    except:
        pass
    return None

def compute_wacc(
    ticker: str,
    blobs: dict,
    rf: float = 0.04,      # 4% default (overrideable in UI)
    mrp: float = 0.05      # 5% equity risk premium default (overrideable in UI)
) -> dict:
    d = blobs[ticker]
    fin = d["financials"]
    bs  = d["balance_sheet"]
    info = d["info"] or {}

    beta = estimate_equity_beta(ticker)
    ke = None if beta is None else rf + beta * mrp

    kd = infer_cost_of_debt(fin, bs)
    tax = infer_tax_rate(fin)
    if tax is None: tax = 0.21  # fallback

    # Capital structure: use market cap from info; debt from BS
    mkt_cap = info.get("marketCap", None)
    debt_lbl = find_label(bs, ["Total Debt", "Total Liab", "Long Term Debt"])
    total_debt = None
    if debt_lbl is not None:
        try: total_debt = float(pd.to_numeric(bs.loc[debt_lbl], errors="coerce").dropna().iloc[0])
        except: total_debt = None

    if mkt_cap is None:
        # try price * shares outstanding
        so = info.get("sharesOutstanding")
        price = None
        try:
            price = d["hist"]["Close"].iloc[-1]
        except:
            pass
        if so and price: mkt_cap = so * price

    if ke is None or kd is None or mkt_cap is None or total_debt is None or (mkt_cap+total_debt) == 0:
        return {
            "beta": beta, "ke": ke, "kd": kd, "tax": tax,
            "E": mkt_cap, "D": total_debt, "wacc": None
        }

    V = mkt_cap + total_debt
    we, wd = mkt_cap / V, total_debt / V
    wacc = we * ke + wd * kd * (1 - tax)

    return {
        "beta": beta, "ke": ke, "kd": kd, "tax": tax,
        "E": mkt_cap, "D": total_debt, "wacc": wacc
    }

# =============================
# Event Study (new data forward)
# =============================
NYC = pytz.timezone("America/New_York")

@st.cache_data(show_spinner=False)
def event_study_series(ticker: str, event_date: str, market="^GSPC", est_days=200, win_days=5):
    """Return abnormal, CAR, index for a given ticker/event_date."""
    try:
        ev_dt = NYC.localize(dt.datetime.strptime(event_date, "%Y-%m-%d"))
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

        est_mask = (df.index >= est_start) & (df.index <= est_end)
        win_mask = (df.index >= win_start) & (df.index <= win_end)
        est, win = df.loc[est_mask], df.loc[win_mask]
        if est.empty or win.empty: return None, None, None

        X = sm.add_constant(est["mkt"])
        model = sm.OLS(est["firm"], X).fit()
        alpha, beta = model.params.get("const", 0.0), model.params.get("mkt", 0.0)

        normal = alpha + beta * win["mkt"]
        abnormal = win["firm"] - normal
        car = abnormal.cumsum()
        return abnormal, car, win.index
    except:
        return None, None, None

# Hardcoded, presentation-ready events (you can add more easily)
EVENTS = {
    "2024-10-22": "Lockheed Martin Q3 2024 Earnings (beat; raised guidance)",
    "2024-10-23": "Boeing Q3 2024 Earnings (miss; production delays)",
    "2024-10-24": "Raytheon & Northrop Q3 2024 Results (mixed)",
    "2024-07-25": "BAE Systems HY2024 (backlog & guidance up)"
}

# =============================
# Sidebar controls
# =============================
st.sidebar.title("Controls")
main_ticker = st.sidebar.text_input("Main Ticker", "LMT").upper().strip()
competitors_raw = st.sidebar.text_input("Competitors (comma-separated)", "BA, RTX, NOC, GD, BAESY")
competitors = [t.strip().upper() for t in competitors_raw.split(",") if t.strip()]
tickers = [main_ticker] + competitors

section = st.sidebar.selectbox(
    "Section",
    ["Financial Overview", "Key Ratios & WACC", "Event Studies", "Competitor Comparison"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: override WACC assumptions here ↓")
rf = st.sidebar.number_input("Risk-free rate (decimal)", value=0.04, step=0.005, format="%.3f")
mrp = st.sidebar.number_input("Market risk premium (decimal)", value=0.05, step=0.005, format="%.3f")

# =============================
# Load blobs & FX
# =============================
with st.spinner("Fetching data..."):
    blobs = fetch_firm_blobs(tickers)
    fx_map = {}
    for t in tickers:
        cur = (blobs[t]["info"] or {}).get("currency", "USD")
        fx_map[t] = {"currency": cur, "fx": get_fx_rate_to_usd(cur)}

# =============================
# Section: Financial Overview
# =============================
if section == "Financial Overview":
    st.title("Financial Overview")
    st.write("Cumulative stock returns, high-level market cap, and quick stats. All amounts converted to USD where possible.")

    # Cumulative returns (1Y)
    fig = go.Figure()
    for t in tickers:
        h = blobs[t]["hist"]
        if h is None or h.empty or "Close" not in h: continue
        pr = h["Close"].dropna()
        # last 1y
        pr = pr.loc[pr.index >= (pr.index.max() - pd.Timedelta(days=365))]
        if pr.empty: continue
        rets = pr.pct_change().fillna(0.0)
        cum = (1 + rets).cumprod() - 1
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name=t))
    fig.update_layout(title="Cumulative Returns (Last 1Y)", template="plotly_white", yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True, key="cum_returns")

    # Market cap table
    caps = {}
    for t in tickers:
        info = blobs[t]["info"] or {}
        mc = info.get("marketCap")
        if mc is None:
            so = info.get("sharesOutstanding"); px = None
            try: px = blobs[t]["hist"]["Close"].iloc[-1]
            except: pass
            if so and px: mc = so * px
        if mc is not None:
            caps[t] = mc * fx_map[t]["fx"]
    if caps:
        cap_df = pd.DataFrame.from_dict(caps, orient="index", columns=["Market Cap (USD)"])
        st.subheader("Market Cap (USD)")
        st.write(styled_money_table(cap_df), unsafe_allow_html=True)

# =============================
# Section: Key Ratios & WACC
# =============================
elif section == "Key Ratios & WACC":
    st.title("Key Ratios & WACC")
    st.write("Gross/Operating/Net margins; ROE & D/E; plus WACC with user-overridable RF/MRP.")

    # Ratios snapshot (simple averages across reported periods)
    ratio_rows = {}
    for t in tickers:
        fin = blobs[t]["financials"]; bs = blobs[t]["balance_sheet"]
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
        if row: ratio_rows[t] = row

    if ratio_rows:
        ratio_df = pd.DataFrame(ratio_rows).T
        st.subheader("Key Ratios (avg across reported periods)")
        st.write(styled_ratio_table(ratio_df), unsafe_allow_html=True)

    # WACC panel — for main ticker plus show peers inline
    st.subheader("WACC Estimation")
    colA, colB = st.columns([1.2, 1])
    with colA:
        wacc_rows = {}
        for t in tickers:
            res = compute_wacc(t, blobs, rf=rf, mrp=mrp)
            wacc_rows[t] = {
                "Beta": res["beta"],
                "Cost of Equity (Ke)": res["ke"],
                "Cost of Debt (Kd)": res["kd"],
                "Tax Rate": res["tax"],
                "Equity (USD)": res["E"],
                "Debt (USD)": res["D"],
                "WACC": res["wacc"]
            }
        wacc_df = pd.DataFrame(wacc_rows).T
        fmt = wacc_df.copy()
        for c in ["Equity (USD)", "Debt (USD)"]:
            fmt[c] = fmt[c].apply(human_format_number)
        for c in ["Cost of Equity (Ke)", "Cost of Debt (Kd)", "Tax Rate", "WACC"]:
            fmt[c] = fmt[c].apply(lambda v: "" if pd.isna(v) else f"{v:.2%}")
        fmt["Beta"] = fmt["Beta"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        st.dataframe(fmt, use_container_width=True)
    with colB:
        # simple bar of WACC
        fig = go.Figure()
        for t in tickers:
            val = wacc_rows[t]["wacc"]
            if pd.isna(val) or val is None: continue
            fig.add_trace(go.Bar(x=[t], y=[val], name=t))
        fig.update_layout(title="WACC by Firm", template="plotly_white", yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True, key="wacc_bar")

# =============================
# Section: Event Studies (dropdown-driven)
# =============================
elif section == "Event Studies":
    st.title("Event Studies — New Data as Main Talking Point")
    ev_key = st.selectbox("Choose event", list(EVENTS.keys()), format_func=lambda k: EVENTS[k])
    st.caption("We model abnormal returns (market model) and accumulate them to CAR over a ±5 day window.")

    # Per-firm panels
    for t in tickers:
        abn, car, idx = event_study_series(t, ev_key)
        st.subheader(f"{t} — {EVENTS[ev_key]}")
        if abn is None:
            st.warning(f"Insufficient data for {t} around {ev_key}.")
            continue
        # quick read
        try:
            net = float(car.iloc[-1] - car.iloc[0])
            direction = "↑ positive" if net > 0 else ("↓ negative" if net < 0 else "flat")
            st.caption(f"Quick read: CAR over window is **{direction}** ({net:.2%}).")
        except:
            pass

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx, y=abn.values, mode="lines+markers", name="Abnormal Return",
                                 hovertemplate="%{x}<br>%{y:.2%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=idx, y=car.values, mode="lines+markers", name="CAR",
                                 hovertemplate="%{x}<br>%{y:.2%}<extra></extra>"))
        fig.update_layout(template="plotly_white", title=f"{t} — Abnormal & CAR", yaxis_tickformat=".2%")
        st.plotly_chart(fig, use_container_width=True, key=f"{t}_{ev_key}_panel")
        st.markdown("---")

    # Cross-company comparison
    st.header("Cross-Company Comparison for Selected Event")
    comp_ab, comp_car = pd.DataFrame(), pd.DataFrame()
    ix = None
    for t in tickers:
        abn, car, idx = event_study_series(t, ev_key)
        if abn is not None:
            comp_ab[t] = abn
            comp_car[t] = car
            ix = idx
    if comp_ab.empty:
        st.info("No comparable data across firms for this event.")
    else:
        # Abnormal comparison (lines)
        fig1 = go.Figure()
        for c in comp_ab.columns:
            fig1.add_trace(go.Scatter(x=ix, y=comp_ab[c], mode="lines+markers", name=c,
                                      hovertemplate="%{x}<br>%{y:.2%}<extra></extra>"))
        fig1.update_layout(title="Abnormal Returns (daily)", template="plotly_white", yaxis_tickformat=".2%", hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True, key=f"ab_comp_{ev_key}")

        # CAR comparison (lines)
        fig2 = go.Figure()
        for c in comp_car.columns:
            fig2.add_trace(go.Scatter(x=ix, y=comp_car[c], mode="lines+markers", name=c,
                                      hovertemplate="%{x}<br>%{y:.2%}<extra></extra>"))
        fig2.update_layout(title="Cumulative Abnormal Returns (CAR)", template="plotly_white", yaxis_tickformat=".2%", hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True, key=f"car_comp_{ev_key}")

# =============================
# Section: Competitor Comparison
# =============================
elif section == "Competitor Comparison":
    st.title("Competitor Comparison")
    st.write("Compare core P&L lines across firms (USD). Use this to back up your event study narrative.")

    metric = st.selectbox("Metric", ["Total Revenue", "Gross Profit", "Net Income"])
    cand = {
        "Total Revenue": ["Total Revenue", "Revenue", "TotalRevenue"],
        "Gross Profit": ["Gross Profit", "GrossProfit"],
        "Net Income": ["Net Income", "NetIncome"]
    }[metric]

    frames = {}
    notes = {}
    for t in tickers:
        fin = blobs[t]["financials"]
        if fin is None or fin.empty: continue
        lbl = find_label(fin, cand)
        if not lbl: continue
        fx = fx_map[t]["fx"]
        s = to_usd(fin.loc[lbl], fx)
        frames[t] = s
        cur = fx_map[t]["currency"]
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
st.sidebar.caption("Built for class: Event Study first, with WACC & updated returns. Add more events in EVENTS{}.")

