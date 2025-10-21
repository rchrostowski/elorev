# app.py — Corporate Finance Dashboard (Final Build)
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
    return d.style.format("{:.2%}").to_html()

def find_label(df: pd.DataFrame, candidates: list) -> str | None:
    if df is None or df.empty: return None
    for c in candidates:
        for i in df.index:
            if c.lower() in str(i).lower(): return i
    return None

# =============================
# Cached Data Loaders
# =============================
@st.cache_data(show_spinner=False)
def get_fx_rate_to_usd(currency: str | None) -> float:
    if not currency: return 1.0
    cur = currency.upper()
    if cur == "USD": return 1.0
    try:
        df = yf.Ticker(f"{cur}USD=X").history(period="5d")
        if not df.empty: return float(df["Close"].iloc[-1])
    except: pass
    try:
        df = yf.Ticker(f"USD{cur}=X").history(period="5d")
        if not df.empty: return 1 / float(df["Close"].iloc[-1])
    except: pass
    return 1.0

@st.cache_data(show_spinner=False)
def fetch_data(tickers):
    out = {}
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

def to_usd(series, fx):
    try: return pd.to_numeric(series, errors="coerce") * fx
    except: return series

# =============================
# WACC calculation
# =============================
@st.cache_data(show_spinner=False)
def estimate_beta(ticker, market="^GSPC", years=2):
    try:
        end = dt.datetime.now()
        start = end - dt.timedelta(days=int(365*years))
        f = yf.Ticker(ticker).history(start=start, end=end)["Close"].pct_change().dropna()
        m = yf.Ticker(market).history(start=start, end=end)["Close"].pct_change().dropna()
        df = pd.concat([f, m], axis=1).dropna()
        df.columns = ["firm", "mkt"]
        if df.empty: return None
        X = sm.add_constant(df["mkt"])
        return sm.OLS(df["firm"], X).fit().params.get("mkt", None)
    except:
        return None

def compute_wacc(ticker, data, rf=0.04, mrp=0.05):
    fin = data[ticker]["financials"]
    bs = data[ticker]["balance"]
    info = data[ticker]["info"] or {}

    beta = estimate_beta(ticker)
    ke = None if beta is None else rf + beta * mrp
    kd = None; tax = 0.21

    try:
        int_lbl = find_label(fin, ["Interest Expense"])
        debt_lbl = find_label(bs, ["Total Debt", "Long Term Debt"])
        if int_lbl and debt_lbl:
            int_exp = abs(pd.to_numeric(fin.loc[int_lbl], errors="coerce"))
            debt = abs(pd.to_numeric(bs.loc[debt_lbl], errors="coerce"))
            kd = (int_exp / debt).mean()
    except: pass

    mcap = info.get("marketCap")
    debt_val = None
    if find_label(bs, ["Total Debt"]):
        debt_val = float(pd.to_numeric(bs.loc[find_label(bs, ["Total Debt"])], errors="coerce").dropna().iloc[0])

    if not mcap or not debt_val or not ke or not kd:
        return {"beta": beta, "ke": ke, "kd": kd, "tax": tax, "E": mcap, "D": debt_val, "wacc": None}

    V = mcap + debt_val
    we, wd = mcap/V, debt_val/V
    wacc = we*ke + wd*kd*(1 - tax)
    return {"beta": beta, "ke": ke, "kd": kd, "tax": tax, "E": mcap, "D": debt_val, "wacc": wacc}

# =============================
# Event Study
# =============================
NYC = pytz.timezone("America/New_York")

@st.cache_data(show_spinner=False)
def event_study(ticker, date_str, est_days=200, win_days=5):
    try:
        ev_dt = NYC.localize(dt.datetime.strptime(date_str, "%Y-%m-%d"))
        est_start = ev_dt - dt.timedelta(days=est_days+win_days)
        est_end = ev_dt - dt.timedelta(days=win_days)
        win_start = ev_dt - dt.timedelta(days=win_days)
        win_end = ev_dt + dt.timedelta(days=win_days)
        f = yf.Ticker(ticker).history(start=est_start, end=win_end)
        m = yf.Ticker("^GSPC").history(start=est_start, end=win_end)
        if f.empty or m.empty: return None, None, None
        fr, mr = f["Close"].pct_change().dropna(), m["Close"].pct_change().dropna()
        df = pd.concat([fr, mr], axis=1).dropna()
        df.columns = ["firm", "mkt"]
        X = sm.add_constant(df.loc[:est_end, "mkt"])
        model = sm.OLS(df.loc[:est_end, "firm"], X).fit()
        abnormal = df.loc[win_start:win_end, "firm"] - (model.params[0] + model.params[1]*df.loc[win_start:win_end, "mkt"])
        car = abnormal.cumsum()
        return abnormal, car, abnormal.index
    except:
        return None, None, None

# Updated events (June + Sept included)
EVENTS = {
    "2024-06-03": "Lockheed Martin awarded major F-35 contract (June 2024)",
    "2024-09-03": "Lockheed Martin announces partnership with NASA for lunar systems (Sept 2024)",
    "2024-10-22": "Lockheed Martin Q3 2024 Earnings (beat; raised guidance)",
    "2024-10-23": "Boeing Q3 2024 Earnings (miss; production delays)",
    "2024-10-24": "Raytheon & Northrop Q3 2024 Results (mixed performance)",
    "2024-07-25": "BAE Systems HY2024 (increased backlog & orders)"
}

# =============================
# Sidebar
# =============================
st.sidebar.title("Dashboard Controls")
main = st.sidebar.text_input("Main Ticker", "LMT").upper()
competitors = st.sidebar.text_input("Competitors (comma separated)", "BA, RTX, NOC, GD, BAESY")
comps = [t.strip().upper() for t in competitors.split(",") if t.strip()]
tickers = [main] + comps

section = st.sidebar.selectbox("Section", ["Financial Overview", "Key Ratios & WACC", "Event Studies", "Competitor Comparison"])
rf = st.sidebar.number_input("Risk-Free Rate", 0.04, 0.10, 0.04, step=0.005)
mrp = st.sidebar.number_input("Market Risk Premium", 0.03, 0.10, 0.05, step=0.005)

with st.spinner("Loading data..."):
    data = fetch_data(tickers)
    fx = {t: get_fx_rate_to_usd(data[t]["info"].get("currency", "USD")) for t in tickers}

# =============================
# Financial Overview
# =============================
if section == "Financial Overview":
    st.title("📊 Financial Overview")

    # Cumulative Returns
    fig = go.Figure()
    for t in tickers:
        h = data[t]["hist"]
        if h is None or h.empty: continue
        rets = h["Close"].pct_change().dropna()
        cum = (1 + rets).cumprod() - 1
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode="lines", name=t))
    fig.update_layout(title="Cumulative Returns (Last 1Y)", template="plotly_white", yaxis_tickformat=".1%")
    st.plotly_chart(fig, use_container_width=True)

# =============================
# Key Ratios & WACC
# =============================
elif section == "Key Ratios & WACC":
    st.title("📈 Key Ratios & WACC")

    ratio_dict = {}
    for t in tickers:
        fin, bs = data[t]["financials"], data[t]["balance"]
        if fin.empty or bs.empty: continue
        rev = find_label(fin, ["Total Revenue", "Revenue"])
        gp = find_label(fin, ["Gross Profit"])
        op = find_label(fin, ["Operating Income"])
        ni = find_label(fin, ["Net Income"])
        eq = find_label(bs, ["Total Stockholder Equity"])
        debt = find_label(bs, ["Total Debt"])
        row = {}
        try:
            if rev and gp: row["Gross Margin"] = (fin.loc[gp] / fin.loc[rev]).mean()
            if rev and op: row["Operating Margin"] = (fin.loc[op] / fin.loc[rev]).mean()
            if rev and ni: row["Net Margin"] = (fin.loc[ni] / fin.loc[rev]).mean()
            if eq and ni: row["ROE"] = (fin.loc[ni] / bs.loc[eq]).mean()
            if eq and debt: row["D/E"] = (bs.loc[debt] / bs.loc[eq]).mean()
        except: pass
        ratio_dict[t] = row

    if ratio_dict:
        df = pd.DataFrame(ratio_dict).T
        st.write(styled_ratio_table(df), unsafe_allow_html=True)

    st.subheader("WACC Summary")
    res = {}
    for t in tickers:
        res[t] = compute_wacc(t, data, rf, mrp)
    dfw = pd.DataFrame(res).T
    dfw["WACC"] = dfw["wacc"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    dfw["Beta"] = dfw["beta"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    st.dataframe(dfw[["Beta", "ke", "kd", "tax", "E", "D", "WACC"]])

# =============================
# Event Studies
# =============================
elif section == "Event Studies":
    st.title("📅 Event Studies")
    ev = st.selectbox("Choose Event", list(EVENTS.keys()), format_func=lambda x: EVENTS[x])
    for t in tickers:
        abn, car, idx = event_study(t, ev)
        st.subheader(f"{t} — {EVENTS[ev]}")
        if abn is None:
            st.warning(f"No data for {t}.")
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=idx, y=abn, mode="lines+markers", name="Abnormal"))
        fig.add_trace(go.Scatter(x=idx, y=car, mode="lines+markers", name="CAR"))
        fig.update_layout(title=f"{t} — Event Reaction", template="plotly_white", yaxis_tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

# =============================
# Competitor Comparison
# =============================
elif section == "Competitor Comparison":
    st.title("🏢 Competitor Comparison")
    metric = st.selectbox("Metric", ["Total Revenue", "Gross Profit", "Net Income"])
    frames = {}
    for t in tickers:
        fin = data[t]["financials"]
        lbl = find_label(fin, [metric])
        if lbl:
            frames[t] = to_usd(fin.loc[lbl], fx[t])
    if frames:
        df = pd.DataFrame(frames)
        st.write(styled_money_table(df), unsafe_allow_html=True)
        fig = go.Figure()
        for c in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines+markers", name=c))
        fig.update_layout(title=f"{metric} Over Time (USD)", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.caption("Built for Corporate Finance module — includes June & Sept 2024 events, WACC, and competitor analysis.")

