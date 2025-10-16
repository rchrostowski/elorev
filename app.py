import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
import datetime
import pytz

st.set_page_config(page_title="Corporate Finance Dashboard", layout="wide")

# -------------------------------------
# Utility Functions
# -------------------------------------

def human_format_number(x):
    if pd.isna(x):
        return ""
    try:
        x = float(x)
    except Exception:
        return str(x)
    absx = abs(x)
    if absx >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f} B"
    elif absx >= 1_000_000:
        return f"{x/1_000_000:.2f} M"
    elif absx >= 1_000:
        return f"{x:,.0f}"
    else:
        return f"{x:.2f}"

@st.cache_data
def get_fx_rate_to_usd(currency):
    if not currency or currency.upper() == "USD":
        return 1.0
    try:
        pair = f"{currency.upper()}USD=X"
        data = yf.Ticker(pair).history(period="5d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except:
        pass
    try:
        pair = f"USD{currency.upper()}=X"
        data = yf.Ticker(pair).history(period="5d")
        if not data.empty:
            return 1 / float(data["Close"].iloc[-1])
    except:
        pass
    return 1.0

@st.cache_data
def get_data(tickers):
    out = {}
    for t in tickers:
        tkr = yf.Ticker(t)
        info = tkr.info
        out[t] = {
            "financials": tkr.financials,
            "balance_sheet": tkr.balance_sheet,
            "cashflow": tkr.cashflow,
            "currency": info.get("currency", "USD"),
            "info": info
        }
    return out

def convert_to_usd(df, rate):
    try:
        return df.apply(pd.to_numeric, errors="coerce") * rate
    except:
        return df

def find_label(df, candidates):
    if df is None or df.empty:
        return None
    idx = [str(i).lower() for i in df.index]
    for c in candidates:
        for i, label in enumerate(idx):
            if c.lower() in label:
                return df.index[i]
    return None

def styled_money_table(df):
    if df.empty:
        return "<i>No data</i>"
    return df.style.format(human_format_number).to_html()

def styled_ratio_table(df):
    if df.empty:
        return "<i>No data</i>"
    return df.style.format("{:.2%}").background_gradient(cmap="RdYlGn").to_html()

# -------------------------------------
# Event Study Functionality
# -------------------------------------

def event_study(ticker, date_str):
    try:
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        nyc = pytz.timezone("America/New_York")
        dt = nyc.localize(dt)

        firm = yf.Ticker(ticker).history(start=dt - datetime.timedelta(days=220), end=dt + datetime.timedelta(days=10))
        market = yf.Ticker("^GSPC").history(start=dt - datetime.timedelta(days=220), end=dt + datetime.timedelta(days=10))

        if firm.empty or market.empty:
            return None, None, None

        fr, mr = firm["Close"].pct_change().dropna(), market["Close"].pct_change().dropna()
        ret = pd.DataFrame({"Firm": fr, "Market": mr}).dropna()

        est = ret.iloc[:-10]
        ev = ret.iloc[-10:]

        X = sm.add_constant(est["Market"])
        model = sm.OLS(est["Firm"], X).fit()
        alpha, beta = model.params
        abnormal = ev["Firm"] - (alpha + beta * ev["Market"])
        car = abnormal.cumsum()

        return abnormal, car, ev.index
    except:
        return None, None, None

# -------------------------------------
# Sidebar
# -------------------------------------

st.sidebar.title("Dashboard Controls")
main_ticker = st.sidebar.text_input("Main Ticker", "LMT").upper()
competitors = st.sidebar.text_input("Competitors (comma separated)", "BA, RTX, NOC, GD, BAESY")
competitor_tickers = [t.strip().upper() for t in competitors.split(",") if t.strip()]
all_tickers = [main_ticker] + competitor_tickers
section = st.sidebar.radio("Choose Section", ["Income Statement", "Balance Sheet", "Cash Flow", "Event Study"])

# -------------------------------------
# Load Data
# -------------------------------------

with st.spinner("Loading data..."):
    data = get_data(all_tickers)
    fx_rates = {t: get_fx_rate_to_usd(data[t]["currency"]) for t in all_tickers}

# -------------------------------------
# Hardcoded Events
# -------------------------------------

EVENTS = {
    "2024-10-22": "Lockheed Martin Q3 2024 Earnings Beat and Raised Guidance",
    "2024-10-23": "Boeing Q3 2024 Earnings Miss and Production Delays",
    "2024-10-24": "Raytheon & Northrop Q3 2024 Results (Mixed Performance)",
    "2024-07-25": "BAE Systems Half-Year 2024 Results Show Increased Order Backlog"
}

# -------------------------------------
# Income Statement
# -------------------------------------

if section == "Income Statement":
    st.title("📈 Income Statement")
    metric = st.selectbox("Choose a Metric", ["Total Revenue", "Gross Profit", "Net Income", "Key Ratios"])

    if metric != "Key Ratios":
        frames = {}
        candidates = {
            "Total Revenue": ["Total Revenue", "Revenue"],
            "Gross Profit": ["Gross Profit"],
            "Net Income": ["Net Income", "NetIncome"]
        }[metric]

        for t in all_tickers:
            fin = data[t]["financials"]
            if fin is None or fin.empty:
                continue
            label = find_label(fin, candidates)
            if label:
                df = convert_to_usd(fin.loc[label], fx_rates[t])
                frames[t] = df
        if not frames:
            st.info("No data found.")
        else:
            df = pd.DataFrame(frames)
            st.markdown(styled_money_table(df), unsafe_allow_html=True)
            fig = go.Figure()
            for col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines+markers", name=col))
            fig.update_layout(template="plotly_white", title=f"{metric} Over Time (USD)")
            st.plotly_chart(fig, use_container_width=True, key=f"{metric}_chart")

    else:
        st.subheader("Key Ratios")
        ratios = {}
        for t in all_tickers:
            fin = data[t]["financials"]
            bs = data[t]["balance_sheet"]
            if fin is None or bs is None or fin.empty or bs.empty:
                continue

            rev = find_label(fin, ["Total Revenue", "Revenue"])
            gp = find_label(fin, ["Gross Profit"])
            op = find_label(fin, ["Operating Income"])
            ni = find_label(fin, ["Net Income", "NetIncome"])
            eq = find_label(bs, ["Total Stockholder Equity"])
            debt = find_label(bs, ["Total Debt", "Total Liab"])

            ratios[t] = {}
            if rev and gp:
                ratios[t]["Gross Margin"] = (fin.loc[gp] / fin.loc[rev]).mean()
            if rev and op:
                ratios[t]["Operating Margin"] = (fin.loc[op] / fin.loc[rev]).mean()
            if rev and ni:
                ratios[t]["Net Margin"] = (fin.loc[ni] / fin.loc[rev]).mean()
            if eq and ni:
                ratios[t]["ROE"] = (fin.loc[ni] / bs.loc[eq]).mean()
            if eq and debt:
                ratios[t]["D/E"] = (bs.loc[debt] / bs.loc[eq]).mean()

        ratio_df = pd.DataFrame(ratios).T
        st.markdown(styled_ratio_table(ratio_df), unsafe_allow_html=True)

# -------------------------------------
# Balance Sheet
# -------------------------------------

elif section == "Balance Sheet":
    st.title("🏦 Balance Sheet")
    metric = st.selectbox("Choose a Metric", ["Total Assets", "Total Liabilities", "Total Equity"])
    frames = {}
    for t in all_tickers:
        bs = data[t]["balance_sheet"]
        if bs is None or bs.empty:
            continue
        label = find_label(bs, [metric])
        if label:
            frames[t] = convert_to_usd(bs.loc[label], fx_rates[t])
    if not frames:
        st.info("No data available.")
    else:
        df = pd.DataFrame(frames)
        st.markdown(styled_money_table(df), unsafe_allow_html=True)
        fig = go.Figure()
        for col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines+markers", name=col))
        fig.update_layout(template="plotly_white", title=f"{metric} Over Time (USD)")
        st.plotly_chart(fig, use_container_width=True, key=f"{metric}_bs")

# -------------------------------------
# Cash Flow
# -------------------------------------

elif section == "Cash Flow":
    st.title("💵 Cash Flow")
    metric = st.selectbox("Choose a Metric", ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow"])
    frames = {}
    for t in all_tickers:
        cf = data[t]["cashflow"]
        if cf is None or cf.empty:
            continue
        label = find_label(cf, [metric])
        if label:
            frames[t] = convert_to_usd(cf.loc[label], fx_rates[t])
    if not frames:
        st.info("No data available.")
    else:
        df = pd.DataFrame(frames)
        st.markdown(styled_money_table(df), unsafe_allow_html=True)
        fig = go.Figure()
        for col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines+markers", name=col))
        fig.update_layout(template="plotly_white", title=f"{metric} Over Time (USD)")
        st.plotly_chart(fig, use_container_width=True, key=f"{metric}_cf")

# -------------------------------------
# Event Study
# -------------------------------------

elif section == "Event Study":
    st.title("📅 Event Study")
    selected_event = st.selectbox("Choose Event", list(EVENTS.keys()), format_func=lambda x: EVENTS[x])

    # Per-company analysis
    for ticker in all_tickers:
        abnormal, car, idx = event_study(ticker, selected_event)
        st.subheader(f"{ticker} — {EVENTS[selected_event]}")
        if abnormal is None:
            st.warning(f"No data for {ticker} on {selected_event}")
            continue

        fig = go.Figure()
        fig.add_trace(go.Bar(x=idx, y=abnormal, name="Abnormal Return", marker_color="indianred"))
        fig.add_trace(go.Scatter(x=idx, y=car, name="CAR", marker_color="royalblue"))
        fig.update_layout(template="plotly_white", title=f"{ticker} — {selected_event}", xaxis_title="Date", yaxis_title="Return")
        st.plotly_chart(fig, use_container_width=True, key=f"{ticker}_{selected_event}_event")

    # Cross-company comparison
    st.header("Cross-Company Comparison")
    df_ab = pd.DataFrame()
    df_car = pd.DataFrame()
    for t in all_tickers:
        abnormal, car, idx = event_study(t, selected_event)
        if abnormal is not None:
            df_ab[t] = abnormal
        if car is not None:
            df_car[t] = car

    if df_ab.empty:
        st.warning("No data available for this event.")
    else:
        fig1 = go.Figure()
        for c in df_ab.columns:
            fig1.add_trace(go.Bar(x=df_ab.index, y=df_ab[c], name=c))
        fig1.update_layout(barmode="group", title=f"Abnormal Returns Comparison ({selected_event})", template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True, key=f"ab_{selected_event}")

        fig2 = go.Figure()
        for c in df_car.columns:
            fig2.add_trace(go.Scatter(x=df_car.index, y=df_car[c], mode="lines+markers", name=c))
        fig2.update_layout(title=f"CAR Comparison ({selected_event})", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True, key=f"car_{selected_event}")

