import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import datetime, pytz

st.set_page_config(page_title="Corporate Finance Dashboard (Interactive)", layout="wide")

# -----------------------
# Helper functions
# -----------------------
def human_format_number(x):
    if pd.isna(x):
        return ""
    try:
        x = float(x)
    except:
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
    pair = f"{currency.upper()}USD=X"
    try:
        data = yf.Ticker(pair).history(period="5d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except:
        pass
    pair = f"USD{currency.upper()}=X"
    try:
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
            "returns": tkr.history(period="1y")["Close"].pct_change().dropna(),
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

# -----------------------
# Streamlit sidebar
# -----------------------
st.sidebar.title("Dashboard Controls")
main_ticker = st.sidebar.text_input("Main Ticker", "LMT").upper()
competitors = st.sidebar.text_input("Competitors (comma separated)", "BA, RTX, NOC, GD, BAESY")
competitor_tickers = [t.strip().upper() for t in competitors.split(",") if t.strip()]
all_tickers = [main_ticker] + competitor_tickers
section = st.sidebar.radio("Choose Section", ["Income Statement", "Balance Sheet", "Cash Flow", "Event Study"])
event_date_str = st.sidebar.text_input("Event Date (YYYY-MM-DD)", "2024-10-22")

# -----------------------
# Data retrieval
# -----------------------
with st.spinner("Loading data..."):
    data = get_data(all_tickers)
    fx_rates = {t: get_fx_rate_to_usd(data[t]["currency"]) for t in all_tickers}

# -----------------------
# Event study function
# -----------------------
def event_study(ticker, date_str):
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

# -----------------------
# SECTION 1: Income Statement
# -----------------------
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
            st.plotly_chart(fig, use_container_width=True)

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

# -----------------------
# SECTION 2: Balance Sheet
# -----------------------
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
        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# SECTION 3: Cash Flow
# -----------------------
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
        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# SECTION 4: Event Study
# -----------------------
elif section == "Event Study":
    st.title("📅 Event Study")
    st.markdown("Analyze how the stock reacted to a specific event date.")

    abnormal, car, dates = event_study(main_ticker, event_date_str)
    if abnormal is None:
        st.warning("Event study data unavailable for selected ticker/date.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dates, y=abnormal, name="Abnormal Return", marker_color="indianred"))
        fig.add_trace(go.Scatter(x=dates, y=car, name="CAR", marker_color="royalblue"))
        fig.update_layout(template="plotly_white", title=f"{main_ticker} Event Study ({event_date_str})",
                          xaxis_title="Date", yaxis_title="Return")
        st.plotly_chart(fig, use_container_width=True)

        df = pd.DataFrame({"Abnormal Return": abnormal, "CAR": car}, index=dates)
        st.dataframe(df)


