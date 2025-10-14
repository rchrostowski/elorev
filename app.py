import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime, pytz

# --- Streamlit Page Config ---
st.set_page_config(page_title="Corporate Finance Dashboard", layout="wide")

# --- Sidebar Inputs ---
st.sidebar.header("📊 Dashboard Controls")
main_ticker = st.sidebar.text_input("Main Company Ticker", "LMT")
competitors = st.sidebar.text_input("Competitor Tickers (comma-separated)", "BA, RTX, NOC, GD, BAESY")
event_date_str = st.sidebar.text_input("Event Date (YYYY-MM-DD)", "2024-10-22")

competitor_tickers = [t.strip().upper() for t in competitors.split(",")]
all_tickers = [main_ticker] + competitor_tickers

# --- Data Fetching ---
@st.cache_data
def get_data(tickers):
    firms = {}
    for t in tickers:
        firm = yf.Ticker(t)
        firms[t] = {
            'financials': firm.financials,
            'balance_sheet': firm.balance_sheet,
            'cashflow': firm.cashflow,
            'stock_rets': firm.history(period="1y")['Close'].pct_change().dropna(),
            'info': firm.info
        }
    return firms

with st.spinner("Collecting data..."):
    data = get_data(all_tickers)
st.success("✅ Data Loaded Successfully")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Income Statement", "🏦 Balance Sheet", "💵 Cash Flow", "📅 Event Study"])

# ========== TAB 1: INCOME STATEMENT ==========
with tab1:
    st.header("Income Statement Analysis")
    st.markdown("""
    This section shows the firm's **revenue**, **gross profit**, and **net income** over time, 
    compared with competitors. These are key indicators of growth, cost efficiency, and profitability.
    """)
    
    metrics = ["Total Revenue", "Gross Profit", "Net Income"]
    for metric in metrics:
        metric_data = pd.DataFrame({t: d['financials'].loc[metric] for t, d in data.items() if metric in d['financials'].index})
        if not metric_data.empty:
            st.subheader(metric)
            st.dataframe(metric_data)
            fig, ax = plt.subplots(figsize=(10, 4))
            for t in metric_data.columns:
                ax.plot(metric_data.index, metric_data[t], marker='o', label=t)
            ax.set_title(f"{metric} Over Time")
            ax.legend()
            st.pyplot(fig)

# ========== TAB 2: BALANCE SHEET ==========
with tab2:
    st.header("Balance Sheet Analysis")
    st.markdown("""
    This section presents **assets**, **liabilities**, and **equity** trends across firms.  
    These reflect financial stability, leverage, and long-term capital structure.
    """)
    
    metrics = ["Total Assets", "Total Liabilities Net Minority Interest", "Total Stockholder Equity"]
    for metric in metrics:
        metric_data = pd.DataFrame({t: d['balance_sheet'].loc[metric] for t, d in data.items() if metric in d['balance_sheet'].index})
        if not metric_data.empty:
            st.subheader(metric)
            st.dataframe(metric_data)
            fig, ax = plt.subplots(figsize=(10, 4))
            for t in metric_data.columns:
                ax.plot(metric_data.index, metric_data[t], marker='o', label=t)
            ax.set_title(f"{metric} Over Time")
            ax.legend()
            st.pyplot(fig)

# ========== TAB 3: CASH FLOW ==========
with tab3:
    st.header("Cash Flow Analysis")
    st.markdown("""
    This section analyzes **operating**, **investing**, and **financing** cash flows to show  
    how firms generate and use cash.  
    Strong operating cash flow with manageable investing and financing activity indicates healthy financial operations.
    """)
    
    metrics = ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow"]
    for metric in metrics:
        metric_data = pd.DataFrame({t: d['cashflow'].loc[metric] for t, d in data.items() if metric in d['cashflow'].index})
        if not metric_data.empty:
            st.subheader(metric)
            st.dataframe(metric_data)
            fig, ax = plt.subplots(figsize=(10, 4))
            for t in metric_data.columns:
                ax.plot(metric_data.index, metric_data[t], marker='o', label=t)
            ax.set_title(f"{metric} Over Time")
            ax.legend()
            st.pyplot(fig)

# ========== TAB 4: EVENT STUDY ==========
with tab4:
    st.header("Event Study")
    st.markdown(f"""
    This section studies the **stock reaction** of {main_ticker} around the chosen event date **{event_date_str}**.  
    It compares actual returns with the expected returns (based on the market model) to identify **abnormal performance**.
    """)
    
    # Event Study Setup
    nyc = pytz.timezone("America/New_York")
    event_date = nyc.localize(datetime.datetime.strptime(event_date_str, "%Y-%m-%d"))
    event_window = 5
    estimation_window = 200

    start_est = event_date - datetime.timedelta(days=estimation_window + event_window)
    end_event = event_date + datetime.timedelta(days=event_window)

    firm_data = yf.Ticker(main_ticker).history(start=start_est, end=end_event)['Close']
    market_data = yf.Ticker("^GSPC").history(start=start_est, end=end_event)['Close']

    firm_ret = firm_data.pct_change().dropna()
    mkt_ret = market_data.pct_change().dropna()
    returns_df = pd.DataFrame({main_ticker: firm_ret, "Market": mkt_ret}).dropna()

    estimation = returns_df.iloc[:-event_window*2]
    event = returns_df.iloc[-event_window*2:]

    X = sm.add_constant(estimation["Market"])
    model = sm.OLS(estimation[main_ticker], X).fit()
    alpha, beta = model.params
    abnormal = event[main_ticker] - (alpha + beta * event["Market"])
    car = abnormal.cumsum()

    # Explanation Section
    st.subheader("📄 Event Explanation")
    st.markdown(f"""
    The model estimates how {main_ticker}'s returns typically move with the market (S&P 500).  
    The difference between actual and expected returns represents **abnormal performance** —  
    how the firm uniquely reacted around the event date.
    """)

    # Visualization Section
    st.subheader("📊 Event Study Chart")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(event.index, abnormal, label="Abnormal Return", marker='o')
    ax.plot(event.index, car, label="Cumulative Abnormal Return", linestyle='--')
    ax.legend()
    ax.set_title(f"Event Study for {main_ticker} ({event_date_str})")
    st.pyplot(fig)

