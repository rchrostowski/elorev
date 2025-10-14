import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime, pytz

st.set_page_config(page_title="Corporate Finance Dashboard", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("📊 Dashboard Sections")
page = st.sidebar.radio(
    "Select a Section:",
    ["Income Statement", "Balance Sheet", "Cash Flow", "Event Study"]
)

main_ticker = st.sidebar.text_input("Main Company Ticker", "LMT")
competitors = st.sidebar.text_input("Competitor Tickers (comma-separated)", "BA, RTX, NOC, GD, BAESY")
event_date_str = st.sidebar.text_input("Event Date (YYYY-MM-DD)", "2024-10-22")

competitor_tickers = [t.strip().upper() for t in competitors.split(",")]
all_tickers = [main_ticker] + competitor_tickers

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

data = get_data(all_tickers)

# --- Income Statement ---
if page == "Income Statement":
    st.title("📈 Income Statement Analysis")
    tabs = st.tabs(["Revenue", "Gross Profit", "Net Income"])
    metrics = ["Total Revenue", "Gross Profit", "Net Income"]

    for tab, metric in zip(tabs, metrics):
        with tab:
            st.subheader(metric)
            st.markdown(f"This shows {metric.lower()} trends over time across companies.")
            metric_data = pd.DataFrame({t: d['financials'].loc[metric] for t, d in data.items() if metric in d['financials'].index})
            st.dataframe(metric_data)
            fig, ax = plt.subplots(figsize=(10, 4))
            for t in metric_data.columns:
                ax.plot(metric_data.index, metric_data[t], marker='o', label=t)
            ax.set_title(f"{metric} Over Time")
            ax.legend()
            st.pyplot(fig)

# --- Balance Sheet ---
elif page == "Balance Sheet":
    st.title("🏦 Balance Sheet Analysis")
    tabs = st.tabs(["Assets", "Liabilities", "Equity"])
    metrics = ["Total Assets", "Total Liabilities Net Minority Interest", "Total Stockholder Equity"]

    for tab, metric in zip(tabs, metrics):
        with tab:
            st.subheader(metric)
            st.markdown(f"This displays {metric.lower()} trends across firms.")
            metric_data = pd.DataFrame({t: d['balance_sheet'].loc[metric] for t, d in data.items() if metric in d['balance_sheet'].index})
            st.dataframe(metric_data)
            fig, ax = plt.subplots(figsize=(10, 4))
            for t in metric_data.columns:
                ax.plot(metric_data.index, metric_data[t], marker='o', label=t)
            ax.set_title(f"{metric} Over Time")
            ax.legend()
            st.pyplot(fig)

# --- Cash Flow ---
elif page == "Cash Flow":
    st.title("💵 Cash Flow Analysis")
    tabs = st.tabs(["Operating", "Investing", "Financing"])
    metrics = ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow"]

    for tab, metric in zip(tabs, metrics):
        with tab:
            st.subheader(metric)
            st.markdown(f"This shows {metric.lower()} activity and cash movement across firms.")
            metric_data = pd.DataFrame({t: d['cashflow'].loc[metric] for t, d in data.items() if metric in d['cashflow'].index})
            st.dataframe(metric_data)
            fig, ax = plt.subplots(figsize=(10, 4))
            for t in metric_data.columns:
                ax.plot(metric_data.index, metric_data[t], marker='o', label=t)
            ax.set_title(f"{metric} Over Time")
            ax.legend()
            st.pyplot(fig)

# --- Event Study ---
elif page == "Event Study":
    st.title("📅 Event Study Analysis")
    st.markdown(f"""
    Examine {main_ticker}'s stock behavior around **{event_date_str}**.  
    The event study compares actual vs expected returns to detect abnormal performance.
    """)

    tabs = st.tabs(["Explanation", "Event Study Chart"])

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

    with tabs[0]:
        st.subheader("Event Explanation")
        st.markdown(f"""
        The model estimates how {main_ticker} typically moves with the S&P 500.  
        Abnormal returns show deviation from expected performance.  
        Cumulative Abnormal Return (CAR) measures total market-adjusted gain/loss around the event.
        """)

    with tabs[1]:
        st.subheader("Event Study Chart")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(event.index, abnormal, label="Abnormal Return", marker='o')
        ax.plot(event.index, car, label="Cumulative Abnormal Return", linestyle='--')
        ax.legend()
        ax.set_title(f"Event Study for {main_ticker} ({event_date_str})")
        st.pyplot(fig)


