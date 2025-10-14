# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime, pytz

st.set_page_config(page_title="Corporate Finance Dashboard", layout="wide")

st.title("📊 Corporate Finance Dashboard")
st.markdown("Analyze financials, ratios, and event studies for a firm and its competitors.")

# --- Inputs ---
st.sidebar.header("Firm Selection")
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

# --- Market Cap and Market Share ---
market_caps = {t: data[t]['info'].get('marketCap') for t in all_tickers if data[t]['info']}
market_cap_df = pd.DataFrame.from_dict(market_caps, orient='index', columns=['Market Cap']).dropna()
market_cap_df['Market Share (%)'] = (market_cap_df['Market Cap'] / market_cap_df['Market Cap'].sum()) * 100

st.subheader("🏢 Market Capitalization and Market Share")
st.dataframe(market_cap_df)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(market_cap_df.index, market_cap_df['Market Share (%)'])
ax.set_ylabel("Market Share (%)")
st.pyplot(fig)

# --- Stock Performance ---
st.subheader("📈 Stock Performance (Cumulative Returns)")
fig, ax = plt.subplots(figsize=(10, 4))
for t in all_tickers:
    ax.plot(data[t]['stock_rets'].cumsum(), label=t)
ax.legend()
ax.set_title("Cumulative Returns (1 Year)")
st.pyplot(fig)

# --- Event Study ---
st.subheader("📅 Event Study Analysis")
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

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(event.index, abnormal, label="Abnormal Return", marker='o')
ax.plot(event.index, car, label="Cumulative Abnormal Return", linestyle='--')
ax.legend()
ax.set_title(f"Event Study for {main_ticker} ({event_date_str})")
st.pyplot(fig)

st.caption("Abnormal returns show how the stock performed relative to the market around the event date.")
