# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import datetime, pytz
from typing import Tuple, Dict

st.set_page_config(page_title="Corporate Finance Dashboard (Enhanced)", layout="wide")

# -----------------------------
# Sidebar - Controls & Navigation
# -----------------------------
st.sidebar.title("📊 Dashboard Sections")
section = st.sidebar.radio(
    "Select a Section:",
    ["Income Statement", "Balance Sheet", "Cash Flow", "Event Study"]
)

main_ticker = st.sidebar.text_input("Main Company Ticker", "LMT").upper()
competitors_input = st.sidebar.text_input("Competitor Tickers (comma-separated)", "BA, RTX, NOC, GD, BAESY")
competitor_tickers = [t.strip().upper() for t in competitors_input.split(",") if t.strip()]
all_tickers = [main_ticker] + competitor_tickers
# event date input retained for single-event quick test (but event_list below drives actual event studies)
event_date_str = st.sidebar.text_input("Quick Event Date (YYYY-MM-DD)", "2024-10-22")

# -----------------------------
# Utility: get_data (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def get_data(tickers):
    firms = {}
    for t in tickers:
        try:
            firm = yf.Ticker(t)
            # financials may be empty; wrap accesses
            financials = firm.financials if hasattr(firm, 'financials') else pd.DataFrame()
            balance_sheet = firm.balance_sheet if hasattr(firm, 'balance_sheet') else pd.DataFrame()
            cashflow = firm.cashflow if hasattr(firm, 'cashflow') else pd.DataFrame()
            hist = firm.history(period="1y")
            stock_rets = hist['Close'].pct_change().dropna() if 'Close' in hist else pd.Series(dtype=float)
            info = {}
            try:
                info = firm.info
            except Exception:
                info = {}
            firms[t] = {
                'financials': financials,
                'balance_sheet': balance_sheet,
                'cashflow': cashflow,
                'stock_rets': stock_rets,
                'info': info
            }
        except Exception as e:
            firms[t] = {
                'financials': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cashflow': pd.DataFrame(),
                'stock_rets': pd.Series(dtype=float),
                'info': {}
            }
    return firms

with st.spinner("Fetching financial & price data..."):
    data = get_data(all_tickers)
st.sidebar.success("Data loaded")

# -----------------------------
# Helpers: robust index finders
# -----------------------------
def find_balance_sheet_label(df: pd.DataFrame, candidates):
    """Return first matching index label from candidates present in df, else None"""
    if df is None or df.empty:
        return None
    idx = df.index.astype(str)
    for c in candidates:
        if c in idx.values:
            return c
    # try case-insensitive contains match
    for c in candidates:
        matches = [i for i in idx if c.lower() in i.lower()]
        if matches:
            return matches[0]
    return None

def find_financials_label(df: pd.DataFrame, candidates):
    return find_balance_sheet_label(df, candidates)

# -----------------------------
# Build Market Cap / Market Share
# -----------------------------
def build_market_caps(firms: Dict[str, dict]) -> pd.DataFrame:
    market_caps = {}
    for t, d in firms.items():
        try:
            cap = d.get('info', {}).get('marketCap', None)
            if cap is None:
                # fallback: compute from sharesOutstanding * previousClose if available
                info = d.get('info', {})
                so = info.get('sharesOutstanding')
                prev_close = None
                try:
                    prev_close = yf.Ticker(t).history(period="1d")['Close'].iloc[-1]
                except Exception:
                    prev_close = None
                if so and prev_close:
                    cap = so * prev_close
            market_caps[t] = cap
        except Exception:
            market_caps[t] = None
    df = pd.DataFrame.from_dict(market_caps, orient='index', columns=['Market Cap']).dropna()
    if not df.empty:
        df['Market Share (%)'] = df['Market Cap'] / df['Market Cap'].sum() * 100
    return df

market_cap_df = build_market_caps(data)

# -----------------------------
# Financial Ratios (robust)
# -----------------------------
def compute_ratios(firms: Dict[str, dict]) -> pd.DataFrame:
    to_concat = []
    # metrics described in your file
    # Gross Profit Margin, Operating Margin, Net Income Margin, Return on Equity, Debt to Equity
    for metric in ["Gross Profit Margin", "Operating Margin", "Net Income Margin", "Return on Equity", "Debt to Equity"]:
        metric_frames = {}
        try:
            if metric == "Gross Profit Margin":
                for t, d in firms.items():
                    fin = d['financials']
                    if fin is None or fin.empty:
                        continue
                    gp = find_financials_label(fin, ["Gross Profit", "GrossProfit", "Gross profit"])
                    rev = find_financials_label(fin, ["Total Revenue", "TotalRevenue", "Revenue"])
                    if gp and rev and gp in fin.index and rev in fin.index:
                        metric_frames[t] = (fin.loc[gp] / fin.loc[rev]).dropna()
            elif metric == "Operating Margin":
                for t, d in firms.items():
                    fin = d['financials']
                    if fin is None or fin.empty:
                        continue
                    op = find_financials_label(fin, ["Operating Income", "OperatingIncome", "Operating income"])
                    rev = find_financials_label(fin, ["Total Revenue", "TotalRevenue", "Revenue"])
                    if op and rev and op in fin.index and rev in fin.index:
                        metric_frames[t] = (fin.loc[op] / fin.loc[rev]).dropna()
            elif metric == "Net Income Margin":
                for t, d in firms.items():
                    fin = d['financials']
                    if fin is None or fin.empty:
                        continue
                    ni = find_financials_label(fin, ["Net Income", "NetIncome", "Net income"])
                    rev = find_financials_label(fin, ["Total Revenue", "TotalRevenue", "Revenue"])
                    if ni and rev and ni in fin.index and rev in fin.index:
                        metric_frames[t] = (fin.loc[ni] / fin.loc[rev]).dropna()
            elif metric == "Return on Equity":
                # need net income and average equity from balance sheet
                roe_frames = {}
                for t, d in firms.items():
                    fin = d['financials']
                    bal = d['balance_sheet']
                    if fin is None or fin.empty or bal is None or bal.empty:
                        continue
                    ni_label = find_financials_label(fin, ["Net Income", "NetIncome", "Net income"])
                    eq_label = find_balance_sheet_label(bal, ["Total Stockholder Equity", "Stockholders Equity", "Stockholders' Equity", "Total Stockholders' Equity", "Total Stockholder Equity", "Total Equity"])
                    if ni_label and eq_label and ni_label in fin.index and eq_label in bal.index:
                        ni = fin.loc[ni_label]
                        eq = bal.loc[eq_label]
                        # average equity: simple two-point rolling mean if multiple entries
                        avg_eq = eq.rolling(window=2).mean()
                        roe = (ni / avg_eq).dropna()
                        if not roe.empty:
                            roe_frames[t] = roe
                metric_frames = roe_frames
            elif metric == "Debt to Equity":
                dte_frames = {}
                for t, d in firms.items():
                    bal = d['balance_sheet']
                    if bal is None or bal.empty:
                        continue
                    td_label = find_balance_sheet_label(bal, ["Total Debt", "Total debt", "Debt"])
                    eq_label = find_balance_sheet_label(bal, ["Total Stockholder Equity", "Stockholders Equity", "Stockholders' Equity", "Total Equity"])
                    if td_label and eq_label and td_label in bal.index and eq_label in bal.index:
                        td = bal.loc[td_label]
                        eq = bal.loc[eq_label]
                        # avoid divide by zero
                        ratio = (td / eq).replace([np.inf, -np.inf], np.nan).dropna()
                        if not ratio.empty:
                            dte_frames[t] = ratio
                metric_frames = dte_frames
            # generic packaging
            if metric not in ["Return on Equity", "Debt to Equity"]:
                # metric_frames is a dict mapping ticker -> Series
                packed = {}
                for t, series in metric_frames.items():
                    packed[t] = series
                if packed:
                    df_metric = pd.DataFrame(packed)
                    df_metric.columns = pd.MultiIndex.from_product([[metric], df_metric.columns])
                    to_concat.append(df_metric)
            else:
                if metric_frames:
                    df_metric = pd.DataFrame(metric_frames)
                    df_metric.columns = pd.MultiIndex.from_product([[metric], df_metric.columns])
                    to_concat.append(df_metric)
        except Exception:
            # skip metric on error to keep app alive
            continue
    if to_concat:
        ratios_df = pd.concat(to_concat, axis=1)
        return ratios_df
    else:
        return pd.DataFrame()

ratios_data = compute_ratios(data)

# -----------------------------
# Balance sheet highlights & Cash flow data
# -----------------------------
def build_balance_sheet_highlights(firms: Dict[str, dict]) -> pd.DataFrame:
    highlights = ["Total Assets", "Total Liabilities", "Total Stockholder Equity", "Cash and Cash Equivalents", "Total Stockholders' Equity", "Total Liab"]
    frames = []
    for metric in highlights:
        metric_frames = {}
        for t, d in firms.items():
            bal = d['balance_sheet']
            if bal is None or bal.empty:
                continue
            label_found = find_balance_sheet_label(bal, [metric])
            if label_found and label_found in bal.index:
                metric_frames[t] = bal.loc[label_found]
        if metric_frames:
            df_metric = pd.DataFrame(metric_frames)
            df_metric.columns = pd.MultiIndex.from_product([[metric], df_metric.columns])
            frames.append(df_metric)
    if frames:
        return pd.concat(frames, axis=1)
    return pd.DataFrame()

def build_cash_flow_highlights(firms: Dict[str, dict]) -> pd.DataFrame:
    metrics = ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow", "Net Cash Provided by Operating Activities", "Net Cash Provided by (Used in) Operating Activities"]
    frames = []
    for metric in metrics:
        metric_frames = {}
        for t, d in firms.items():
            cf = d['cashflow']
            if cf is None or cf.empty:
                continue
            label_found = find_balance_sheet_label(cf, [metric])
            if label_found and label_found in cf.index:
                metric_frames[t] = cf.loc[label_found]
        if metric_frames:
            df_metric = pd.DataFrame(metric_frames)
            # normalize column name
            df_metric.columns = pd.MultiIndex.from_product([[metric], df_metric.columns])
            frames.append(df_metric)
    if frames:
        return pd.concat(frames, axis=1)
    return pd.DataFrame()

balance_sheet_data = build_balance_sheet_highlights(data)
cash_flow_data = build_cash_flow_highlights(data)

# -----------------------------
# Event study code copied/adapted from the file
# -----------------------------
def perform_event_study(firm_ticker: str, market_ticker: str, event_date: datetime.datetime,
                        event_window_days: int = 5, estimation_window_days: int = 200) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    Perform an event study using market model regression (firm ~ market).
    Returns (abnormal_returns, car, event_returns_df) or (None, None, None) on failure.
    """
    try:
        nyc = pytz.timezone("America/New_York")
        # ensure event_date is timezone-aware
        if event_date.tzinfo is None:
            event_date = nyc.localize(event_date)

        # estimation window
        estimation_end = event_date - datetime.timedelta(days=event_window_days + 1)
        estimation_start = estimation_end - datetime.timedelta(days=estimation_window_days)

        # event window
        event_start = event_date - datetime.timedelta(days=event_window_days)
        event_end = event_date + datetime.timedelta(days=event_window_days)

        firm_hist = yf.Ticker(firm_ticker).history(start=estimation_start, end=event_end)
        market_hist = yf.Ticker(market_ticker).history(start=estimation_start, end=event_end)

        if firm_hist.empty or market_hist.empty:
            return None, None, None

        firm_close = firm_hist['Close'].dropna()
        market_close = market_hist['Close'].dropna()

        firm_ret = firm_close.pct_change().dropna()
        market_ret = market_close.pct_change().dropna()

        returns_df = pd.DataFrame({firm_ticker: firm_ret, market_ticker: market_ret}).dropna()
        # define estimation and event subsets
        estimation_mask = (returns_df.index >= estimation_start) & (returns_df.index <= estimation_end)
        event_mask = (returns_df.index >= event_start) & (returns_df.index <= event_end)

        estimation_returns = returns_df.loc[estimation_mask]
        event_returns = returns_df.loc[event_mask]

        if estimation_returns.empty or event_returns.empty:
            return None, None, None

        X = sm.add_constant(estimation_returns[market_ticker])
        y = estimation_returns[firm_ticker]
        model = sm.OLS(y, X).fit()
        alpha = model.params['const']
        beta = model.params[market_ticker]

        estimated_normal = alpha + beta * event_returns[market_ticker]
        abnormal = event_returns[firm_ticker] - estimated_normal
        car = abnormal.cumsum()

        return abnormal, car, event_returns

    except Exception as e:
        return None, None, None

# Hard-coded event_list copied from your file (with updated descriptions)
event_list = [
    {"firm_ticker": "LMT", "event_date": "2024-10-22", "event_description": "Q3 2024 Earnings Beat and Raised Guidance"},
    {"firm_ticker": "BA",  "event_date": "2024-10-23", "event_description": "Q3 2024 Earnings Miss and Production Delays Announced"},
    {"firm_ticker": "RTX", "event_date": "2024-10-24", "event_description": "Q3 2024 Earnings In Line with Expectations"},
    {"firm_ticker": "NOC", "event_date": "2024-10-24", "event_description": "Q3 2024 Earnings Miss Due to Supply Chain Issues"},
    {"firm_ticker": "GD",  "event_date": "2024-10-23", "event_description": "Q3 2024 Earnings Beat on Strong Demand"},
    {"firm_ticker": "BAESY","event_date":"2024-07-25","event_description":"Half Year 2024 Results Show Increased Order Backlog"}
]

# compute all event studies (cache lightly)
@st.cache_data(show_spinner=False)
def compute_all_event_studies(events):
    results = {}
    for ev in events:
        try:
            tkr = ev['firm_ticker']
            dt = datetime.datetime.strptime(ev['event_date'], "%Y-%m-%d")
            abnormal, car, event_returns = perform_event_study(tkr, "^GSPC", dt, event_window_days=5, estimation_window_days=200)
            if abnormal is not None:
                results[f"{tkr}_{ev['event_date']}"] = {
                    "abnormal": abnormal,
                    "car": car,
                    "event_returns": event_returns,
                    "description": ev.get('event_description', "")
                }
        except Exception:
            continue
    return results

with st.spinner("Running event studies (may take a moment)..."):
    event_study_results = compute_all_event_studies(event_list)

# group event results by description
event_groups = {}
for key, val in event_study_results.items():
    desc = val['description']
    event_groups.setdefault(desc, []).append( (key, val) )

# -----------------------------
# UI: Sections & Tabs (preserve structure)
# -----------------------------
# Income Statement section
if section == "Income Statement":
    st.title("📈 Income Statement")
    st.markdown(
        "This section presents key income-statement metrics (revenue, gross profit, net income). "
        "Each metric has an explanation and a corresponding chart/table so you can quickly compare trends across firms."
    )

    tabs = st.tabs(["Revenue", "Gross Profit", "Net Income", "Key Ratios"])
    # Revenue tab
    with tabs[0]:
        st.subheader("Total Revenue")
        st.markdown("Total revenue (top-line) shows how the business is growing sales over time. Look for persistent growth or declines.")
        revenue_label_candidates = ["Total Revenue", "TotalRevenue", "Revenue"]
        rev_frames = {}
        for t, d in data.items():
            fin = d['financials']
            if fin is None or fin.empty:
                continue
            lbl = find_financials_label(fin, revenue_label_candidates)
            if lbl and lbl in fin.index:
                rev_frames[t] = fin.loc[lbl]
        if rev_frames:
            rev_df = pd.DataFrame(rev_frames)
            st.dataframe(rev_df)
            fig, ax = plt.subplots(figsize=(10,4))
            for col in rev_df.columns:
                ax.plot(rev_df.index, rev_df[col], marker='o', label=col)
            ax.set_title("Total Revenue Over Time")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Revenue data not available for selected tickers.")

    # Gross Profit tab
    with tabs[1]:
        st.subheader("Gross Profit")
        st.markdown("Gross profit shows the revenue left after direct costs of goods sold — a measure of product-level profitability.")
        gp_candidates = ["Gross Profit", "GrossProfit", "Gross profit"]
        gp_frames = {}
        for t, d in data.items():
            fin = d['financials']
            if fin is None or fin.empty:
                continue
            lbl = find_financials_label(fin, gp_candidates)
            if lbl and lbl in fin.index:
                gp_frames[t] = fin.loc[lbl]
        if gp_frames:
            gp_df = pd.DataFrame(gp_frames)
            st.dataframe(gp_df)
            fig, ax = plt.subplots(figsize=(10,4))
            for col in gp_df.columns:
                ax.plot(gp_df.index, gp_df[col], marker='o', label=col)
            ax.set_title("Gross Profit Over Time")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Gross profit data not available for selected tickers.")

    # Net Income tab
    with tabs[2]:
        st.subheader("Net Income")
        st.markdown("Net income is the 'bottom line' after all expenses, taxes, and interest — a key measure of profitability.")
        ni_candidates = ["Net Income", "NetIncome", "Net income"]
        ni_frames = {}
        for t, d in data.items():
            fin = d['financials']
            if fin is None or fin.empty:
                continue
            lbl = find_financials_label(fin, ni_candidates)
            if lbl and lbl in fin.index:
                ni_frames[t] = fin.loc[lbl]
        if ni_frames:
            ni_df = pd.DataFrame(ni_frames)
            st.dataframe(ni_df)
            fig, ax = plt.subplots(figsize=(10,4))
            for col in ni_df.columns:
                ax.plot(ni_df.index, ni_df[col], marker='o', label=col)
            ax.set_title("Net Income Over Time")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Net income data not available for selected tickers.")

    # Key Ratios tab
    with tabs[3]:
        st.subheader("Key Financial Ratios")
        st.markdown("Important ratios (margins, ROE, debt/equity) help compare profitability and leverage across peers.")
        if not ratios_data.empty:
            st.dataframe(ratios_data)
            # show example ratio plots if present
            try:
                # show gross profit margin if available
                if ("Gross Profit Margin",) in ratios_data.columns.tolist() or "Gross Profit Margin" in [c[0] for c in ratios_data.columns]:
                    # fetch columns with that metric
                    cols = [c for c in ratios_data.columns if c[0]=="Gross Profit Margin"]
                    if cols:
                        df_plot = ratios_data[cols]
                        fig, ax = plt.subplots(figsize=(10,4))
                        for col in df_plot.columns:
                            ax.plot(df_plot.index, df_plot[col], marker='o', label=col[1])
                        ax.set_title("Gross Profit Margin Over Time")
                        ax.legend()
                        st.pyplot(fig)
            except Exception:
                pass
        else:
            st.info("No ratio data available for the selected tickers.")

# Balance Sheet section
elif section == "Balance Sheet":
    st.title("🏦 Balance Sheet")
    st.markdown("Balance sheet highlights (assets, liabilities, equity). Check liquidity and capital structure trends.")
    tabs = st.tabs(["Assets", "Liabilities", "Equity", "Market Cap / Market Share"])

    with tabs[0]:
        st.subheader("Assets")
        st.markdown("Total assets reflect the scale of the firm's operations and resource base.")
        candidates = ["Total Assets", "Assets"]
        frames = {}
        for t, d in data.items():
            bal = d['balance_sheet']
            if bal is None or bal.empty:
                continue
            lbl = find_balance_sheet_label(bal, candidates)
            if lbl and lbl in bal.index:
                frames[t] = bal.loc[lbl]
        if frames:
            assets_df = pd.DataFrame(frames)
            st.dataframe(assets_df)
            fig, ax = plt.subplots(figsize=(10,4))
            for col in assets_df.columns:
                ax.plot(assets_df.index, assets_df[col], marker='o', label=col)
            ax.set_title("Total Assets Over Time")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Assets data not available for selected tickers.")

    with tabs[1]:
        st.subheader("Liabilities")
        st.markdown("Total liabilities indicate claims on assets. Compare liabilities to assets and equity to understand leverage.")
        candidates = ["Total Liabilities", "Total Liabilities Net Minority Interest", "Total Liab"]
        frames = {}
        for t, d in data.items():
            bal = d['balance_sheet']
            if bal is None or bal.empty:
                continue
            lbl = find_balance_sheet_label(bal, candidates)
            if lbl and lbl in bal.index:
                frames[t] = bal.loc[lbl]
        if frames:
            liab_df = pd.DataFrame(frames)
            st.dataframe(liab_df)
            fig, ax = plt.subplots(figsize=(10,4))
            for col in liab_df.columns:
                ax.plot(liab_df.index, liab_df[col], marker='o', label=col)
            ax.set_title("Total Liabilities Over Time")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Liabilities data not available for selected tickers.")

    with tabs[2]:
        st.subheader("Equity")
        st.markdown("Equity shows the residual claim of shareholders; watch how equity moves relative to profit and buybacks.")
        candidates = ["Total Stockholder Equity", "Stockholders Equity", "Total Stockholders' Equity", "Total Equity"]
        frames = {}
        for t, d in data.items():
            bal = d['balance_sheet']
            if bal is None or bal.empty:
                continue
            lbl = find_balance_sheet_label(bal, candidates)
            if lbl and lbl in bal.index:
                frames[t] = bal.loc[lbl]
        if frames:
            eq_df = pd.DataFrame(frames)
            st.dataframe(eq_df)
            fig, ax = plt.subplots(figsize=(10,4))
            for col in eq_df.columns:
                ax.plot(eq_df.index, eq_df[col], marker='o', label=col)
            ax.set_title("Total Equity Over Time")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Equity data not available for selected tickers.")

    with tabs[3]:
        st.subheader("Market Capitalization & Market Share")
        st.markdown("Market cap and market share (by market cap) provide a market view of firm size relative to peers.")
        if not market_cap_df.empty:
            st.dataframe(market_cap_df)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(market_cap_df.index, market_cap_df['Market Share (%)'])
            ax.set_ylabel("Market Share (%)")
            st.pyplot(fig)
        else:
            st.info("Market cap data not available for the selected tickers.")

# Cash Flow section
elif section == "Cash Flow":
    st.title("💵 Cash Flow")
    st.markdown("Analyze operating, investing, and financing cash flows. Healthy firms generate positive operating cash flow and deploy it sensibly.")
    tabs = st.tabs(["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow", "Cash Flow Table"])

    with tabs[0]:
        st.subheader("Operating Cash Flow")
        st.markdown("Operating cash flow shows cash generated by core business operations — critical for sustainability.")
        candidates = ["Operating Cash Flow", "Net Cash Provided by Operating Activities", "Net Cash Provided by (Used in) Operating Activities"]
        frames = {}
        for t, d in data.items():
            cf = d['cashflow']
            if cf is None or cf.empty:
                continue
            lbl = find_balance_sheet_label(cf, candidates)
            if lbl and lbl in cf.index:
                frames[t] = cf.loc[lbl]
        if frames:
            op_df = pd.DataFrame(frames)
            st.dataframe(op_df)
            fig, ax = plt.subplots(figsize=(10,4))
            for col in op_df.columns:
                ax.plot(op_df.index, op_df[col], marker='o', label=col)
            ax.set_title("Operating Cash Flow Over Time")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Operating cash flow data not available for selected tickers.")

    with tabs[1]:
        st.subheader("Investing Cash Flow")
        st.markdown("Investing cash flow reflects capex and investments. Negative investing cash flow is common for growth firms investing in the business.")
        candidates = ["Investing Cash Flow", "Net Cash Used in Investing Activities", "Capital Expenditures"]
        frames = {}
        for t, d in data.items():
            cf = d['cashflow']
            if cf is None or cf.empty:
                continue
            lbl = find_balance_sheet_label(cf, ["Investing Cash Flow", "Net Cash Used in Investing Activities", "Capital Expenditures", "Purchase of property and equipment"])
            if lbl and lbl in cf.index:
                frames[t] = cf.loc[lbl]
        if frames:
            inv_df = pd.DataFrame(frames)
            st.dataframe(inv_df)
            fig, ax = plt.subplots(figsize=(10,4))
            for col in inv_df.columns:
                ax.plot(inv_df.index, inv_df[col], marker='o', label=col)
            ax.set_title("Investing Cash Flow Over Time")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Investing cash flow data not available for selected tickers.")

    with tabs[2]:
        st.subheader("Financing Cash Flow")
        st.markdown("Financing cash flow shows share issuance/buybacks and debt issuance/repayment — signals capital structure decisions.")
        candidates = ["Financing Cash Flow", "Net Cash Provided by (Used in) Financing Activities"]
        frames = {}
        for t, d in data.items():
            cf = d['cashflow']
            if cf is None or cf.empty:
                continue
            lbl = find_balance_sheet_label(cf, candidates)
            if lbl and lbl in cf.index:
                frames[t] = cf.loc[lbl]
        if frames:
            fin_df = pd.DataFrame(frames)
            st.dataframe(fin_df)
            fig, ax = plt.subplots(figsize=(10,4))
            for col in fin_df.columns:
                ax.plot(fin_df.index, fin_df[col], marker='o', label=col)
            ax.set_title("Financing Cash Flow Over Time")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Financing cash flow data not available for selected tickers.")

    with tabs[3]:
        st.subheader("Cash Flow Summary Table")
        st.markdown("A consolidated table of operating / investing / financing cash flow metrics where available.")
        if not cash_flow_data.empty:
            st.dataframe(cash_flow_data)
        else:
            st.info("Cash flow tables not available.")

# Event Study section
elif section == "Event Study":
    st.title("📅 Event Study")
    st.markdown(
        "This section runs event studies around specific firm events (earnings, guidance, or other key announcements). "
        "It compares the firm's actual returns with expected (market-model) returns to compute abnormal returns and CARs."
    )

    tabs = st.tabs(["Event List & Descriptions", "Pick an Event (Explanation)", "Event Chart", "CAR Comparison"])

    # Event list tab
    with tabs[0]:
        st.subheader("Event List & Descriptions")
        st.markdown("Below are the events (manually provided). You can pick an event in the next tab to see the detailed analysis.")
        if event_list:
            el_df = pd.DataFrame(event_list)
            st.dataframe(el_df)
        else:
            st.info("No events available.")

    # Pick an event
    with tabs[1]:
        st.subheader("Pick an Event to Explain")
        if not event_study_results:
            st.info("No event study results computed (insufficient data for selected tickers / events).")
        else:
            keys = list(event_study_results.keys())
            choice = st.selectbox("Choose event (ticker_date)", keys)
            if choice:
                ev = event_study_results[choice]
                desc = ev.get("description", "")
                st.markdown(f"**Event:** {choice}")
                st.markdown(f"**Description:** {desc}")
                # simple narrative generation similar to file's interpretation logic
                car = ev['car']
                if car is not None and not car.empty:
                    car_start = car.iloc[0]
                    car_end = car.iloc[-1]
                    trend = "varied"
                    if car_end > car_start and (car.diff().dropna() >= 0).all():
                        trend = "steady increase"
                    elif car_end < car_start and (car.diff().dropna() <= 0).all():
                        trend = "steady decline"
                    elif car_end > car_start:
                        trend = "overall increase"
                    elif car_end < car_start:
                        trend = "overall decline"
                    interpretation = "Market reaction is mixed."
                    if "Beat" in desc and car_end > 0:
                        interpretation = "Positive CAR suggests favorable market reaction to the positive surprise."
                    elif "Miss" in desc and car_end < 0:
                        interpretation = "Negative CAR suggests unfavorable market reaction to a miss."
                    st.markdown(f"**CAR Trend:** {trend} (start ≈ {car_start:.4f}, end ≈ {car_end:.4f})")
                    st.markdown(f"**Interpretation:** {interpretation}")
                else:
                    st.info("CAR not available for this event (insufficient data).")

    # Event Chart tab
    with tabs[2]:
        st.subheader("Event Chart (Abnormal Returns & CAR)")
        st.markdown("Chart shows abnormal returns (daily) and cumulative abnormal returns (CAR).")
        if not event_study_results:
            st.info("No event study results available.")
        else:
            keys = list(event_study_results.keys())
            choice = st.selectbox("Which event to chart?", keys, key="chart_choice")
            if choice:
                ev = event_study_results[choice]
                abnormal = ev['abnormal']
                car = ev['car']
                event_returns = ev['event_returns']
                if abnormal is None or car is None:
                    st.info("Event data incomplete for this event.")
                else:
                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(abnormal.index, abnormal.values, marker='o', label='Abnormal Return')
                    ax.plot(car.index, car.values, linestyle='--', marker='o', label='Cumulative Abnormal Return (CAR)')
                    ax.axvline(x=abnormal.index[len(abnormal)//2], color='gray', linestyle=':', alpha=0.6)  # visual separator near event day
                    ax.legend()
                    ax.set_title(f"Event Study: {choice} - {ev.get('description','')}")
                    st.pyplot(fig)
                    st.markdown("**Event window returns (table):**")
                    st.dataframe(event_returns)

    # CAR Comparison tab
    with tabs[3]:
        st.subheader("CAR Comparison across similar events")
        st.markdown("Compare Cumulative Abnormal Returns (CAR) across firms that experienced the same event description.")
        if not event_groups:
            st.info("No grouped events available.")
        else:
            group_names = list(event_groups.keys())
            sel = st.selectbox("Choose event group", group_names)
            if sel:
                group_items = event_groups[sel]
                # build combined df
                car_df = pd.DataFrame()
                for key, val in group_items:
                    car = val['car']
                    # rename series to firm only
                    firm = key.split('_')[0]
                    if car is not None:
                        car_df[firm] = car
                if car_df.empty:
                    st.info("No CAR data available for this group.")
                else:
                    st.dataframe(car_df)
                    fig, ax = plt.subplots(figsize=(10,5))
                    for col in car_df.columns:
                        ax.plot(car_df.index, car_df[col], marker='o', label=col)
                    ax.set_title(f"CAR Comparison: {sel}")
                    ax.legend()
                    st.pyplot(fig)

# End of app
st.sidebar.markdown("---")
st.sidebar.markdown("Built from combined `15_cf_activities` logic. Ask me to add interactive Plotly charts or a download/export feature.")

