# app.py
# Enhanced Streamlit corporate finance dashboard
# - Sidebar with 4 sections (Income Statement, Balance Sheet, Cash Flow, Event Study)
# - Dropdowns (selectbox) inside each section to choose specific metric to display
# - All financials converted to USD using live FX rates (cached)
# - Smart number formatting (commas, M/B units) and percent formatting for ratios
# - Interactive Plotly charts and downloadable CSV exports
#
# Requirements (requirements.txt):
# streamlit, yfinance, pandas, numpy, plotly, statsmodels, pytz

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import datetime, pytz
from typing import Dict, Tuple, Optional

st.set_page_config(page_title="Corporate Finance Dashboard (Interactive)", layout="wide")

# -----------------------
# Helper utilities
# -----------------------
def human_format_number(x: float) -> str:
    """Format large numbers to readable human format with units (K, M, B)."""
    if pd.isna(x):
        return ""
    try:
        x = float(x)
    except Exception:
        return str(x)
    absx = abs(x)
    if absx >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f} B"
    if absx >= 1_000_000:
        return f"{x/1_000_000:.2f} M"
    if absx >= 1_000:
        return f"{x:,.0f}"
    return f"{x:.2f}"

def fmt_comma(x):
    if pd.isna(x):
        return ""
    try:
        x = float(x)
        if abs(x) >= 1_000_000_000:
            return f"{x:,.0f}"
        elif abs(x) >= 1_000_000:
            return f"{x:,.0f}"
        else:
            return f"{x:,.0f}"
    except Exception:
        return x

def pct_format(x):
    if pd.isna(x):
        return ""
    try:
        return f"{100*x:.2f} %"
    except Exception:
        return x

# -----------------------
# Caching: FX rates and data
# -----------------------
@st.cache_data(show_spinner=False)
def get_fx_rate_to_usd(currency: str) -> Optional[float]:
    """
    Returns the multiplier to convert from `currency` to USD.
    i.e., USD_value = original_value * rate
    Uses Yahoo finance FX pairs like EURUSD=X (EUR -> USD).
    Falls back to inverting USD<currency>=X if needed.
    """
    if currency is None:
        return None
    currency = currency.upper()
    if currency == "USD":
        return 1.0
    pair = f"{currency}USD=X"
    try:
        ticker = yf.Ticker(pair)
        hist = ticker.history(period="5d")
        if not hist.empty and 'Close' in hist:
            rate = hist['Close'].iloc[-1]
            if rate is not None and not np.isnan(rate):
                return float(rate)
    except Exception:
        pass
    # try inverted pair USD<CUR>=X and invert
    inv_pair = f"USD{currency}=X"
    try:
        ticker = yf.Ticker(inv_pair)
        hist = ticker.history(period="5d")
        if not hist.empty and 'Close' in hist:
            inv_rate = hist['Close'].iloc[-1]
            if inv_rate and inv_rate != 0:
                return float(1.0 / inv_rate)
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def get_data_for_tickers(tickers: list) -> Dict[str, dict]:
    firms = {}
    for t in tickers:
        t = t.upper()
        try:
            yc = yf.Ticker(t)
            fin = yc.financials if hasattr(yc, "financials") else pd.DataFrame()
            bs = yc.balance_sheet if hasattr(yc, "balance_sheet") else pd.DataFrame()
            cf = yc.cashflow if hasattr(yc, "cashflow") else pd.DataFrame()
            hist = yc.history(period="1y")
            close = hist['Close'] if 'Close' in hist else pd.Series(dtype=float)
            rets = close.pct_change().dropna() if not close.empty else pd.Series(dtype=float)
            info = {}
            try:
                info = yc.info or {}
            except Exception:
                info = {}
            currency = info.get("currency", "USD")
            firms[t] = {
                "financials": fin,
                "balance_sheet": bs,
                "cashflow": cf,
                "price_close": close,
                "stock_rets": rets,
                "info": info,
                "currency": currency
            }
        except Exception:
            firms[t] = {
                "financials": pd.DataFrame(),
                "balance_sheet": pd.DataFrame(),
                "cashflow": pd.DataFrame(),
                "price_close": pd.Series(dtype=float),
                "stock_rets": pd.Series(dtype=float),
                "info": {},
                "currency": "USD"
            }
    return firms

# -----------------------
# Robust label finders
# -----------------------
def find_label(df: pd.DataFrame, candidates: list) -> Optional[str]:
    if df is None or df.empty:
        return None
    idx = [str(i) for i in df.index]
    for c in candidates:
        if c in idx:
            return c
    # case-insensitive contains
    for c in candidates:
        for i in idx:
            if c.lower() in i.lower():
                return i
    return None

# -----------------------
# Formatting helpers for display
# -----------------------
def styled_money_table(df: pd.DataFrame, scale: str = "auto") -> str:
    """
    Returns HTML (safe) of a pandas Styler with money formatting.
    scale: 'auto' (choose M/B) or 'raw' to keep raw numbers (but formatted with commas)
    """
    if df is None or df.empty:
        return "<i>No data</i>"
    # convert numeric to floats for formatting
    d = df.copy()
    # Try to convert columns to numeric where possible
    for col in d.columns:
        try:
            d[col] = pd.to_numeric(d[col], errors='coerce')
        except Exception:
            pass

    # Decide a scale per-cell: we will format with human_format_number
    sty = d.style.format(human_format_number).set_table_attributes("style='width:100%'")
    return sty.to_html()

def styled_ratio_table(df: pd.DataFrame) -> str:
    """
    Formats ratio table with percent formatting and color gradient.
    """
    if df is None or df.empty:
        return "<i>No data</i>"
    d = df.copy()
    # Convert ratio numbers (assume in 0..1)
    for col in d.columns:
        try:
            d[col] = pd.to_numeric(d[col], errors='coerce')
        except Exception:
            pass
    sty = d.style.format("{:.2%}").background_gradient(cmap="RdYlGn", axis=0, subset=None).set_table_attributes("style='width:100%'")
    return sty.to_html()

# -----------------------
# Event study logic
# -----------------------
def perform_event_study(firm_ticker: str, market_ticker: str, event_date: datetime.datetime,
                        event_window_days: int = 5, estimation_window_days: int = 200):
    nyc = pytz.timezone("America/New_York")
    if event_date.tzinfo is None:
        event_date = nyc.localize(event_date)
    estimation_end = event_date - datetime.timedelta(days=event_window_days + 1)
    estimation_start = estimation_end - datetime.timedelta(days=estimation_window_days)
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
    returns = pd.DataFrame({firm_ticker: firm_ret, market_ticker: market_ret}).dropna()

    est_mask = (returns.index >= estimation_start) & (returns.index <= estimation_end)
    ev_mask = (returns.index >= event_start) & (returns.index <= event_end)
    est_returns = returns.loc[est_mask]
    ev_returns = returns.loc[ev_mask]
    if est_returns.empty or ev_returns.empty:
        return None, None, None

    X = sm.add_constant(est_returns[market_ticker])
    y = est_returns[firm_ticker]
    model = sm.OLS(y, X).fit()
    alpha = model.params['const']
    beta = model.params[market_ticker]
    estimated = alpha + beta * ev_returns[market_ticker]
    abnormal = ev_returns[firm_ticker] - estimated
    car = abnormal.cumsum()
    return abnormal, car, ev_returns

# -----------------------
# Sidebar / inputs
# -----------------------
st.sidebar.title("Dashboard Controls")
main_ticker = st.sidebar.text_input("Main company ticker", "LMT").upper()
competitors = st.sidebar.text_input("Competitors (comma separated)", "BA, RTX, NOC, GD, BAESY")
competitor_tickers = [t.strip().upper() for t in competitors.split(",") if t.strip()]
all_tickers = [main_ticker] + competitor_tickers

# quick event date (not required; events list will be used)
quick_event_date = st.sidebar.text_input("Quick event date (YYYY-MM-DD)", "2024-10-22")

section = st.sidebar.radio("Choose section", ["Income Statement", "Balance Sheet", "Cash Flow", "Event Study"])

# -----------------------
# Fetch data & FX rates
# -----------------------
with st.spinner("Fetching data and FX rates..."):
    firms = get_data_for_tickers(all_tickers)
    # compute fx rates map
    fx_map = {}
    for t, info in firms.items():
        cur = info.get("currency", "USD") or info['info'].get('currency', 'USD') if isinstance(info.get('info', {}), dict) else 'USD'
        rate = get_fx_rate_to_usd(cur)
        fx_map[t] = {"orig_currency": cur, "fx_to_usd": rate or 1.0}
st.sidebar.success("Data ready")

# -----------------------
# Helpers to convert financial tables to USD
# -----------------------
def convert_df_to_usd(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Assumes df rows are items and columns are periods; converts numeric values by fx_map[ticker]['fx_to_usd']"""
    if df is None or df.empty:
        return df
    rate = fx_map.get(ticker, {}).get("fx_to_usd", 1.0)
    try:
        converted = df.apply(pd.to_numeric, errors='coerce') * rate
        # preserve index/names
        converted.index = df.index
        converted.columns = df.columns
        return converted
    except Exception:
        return df

# -----------------------
# Compute market cap df (USD-normalized)
# -----------------------
def build_marketcap_df(firms: Dict[str, dict]) -> pd.DataFrame:
    caps = {}
    for t, d in firms.items():
        info = d.get("info", {}) or {}
        cap = info.get("marketCap")
        if cap is None:
            # try compute from sharesOutstanding and last price
            so = info.get("sharesOutstanding")
            price = None
            try:
                price = d.get("price_close").iloc[-1]
            except Exception:
                price = None
            if so and price:
                cap = so * price
        if cap is None:
            caps[t] = np.nan
        else:
            # convert to USD if necessary
            rate = fx_map.get(t, {}).get("fx_to_usd", 1.0)
            caps[t] = cap * rate
    df = pd.DataFrame.from_dict(caps, orient="index", columns=["market_cap"]).dropna()
    if not df.empty:
        df['market_share_%'] = df['market_cap'] / df['market_cap'].sum() * 100
    return df

market_cap_df = build_marketcap_df(firms)

# -----------------------
# Precompute event list (copied from uploaded file)
# -----------------------
EVENT_LIST = [
    {"firm_ticker": "LMT", "event_date": "2024-10-22", "event_description": "Q3 2024 Earnings Beat and Raised Guidance"},
    {"firm_ticker": "BA",  "event_date": "2024-10-23", "event_description": "Q3 2024 Earnings Miss and Production Delays Announced"},
    {"firm_ticker": "RTX", "event_date": "2024-10-24", "event_description": "Q3 2024 Earnings In Line with Expectations"},
    {"firm_ticker": "NOC", "event_date": "2024-10-24", "event_description": "Q3 2024 Earnings Miss Due to Supply Chain Issues"},
    {"firm_ticker": "GD",  "event_date": "2024-10-23", "event_description": "Q3 2024 Earnings Beat on Strong Demand"},
    {"firm_ticker":"BAESY","event_date":"2024-07-25","event_description":"Half Year 2024 Results Show Increased Order Backlog"}
]

# precompute event studies (cached)
@st.cache_data(show_spinner=False)
def compute_all_events(events):
    out = {}
    for ev in events:
        tkr = ev['firm_ticker']
        try:
            dt = datetime.datetime.strptime(ev['event_date'], "%Y-%m-%d")
            abnormal, car, ev_returns = perform_event_study(tkr, "^GSPC", dt)
            if abnormal is not None:
                out[f"{tkr}_{ev['event_date']}"] = {
                    "abnormal": abnormal,
                    "car": car,
                    "event_returns": ev_returns,
                    "description": ev.get("event_description", "")
                }
        except Exception:
            continue
    return out

with st.spinner("Computing event studies... (this may take ~30s)"):
    event_results = compute_all_events(EVENT_LIST)

# -----------------------
# UI: Sections with dropdowns
# -----------------------
def download_button_for_df(df: pd.DataFrame, label: str = "Download CSV"):
    if df is None or df.empty:
        st.info("No data to download")
        return
    csv = df.to_csv(index=True).encode('utf-8')
    st.download_button(label, csv, file_name=f"{label.replace(' ','_')}.csv", mime="text/csv")

# Income Statement section
if section == "Income Statement":
    st.title("Income Statement")
    st.markdown("Choose a metric below. Numbers are converted to USD when possible; formatted for readability.")

    metric_choice = st.selectbox("Metric", ["Total Revenue", "Gross Profit", "Net Income", "Key Ratios"])
    if metric_choice != "Key Ratios":
        # gather series across firms
        frames = {}
        candidates_map = {
            "Total Revenue": ["Total Revenue", "TotalRevenue", "Revenue"],
            "Gross Profit": ["Gross Profit", "GrossProfit", "Gross profit"],
            "Net Income": ["Net Income", "NetIncome", "Net income"]
        }
        cands = candidates_map[metric_choice]
        currency_notes = {}
        for t, d in firms.items():
            fin = d['financials']
            if fin is None or fin.empty:
                continue
            lbl = find_label(fin, cands)
            if lbl:
                converted = convert_df_to_usd(fin.loc[lbl], t)
                # fin.loc[lbl] may be a Series indexed by periods; ensure we convert to Series
                frames[t] = converted
                if fx_map.get(t, {}).get("orig_currency") and fx_map.get(t, {}).get("fx_to_usd") and fx_map[t]["orig_currency"] != "USD":
                    currency_notes[t] = f"Converted from {fx_map[t]['orig_currency']} to USD @ {fx_map[t]['fx_to_usd']:.4f}"
        if not frames:
            st.info("No data available for this metric.")
        else:
            df = pd.DataFrame(frames)
            st.write("### Table")
            st.write(pd.DataFrame(df))  # raw numeric table
            # styled HTML table for human readable
            st.markdown("### Formatted (human readable)")
            st.write(styled_money_table(df), unsafe_allow_html=True)
            download_button_for_df(df, f"{metric_choice}_table")
            # interactive plot
            df_plot = df.dropna(how='all', axis=1)
            if not df_plot.empty:
                fig = px.line(df_plot.reset_index().melt(id_vars=df_plot.index.name or df_plot.index.names),
                              x="index", y="value", color="variable",
                              labels={"index": "Period", "value": "USD"},
                              title=f"{metric_choice} (USD) Over Time")
                # Because melt above might be clumsy, create more direct fig:
                fig = go.Figure()
                for col in df_plot.columns:
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], mode="lines+markers", name=col,
                                             hovertemplate="%{x}<br>%{y:,.0f}"))
                fig.update_layout(yaxis_title="USD", xaxis_title="Period", template="plotly_white", height=450)
                st.plotly_chart(fig, use_container_width=True)

            # show currency note if any conversion happened
            if currency_notes:
                st.markdown("**Currency conversion notes:**")
                for k, v in currency_notes.items():
                    st.markdown(f"- {k}: {v}")

    else:
        # Key Ratios
        st.write("### Key Financial Ratios")
        if event := ratios := (lambda: None)():
            pass
        if 'ratios' not in locals():
            pass
        # compute ratios on the fly (reuse compute function from before)
        ratios_df = None
        try:
            # compute gross/operating/net margins, ROE, D/E
            # Reuse compute_ratios-like logic but simpler and inline for display
            parts = []
            # build dict per metric
            def build_metric(metric):
                frames = {}
                if metric == "Gross Profit Margin":
                    for t, d in firms.items():
                        fin = d['financials']
                        if fin is None or fin.empty:
                            continue
                        gp = find_label(fin, ["Gross Profit", "GrossProfit"])
                        rev = find_label(fin, ["Total Revenue", "Revenue"])
                        if gp and rev:
                            try:
                                series = (fin.loc[gp] / fin.loc[rev]).dropna()
                                frames[t] = series
                            except Exception:
                                continue
                if metric == "Operating Margin":
                    for t, d in firms.items():
                        fin = d['financials']
                        if fin is None or fin.empty:
                            continue
                        op = find_label(fin, ["Operating Income", "OperatingIncome"])
                        rev = find_label(fin, ["Total Revenue", "Revenue"])
                        if op and rev:
                            try:
                                frames[t] = (fin.loc[op] / fin.loc[rev]).dropna()
                            except Exception:
                                continue
                if metric == "Net Income Margin":
                    for t, d in firms.items():
                        fin = d['financials']
                        if fin is None or fin.empty:
                            continue
                        ni = find_label(fin, ["Net Income", "NetIncome"])
                        rev = find_label(fin, ["Total Revenue", "Revenue"])
                        if ni and rev:
                            try:
                                frames[t] = (fin.loc[ni] / fin.loc[rev]).dropna()
                            except Exception:
                                continue
                if metric == "Return on Equity":
                    frames = {}
                    for t, d in firms.items():
                        fin = d['financials']
                        bal = d['balance_sheet']
                        if fin is None or fin.empty or bal is None or bal.empty:
                            continue
                        ni = find_label(fin, ["Net Income", "NetIncome"])
                        eq = find_label(bal, ["Total Stockholder Equity", "Stockholders Equity", "Total Equity"])
                        if ni and eq:
                            try:
                                ni_s = fin.loc[ni]
                                eq_s = bal.loc[eq]
                                avg_eq = eq_s.rolling(2).mean()
                                roe = (ni_s / avg_eq).dropna()
                                if not roe.empty:
                                    frames[t] = roe
                            except Exception:
                                continue
                if metric == "Debt to Equity":
                    frames = {}
                    for t, d in firms.items():
                        bal = d['balance_sheet']
                        if bal is None or bal.empty:
                            continue
                        debt = find_label(bal, ["Total Debt", "Total liabilities", "Total Liab", "Long Term Debt", "Short Long Term Debt"])
                        eq = find_label(bal, ["Total Stockholder Equity", "Stockholders Equity", "Total Equity"])
                        if debt and eq:
                            try:
                                frames[t] = (bal.loc[debt] / bal.loc[eq]).replace([np.inf, -np.inf], np.nan).dropna()
                            except Exception:
                                continue
                return frames

            metrics = ["Gross Profit Margin", "Operating Margin", "Net Income Margin", "Return on Equity", "Debt to Equity"]
            pieces = []
            for m in metrics:
                frames = build_metric(m)
                if frames:
                    dfm = pd.DataFrame(frames)
                    # for ratios, do not convert currency; ratios are unitless
                    dfm.columns = pd.MultiIndex.from_product([[m], dfm.columns])
                    pieces.append(dfm)
            if pieces:
                ratios_df = pd.concat(pieces, axis=1)
        except Exception:
            ratios_df = pd.DataFrame()

        if ratios_df is None or ratios_df.empty:
            st.info("No ratio data available for selected tickers.")
        else:
            # show raw table and styled percent table
            st.write("#### Raw Ratios")
            st.dataframe(ratios_df)
            # Build a percent formatted view (convert to 0..1 expected)
            # Create a flat table for styling: columns = (metric, ticker) -> convert to single level for style
            flat = ratios_df.copy()
            # apply percent format in styled table
            st.markdown("#### Formatted Ratios")
            st.write(styled_ratio_table(flat), unsafe_allow_html=True)
            download_button_for_df(ratios_df, "ratios_table")

# Balance Sheet section
elif section == "Balance Sheet":
    st.title("Balance Sheet")
    st.markdown("Choose an item to view. Values are converted to USD where possible.")

    item_choice = st.selectbox("Item", ["Total Assets", "Total Liabilities", "Total Equity", "Market Cap / Share"])
    if item_choice == "Market Cap / Share":
        if market_cap_df is None or market_cap_df.empty:
            st.info("Market cap not available for these tickers.")
        else:
            st.write("### Market Capitalization (USD)")
            st.dataframe(market_cap_df.assign(market_cap=lambda d: d['market_cap'].apply(human_format_number)))
            # bar chart
            fig = px.bar(market_cap_df.reset_index(), x="index", y="market_cap", hover_data=["market_share_%"],
                         labels={"index": "Ticker", "market_cap": "Market Cap (USD)"},
                         title="Market Capitalization (USD)")
            fig.update_traces(hovertemplate="%{x}<br>%{y:,.0f} USD<br>Share: %{customdata[0]:.2f}%")
            st.plotly_chart(fig, use_container_width=True)
            download_button_for_df(market_cap_df, "market_cap")
    else:
        candidates_map = {
            "Total Assets": ["Total Assets", "Assets"],
            "Total Liabilities": ["Total Liabilities", "Total Liab", "Total liabilities"],
            "Total Equity": ["Total Stockholder Equity", "Stockholders Equity", "Total Equity"]
        }
        cands = candidates_map[item_choice]
        frames = {}
        for t, d in firms.items():
            bal = d['balance_sheet']
            if bal is None or bal.empty:
                continue
            lbl = find_label(bal, cands)
            if lbl:
                conv = convert_df_to_usd(bal.loc[lbl], t)
                frames[t] = conv
        if not frames:
            st.info("No balance sheet data available for these tickers.")
        else:
            df = pd.DataFrame(frames)
            st.write("Raw table:")
            st.dataframe(df)
            st.markdown("Formatted:")
            st.write(styled_money_table(df), unsafe_allow_html=True)
            download_button_for_df(df, f"{item_choice}_table")
            # interactive plot
            fig = go.Figure()
            for col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines+markers", name=col,
                                         hovertemplate="%{x}<br>%{y:,.0f} USD"))
            fig.update_layout(title=f"{item_choice} Over Time (USD)", xaxis_title="Period", yaxis_title="USD", template="plotly_white", height=450)
            st.plotly_chart(fig, use_container_width=True)

# Cash Flow section
elif section == "Cash Flow":
    st.title("Cash Flow")
    st.markdown("Choose the cash flow metric to inspect. Values converted to USD when possible.")

    cf_choice = st.selectbox("Cash flow metric", ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow", "Cash Flow Table"])
    if cf_choice == "Cash Flow Table":
        # build a consolidated cash flow table
        frames = {}
        for t, d in firms.items():
            cf = d['cashflow']
            if cf is None or cf.empty:
                continue
            # try to take common keys
            candidates = ["Operating Cash Flow", "Net Cash Provided by Operating Activities", "Net Cash Provided by (Used in) Operating Activities"]
            found = None
            for c in candidates:
                lbl = find_label(cf, [c])
                if lbl:
                    found = lbl
                    break
            if found:
                frames[t] = convert_df_to_usd(cf.loc[found], t)
        if not frames:
            st.info("Not enough cashflow data available.")
        else:
            df = pd.DataFrame(frames)
            st.dataframe(df)
            st.markdown("Formatted:")
            st.write(styled_money_table(df), unsafe_allow_html=True)
            download_button_for_df(df, "cash_flow_table")
    else:
        # single metric
        candidate_groups = {
            "Operating Cash Flow": ["Operating Cash Flow", "Net Cash Provided by Operating Activities", "Net Cash Provided by (Used in) Operating Activities"],
            "Investing Cash Flow": ["Investing Cash Flow", "Net Cash Used in Investing Activities", "Capital Expenditures"],
            "Financing Cash Flow": ["Financing Cash Flow", "Net Cash Provided by (Used in) Financing Activities"]
        }
        cands = candidate_groups[cf_choice]
        frames = {}
        for t, d in firms.items():
            cf = d['cashflow']
            if cf is None or cf.empty:
                continue
            lbl = find_label(cf, cands)
            if lbl:
                frames[t] = convert_df_to_usd(cf.loc[lbl], t)
        if not frames:
            st.info("No data available for this cash flow metric.")
        else:
            df = pd.DataFrame(frames)
            st.write("Raw table")
            st.dataframe(df)
            st.markdown("Formatted:")
            st.write(styled_money_table(df), unsafe_allow_html=True)
            # interactive plot
            fig = go.Figure()
            for col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines+markers", name=col,
                                         hovertemplate="%{x}<br>%{y:,.0f} USD"))
            fig.update_layout(title=f"{cf_choice} Over Time (USD)", xaxis_title="Period", yaxis_title="USD", template="plotly_white", height=450)
            st.plotly_chart(fig, use_container_width=True)
            download_button_for_df(df, f"{cf_choice}_table")

# Event Study section
elif section == "Event Study":
    st.title("Event Study")
    st.markdown("Interactive event study analysis. Choose an event to view abnormal returns and CAR (market-model).")

    if not event_results:
        st.info("No computed event studies available (insufficient data for the provided events/tickers).")
    else:
        # show event list
        ev_df = pd.DataFrame([{"key": k, "ticker": k.split("_")[0], "date": k.split("_")[1], "description": v["description"]} for k, v in event_results.items()])
        st.write("### Available events")
        st.dataframe(ev_df)

        pick = st.selectbox("Select event", list(event_results.keys()))
        if pick:
            ev = event_results[pick]
            desc = ev.get("description", "")
            st.subheader(f"{pick} — {desc}")
            abnormal = ev["abnormal"]
            car = ev["car"]
            ev_returns = ev["event_returns"]

            # Narrative: quick auto-interpretation
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
                interpretation = "Market reaction appears mixed."
                if "Beat" in desc and car_end > 0:
                    interpretation = "Positive CAR suggests favorable market reaction."
                elif "Miss" in desc and car_end < 0:
                    interpretation = "Negative CAR suggests unfavorable reaction."
                st.markdown(f"**Event window CAR trend:** {trend} (start ≈ {car_start:.4f}, end ≈ {car_end:.4f})")
                st.markdown(f"**Interpretation:** {interpretation}")

            # Plot interactive: abnormal and CAR
            fig = go.Figure()
            if abnormal is not None:
                fig.add_trace(go.Bar(x=abnormal.index, y=abnormal.values, name="Abnormal Return", marker_color="indianred",
                                     hovertemplate="%{x}<br>%{y:.4f}"))
            if car is not None:
                fig.add_trace(go.Scatter(x=car.index, y=car.values, mode="lines+markers", name="CAR", marker_color="royalblue",
                                         hovertemplate="%{x}<br>%{y:.4f}"))
            fig.update_layout(title=f"Event Study: {pick} — {desc}", xaxis_title="Date", yaxis_title="Return", template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Event window returns (raw)**")
            st.dataframe(ev_returns)
            download_button_for_df(ev_returns, f"{pick}_event_returns")
            # show CAR table
            if car is not None and not car.empty:
                car_df = car.to_frame(name="CAR")
                st.markdown("**Cumulative Abnormal Returns (CAR) table**")
                st.dataframe(car_df)
                download_button_for_df(car_df, f"{pick}_CAR")

        # Grouped CAR comparison
        st.markdown("---")
        st.subheader("Compare CAR across events of same type")
        # group by description
        grp_map = {}
        for k, v in event_results.items():
            desc = v['description'] or "Other"
            grp_map.setdefault(desc, []).append((k, v))
        group_sel = st.selectbox("Choose event description group", list(grp_map.keys()))
        if group_sel:
            items = grp_map[group_sel]
            # build CAR dataframe aligned on dates (outer join)
            car_combined = pd.DataFrame()
            for k, v in items:
                c = v.get("car")
                if c is not None:
                    firm = k.split("_")[0]
                    car_combined[firm] = c
            if car_combined.empty:
                st.info("No CAR data for this group.")
            else:
                st.dataframe(car_combined)
                fig2 = go.Figure()
                for col in car_combined.columns:
                    fig2.add_trace(go.Scatter(x=car_combined.index, y=car_combined[col], mode="lines+markers", name=col,
                                              hovertemplate="%{x}<br>%{y:.4f}"))
                fig2.update_layout(title=f"CAR Comparison: {group_sel}", xaxis_title="Date", yaxis_title="CAR", template="plotly_white", height=500)
                st.plotly_chart(fig2, use_container_width=True)
                download_button_for_df(car_combined, f"CAR_comparison_{group_sel.replace(' ','_')}")

# Footer note
st.sidebar.markdown("---")
st.sidebar.markdown("Built from your `15_cf_activities` logic. Charts are interactive. Ask me to add more metrics or export formats.")


