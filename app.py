# app.py
# Full interactive corporate-finance dashboard with multi-ticker multi-event event studies
# Requirements: streamlit, yfinance, pandas, numpy, plotly, statsmodels, pytz

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import datetime
import pytz
from typing import Dict, Tuple, Optional

st.set_page_config(page_title="Corporate Finance Dashboard (Full)", layout="wide")

# -------------------------
# Helper formatters & utils
# -------------------------
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
    if absx >= 1_000_000:
        return f"{x/1_000_000:.2f} M"
    if absx >= 1_000:
        return f"{x:,.0f}"
    return f"{x:.2f}"

def fmt_percent(x):
    if pd.isna(x):
        return ""
    try:
        return f"{100*x:.2f} %"
    except Exception:
        return str(x)

# -------------------------
# Caching data and FX calls
# -------------------------
@st.cache_data(show_spinner=False)
def get_fx_rate_to_usd(currency: Optional[str]) -> float:
    """
    Returns multiplier to convert from `currency` to USD.
    Falls back to 1.0 if unknown.
    """
    if not currency:
        return 1.0
    cur = currency.upper()
    if cur == "USD":
        return 1.0
    # Try XXUSD=X
    try:
        pair = f"{cur}USD=X"
        df = yf.Ticker(pair).history(period="7d")
        if not df.empty and "Close" in df:
            val = df["Close"].iloc[-1]
            if pd.notna(val) and val != 0:
                return float(val)
    except Exception:
        pass
    # Try USDXX=X (invert)
    try:
        pair = f"USD{cur}=X"
        df = yf.Ticker(pair).history(period="7d")
        if not df.empty and "Close" in df:
            val = df["Close"].iloc[-1]
            if pd.notna(val) and val != 0:
                return float(1.0 / val)
    except Exception:
        pass
    return 1.0

@st.cache_data(show_spinner=False)
def get_data_for_tickers(tickers: list) -> Dict[str, dict]:
    """
    Returns a dict keyed by ticker with:
    financials, balance_sheet, cashflow, price_close, stock_rets, info, currency
    """
    out = {}
    for t in tickers:
        t = t.upper()
        try:
            tk = yf.Ticker(t)
            info = {}
            try:
                info = tk.info or {}
            except Exception:
                info = {}
            fin = tk.financials if hasattr(tk, "financials") else pd.DataFrame()
            bal = tk.balance_sheet if hasattr(tk, "balance_sheet") else pd.DataFrame()
            cf = tk.cashflow if hasattr(tk, "cashflow") else pd.DataFrame()
            hist = tk.history(period="1y")
            price_close = hist["Close"] if "Close" in hist else pd.Series(dtype=float)
            stock_rets = price_close.pct_change().dropna() if not price_close.empty else pd.Series(dtype=float)
            currency = info.get("currency", "USD")
            out[t] = {
                "financials": fin,
                "balance_sheet": bal,
                "cashflow": cf,
                "price_close": price_close,
                "stock_rets": stock_rets,
                "info": info,
                "currency": currency
            }
        except Exception:
            out[t] = {
                "financials": pd.DataFrame(),
                "balance_sheet": pd.DataFrame(),
                "cashflow": pd.DataFrame(),
                "price_close": pd.Series(dtype=float),
                "stock_rets": pd.Series(dtype=float),
                "info": {},
                "currency": "USD"
            }
    return out

# -------------------------
# Robust label finder
# -------------------------
def find_label(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """
    Find first candidate present in df.index (case-insensitive, substring allowed).
    Returns the actual index label if found.
    """
    if df is None or df.empty:
        return None
    idx = [str(i) for i in df.index]
    for c in candidates:
        # exact match prefers
        for i in idx:
            if i == c:
                return i
    for c in candidates:
        for i in idx:
            if c.lower() in i.lower():
                return i
    return None

# -------------------------
# Functions to convert & style tables
# -------------------------
def convert_series_to_usd(series: pd.Series, rate: float) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="coerce") * rate
    except Exception:
        return series

def styled_money_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<i>No data</i>"
    # attempt numeric conversion
    d = df.copy()
    for col in d.columns:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    sty = d.style.format(human_format_number).set_table_attributes('style="width:100%"')
    return sty.to_html()

def styled_ratio_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<i>No data</i>"
    d = df.copy()
    for col in d.columns:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    sty = d.style.format("{:.2%}").background_gradient(cmap="RdYlGn", axis=0).set_table_attributes('style="width:100%"')
    return sty.to_html()

def download_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")

# -------------------------
# Event study logic
# -------------------------
def perform_event_study(firm_ticker: str, market_ticker: str, event_date: datetime.datetime,
                        event_window_days: int = 5, estimation_window_days: int = 200) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.DatetimeIndex]]:
    """
    Returns (abnormal_returns, car, event_index) or (None, None, None) on fail.
    """
    try:
        nyc = pytz.timezone("America/New_York")
        if event_date.tzinfo is None:
            event_date = nyc.localize(event_date)

        estimation_end = event_date - datetime.timedelta(days=event_window_days + 1)
        estimation_start = estimation_end - datetime.timedelta(days=estimation_window_days)
        event_start = event_date - datetime.timedelta(days=event_window_days)
        event_end = event_date + datetime.timedelta(days=event_window_days)

        firm_hist = yf.Ticker(firm_ticker).history(start=estimation_start, end=event_end)
        market_hist = yf.Ticker(market_ticker).history(start=estimation_start, end=event_end)

        if firm_hist.empty or market_hist.empty or 'Close' not in firm_hist or 'Close' not in market_hist:
            return None, None, None

        firm_close = firm_hist['Close'].dropna()
        market_close = market_hist['Close'].dropna()

        firm_ret = firm_close.pct_change().dropna()
        market_ret = market_close.pct_change().dropna()

        returns_df = pd.DataFrame({firm_ticker: firm_ret, market_ticker: market_ret}).dropna()
        if returns_df.empty:
            return None, None, None

        est_mask = (returns_df.index >= estimation_start) & (returns_df.index <= estimation_end)
        ev_mask = (returns_df.index >= event_start) & (returns_df.index <= event_end)

        estimation_returns = returns_df.loc[est_mask]
        event_returns = returns_df.loc[ev_mask]

        if estimation_returns.empty or event_returns.empty:
            return None, None, None

        X = sm.add_constant(estimation_returns[market_ticker])
        model = sm.OLS(estimation_returns[firm_ticker], X).fit()
        alpha = float(model.params.get('const', 0.0))
        beta = float(model.params.get(market_ticker, 0.0))

        estimated_normal = alpha + beta * event_returns[market_ticker]
        abnormal = event_returns[firm_ticker] - estimated_normal
        car = abnormal.cumsum()

        return abnormal, car, event_returns.index

    except Exception:
        return None, None, None

# -------------------------
# Hardcoded event list (from earlier file)
# -------------------------
EVENT_LIST = [
    {"firm_ticker": "LMT", "event_date": "2024-10-22", "event_description": "Q3 2024 Earnings Beat and Raised Guidance"},
    {"firm_ticker": "BA",  "event_date": "2024-10-23", "event_description": "Q3 2024 Earnings Miss and Production Delays Announced"},
    {"firm_ticker": "RTX", "event_date": "2024-10-24", "event_description": "Q3 2024 Earnings In Line with Expectations"},
    {"firm_ticker": "NOC", "event_date": "2024-10-24", "event_description": "Q3 2024 Earnings Miss Due to Supply Chain Issues"},
    {"firm_ticker": "GD",  "event_date": "2024-10-23", "event_description": "Q3 2024 Earnings Beat on Strong Demand"},
    {"firm_ticker":"BAESY","event_date":"2024-07-25","event_description":"Half Year 2024 Results Show Increased Order Backlog"}
]

# -------------------------
# Sidebar inputs
# -------------------------
st.sidebar.title("Controls")
main_ticker = st.sidebar.text_input("Main ticker", "LMT").upper()
competitors = st.sidebar.text_input("Competitors (comma-separated)", "BA, RTX, NOC, GD, BAESY")
competitor_tickers = [t.strip().upper() for t in competitors.split(",") if t.strip()]
all_tickers = [main_ticker] + competitor_tickers
section = st.sidebar.radio("Select section", ["Income Statement", "Balance Sheet", "Cash Flow", "Event Study"])
st.sidebar.markdown("---")
st.sidebar.info("Tip: add more tickers in the competitors field (comma-separated).")
st.sidebar.markdown("---")

# -------------------------
# Load data & FX rates (cached)
# -------------------------
with st.spinner("Fetching financials & FX..."):
    firms = get_data_for_tickers(all_tickers)
    fx_map = {}
    for t in all_tickers:
        cur = firms[t].get("currency", "USD") or firms[t].get("info", {}).get("currency", "USD")
        fx_map[t] = {"orig_currency": cur, "fx_to_usd": get_fx_rate_to_usd(cur)}

# -------------------------
# Market cap helper
# -------------------------
def build_market_caps(firms: dict) -> pd.DataFrame:
    caps = {}
    for t, d in firms.items():
        info = d.get("info", {}) or {}
        cap = info.get("marketCap")
        if cap is None:
            so = info.get("sharesOutstanding")
            price = None
            try:
                price = d.get("price_close").iloc[-1]
            except Exception:
                price = None
            if so and price is not None:
                cap = so * price
        if cap is None:
            caps[t] = np.nan
        else:
            rate = fx_map.get(t, {}).get("fx_to_usd", 1.0)
            caps[t] = cap * rate
    df = pd.DataFrame.from_dict(caps, orient="index", columns=["market_cap"]).dropna()
    if not df.empty:
        df["market_share_%"] = df["market_cap"] / df["market_cap"].sum() * 100
    return df

market_cap_df = build_market_caps(firms)

# -------------------------
# Precompute event studies for all events and all tickers (cached)
# -------------------------
@st.cache_data(show_spinner=False)
def compute_all_event_studies(events: list, tickers: list) -> Dict[str, dict]:
    """
    Returns dict keyed by f"{ticker}_{event_date}" -> {"abnormal": Series, "car": Series, "index": DatetimeIndex, "description": str}
    """
    results = {}
    for ev in events:
        edate = ev["event_date"]
        desc = ev.get("event_description", "")
        # run for each ticker
        for t in tickers:
            try:
                dt = datetime.datetime.strptime(edate, "%Y-%m-%d")
                abnormal, car, idx = perform_event_study(t, "^GSPC", dt)
                key = f"{t}_{edate}"
                if abnormal is not None and car is not None:
                    results[key] = {"abnormal": abnormal, "car": car, "index": idx, "description": desc}
            except Exception:
                continue
    return results

with st.spinner("Computing event studies for all tickers/events..."):
    all_event_results = compute_all_event_studies(EVENT_LIST, all_tickers)

# -------------------------
# UI: Sections
# -------------------------
def download_df(df: pd.DataFrame, label: str):
    if df is None or df.empty:
        st.info("No data to download")
        return
    st.download_button(label, download_csv_bytes(df), file_name=f"{label.replace(' ','_')}.csv", mime="text/csv")

# Income statement
if section == "Income Statement":
    st.title("Income Statement")
    st.write("Numbers converted to USD where available. Use the dropdown to select the metric you want to view.")
    metric = st.selectbox("Metric", ["Total Revenue", "Gross Profit", "Net Income", "Key Ratios"])
    cand_map = {
        "Total Revenue": ["Total Revenue", "Revenue", "TotalRevenue"],
        "Gross Profit": ["Gross Profit", "GrossProfit"],
        "Net Income": ["Net Income", "NetIncome"]
    }
    if metric != "Key Ratios":
        candidates = cand_map[metric]
        series_map = {}
        conv_notes = {}
        for t, d in firms.items():
            fin = d.get("financials")
            if fin is None or fin.empty:
                continue
            lbl = find_label(fin, candidates)
            if lbl:
                s = fin.loc[lbl]
                # convert to USD using fx_map[t]
                rate = fx_map.get(t, {}).get("fx_to_usd", 1.0)
                s_usd = convert_series_to_usd(s, rate)
                # name the series by ticker
                series_map[t] = s_usd
                if fx_map[t]["orig_currency"] != "USD":
                    conv_notes[t] = f"Converted from {fx_map[t]['orig_currency']} to USD @ {fx_map[t]['fx_to_usd']:.4f}"
        if not series_map:
            st.info("No data available for this metric.")
        else:
            df = pd.DataFrame(series_map)
            st.subheader("Raw table")
            st.dataframe(df)
            st.subheader("Formatted")
            st.write(styled_money_html(df), unsafe_allow_html=True)
            download_df(df, f"{metric}_table")
            # interactive plot
            fig = go.Figure()
            for col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines+markers", name=col,
                                         hovertemplate="%{x}<br>%{y:,.0f} USD"))
            fig.update_layout(title=f"{metric} (USD) Over Time", xaxis_title="Period", yaxis_title="USD", template="plotly_white", height=450)
            st.plotly_chart(fig, use_container_width=True)
            if conv_notes:
                st.markdown("**Currency conversion notes:**")
                for k, v in conv_notes.items():
                    st.markdown(f"- {k}: {v}")

    else:
        # Key ratios
        st.subheader("Key Ratios")
        ratio_pieces = {}
        metrics = ["Gross Profit Margin", "Operating Margin", "Net Income Margin", "Return on Equity", "Debt to Equity"]
        for t, d in firms.items():
            fin = d.get("financials")
            bal = d.get("balance_sheet")
            if fin is None or fin.empty:
                continue
            piece = {}
            # Gross margin
            gp_lbl = find_label(fin, ["Gross Profit", "GrossProfit"])
            rev_lbl = find_label(fin, ["Total Revenue", "Revenue"])
            if gp_lbl and rev_lbl:
                try:
                    piece["Gross Margin"] = (fin.loc[gp_lbl] / fin.loc[rev_lbl]).mean()
                except Exception:
                    pass
            # Operating margin
            op_lbl = find_label(fin, ["Operating Income", "OperatingIncome"])
            if op_lbl and rev_lbl:
                try:
                    piece["Operating Margin"] = (fin.loc[op_lbl] / fin.loc[rev_lbl]).mean()
                except Exception:
                    pass
            # Net margin
            ni_lbl = find_label(fin, ["Net Income", "NetIncome"])
            if ni_lbl and rev_lbl:
                try:
                    piece["Net Margin"] = (fin.loc[ni_lbl] / fin.loc[rev_lbl]).mean()
                except Exception:
                    pass
            # ROE
            if ni_lbl and bal is not None and not bal.empty:
                eq_lbl = find_label(bal, ["Total Stockholder Equity", "Total Equity", "Stockholders Equity"])
                if eq_lbl:
                    try:
                        piece["ROE"] = (fin.loc[ni_lbl] / bal.loc[eq_lbl]).mean()
                    except Exception:
                        pass
            # D/E
            if bal is not None and not bal.empty:
                debt_lbl = find_label(bal, ["Total Debt", "Total Liab", "Long Term Debt"])
                if debt_lbl and eq_lbl:
                    try:
                        piece["D/E"] = (bal.loc[debt_lbl] / bal.loc[eq_lbl]).mean()
                    except Exception:
                        pass
            if piece:
                ratio_pieces[t] = piece
        if not ratio_pieces:
            st.info("No ratio data available.")
        else:
            ratio_df = pd.DataFrame(ratio_pieces).T
            st.subheader("Raw")
            st.dataframe(ratio_df)
            st.subheader("Formatted")
            st.write(styled_ratio_html(ratio_df), unsafe_allow_html=True)
            download_df(ratio_df, "key_ratios")

# Balance sheet
elif section == "Balance Sheet":
    st.title("Balance Sheet")
    choice = st.selectbox("Item", ["Total Assets", "Total Liabilities", "Total Equity", "Market Cap / Market Share"])
    if choice == "Market Cap / Market Share":
        if market_cap_df is None or market_cap_df.empty:
            st.info("Market cap not available.")
        else:
            st.subheader("Market Capitalization (USD)")
            df2 = market_cap_df.copy()
            df2["market_cap_formatted"] = df2["market_cap"].apply(human_format_number)
            st.dataframe(df2[["market_cap_formatted", "market_share_%"]].rename(columns={"market_cap_formatted":"Market Cap (formatted)","market_share_%":"Market Share (%)"}))
            # bar
            fig = px.bar(market_cap_df.reset_index().rename(columns={"index":"Ticker"}), x="Ticker", y="market_cap",
                         hover_data=["market_share_%"], labels={"market_cap":"Market Cap (USD)"}, title="Market Cap (USD)")
            st.plotly_chart(fig, use_container_width=True)
            download_df(market_cap_df, "market_cap")
    else:
        cand_map = {
            "Total Assets": ["Total Assets", "Assets"],
            "Total Liabilities": ["Total Liabilities", "Total Liab"],
            "Total Equity": ["Total Stockholder Equity", "Total Equity", "Stockholders Equity"]
        }
        candidates = cand_map[choice]
        frames = {}
        conv_notes = {}
        for t, d in firms.items():
            bs = d.get("balance_sheet")
            if bs is None or bs.empty:
                continue
            lbl = find_label(bs, candidates)
            if lbl:
                s = bs.loc[lbl]
                rate = fx_map.get(t, {}).get("fx_to_usd", 1.0)
                s_usd = convert_series_to_usd(s, rate)
                frames[t] = s_usd
                if fx_map[t]["orig_currency"] != "USD":
                    conv_notes[t] = f"Converted from {fx_map[t]['orig_currency']} to USD @ {fx_map[t]['fx_to_usd']:.4f}"
        if not frames:
            st.info("No data available for that item.")
        else:
            df = pd.DataFrame(frames)
            st.subheader("Raw")
            st.dataframe(df)
            st.subheader("Formatted")
            st.write(styled_money_html(df), unsafe_allow_html=True)
            download_df(df, f"{choice}_table")
            fig = go.Figure()
            for col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines+markers", name=col",
                                         hovertemplate="%{x}<br>%{y:,.0f} USD"))
            fig.update_layout(title=f"{choice} Over Time (USD)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            if conv_notes:
                st.markdown("**Currency conversion notes:**")
                for k, v in conv_notes.items():
                    st.markdown(f"- {k}: {v}")

# Cash flow
elif section == "Cash Flow":
    st.title("Cash Flow")
    choice = st.selectbox("Metric", ["Operating Cash Flow", "Investing Cash Flow", "Financing Cash Flow", "Cash Flow Table"])
    if choice == "Cash Flow Table":
        frames = {}
        for t, d in firms.items():
            cf = d.get("cashflow")
            if cf is None or cf.empty:
                continue
            # try several candidates
            for cand in ["Operating Cash Flow", "Net Cash Provided by Operating Activities", "Net Cash Provided by (Used in) Operating Activities"]:
                if find_label(cf, [cand]):
                    lbl = find_label(cf, [cand])
                    frames[t] = convert_series_to_usd(cf.loc[lbl], fx_map[t]["fx_to_usd"])
                    break
        if not frames:
            st.info("No consolidated cash flow data available.")
        else:
            df = pd.DataFrame(frames)
            st.dataframe(df)
            st.write(styled_money_html(df), unsafe_allow_html=True)
            download_df(df, "cash_flow_table")
    else:
        cand_groups = {
            "Operating Cash Flow": ["Operating Cash Flow", "Net Cash Provided by Operating Activities"],
            "Investing Cash Flow": ["Investing Cash Flow", "Net Cash Used in Investing Activities", "Capital Expenditures"],
            "Financing Cash Flow": ["Financing Cash Flow", "Net Cash Provided by (Used in) Financing Activities"]
        }
        candidates = cand_groups[choice]
        frames = {}
        for t, d in firms.items():
            cf = d.get("cashflow")
            if cf is None or cf.empty:
                continue
            lbl = find_label(cf, candidates)
            if lbl:
                frames[t] = convert_series_to_usd(cf.loc[lbl], fx_map[t]["fx_to_usd"])
        if not frames:
            st.info("No data available.")
        else:
            df = pd.DataFrame(frames)
            st.dataframe(df)
            st.write(styled_money_html(df), unsafe_allow_html=True)
            download_df(df, f"{choice}_table")
            fig = go.Figure()
            for col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines+markers", name=col,
                                         hovertemplate="%{x}<br>%{y:,.0f} USD"))
            fig.update_layout(title=f"{choice} Over Time (USD)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

# Event Study
elif section == "Event Study":
    st.title("Event Study (multi-ticker, multi-event)")
    st.markdown("This section runs event studies for each hardcoded event and each ticker. Scroll to see each company's events, then a cross-company comparison for each event.")

    # 1) Per-ticker event studies
    for ticker in all_tickers:
        st.header(f"{ticker} — Event Studies")
        # find events that mention this ticker in file (if present), but we also want to show all events regardless
        # We'll show all hardcoded events; user asked to run each event for each ticker
        for ev in EVENT_LIST:
            ev_date = ev["event_date"]
            ev_desc = ev.get("event_description", "")
            key = f"{ticker}_{ev_date}"
            res = all_event_results.get(key)
            st.subheader(f"{ev_date} — {ev_desc}")
            if res is None:
                st.info(f"No event-study data for {ticker} around {ev_date}. (data insufficient or market data missing)")
                continue
            abnormal = res["abnormal"]
            car = res["car"]
            idx = res["index"]
            # quick interpretation snippet
            if car is not None and not car.empty:
                car_start = car.iloc[0]
                car_end = car.iloc[-1]
                trend = "mixed"
                if car_end > car_start and (car.diff().dropna() >= 0).all():
                    trend = "steady increase"
                elif car_end < car_start and (car.diff().dropna() <= 0).all():
                    trend = "steady decline"
                elif car_end > car_start:
                    trend = "overall increase"
                elif car_end < car_start:
                    trend = "overall decline"
                st.markdown(f"**Quick interpretation:** CAR shows *{trend}* (start ≈ {car_start:.4f}, end ≈ {car_end:.4f}).")
            # interactive chart
            fig = go.Figure()
            if abnormal is not None:
                fig.add_trace(go.Bar(x=idx, y=abnormal.values, name="Abnormal Return", marker_color="indianred",
                                     hovertemplate="%{x}<br>Abnormal: %{y:.4f}"))
            if car is not None:
                fig.add_trace(go.Scatter(x=idx, y=car.values, name="CAR", mode="lines+markers", marker_color="royalblue",
                                         hovertemplate="%{x}<br>CAR: %{y:.4f}"))
            fig.update_layout(title=f"{ticker} — {ev_date} — Abnormal & CAR", xaxis_title="Date", yaxis_title="Return", template="plotly_white", height=450)
            st.plotly_chart(fig, use_container_width=True)
            # show raw event returns table if desired
            # Build event returns table if available by recomputing or extracting from abnormal.index
            # For simplicity, reconstruct event returns (firm & market) for display
            try:
                dt = datetime.datetime.strptime(ev_date, "%Y-%m-%d")
                est_end = dt - datetime.timedelta(days=6)
                ev_start = dt - datetime.timedelta(days=5)
                ev_end = dt + datetime.timedelta(days=5)
                firm_hist = yf.Ticker(ticker).history(start=ev_start - datetime.timedelta(days=1), end=ev_end + datetime.timedelta(days=1))
                market_hist = yf.Ticker("^GSPC").history(start=ev_start - datetime.timedelta(days=1), end=ev_end + datetime.timedelta(days=1))
                if not firm_hist.empty and not market_hist.empty:
                    fr = firm_hist["Close"].pct_change().dropna()
                    mr = market_hist["Close"].pct_change().dropna()
                    ev_df = pd.DataFrame({ticker: fr, "Market": mr}).dropna().loc[ev_start:ev_end]
                    if not ev_df.empty:
                        st.markdown("Event window returns (firm vs market):")
                        st.dataframe(ev_df)
                        download_df(ev_df, f"{ticker}_{ev_date}_returns")
            except Exception:
                pass

            st.markdown("---")

    # 2) Cross-company comparison per event
    st.header("Cross-company comparison (same event across tickers)")
    # Build mapping by event (unique by date+description)
    events_grouped = {}
    for ev in EVENT_LIST:
        key = f"{ev['event_date']}|{ev.get('event_description','')}"
        events_grouped.setdefault(key, ev)

    for key, ev in events_grouped.items():
        ev_date = ev["event_date"]
        ev_desc = ev.get("event_description", "")
        st.subheader(f"{ev_date} — {ev_desc} — Cross-company comparison")
        # collect CARs and abnormal returns for all tickers for this event
        car_df = pd.DataFrame()
        abnormal_df = pd.DataFrame()
        for t in all_tickers:
            k = f"{t}_{ev_date}"
            res = all_event_results.get(k)
            if res is None:
                continue
            car = res["car"]
            abnormal = res["abnormal"]
            # rename series to ticker
            if car is not None and not car.empty:
                car_df[t] = car
            if abnormal is not None and not abnormal.empty:
                abnormal_df[t] = abnormal
        if car_df.empty and abnormal_df.empty:
            st.info("No cross-company data available for this event.")
            continue
        # display CAR comparison
        if not car_df.empty:
            st.markdown("Cumulative Abnormal Returns (CAR) comparison")
            st.dataframe(car_df)
            fig = go.Figure()
            for col in car_df.columns:
                fig.add_trace(go.Scatter(x=car_df.index, y=car_df[col], mode="lines+markers", name=col,
                                         hovertemplate="%{x}<br>%{y:.4f}"))
            fig.update_layout(title=f"CAR Comparison - {ev_date}", xaxis_title="Date", yaxis_title="CAR", template="plotly_white", height=450)
            st.plotly_chart(fig, use_container_width=True)
            download_df(car_df, f"CAR_comparison_{ev_date}")
        # display abnormal comparison
        if not abnormal_df.empty:
            st.markdown("Abnormal Returns comparison (daily)")
            st.dataframe(abnormal_df)
            fig2 = go.Figure()
            for col in abnormal_df.columns:
                fig2.add_trace(go.Bar(x=abnormal_df.index, y=abnormal_df[col], name=col, opacity=0.8))
            fig2.update_layout(barmode='group', title=f"Abnormal Returns Comparison - {ev_date}", xaxis_title="Date", yaxis_title="Abnormal Return", template="plotly_white", height=450)
            st.plotly_chart(fig2, use_container_width=True)
            download_df(abnormal_df, f"Abnormal_comparison_{ev_date}")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Built from combined `15_cf_activities` logic. Ask me to add filters, more events, or export options.")



