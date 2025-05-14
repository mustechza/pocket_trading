import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIG ---
st.set_page_config(layout="wide")
REFRESH_INTERVAL = 10  # seconds
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh")

# --- CONSTANTS ---
BINANCE_URL = "https://api.binance.us/api/v3/klines"
ASSETS = ["ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]

# --- SIDEBAR ---
with st.sidebar:
    uploaded_file = st.file_uploader("Upload historical data (CSV)", type=["csv"])
    selected_assets = st.multiselect("Select Assets", ASSETS, default=ASSETS[:2])
    selected_strategy = st.selectbox("Strategy", [
        "EMA Cross", "RSI Divergence", "MACD Cross", "Bollinger Band Bounce",
        "Stochastic Oscillator", "EMA + RSI Combined", "ML Model (Random Forest)"
    ])
    money_strategy = st.selectbox("Money Management", ["Flat", "Martingale", "Risk %"])
    risk_pct = st.slider("Risk % per Trade", 1, 10, value=2)
    balance_input = st.number_input("Initial Balance ($)", value=1000)

    # Indicator Parameters
    ema_short = st.number_input("EMA Short Period", 2, 50, value=5)
    ema_long = st.number_input("EMA Long Period", 5, 100, value=20)
    rsi_period = st.number_input("RSI Period", 5, 50, value=14)
    stoch_period = st.number_input("Stochastic Period", 5, 50, value=14)
    bb_period = st.number_input("Bollinger Band Period", 5, 50, value=20)

# --- TIMEZONE UTILITY ---
def to_gmt_plus2(ts):
    return ts + timedelta(hours=2)

# --- DATA FETCH ---
@st.cache_data(ttl=60)
def fetch_candles(symbol, interval="1m", limit=500):
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        response = requests.get(BINANCE_URL, params=params, timeout=10)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').apply(to_gmt_plus2)
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        return df
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {e}")
        return None

# --- TITLE (FIXED UNICODE BUG) ---
st.title("📈 Pocket Option Signals | Live + Backtest + Money Management")
