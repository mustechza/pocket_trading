import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from streamlit_autorefresh import st_autorefresh

# --- SETTINGS ---
REFRESH_INTERVAL = 10  # seconds
CANDLE_LIMIT = 100
TRADE_DURATION_MINUTES = 2
ASSETS = ["ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]
BINANCE_URL = "https://api.binance.com/api/v3/klines"

# --- HELPER FUNCTIONS ---

def fetch_candles(symbol, interval="1m", limit=500):
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        response = requests.get(BINANCE_URL, params=params, timeout=10)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        return df
    except Exception as e:
        st.error(f"Error fetching candles: {e}")
        return None

def calculate_indicators(df):
    df['EMA5'] = df['close'].ewm(span=5).mean()
    df['EMA20'] = df['close'].ewm(span=20).mean()
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

    df['BB_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['BB_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()

    df['Stochastic'] = (df['close'] - df['low'].rolling(14).min()) / \
                       (df['high'].rolling(14).max() - df['low'].rolling(14).min()) * 100
    return df

def format_signal(ts, text, price):
    start_time = ts.strftime("%H:%M")
    end_time = (ts + pd.Timedelta(minutes=TRADE_DURATION_MINUTES)).strftime("%H:%M")
    return f"{start_time} - {end_time}: {text} at ${price:.2f}"

def detect_signals(df, strategy):
    signals = []
    for i in range(1, len(df)):
        ts = df['timestamp'].iloc[i]
        price = df['close'].iloc[i]

        if strategy == "EMA Cross":
            if df['EMA5'].iloc[i-1] < df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] > df['EMA20'].iloc[i]:
                signals.append(format_signal(ts, "Buy Signal (EMA Cross UP)", price))
            elif df['EMA5'].iloc[i-1] > df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] < df['EMA20'].iloc[i]:
                signals.append(format_signal(ts, "Sell Signal (EMA Cross DOWN)", price))

        elif strategy == "RSI Divergence":
            rsi = df['RSI'].iloc[i]
            if rsi < 30:
                signals.append(format_signal(ts, "Potential Buy (RSI Oversold)", price))
            elif rsi > 70:
                signals.append(format_signal(ts, "Potential Sell (RSI Overbought)", price))

        elif strategy == "MACD Cross":
            if df['MACD'].iloc[i-1] < df['MACD_signal'].iloc[i-1] and df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
                signals.append(format_signal(ts, "Buy Signal (MACD Cross UP)", price))
            elif df['MACD'].iloc[i-1] > df['MACD_signal'].iloc[i-1] and df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
                signals.append(format_signal(ts, "Sell Signal (MACD Cross DOWN)", price))

        elif strategy == "Bollinger Band Bounce":
            if df['close'].iloc[i] < df['BB_lower'].iloc[i]:
                signals.append(format_signal(ts, "Buy Signal (BB Lower Bounce)", price))
            elif df['close'].iloc[i] > df['BB_upper'].iloc[i]:
                signals.append(format_signal(ts, "Sell Signal (BB Upper Bounce)", price))

        elif strategy == "Stochastic Oscillator":
            stochastic = df['Stochastic'].iloc[i]
            if stochastic < 20:
                signals.append(format_signal(ts, "Potential Buy (Stochastic Oversold)", price))
            elif stochastic > 80:
                signals.append(format_signal(ts, "Potential Sell (Stochastic Overbought)", price))

        elif strategy == "EMA + RSI Combined":
            buy_cross = df['EMA5'].iloc[i-1] < df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] > df['EMA20'].iloc[i]
            sell_cross = df['EMA5'].iloc[i-1] > df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] < df['EMA20'].iloc[i]
            rsi = df['RSI'].iloc[i]

            if buy_cross and rsi < 40:
                signals.append(format_signal(ts, "Buy Signal (EMA+RSI Combo)", price))
            elif sell_cross and rsi > 60:
                signals.append(format_signal(ts, "Sell Signal (EMA+RSI Combo)", price))
    return signals

# --- STREAMLIT APP ---
st.set_page_config(page_title="Live Crypto Signals with Trade Timing", layout="wide")
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh_key")

st.title("📈 Real-Time Crypto Signals (with Trade Timing)")

selected_assets = st.sidebar.multiselect("Select Assets", ASSETS, default=["ETHUSDT"])
selected_strategy = st.sidebar.selectbox("Select Strategy", ["EMA Cross", "RSI Divergence", "MACD Cross", "Bollinger Band Bounce", "Stochastic Oscillator", "EMA + RSI Combined"])

if 'seen_signals' not in st.session_state:
    st.session_state.seen_signals = {}

for asset in selected_assets:
    df = fetch_candles(asset, limit=CANDLE_LIMIT)
    if df is not None:
        df = calculate_indicators(df)
        signals = detect_signals(df, selected_strategy)

        new_signals = [s for s in signals if s not in st.session_state.seen_signals.get(asset, [])]
        if new_signals:
            st.subheader(f"🔔 {asset} - New Signals")
            for signal in new_signals:
                st.success(signal)
            st.session_state.seen_signals[asset] = signals
        else:
            st.write(f"✅ {asset} - No new signals yet")
