import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import plotly.graph_objects as go
import datetime
import telegram

# --- SETTINGS ---
REFRESH_INTERVAL = 5  # seconds
CANDLE_LIMIT = 500
BINANCE_URL = "https://api.binance.com/api/v3/klines"
ASSETS = ["ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]

# --- TELEGRAM SETTINGS (optional) ---
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

# --- FUNCTIONS ---

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
        st.warning(f"Fetching data failed, retrying... Error: {e}")
        return None

def calculate_indicators(df):
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def detect_ema_cross(df):
    if len(df) < 2:
        return None
    if df['EMA5'].iloc[-2] < df['EMA20'].iloc[-2] and df['EMA5'].iloc[-1] > df['EMA20'].iloc[-1]:
        return "Buy Signal (EMA Cross UP)"
    elif df['EMA5'].iloc[-2] > df['EMA20'].iloc[-2] and df['EMA5'].iloc[-1] < df['EMA20'].iloc[-1]:
        return "Sell Signal (EMA Cross DOWN)"
    return None

def detect_rsi_divergence(df):
    if len(df) < 15:
        return None
    rsi = df['RSI'].iloc[-1]
    price = df['close'].iloc[-1]
    if rsi < 30:
        return "Potential Buy (RSI Oversold)"
    elif rsi > 70:
        return "Potential Sell (RSI Overbought)"
    return None

def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candles'
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['EMA5'],
        line=dict(color='blue', width=1),
        name='EMA5'
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['EMA20'],
        line=dict(color='red', width=1),
        name='EMA20'
    ))
    fig.update_layout(
        title=f"Live Chart: {asset}",
        yaxis_title="Price (USDT)",
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    return fig

def send_telegram_alert(message):
    try:
        bot = telegram.Bot(token=TELEGRAM_TOKEN)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        st.warning(f"Failed to send Telegram alert: {e}")

# --- STREAMLIT APP START ---
st.set_page_config(page_title="Pocket Option Signals", layout="wide")

st.title("Pocket Option Trading Signals App")

selected_assets = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS[:3])
selected_strategy = st.sidebar.selectbox("Select Strategy", ["EMA Cross", "RSI Divergence"])
enable_telegram = st.sidebar.checkbox("Enable Telegram Alerts", value=False)

placeholder = st.empty()

seen_signals = set()

while True:
    with placeholder.container():
        for asset in selected_assets:
            df = fetch_candles(asset, limit=CANDLE_LIMIT)
            if df is None:
                continue
            df = calculate_indicators(df)
            signal = None

            if selected_strategy == "EMA Cross":
                signal = detect_ema_cross(df)
            elif selected_strategy == "RSI Divergence":
                signal = detect_rsi_divergence(df)

            st.subheader(f"Asset: {asset}")
            st.plotly_chart(plot_chart(df, asset), use_container_width=True)

            if signal and (asset, signal) not in seen_signals:
                st.success(f"✅ {signal} detected on {asset}")
                seen_signals.add((asset, signal))
                if enable_telegram:
                    send_telegram_alert(f"{signal} on {asset}")

            st.info("Fetching latest data...")

    time.sleep(REFRESH_INTERVAL)
    st.experimental_rerun()
