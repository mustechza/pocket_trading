import streamlit as st
st.set_page_config(layout="wide")

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from collections import deque
from threading import Thread

# --- SETTINGS ---
ASSETS = ["ethusdt", "solusdt", "adausdt", "bnbusdt", "xrpusdt", "ltcusdt"]
selected_assets = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS[:2])
selected_strategy = st.sidebar.selectbox("Strategy", ["EMA Cross"])
ema_short = st.sidebar.number_input("EMA Short Period", 2, 50, value=5)
ema_long = st.sidebar.number_input("EMA Long Period", 5, 100, value=20)

# Global storage for live data
live_data = {symbol: deque(maxlen=500) for symbol in ASSETS}
last_signals = {symbol: None for symbol in ASSETS}

# --- Indicator Calculation ---
def calculate_ema_cross(df):
    df['EMA_Short'] = df['close'].ewm(span=ema_short, adjust=False).mean()
    df['EMA_Long'] = df['close'].ewm(span=ema_long, adjust=False).mean()
    return df

def detect_ema_signal(df):
    signals = []
    if len(df) < 2:
        return signals
    i = -1
    if df['EMA_Short'].iloc[i - 1] < df['EMA_Long'].iloc[i - 1] and df['EMA_Short'].iloc[i] > df['EMA_Long'].iloc[i]:
        signals.append(("Buy (EMA Cross)", df['timestamp'].iloc[i], df['close'].iloc[i]))
    elif df['EMA_Short'].iloc[i - 1] > df['EMA_Long'].iloc[i - 1] and df['EMA_Short'].iloc[i] < df['EMA_Long'].iloc[i]:
        signals.append(("Sell (EMA Cross)", df['timestamp'].iloc[i], df['close'].iloc[i]))
    return signals

# --- WebSocket Data Fetching ---
async def stream_binance(symbol):
    url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_1m"
    async with websockets.connect(url) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            k = data['k']
            candle = {
                "timestamp": pd.to_datetime(k['t'], unit='ms'),
                "open": float(k['o']),
                "high": float(k['h']),
                "low": float(k['l']),
                "close": float(k['c']),
                "volume": float(k['v'])
            }
            live_data[symbol].append(candle)

# --- Background Thread for WebSocket ---
def start_websocket_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [stream_binance(symbol) for symbol in selected_assets]
    loop.run_until_complete(asyncio.gather(*tasks))

if 'websocket_thread_started' not in st.session_state:
    thread = Thread(target=start_websocket_loop, daemon=True)
    thread.start()
    st.session_state.websocket_thread_started = True

# --- Live Stream Display ---
st.title("📡 Binance WebSocket Crypto Signal App")

for symbol in selected_assets:
    if len(live_data[symbol]) < 20:
        st.write(f"⏳ Waiting for data: {symbol.upper()}")
        continue

    df = pd.DataFrame(live_data[symbol])
    df = calculate_ema_cross(df)
    signals = detect_ema_signal(df)

    st.subheader(symbol.upper())
    st.plotly_chart(
        go.Figure(data=[
            go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'],
                           low=df['low'], close=df['close'], name="Candles"),
            go.Scatter(x=df['timestamp'], y=df['EMA_Short'], name="EMA Short", line=dict(color='blue')),
            go.Scatter(x=df['timestamp'], y=df['EMA_Long'], name="EMA Long", line=dict(color='red')),
        ]).update_layout(xaxis_rangeslider_visible=False),
        use_container_width=True
    )

    if signals:
        signal_text = f"📍 **{signals[-1][0]}** at {signals[-1][1].strftime('%H:%M:%S')} – Price: {signals[-1][2]}"
        if last_signals[symbol] != signal_text:
            st.success(signal_text)
            last_signals[symbol] = signal_text
