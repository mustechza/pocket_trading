import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import asyncio
import json
import websockets
import time
import threading
from streamlit_autorefresh import st_autorefresh

# --- Streamlit Config ---
st.set_page_config(layout="wide")
st.title("📡 Live Binance Signals Dashboard")

ASSETS = ["btcusdt", "ethusdt", "solusdt"]
selected_assets = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS[:2])

# Initialize state
if "live_data" not in st.session_state:
    st.session_state.live_data = {symbol: pd.DataFrame() for symbol in ASSETS}

if "signals" not in st.session_state:
    st.session_state.signals = {symbol: "⏳ Waiting..." for symbol in ASSETS}

if "last_update" not in st.session_state:
    st.session_state.last_update = {symbol: 0 for symbol in ASSETS}

if "ws_threads" not in st.session_state:
    st.session_state.ws_threads = {}

# --- Function: Compute EMA + RSI Signals ---
def compute_signals(df):
    df = df.copy()
    df['EMA20'] = df['close'].ewm(span=20).mean()
    df['EMA50'] = df['close'].ewm(span=50).mean()
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    latest = df.iloc[-1]
    if latest['EMA20'] > latest['EMA50'] and latest['RSI'] < 70:
        return "🟢 Buy Signal"
    elif latest['EMA20'] < latest['EMA50'] and latest['RSI'] > 30:
        return "🔴 Sell Signal"
    else:
        return "⚪ No Clear Signal"

# --- Function: Plot Chart ---
def plot_chart(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='Candles'
    ))
    if 'EMA20' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA20'], line=dict(color='blue', width=1), name='EMA20'))
    if 'EMA50' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA50'], line=dict(color='orange', width=1), name='EMA50'))
    fig.update_layout(title=symbol.upper(), xaxis_rangeslider_visible=False)
    return fig

# --- Background Thread: Binance WebSocket Stream ---
def run_ws(symbol):
    url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_1m"
    async def ws_loop():
        while True:
            try:
                async with websockets.connect(url) as ws:
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        k = data['k']
                        ts = pd.to_datetime(int(k['t']), unit='ms')
                        row = {
                            "timestamp": ts,
                            "open": float(k['o']),
                            "high": float(k['h']),
                            "low": float(k['l']),
                            "close": float(k['c'])
                        }
                        df = st.session_state.live_data[symbol]
                        df = pd.concat([df, pd.DataFrame([row])]).drop_duplicates(subset='timestamp', keep='last')
                        df = df.sort_values('timestamp').tail(100)
                        st.session_state.live_data[symbol] = df

                        st.session_state.last_update[symbol] = time.time()

                        if len(df) > 50:
                            signal = compute_signals(df)
                            st.session_state.signals[symbol] = signal

            except Exception as e:
                st.session_state.signals[symbol] = f"❌ Reconnecting..."
                await asyncio.sleep(3)

    asyncio.run(ws_loop())

# --- Start or Restart WebSocket Thread ---
def start_ws(symbol):
    # Skip if already running
    thread = st.session_state.ws_threads.get(symbol)
    if thread and thread.is_alive():
        return

    # Start new thread
    t = threading.Thread(target=run_ws, args=(symbol,), daemon=True)
    t.start()
    st.session_state.ws_threads[symbol] = t

# --- Start Streams for selected assets ---
for symbol in selected_assets:
    start_ws(symbol)

# --- Auto-refresh every 2 seconds ---
st_autorefresh(interval=2000, key="refresh")

# --- Streamlit Live Dashboard ---
cols = st.columns(len(selected_assets))
for i, symbol in enumerate(selected_assets):
    df = st.session_state.live_data[symbol]
    signal = st.session_state.signals.get(symbol, "⏳ Waiting...")
    last_time = st.session_state.last_update.get(symbol, 0)
    seconds_since_update = time.time() - last_time

    # Auto-reconnect if stale
    if seconds_since_update > 15:
        cols[i].error(f"❌ {symbol.upper()} Stale! No update for {int(seconds_since_update)} sec")
        start_ws(symbol)
        continue
    else:
        cols[i].success(f"🟢 {symbol.upper()} Live")

    if not df.empty:
        price = df['close'].iloc[-1]
        cols[i].metric(label=f"{symbol.upper()} Price", value=f"${price:.2f}", delta=signal)
        cols[i].plotly_chart(plot_chart(df, symbol), use_container_width=True)
    else:
        cols[i].write(f"Waiting for live data for {symbol.upper()}...")

time_now = datetime.datetime.now().strftime('%H:%M:%S')
st.caption(f"🔄 Last updated: {time_now}")
