import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import asyncio
import json
import websockets
import time

# --- Streamlit Config ---
st.set_page_config(layout="wide")
st.title("📡 Live Binance Signals Dashboard")

# Assets
ASSETS = ["btcusdt", "ethusdt", "solusdt"]
selected_assets = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS[:2])

# Initialize session state
if "live_data" not in st.session_state:
    st.session_state.live_data = {symbol: pd.DataFrame() for symbol in ASSETS}

if "signals" not in st.session_state:
    st.session_state.signals = {symbol: "⏳ Waiting..." for symbol in ASSETS}

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

    # Signal logic
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

# --- Async Function: Binance WebSocket Stream ---
async def binance_ws(symbol):
    url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_1m"
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

            # Compute signal
            if len(df) > 50:
                signal = compute_signals(df)
                st.session_state.signals[symbol] = signal

# --- Start All Streams ---
async def run_streams():
    tasks = [asyncio.create_task(binance_ws(symbol)) for symbol in selected_assets]
    await asyncio.gather(*tasks)

# --- Start WebSocket in Background ---
if "ws_started" not in st.session_state:
    st.session_state.ws_started = True
    asyncio.get_event_loop().run_in_executor(None, lambda: asyncio.run(run_streams()))

# --- Streamlit Live Dashboard ---
placeholder = st.empty()

while True:
    with placeholder.container():
        cols = st.columns(len(selected_assets))
        for i, symbol in enumerate(selected_assets):
            df = st.session_state.live_data[symbol]
            signal = st.session_state.signals.get(symbol, "⏳ Waiting...")
            if not df.empty:
                price = df['close'].iloc[-1]
                cols[i].metric(label=f"{symbol.upper()} Price", value=f"${price:.2f}", delta=signal)
                cols[i].plotly_chart(plot_chart(df, symbol), use_container_width=True)
            else:
                cols[i].write(f"Waiting for live data for {symbol.upper()}...")

    # Refresh every 2 sec
    time_now = datetime.datetime.now().strftime('%H:%M:%S')
    st.caption(f"🔄 Last updated: {time_now}")
    time.sleep(2)
