import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import websockets
import json
from datetime import datetime, timedelta
import pytz
import threading

st.set_page_config(page_title="Deriv Signal App", layout="wide")

# SETTINGS
ASSET_SYMBOL = "R_100"  # Replace with your preferred Deriv symbol, e.g., "R_100", "1HZ100V", etc.
INTERVAL = 1  # Candle interval in minutes
EMA_SHORT = 5
EMA_LONG = 10
TIMEZONE = pytz.timezone("Africa/Johannesburg")

# Streamlit State
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

if "signals" not in st.session_state:
    st.session_state.signals = []

# ----------------------------------------
# WebSocket Client for Deriv
# ----------------------------------------
def deriv_subscribe_candles(symbol, granularity):
    return {
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": 100,
        "granularity": granularity * 60,
        "style": "candles",
        "subscribe": 1
    }

def format_time(timestamp):
    return datetime.fromtimestamp(timestamp, tz=TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")

async def fetch_and_stream_data():
    uri = "wss://ws.derivws.com/websockets/v3?app_id=1089"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps(deriv_subscribe_candles(ASSET_SYMBOL, INTERVAL)))
        while True:
            response = await ws.recv()
            data = json.loads(response)

            if "candles" in data:
                candles = data["candles"]
            elif "ohlc" in data:
                candle = data["ohlc"]
                candles = [{
                    "epoch": candle["open_time"],
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                }]
            else:
                continue

            update_data(candles)

# ----------------------------------------
# Process and Signal Logic
# ----------------------------------------
def update_data(candles):
    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["epoch"], unit="s").dt.tz_localize("UTC").dt.tz_convert(TIMEZONE)
    df = df.sort_values("time")

    df["ema_short"] = df["close"].ewm(span=EMA_SHORT, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=EMA_LONG, adjust=False).mean()

    df["signal"] = np.where(df["ema_short"] > df["ema_long"], "Buy", "Sell")
    df["crossover"] = df["signal"] != df["signal"].shift(1)

    # Append new signals
    latest = df.iloc[-1]
    if latest["crossover"]:
        signal_time = latest["time"].strftime("%Y-%m-%d %H:%M:%S")
        direction = latest["signal"]
        price = latest["close"]

        signal = {
            "time": signal_time,
            "price": price,
            "direction": direction
        }

        if len(st.session_state.signals) == 0 or signal != st.session_state.signals[-1]:
            st.session_state.signals.append(signal)

    st.session_state.df = df

# ----------------------------------------
# Background Thread for Async Loop
# ----------------------------------------
def start_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(fetch_and_stream_data())

threading.Thread(target=start_loop, daemon=True).start()

# ----------------------------------------
# Streamlit UI
# ----------------------------------------

st.title("📡 Deriv Live Signal App")
st.markdown(f"**Asset:** `{ASSET_SYMBOL}` | **Interval:** {INTERVAL}m | **EMA:** {EMA_SHORT}/{EMA_LONG}")
st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Live Chart")
    df = st.session_state.df
    if not df.empty:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Candles"
        ))
        fig.add_trace(go.Scatter(x=df["time"], y=df["ema_short"], mode="lines", name=f"EMA {EMA_SHORT}"))
        fig.add_trace(go.Scatter(x=df["time"], y=df["ema_long"], mode="lines", name=f"EMA {EMA_LONG}"))
        fig.update_layout(xaxis_rangeslider_visible=False, height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Waiting for data...")

with col2:
    st.subheader("🔔 Signals")
    signals = st.session_state.signals[-5:][::-1]  # Last 5
    for sig in signals:
        st.markdown(f"""
        <div style="padding:10px;border-radius:10px;margin:5px 0;background-color:{'#d1ffd6' if sig['direction']=='Buy' else '#ffd6d6'}">
        <strong>{sig['direction']} Signal</strong><br>
        <small>{sig['time']}</small><br>
        <b>Price:</b> {sig['price']}
        </div>
        """, unsafe_allow_html=True)
