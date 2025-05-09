import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import websockets
import json
import datetime
import plotly.graph_objects as go
from threading import Thread

# --- SETTINGS ---
ASSETS = ["ethusdt", "bnbusdt", "adausdt", "solusdt"]
INTERVAL = "1m"
live_data = {asset: pd.DataFrame() for asset in ASSETS}

# --- WEBSOCKET CLIENT ---

async def binance_ws(asset):
    url = f"wss://stream.binance.com:9443/ws/{asset}@kline_{INTERVAL}"
    async with websockets.connect(url) as ws:
        while True:
            msg = await ws.recv()
            msg = json.loads(msg)
            kline = msg['k']
            candle = {
                'timestamp': pd.to_datetime(kline['t'], unit='ms'),
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v'])
            }
            df = live_data[asset]
            df = pd.concat([df, pd.DataFrame([candle])]).drop_duplicates(subset='timestamp', keep='last')
            live_data[asset] = df.tail(100)  # keep last 100 rows
            print(f"✅ Received live data for {asset} at {candle['timestamp']}")
            

def start_ws():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [binance_ws(asset) for asset in ASSETS]
    loop.run_until_complete(asyncio.gather(*tasks))

# Start websocket in separate thread
t = Thread(target=start_ws)
t.daemon = True
t.start()

# --- INDICATORS ---

def calculate_indicators(df):
    df = df.copy()
    df['EMA5'] = df['close'].ewm(span=5).mean()
    df['EMA20'] = df['close'].ewm(span=20).mean()
    df['RSI'] = compute_rsi(df['close'])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_signal(df):
    if len(df) < 20:
        return "Waiting..."
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    # EMA Crossover
    if prev['EMA5'] < prev['EMA20'] and latest['EMA5'] > latest['EMA20']:
        return "📈 Buy (EMA Cross)"
    elif prev['EMA5'] > prev['EMA20'] and latest['EMA5'] < latest['EMA20']:
        return "📉 Sell (EMA Cross)"
    # RSI
    if latest['RSI'] < 30:
        return "📈 Buy (RSI Oversold)"
    elif latest['RSI'] > 70:
        return "📉 Sell (RSI Overbought)"
    return "No Signal"

def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Candles'
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['EMA5'], name="EMA5", line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['EMA20'], name="EMA20", line=dict(color='red')
    ))
    fig.update_layout(title=asset.upper(), xaxis_rangeslider_visible=False)
    return fig

# --- STREAMLIT DASHBOARD ---

st.set_page_config(layout="wide")
st.title("🚀 Real-Time Crypto Signals (WebSocket + Dashboard Cards)")

selected_assets = st.multiselect("Select Assets", ASSETS, default=ASSETS[:2])

for asset in selected_assets:
    df = live_data[asset]
    if df.empty or len(df) < 5:
        st.info(f"Waiting for live data for {asset.upper()}...")
        continue

    df = calculate_indicators(df)
    signal = detect_signal(df)

    # Dashboard card
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label=f"{asset.upper()} Signal", value=signal)
    with col2:
        st.plotly_chart(plot_chart(df, asset), use_container_width=True)

st.caption("✅ Live data via Binance WebSocket | Updated every few seconds")
    
