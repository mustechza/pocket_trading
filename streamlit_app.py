import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import asyncio
import websockets
import json
from datetime import datetime
import streamlit.components.v1 as components

# --- SETTINGS ---
st.set_page_config(layout="wide")
ASSETS = ["ethusdt", "btcusdt", "bnbusdt", "xrpusdt"]
INTERVAL = "1m"

# --- SIDEBAR ---
selected_assets = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS[:2])
selected_strategy = st.sidebar.selectbox("Strategy", ["EMA Cross", "RSI", "MACD"])
ema_short = st.sidebar.number_input("EMA Short", 2, 50, 5)
ema_long = st.sidebar.number_input("EMA Long", 5, 100, 20)

# --- INDICATOR FUNCTIONS ---
def calculate_indicators(df):
    df['EMA_Short'] = df['close'].ewm(span=ema_short, adjust=False).mean()
    df['EMA_Long'] = df['close'].ewm(span=ema_long, adjust=False).mean()
    df['RSI'] = compute_rsi(df['close'])
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_signal(df, strategy):
    if len(df) < 2:
        return None
    last = df.iloc[-1]
    prev = df.iloc[-2]
    if strategy == "EMA Cross":
        if prev['EMA_Short'] < prev['EMA_Long'] and last['EMA_Short'] > last['EMA_Long']:
            return "Buy (EMA Cross)"
        elif prev['EMA_Short'] > prev['EMA_Long'] and last['EMA_Short'] < last['EMA_Long']:
            return "Sell (EMA Cross)"
    elif strategy == "RSI":
        if last['RSI'] < 30:
            return "Buy (RSI Oversold)"
        elif last['RSI'] > 70:
            return "Sell (RSI Overbought)"
    elif strategy == "MACD":
        # Add MACD logic if needed
        pass
    return None

def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candles'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['EMA_Short'], name="EMA Short", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['EMA_Long'], name="EMA Long", line=dict(color='red')))
    fig.update_layout(title=asset.upper(), xaxis_rangeslider_visible=False)
    return fig

# --- STREAMLIT PLACEHOLDERS ---
st.title("📡 Real-Time Crypto Signals (WebSocket Powered)")

cards = {}
charts = {}

for asset in selected_assets:
    col1, col2 = st.columns([1, 3])
    cards[asset] = col1.empty()
    charts[asset] = col2.empty()

# --- ASYNC BINANCE STREAM ---
async def binance_ws(symbol):
    url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{INTERVAL}"
    df = pd.DataFrame()
    async with websockets.connect(url) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            k = data['k']
            row = {
                'time': datetime.fromtimestamp(k['t'] / 1000),
                'open': float(k['o']),
                'high': float(k['h']),
                'low': float(k['l']),
                'close': float(k['c']),
                'volume': float(k['v'])
            }
            df = pd.concat([df, pd.DataFrame([row])]).drop_duplicates(subset='time').reset_index(drop=True)
            df = calculate_indicators(df)

            signal = detect_signal(df, selected_strategy)

            # --- Update Card ---
            price = row['close']
            color = "green" if signal and "Buy" in signal else "red" if signal else "gray"
            cards[symbol].markdown(f"""
                <div style="padding:10px;border-radius:10px;background-color:{color};color:white;text-align:center">
                <h3>{symbol.upper()}</h3>
                <p>Price: {price:.4f}</p>
                <b>{signal or 'Waiting...'}</b>
                </div>
            """, unsafe_allow_html=True)

            # --- Update Chart ---
            charts[symbol].plotly_chart(plot_chart(df.tail(100), symbol), use_container_width=True)

# --- MAIN EVENT LOOP ---
async def main():
    tasks = [binance_ws(asset) for asset in selected_assets]
    await asyncio.gather(*tasks)

# --- RUN ---
asyncio.run(main())
