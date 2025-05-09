import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import asyncio
import websockets
import json
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components

# --- SETTINGS ---
REFRESH_INTERVAL = 10  # seconds
ASSETS = ["ethusdt", "solusdt", "adausdt", "bnbusdt", "xrpusdt", "ltcusdt"]
STREAM_URL = "wss://stream.binance.com:9443/ws"

# --- SIDEBAR: Parameter Inputs ---
uploaded_file = st.sidebar.file_uploader("Upload historical data (CSV)", type=["csv"])
selected_assets = st.sidebar.multiselect("Select Assets", [s.upper() for s in ASSETS], default=[s.upper() for s in ASSETS[:2]])
selected_strategy = st.sidebar.selectbox("Strategy", [
    "EMA Cross", "RSI Divergence", "MACD Cross", "Bollinger Band Bounce",
    "Stochastic Oscillator", "EMA + RSI Combined", "ML Model (Random Forest)"
])
money_strategy = st.sidebar.selectbox("Money Management", ["Flat", "Martingale"])

# Indicator parameter inputs
ema_short = st.sidebar.number_input("EMA Short Period", 2, 50, value=5)
ema_long = st.sidebar.number_input("EMA Long Period", 5, 100, value=20)
rsi_period = st.sidebar.number_input("RSI Period", 5, 50, value=14)
stoch_period = st.sidebar.number_input("Stochastic Period", 5, 50, value=14)
bb_period = st.sidebar.number_input("Bollinger Band Period", 5, 50, value=20)

# --- FUNCTIONS ---
def calculate_indicators(df):
    df['EMA5'] = df['close'].ewm(span=ema_short, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=ema_long, adjust=False).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['BB_upper'] = df['close'].rolling(window=bb_period).mean() + 2 * df['close'].rolling(window=bb_period).std()
    df['BB_lower'] = df['close'].rolling(window=bb_period).mean() - 2 * df['close'].rolling(window=bb_period).std()

    low_min = df['low'].rolling(window=stoch_period).min()
    high_max = df['high'].rolling(window=stoch_period).max()
    df['Stochastic'] = (df['close'] - low_min) / (high_max - low_min) * 100

    return df

def generate_signal(timestamp, signal_type, price):
    duration = 5
    return {
        "Time": timestamp,
        "Signal": signal_type,
        "Price": price,
        "Trade Duration (min)": duration
    }

def detect_signals(df, strategy):
    signals = []
    for i in range(1, len(df)):
        t = df['timestamp'].iloc[i]
        price = df['close'].iloc[i]

        if strategy == "EMA Cross":
            if df['EMA5'].iloc[i-1] < df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] > df['EMA20'].iloc[i]:
                signals.append(generate_signal(t, "Buy (EMA Cross)", price))
            elif df['EMA5'].iloc[i-1] > df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] < df['EMA20'].iloc[i]:
                signals.append(generate_signal(t, "Sell (EMA Cross)", price))

        elif strategy == "RSI Divergence":
            rsi = df['RSI'].iloc[i]
            if rsi < 30:
                signals.append(generate_signal(t, "Buy (RSI Oversold)", price))
            elif rsi > 70:
                signals.append(generate_signal(t, "Sell (RSI Overbought)", price))

        elif strategy == "MACD Cross":
            if df['MACD'].iloc[i-1] < df['MACD_signal'].iloc[i-1] and df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
                signals.append(generate_signal(t, "Buy (MACD Cross)", price))
            elif df['MACD'].iloc[i-1] > df['MACD_signal'].iloc[i-1] and df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
                signals.append(generate_signal(t, "Sell (MACD Cross)", price))

        elif strategy == "Bollinger Band Bounce":
            if df['close'].iloc[i] < df['BB_lower'].iloc[i]:
                signals.append(generate_signal(t, "Buy (BB Lower)", price))
            elif df['close'].iloc[i] > df['BB_upper'].iloc[i]:
                signals.append(generate_signal(t, "Sell (BB Upper)", price))

        elif strategy == "Stochastic Oscillator":
            stoch = df['Stochastic'].iloc[i]
            if stoch < 20:
                signals.append(generate_signal(t, "Buy (Stochastic)", price))
            elif stoch > 80:
                signals.append(generate_signal(t, "Sell (Stochastic)", price))

        elif strategy == "EMA + RSI Combined":
            if df['EMA5'].iloc[i-1] < df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] > df['EMA20'].iloc[i] and df['RSI'].iloc[i] < 40:
                signals.append(generate_signal(t, "Buy (EMA+RSI)", price))
            elif df['EMA5'].iloc[i-1] > df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] < df['EMA20'].iloc[i] and df['RSI'].iloc[i] > 60:
                signals.append(generate_signal(t, "Sell (EMA+RSI)", price))
    return signals

def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA5'], name="EMA5", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA20'], name="EMA20", line=dict(color='red')))
    fig.update_layout(title=asset.upper(), xaxis_rangeslider_visible=False)
    return fig

def show_browser_alert(message):
    js_code = f"""
    <script>
    alert("{message}");
    </script>
    """
    components.html(js_code)

# --- TITLE ---
st.title("📈 Pocket Option Signals | Live WebSocket Streaming")

# --- LIVE SIGNALS FROM BINANCE WEBSOCKET ---
st.subheader("📡 Live Signals (WebSocket)")
placeholder = st.empty()

async def stream_data():
    symbols = [f"{s.lower()}@kline_1m" for s in selected_assets]
    url = f"{STREAM_URL}/stream?streams=" + "/".join(symbols)
    async with websockets.connect(url) as ws:
        data_buffer = {symbol: [] for symbol in selected_assets}
        while True:
            msg = await ws.recv()
            parsed = json.loads(msg)
            s = parsed['stream'].split('@')[0].upper()
            k = parsed['data']['k']
            row = {
                "timestamp": pd.to_datetime(k['t'], unit='ms'),
                "open": float(k['o']),
                "high": float(k['h']),
                "low": float(k['l']),
                "close": float(k['c']),
                "volume": float(k['v'])
            }
            data_buffer[s].append(row)
            for asset, rows in data_buffer.items():
                if len(rows) > 50:
                    df = pd.DataFrame(rows[-100:])
                    df = calculate_indicators(df)
                    signals = detect_signals(df, selected_strategy)
                    with placeholder.container():
                        st.markdown(f"### {asset}")
                        if signals:
                            st.dataframe(pd.DataFrame(signals[-3:]))
                            show_browser_alert(f"New signal for {asset}: {signals[-1]['Signal']}")
                        st.plotly_chart(plot_chart(df, asset), use_container_width=True)

asyncio.run(stream_data())
