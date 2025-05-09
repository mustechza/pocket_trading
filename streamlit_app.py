import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import asyncio
import websockets
import json
from threading import Thread
from streamlit_autorefresh import st_autorefresh

# --- SETTINGS ---
st.set_page_config(layout="wide")
REFRESH_INTERVAL = 10  # seconds
ASSETS = ["ethusdt", "solusdt", "adausdt", "bnbusdt", "xrpusdt", "ltcusdt"]
live_data = {}
signal_memory = {}  # to track last signal per asset

# --- SIDEBAR ---
uploaded_file = st.sidebar.file_uploader("Upload historical data (CSV)", type=["csv"])
selected_assets = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS[:2])
selected_strategy = st.sidebar.selectbox("Strategy", [
    "EMA Cross", "RSI Divergence", "MACD Cross", "Bollinger Band Bounce",
    "Stochastic Oscillator", "EMA + RSI Combined"
])
money_strategy = st.sidebar.selectbox("Money Management", ["Flat", "Martingale"])
ema_short = st.sidebar.number_input("EMA Short Period", 2, 50, value=5)
ema_long = st.sidebar.number_input("EMA Long Period", 5, 100, value=20)
rsi_period = st.sidebar.number_input("RSI Period", 5, 50, value=14)
stoch_period = st.sidebar.number_input("Stochastic Period", 5, 50, value=14)
bb_period = st.sidebar.number_input("Bollinger Band Period", 5, 50, value=20)

# --- INDICATOR CALCULATIONS ---
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

# --- SIGNAL DETECTION ---
def generate_signal(timestamp, signal_type, price):
    return {"Time": timestamp, "Signal": signal_type, "Price": price, "Trade Duration (min)": 5}

def detect_signals(df, strategy):
    signals = []
    for i in range(1, len(df)):
        t = df['timestamp'].iloc[i]
        price = df['close'].iloc[i]
        if strategy == "EMA Cross":
            if df['EMA5'].iloc[i-1] < df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] > df['EMA20'].iloc[i]:
                signals.append(generate_signal(t, "Buy", price))
            elif df['EMA5'].iloc[i-1] > df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] < df['EMA20'].iloc[i]:
                signals.append(generate_signal(t, "Sell", price))
        elif strategy == "RSI Divergence":
            rsi = df['RSI'].iloc[i]
            if rsi < 30:
                signals.append(generate_signal(t, "Buy", price))
            elif rsi > 70:
                signals.append(generate_signal(t, "Sell", price))
        elif strategy == "MACD Cross":
            if df['MACD'].iloc[i-1] < df['MACD_signal'].iloc[i-1] and df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
                signals.append(generate_signal(t, "Buy", price))
            elif df['MACD'].iloc[i-1] > df['MACD_signal'].iloc[i-1] and df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
                signals.append(generate_signal(t, "Sell", price))
        elif strategy == "Bollinger Band Bounce":
            if df['close'].iloc[i] < df['BB_lower'].iloc[i]:
                signals.append(generate_signal(t, "Buy", price))
            elif df['close'].iloc[i] > df['BB_upper'].iloc[i]:
                signals.append(generate_signal(t, "Sell", price))
        elif strategy == "Stochastic Oscillator":
            stoch = df['Stochastic'].iloc[i]
            if stoch < 20:
                signals.append(generate_signal(t, "Buy", price))
            elif stoch > 80:
                signals.append(generate_signal(t, "Sell", price))
        elif strategy == "EMA + RSI Combined":
            if df['EMA5'].iloc[i-1] < df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] > df['EMA20'].iloc[i] and df['RSI'].iloc[i] < 40:
                signals.append(generate_signal(t, "Buy", price))
            elif df['EMA5'].iloc[i-1] > df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] < df['EMA20'].iloc[i] and df['RSI'].iloc[i] > 60:
                signals.append(generate_signal(t, "Sell", price))
    return signals

# --- CHART PLOT ---
def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'], name='Candles'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA5'], name="EMA5", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA20'], name="EMA20", line=dict(color='red')))
    fig.update_layout(title=asset.upper(), xaxis_rangeslider_visible=False, height=300)
    return fig

# --- LIVE DATA HANDLER ---
async def binance_ws(asset):
    url = f"wss://stream.binance.com:9443/ws/{asset}@kline_1m"
    async with websockets.connect(url) as ws:
        while True:
            data = await ws.recv()
            json_data = json.loads(data)
            kline = json_data['k']
            candle = {
                "timestamp": pd.to_datetime(int(kline['t']), unit='ms'),
                "open": float(kline['o']),
                "high": float(kline['h']),
                "low": float(kline['l']),
                "close": float(kline['c']),
                "volume": float(kline['v'])
            }
            if asset not in live_data:
                live_data[asset] = []
            live_data[asset].append(candle)
            if len(live_data[asset]) > 200:
                live_data[asset].pop(0)

def start_ws_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [binance_ws(asset) for asset in selected_assets]
    loop.run_until_complete(asyncio.gather(*tasks))

if "ws_started" not in st.session_state:
    Thread(target=start_ws_loop, daemon=True).start()
    st.session_state["ws_started"] = True

# --- AUTO REFRESH ---
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh")

# --- TITLE ---
st.title("📡 Real-Time Pocket Option Signals + Animated Dashboard")

# --- LIVE SIGNALS ---
st.subheader("🔴 Live Signals & Charts")

cols = st.columns(len(selected_assets))
for idx, asset in enumerate(selected_assets):
    with cols[idx]:
        st.markdown(f"### {asset.upper()}")
        if asset in live_data and len(live_data[asset]) > 30:
            df_live = pd.DataFrame(live_data[asset])
            df_live = calculate_indicators(df_live)
            signals = detect_signals(df_live, selected_strategy)
            last_signal = signals[-1] if signals else None

            # Remember last signal per asset
            prev_signal = signal_memory.get(asset)
            if last_signal and (prev_signal != last_signal['Time']):
                signal_memory[asset] = last_signal['Time']
                # Flashing effect
                color = "green" if last_signal["Signal"] == "Buy" else "red"
                st.markdown(
                    f"""
                    <div style='background-color:{color};padding:10px;border-radius:10px;text-align:center;animation: flash 1s infinite;'>
                    <strong style='color:white;font-size:18px'>{last_signal['Signal']} @ {last_signal['Price']}</strong>
                    </div>
                    <style>
                    @keyframes flash {{
                        0% {{opacity: 1;}}
                        50% {{opacity: 0.5;}}
                        100% {{opacity: 1;}}
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
            elif last_signal:
                color = "green" if last_signal["Signal"] == "Buy" else "red"
                st.markdown(
                    f"""
                    <div style='background-color:{color};padding:10px;border-radius:10px;text-align:center;'>
                    <strong style='color:white;font-size:18px'>{last_signal['Signal']} @ {last_signal['Price']}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.info("No signal yet.")

            st.plotly_chart(plot_chart(df_live, asset), use_container_width=True)
        else:
            st.warning("Waiting for live data...")

# --- BACKTESTING ---
if uploaded_file:
    st.subheader("📊 Backtest Mode")
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = calculate_indicators(df)
    backtest_signals = detect_signals(df, selected_strategy)
    st.dataframe(pd.DataFrame(backtest_signals))
