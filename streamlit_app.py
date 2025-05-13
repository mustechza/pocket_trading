import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from binance.client import Client
import time
import plotly.graph_objects as go

# Binance API Setup (read-only)
client = Client()

st.set_page_config(page_title="Trading Strategy Tester", layout="wide")
st.title("📈 Trading Strategy Tester & Live Signals")

st.sidebar.header("Settings")
strategy = st.sidebar.selectbox("Select Strategy", [
    "EMA Cross",
    "RSI Divergence",
    "MACD Cross",
    "Bollinger Band Bounce",
    "Stochastic Oscillator",
    "EMA + RSI Combined"
])

symbol = st.sidebar.text_input("Symbol (e.g., BTCUSDT)", value="BTCUSDT")
interval = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m"], index=0)
start_date = st.sidebar.date_input("Start Date", value=datetime.now() - timedelta(days=7))
end_date = st.sidebar.date_input("End Date", value=datetime.now())
run_backtest = st.sidebar.button("Run Backtest")

# ========== Strategy Logic ==========
def calculate_indicators(df):
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = compute_rsi(df['close'])
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
    df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
    df['Stochastic'] = ((df['close'] - df['low'].rolling(14).min()) / 
                        (df['high'].rolling(14).max() - df['low'].rolling(14).min())) * 100
    return df

def compute_confidence(df, strategy):
    scores = []
    for i in range(len(df)):
        score = 0
        if i < 1:
            scores.append(score)
            continue

        if strategy in ["EMA Cross", "EMA + RSI Combined"]:
            if df['EMA5'].iloc[i] > df['EMA20'].iloc[i]:
                score += 1
            elif df['EMA5'].iloc[i] < df['EMA20'].iloc[i]:
                score -= 1

        if strategy in ["RSI Divergence", "EMA + RSI Combined"]:
            if df['RSI'].iloc[i] < 30:
                score += 1
            elif df['RSI'].iloc[i] > 70:
                score -= 1

        if strategy == "MACD Cross":
            if df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
                score += 1
            elif df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
                score -= 1

        if strategy == "Bollinger Band Bounce":
            if df['close'].iloc[i] < df['BB_lower'].iloc[i]:
                score += 1
            elif df['close'].iloc[i] > df['BB_upper'].iloc[i]:
                score -= 1

        if strategy == "Stochastic Oscillator":
            if df['Stochastic'].iloc[i] < 20:
                score += 1
            elif df['Stochastic'].iloc[i] > 80:
                score -= 1

        scores.append(score)
    df['Confidence'] = scores
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ========== Signal Generation ==========
def generate_signal(time, signal, price):
    return {"Time": time, "Signal": signal, "Price": price}

def detect_signals(df, strategy):
    signals = []
    for i in range(1, len(df)):
        t = df.index[i]
        price = df['close'].iloc[i]
        confidence = df['Confidence'].iloc[i]

        if strategy == "EMA Cross":
            if df['EMA5'].iloc[i] > df['EMA20'].iloc[i] and df['EMA5'].iloc[i-1] <= df['EMA20'].iloc[i-1]:
                signal = generate_signal(t, "Buy (EMA Cross)", price)
                signal["Confidence"] = confidence
                signals.append(signal)

        elif strategy == "RSI Divergence":
            if df['RSI'].iloc[i] < 30:
                signal = generate_signal(t, "Buy (RSI < 30)", price)
                signal["Confidence"] = confidence
                signals.append(signal)

        elif strategy == "MACD Cross":
            if df['MACD'].iloc[i] > df['MACD_signal'].iloc[i] and df['MACD'].iloc[i-1] <= df['MACD_signal'].iloc[i-1]:
                signal = generate_signal(t, "Buy (MACD Cross)", price)
                signal["Confidence"] = confidence
                signals.append(signal)

        elif strategy == "Bollinger Band Bounce":
            if df['close'].iloc[i] < df['BB_lower'].iloc[i]:
                signal = generate_signal(t, "Buy (BB Lower)", price)
                signal["Confidence"] = confidence
                signals.append(signal)

        elif strategy == "Stochastic Oscillator":
            if df['Stochastic'].iloc[i] < 20:
                signal = generate_signal(t, "Buy (Stochastic)", price)
                signal["Confidence"] = confidence
                signals.append(signal)

        elif strategy == "EMA + RSI Combined":
            if (df['EMA5'].iloc[i] > df['EMA20'].iloc[i] and df['RSI'].iloc[i] < 30):
                signal = generate_signal(t, "Buy (EMA+RSI)", price)
                signal["Confidence"] = confidence
                signals.append(signal)

    return signals

# ========== Fetch Candles ==========
def fetch_candles(symbol, interval="1m", lookback="2 day ago UTC"):
    klines = client.get_historical_klines(symbol, interval, lookback)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_volume',
        'taker_buy_quote_volume', 'ignore'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    return df

# ========== Live Signal Detection ==========
st.subheader("🔍 Live Signal Detection")
live_signals = []
assets = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
for asset in assets:
    df_live = fetch_candles(asset, interval=interval)
    df_live = calculate_indicators(df_live)
    df_live = compute_confidence(df_live, strategy)
    signals = detect_signals(df_live, strategy)
    live_signals.extend(signals)

if live_signals:
    live_signals = sorted(live_signals, key=lambda x: x["Confidence"], reverse=True)
    latest = live_signals[:3]  # top 3 by confidence
    for s in latest:
        st.write(f"{s['Time']} - {s['Signal']} at {s['Price']:.2f} (Confidence: {s['Confidence']})")
else:
    st.write("No current signals.")

# ========== Backtesting ==========
if run_backtest:
    df = fetch_candles(symbol, interval=interval, lookback="60 day ago UTC")
    df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
    df = calculate_indicators(df)
    df = compute_confidence(df, strategy)
    signals = detect_signals(df, strategy)
    signals = sorted(signals, key=lambda x: x["Confidence"], reverse=True)

    st.subheader("📊 Backtest Results")
    if signals:
        st.write(f"Total Signals: {len(signals)}")
        st.dataframe(pd.DataFrame(signals))

        # Plot chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['open'],
                                     high=df['high'],
                                     low=df['low'],
                                     close=df['close'],
                                     name='Price'))
        signal_times = [s['Time'] for s in signals]
        signal_prices = [s['Price'] for s in signals]
        fig.add_trace(go.Scatter(x=signal_times, y=signal_prices,
                                 mode='markers', marker=dict(color='red', size=8),
                                 name='Signals'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No signals found for selected period.")
            
