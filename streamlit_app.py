# Updated Streamlit App with Audio Alerts, Custom EMA, TP/SL Logic, Real-Time Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import datetime
from streamlit_autorefresh import st_autorefresh

# --- SETTINGS ---
REFRESH_INTERVAL = 5  # seconds
CANDLE_LIMIT = 500
BINANCE_URL = "https://api.binance.com/api/v3/klines"
ASSETS = ["ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]

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
        st.warning(f"Fetching data failed: {e}")
        return None

def calculate_indicators(df, short=5, long=20):
    df['EMA_short'] = df['close'].ewm(span=short, adjust=False).mean()
    df['EMA_long'] = df['close'].ewm(span=long, adjust=False).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['BB_upper'] = df['close'].rolling(window=20).mean() + (df['close'].rolling(window=20).std() * 2)
    df['BB_lower'] = df['close'].rolling(window=20).mean() - (df['close'].rolling(window=20).std() * 2)
    df['Stochastic'] = (df['close'] - df['low'].rolling(window=14).min()) / (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()) * 100
    return df

def detect_ema_cross(df):
    signals = []
    for i in range(1, len(df)):
        if df['EMA_short'].iloc[i-1] < df['EMA_long'].iloc[i-1] and df['EMA_short'].iloc[i] > df['EMA_long'].iloc[i]:
            signals.append((df['timestamp'].iloc[i], "Buy Signal (EMA Cross UP)", df['close'].iloc[i]))
        elif df['EMA_short'].iloc[i-1] > df['EMA_long'].iloc[i-1] and df['EMA_short'].iloc[i] < df['EMA_long'].iloc[i]:
            signals.append((df['timestamp'].iloc[i], "Sell Signal (EMA Cross DOWN)", df['close'].iloc[i]))
    return signals

def detect_rsi_divergence(df):
    signals = []
    for i in range(len(df)):
        rsi = df['RSI'].iloc[i]
        if rsi < 30:
            signals.append((df['timestamp'].iloc[i], "Potential Buy (RSI Oversold)", df['close'].iloc[i]))
        elif rsi > 70:
            signals.append((df['timestamp'].iloc[i], "Potential Sell (RSI Overbought)", df['close'].iloc[i]))
    return signals

def simulate_money_management(signals, df, initial_balance=1000, bet_size=10, strategy="Flat", tp_pct=2.0, sl_pct=1.0):
    balance = initial_balance
    results = []
    for ts, signal, price in signals:
        entry_price = price
        exit_price = None
        result = "Open"
        i = df.index[df['timestamp'] == ts][0]
        for j in range(i + 1, min(i + 10, len(df))):
            high = df['high'].iloc[j]
            low = df['low'].iloc[j]
            if "Buy" in signal:
                tp = entry_price * (1 + tp_pct / 100)
                sl = entry_price * (1 - sl_pct / 100)
                if high >= tp:
                    exit_price = tp
                    result = "Win"
                    break
                elif low <= sl:
                    exit_price = sl
                    result = "Loss"
                    break
            elif "Sell" in signal:
                tp = entry_price * (1 - tp_pct / 100)
                sl = entry_price * (1 + sl_pct / 100)
                if low <= tp:
                    exit_price = tp
                    result = "Win"
                    break
                elif high >= sl:
                    exit_price = sl
                    result = "Loss"
                    break
        if result == "Win":
            balance += bet_size
        elif result == "Loss":
            balance -= bet_size
        results.append({"Time": ts, "Signal": signal, "Entry": entry_price, "Exit": exit_price, "Result": result, "Balance": balance})
    return pd.DataFrame(results)

def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_short'], line=dict(color='blue', width=1), name='EMA Short'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_long'], line=dict(color='red', width=1), name='EMA Long'))
    fig.update_layout(title=f"Live/Backtest Chart: {asset}", yaxis_title="Price (USDT)", xaxis_rangeslider_visible=False, template="plotly_white")
    return fig

# --- STREAMLIT APP START ---
st.set_page_config(page_title="Trading Signals + Backtesting + Alerts", layout="wide")
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh")

st.title("Trading Signals + Backtesting + Alerts")

# --- SIDEBAR ---
uploaded_file = st.sidebar.file_uploader("Upload historical data (CSV)", type=["csv"])
selected_assets = st.sidebar.multiselect("Select Live Assets", ASSETS, default=ASSETS[:3])
selected_strategy = st.sidebar.selectbox("Select Strategy", ["EMA Cross", "RSI Divergence", "EMA + RSI Combined"])
money_management_strategy = st.sidebar.selectbox("Money Management Strategy", ["Flat"])
take_profit_pct = st.sidebar.number_input("Take Profit %", min_value=0.1, max_value=20.0, value=2.0)
stop_loss_pct = st.sidebar.number_input("Stop Loss %", min_value=0.1, max_value=20.0, value=1.0)
ema_short = st.sidebar.number_input("Short EMA Period", min_value=1, max_value=50, value=5)
ema_long = st.sidebar.number_input("Long EMA Period", min_value=5, max_value=100, value=20)

if 'seen_signals' not in st.session_state:
    st.session_state.seen_signals = set()

# --- HANDLE UPLOAD ---
if uploaded_file:
    st.subheader("Backtesting Uploaded Data")
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = calculate_indicators(df, short=ema_short, long=ema_long)

    if selected_strategy == "EMA Cross":
        signals = detect_ema_cross(df)
    elif selected_strategy == "RSI Divergence":
        signals = detect_rsi_divergence(df)
    elif selected_strategy == "EMA + RSI Combined":
        signals = [s for s in detect_ema_cross(df) if s in detect_rsi_divergence(df)]

    performance = simulate_money_management(signals, df, strategy=money_management_strategy, tp_pct=take_profit_pct, sl_pct=stop_loss_pct)
    st.dataframe(performance)

    if not performance.empty:
        win_rate = (performance['Result'] == 'Win').mean() * 100
        final_balance = performance['Balance'].iloc[-1]
        total_signals = len(performance)
        col1, col2, col3 = st.columns(3)
        col1.metric("Win Rate", f"{win_rate:.2f}%")
        col2.metric("Total Signals", total_signals)
        col3.metric("Final Balance", f"${final_balance:.2f}")

    fig = plot_chart(df, selected_assets[0])
    st.plotly_chart(fig)

    new_signals = [sig for sig in signals if sig not in st.session_state.seen_signals]
    for ts, signal, price in new_signals:
        st.toast(f"{ts} - {signal} at ${price:.2f}", icon="⚡")
        st.audio("https://actions.google.com/sounds/v1/alarms/beep_short.ogg", autoplay=True)
        st.session_state.seen_signals.add((ts, signal, price))
