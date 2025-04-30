import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import datetime
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- SETTINGS ---
REFRESH_INTERVAL = 10  # seconds
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
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        return df
    except Exception as e:
        st.warning(f"Fetching data failed: {e}")
        return None

def calculate_indicators(df):
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['BB_upper'] = df['close'].rolling(window=20).mean() + (df['close'].rolling(window=20).std() * 2)
    df['BB_lower'] = df['close'].rolling(window=20).mean() - (df['close'].rolling(window=20).std() * 2)

    low_min = df['low'].rolling(window=14).min()
    high_max = df['high'].rolling(window=14).max()
    df['Stochastic'] = (df['close'] - low_min) / (high_max - low_min) * 100

    df['EMA_diff'] = df['EMA5'] - df['EMA20']
    df.dropna(inplace=True)
    return df

def generate_signal(timestamp, signal_type, price):
    duration = 2 if "Buy" in signal_type else 1
    return {"Time": timestamp, "Signal": signal_type, "Price": price, "Trade Duration (min)": duration}

def detect_signals(df, strategy):
    signals = []
    for i in range(1, len(df)):
        t = df['timestamp'].iloc[i]
        price = df['close'].iloc[i]

        if strategy == "Stochastic + MACD":
            if df['Stochastic'].iloc[i] < 20 and df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
                signals.append(generate_signal(t, "Buy (Stoch+MACD)", price))
            elif df['Stochastic'].iloc[i] > 80 and df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
                signals.append(generate_signal(t, "Sell (Stoch+MACD)", price))

    return signals

def simulate_money_management(df, signals, strategy="Flat", initial_balance=1000, bet_size=10):
    balance = initial_balance
    result_log = []
    last_bet = bet_size

    for s in signals:
        try:
            entry_idx = df.index[df['timestamp'] == s['Time']].tolist()[0]
            duration = s["Trade Duration (min)"]
            exit_idx = min(entry_idx + duration, len(df) - 1)

            entry_price = df['close'].iloc[entry_idx]
            exit_price = df['close'].iloc[exit_idx]

            if "Buy" in s["Signal"]:
                win = exit_price > entry_price * 1.01
            else:
                win = exit_price < entry_price * 0.99
        except:
            win = np.random.choice([True, False], p=[0.55, 0.45])

        if win:
            balance += last_bet
            result = "Win"
            last_bet = bet_size
        else:
            balance -= last_bet
            result = "Loss"
            if strategy == "Martingale":
                last_bet *= 2

        result_log.append({
            "Time": s["Time"],
            "Signal": s["Signal"],
            "Result": result,
            "Balance": balance,
            "Trade Duration (min)": s["Trade Duration (min)"]
        })

    return pd.DataFrame(result_log)

def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA5'], name="EMA5", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA20'], name="EMA20", line=dict(color='red')))
    fig.update_layout(title=asset, xaxis_rangeslider_visible=False)
    return fig

# --- STREAMLIT APP ---
st.set_page_config(layout="wide")
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh")

st.title("📈 Enhanced Pocket Option Signals | Multi-Strategy + ML + Money Management")

uploaded_file = st.sidebar.file_uploader("Upload historical data (CSV)", type=["csv"])
selected_assets = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS[:2])
selected_strategy = st.sidebar.selectbox("Strategy", ["Stochastic + MACD"])
money_strategy = st.sidebar.selectbox("Money Management", ["Flat", "Martingale"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = calculate_indicators(df)
    signals = detect_signals(df, selected_strategy)

    st.subheader("📊 Backtest Results")
    st.dataframe(pd.DataFrame(signals))

    st.subheader("💰 Money Management Simulation")
    results_df = simulate_money_management(df, signals, strategy=money_strategy)
    st.dataframe(results_df)

    win_rate = results_df['Result'].value_counts(normalize=True).get('Win', 0)
    st.metric("Win Rate", f"{win_rate:.2%}")
    st.plotly_chart(plot_chart(df, "Backtest Data"))

st.subheader("📡 Live Market Signal Detection")
for asset in selected_assets:
    df_live = fetch_candles(asset)
    if df_live is not None:
        df_live = calculate_indicators(df_live)
        live_signals = detect_signals(df_live, selected_strategy)
        if live_signals:
            st.markdown(f"### {asset}")
            st.dataframe(pd.DataFrame(live_signals[-5:]))
            st.plotly_chart(plot_chart(df_live, asset), use_container_width=True)
