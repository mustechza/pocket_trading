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

def calculate_indicators(df):
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def detect_ema_cross(df):
    signals = []
    for i in range(1, len(df)):
        if df['EMA5'].iloc[i-1] < df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] > df['EMA20'].iloc[i]:
            signals.append((df['timestamp'].iloc[i], "Buy Signal (EMA Cross UP)", df['close'].iloc[i]))
        elif df['EMA5'].iloc[i-1] > df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] < df['EMA20'].iloc[i]:
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

def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candles'
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['EMA5'],
        line=dict(color='blue', width=1),
        name='EMA5'
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['EMA20'],
        line=dict(color='red', width=1),
        name='EMA20'
    ))
    fig.update_layout(
        title=f"Live/Backtest Chart: {asset}",
        yaxis_title="Price (USDT)",
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    return fig

def simulate_money_management(signals, initial_balance=1000, bet_size=10, strategy="Flat"):
    balance = initial_balance
    results = []
    last_bet_size = bet_size

    for ts, signal, price in signals:
        win = np.random.choice([True, False], p=[0.55, 0.45])  # 55% win chance

        if win:
            balance += last_bet_size
            result = "Win"
            if strategy == "Martingale":
                last_bet_size = bet_size
        else:
            balance -= last_bet_size
            result = "Loss"
            if strategy == "Martingale":
                last_bet_size *= 2

        results.append({"Time": ts, "Signal": signal, "Price": price, "Result": result, "Balance": balance})

    return pd.DataFrame(results)

# --- STREAMLIT APP START ---
st.set_page_config(page_title="Pocket Option Signals + Backtest", layout="wide")
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh")

st.title("Pocket Option Trading Signals + Backtesting + MM Simulation")

# --- SIDEBAR ---
uploaded_file = st.sidebar.file_uploader("Upload historical data (CSV)", type=["csv"])

if st.sidebar.button("Download Sample CSV"):
    sample_data = {
        "timestamp": pd.date_range(end=datetime.datetime.now(), periods=500, freq='1T'),
        "open": np.random.rand(500) * 100,
        "high": np.random.rand(500) * 100 + 1,
        "low": np.random.rand(500) * 100 - 1,
        "close": np.random.rand(500) * 100,
        "volume": np.random.randint(1, 1000, size=500)
    }
    sample_df = pd.DataFrame(sample_data)
    st.download_button("Click to download", data=sample_df.to_csv(index=False), file_name="sample_data.csv", mime='text/csv')

selected_assets = st.sidebar.multiselect("Select Live Assets", ASSETS, default=ASSETS[:3])
selected_strategy = st.sidebar.selectbox("Select Strategy", ["EMA Cross", "RSI Divergence"])
money_management_strategy = st.sidebar.selectbox("Money Management Strategy", ["Flat", "Martingale"])

if 'seen_signals' not in st.session_state:
    st.session_state.seen_signals = set()

# --- HANDLE UPLOAD ---
if uploaded_file:
    st.subheader("Backtesting Uploaded Data")
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = calculate_indicators(df)

    if selected_strategy == "EMA Cross":
        signals = detect_ema_cross(df)
    elif selected_strategy == "RSI Divergence":
        signals = detect_rsi_divergence(df)

    st.plotly_chart(plot_chart(df, "Uploaded CSV"), use_container_width=True)
    st.subheader(f"Detected {len(signals)} signals:")
    st.dataframe(pd.DataFrame(signals, columns=["Time", "Signal", "Price"]))

    st.subheader("Simulated Money Management Results:")
    mm_df = simulate_money_management(signals, strategy=money_management_strategy)
    st.dataframe(mm_df)

    st.download_button("Download Signals as CSV", data=pd.DataFrame(signals).to_csv(index=False), file_name="signals_results.csv", mime="text/csv")

else:
    for asset in selected_assets:
        df = fetch_candles(asset, limit=CANDLE_LIMIT)
        if df is None:
            continue
        df = calculate_indicators(df)
        if selected_strategy == "EMA Cross":
            signals = detect_ema_cross(df)
        elif selected_strategy == "RSI Divergence":
            signals = detect_rsi_divergence(df)

        st.subheader(f"Asset: {asset}")
        st.plotly_chart(plot_chart(df, asset), use_container_width=True)

        if signals:
            latest_signal = signals[-1]
            if (asset, latest_signal[1]) not in st.session_state.seen_signals:
                st.success(f"{latest_signal[1]} on {asset} at {latest_signal[0]}")
                st.session_state.seen_signals.add((asset, latest_signal[1]))

st.info("Fetching or analyzing latest data...")
