import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import datetime
import telegram
from streamlit_autorefresh import st_autorefresh

# --- SETTINGS ---
REFRESH_INTERVAL = 5
CANDLE_LIMIT = 500
BINANCE_URL = "https://api.binance.com/api/v3/klines"
ASSETS = ["ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]

# --- TELEGRAM SETTINGS ---
TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_TELEGRAM_CHAT_ID"

# --- INITIALIZATION ---
st.set_page_config(page_title="Pocket Option Analyzer", layout="wide")
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh")
if 'seen_signals' not in st.session_state:
    st.session_state.seen_signals = set()

# --- FUNCTIONS ---
def fetch_candles(symbol, interval="1m", limit=500):
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        res = requests.get(BINANCE_URL, params=params, timeout=10)
        data = res.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        st.warning(f"Error fetching data: {e}")
        return None

def calculate_indicators(df):
    df['EMA5'] = df['close'].ewm(span=5).mean()
    df['EMA20'] = df['close'].ewm(span=20).mean()
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

def send_telegram_alert(msg):
    try:
        telegram.Bot(token=TELEGRAM_TOKEN).send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
    except Exception as e:
        st.warning(f"Telegram Error: {e}")

def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA5'], line=dict(color='blue'), name='EMA5'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA20'], line=dict(color='red'), name='EMA20'))
    fig.update_layout(title=f"{asset} Candlestick + EMA", template="plotly_white", xaxis_rangeslider_visible=False)
    return fig

def plot_rsi(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['RSI'], line=dict(color='purple'), name='RSI'))
    fig.update_layout(title="RSI Indicator", yaxis_title="RSI", height=300, template="plotly_white")
    return fig

def simulate_money_management(signals, initial_balance=1000, bet_size=10, strategy="Flat", win_rate=0.55):
    balance = initial_balance
    last_bet_size = bet_size
    results = []

    for ts, signal, price in signals:
        win = np.random.rand() < win_rate
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

# --- SIDEBAR CONFIG ---
st.sidebar.header("Options")
strategy = st.sidebar.selectbox("Signal Strategy", ["EMA Cross", "RSI Divergence"])
mm_strategy = st.sidebar.selectbox("Money Management", ["Flat", "Martingale"])
win_rate = st.sidebar.slider("Simulation Win Rate", 0.0, 1.0, 0.55)
enable_telegram = st.sidebar.checkbox("Enable Telegram Alerts", value=False)
assets_selected = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS[:2])

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["Live Signals", "Backtest CSV", "Simulations"])

# --- LIVE SIGNAL TAB ---
with tab1:
    for asset in assets_selected:
        df = fetch_candles(asset, limit=CANDLE_LIMIT)
        if df is None:
            continue
        df = calculate_indicators(df)

        signals = detect_ema_cross(df) if strategy == "EMA Cross" else detect_rsi_divergence(df)
        st.subheader(f"{asset} - Live Chart")
        st.plotly_chart(plot_chart(df, asset), use_container_width=True)
        st.plotly_chart(plot_rsi(df), use_container_width=True)

        if signals:
            latest_signal = signals[-1]
            key = (asset, latest_signal[1], str(latest_signal[0]))
            if key not in st.session_state.seen_signals:
                st.success(f"{latest_signal[1]} at {latest_signal[0]} for {asset}")
                st.session_state.seen_signals.add(key)
                if enable_telegram:
                    send_telegram_alert(f"{latest_signal[1]} on {asset} at {latest_signal[0]}")

# --- BACKTEST TAB ---
with tab2:
    uploaded = st.file_uploader("Upload Historical CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = calculate_indicators(df)
        signals = detect_ema_cross(df) if strategy == "EMA Cross" else detect_rsi_divergence(df)

        st.plotly_chart(plot_chart(df, "CSV"), use_container_width=True)
        st.plotly_chart(plot_rsi(df), use_container_width=True)
        st.subheader(f"Detected {len(signals)} Signals")
        signal_df = pd.DataFrame(signals, columns=["Time", "Signal", "Price"])
        st.dataframe(signal_df)

        download = signal_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Signals CSV", data=download, file_name="backtest_signals.csv", mime='text/csv')

# --- SIMULATION TAB ---
with tab3:
    if uploaded:
        st.subheader("Simulated Trading")
        mm_results = simulate_money_management(signals, strategy=mm_strategy, win_rate=win_rate)
        st.dataframe(mm_results)

        st.plotly_chart(
            go.Figure(data=[go.Scatter(x=mm_results["Time"], y=mm_results["Balance"], mode="lines+markers")])
            .update_layout(title="Balance Over Time", template="plotly_white"),
            use_container_width=True
        )
    else:
        st.info("Upload a CSV file in the Backtest tab to run simulations.")
