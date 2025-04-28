import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import time
import datetime

# ---------------- SETTINGS ----------------

TELEGRAM_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"

POCKET_ASSETS = ['ETHUSD', 'SOLUSD', 'ADAUSD', 'BNBUSD', 'XRPUSD', 'LTCUSD']

# --------------- FUNCTIONS ----------------

def fetch_data(asset, timeframe, candles):
    url = f"https://api.polygon.io/v2/aggs/ticker/X:{asset}/range/{timeframe}/minute/{candles}?adjusted=true&sort=asc&apiKey=YOUR_POLYGON_API_KEY"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['results']
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close'})
        return df[['timestamp', 'open', 'high', 'low', 'close']]
    else:
        return pd.DataFrame()

def calculate_indicators(df):
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    return df

def detect_ema_cross(df):
    if len(df) < 2:
        return None
    if (df['EMA5'].iloc[-1] > df['EMA20'].iloc[-1]) and (df['EMA5'].iloc[-2] <= df['EMA20'].iloc[-2]):
        return 'CALL'
    if (df['EMA5'].iloc[-1] < df['EMA20'].iloc[-1]) and (df['EMA5'].iloc[-2] >= df['EMA20'].iloc[-2]):
        return 'PUT'
    return None

def detect_rsi_divergence(df):
    if len(df) < 20:
        return None
    if df['RSI'].iloc[-1] > 70:
        return 'PUT'
    if df['RSI'].iloc[-1] < 30:
        return 'CALL'
    return None

def generate_signal(df, strategy):
    if strategy == "EMA Cross":
        return detect_ema_cross(df)
    elif strategy == "RSI Divergence":
        return detect_rsi_divergence(df)
    return None

def send_telegram_alert(asset, timeframe, signal):
    message = f"New Signal:\nAsset: {asset}\nTimeframe: {timeframe}\nSignal: {signal}"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    requests.post(url, data=data)

def simulate_trade(df, entry_index, direction, candles_ahead):
    if entry_index + candles_ahead >= len(df):
        return None
    entry_price = df['close'].iloc[entry_index]
    exit_price = df['close'].iloc[entry_index + candles_ahead]
    if direction == 'CALL':
        return 'WIN' if exit_price > entry_price else 'LOSS'
    else:
        return 'WIN' if exit_price < entry_price else 'LOSS'

def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'],
                                 open=df['open'],
                                 high=df['high'],
                                 low=df['low'],
                                 close=df['close'],
                                 name=asset))
    fig.update_layout(title=f"Live Chart - {asset}", xaxis_rangeslider_visible=False)
    return fig

# ------------------ UI -------------------

st.set_page_config(page_title="Pocket Option Signal App", layout="wide")

st.title("Pocket Option Signals - EMA Cross & RSI Divergence")

col1, col2, col3 = st.columns(3)
with col1:
    selected_asset = st.selectbox("Select Asset", POCKET_ASSETS)
with col2:
    selected_strategy = st.selectbox("Select Strategy", ["EMA Cross", "RSI Divergence"])
with col3:
    timeframe = st.selectbox("Timeframe", ["1", "5"], format_func=lambda x: f"{x}m")

candle_count = 500
simulation_candles = st.slider("Candles Ahead for Simulation (Win/Loss Check)", 1, 10, 3)

trade_log = []

balance = 100  # Starting Balance

# ----------------- MAIN LOOP -----------------

status_placeholder = st.empty()
chart_placeholder = st.empty()
table_placeholder = st.empty()

while True:
    df = fetch_data(selected_asset, timeframe, candle_count)
    if df.empty:
        status_placeholder.warning("Fetching data failed, retrying...")
        time.sleep(5)
        continue

    df = calculate_indicators(df)
    signal = generate_signal(df, selected_strategy)

    if signal:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        result = simulate_trade(df, -1, signal, simulation_candles)
        if result == 'WIN':
            balance += 5
        elif result == 'LOSS':
            balance -= 5

        trade_log.append({
            "Time": now,
            "Asset": selected_asset,
            "Strategy": selected_strategy,
            "Signal": signal,
            "Result": result,
            "Balance": balance
        })

        send_telegram_alert(selected_asset, timeframe, signal)
        status_placeholder.success(f"Signal Found: {signal} ({now})")
    else:
        status_placeholder.info("Searching for signals...")

    chart_placeholder.plotly_chart(plot_chart(df, selected_asset), use_container_width=True)

    if trade_log:
        log_df = pd.DataFrame(trade_log)
        winrate = (log_df['Result'] == 'WIN').mean() * 100
        table_placeholder.dataframe(log_df)
        st.metric("Winrate %", f"{winrate:.2f}%")
        st.metric("Current Balance", f"${balance:.2f}")

    time.sleep(5)  # Auto-refresh every 5 seconds
    st.experimental_rerun()
