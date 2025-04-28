import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import requests
import time

# Constants
BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
ASSETS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT']
CANDLE_INTERVAL = '1m'
CANDLE_LIMIT = 500
TRADE_AMOUNT = 100  # Fixed $100

# Streamlit config
st.set_page_config(page_title="PocketOption Signals", layout="wide")

if "balance" not in st.session_state:
    st.session_state.balance = 1000

if "trade_history" not in st.session_state:
    st.session_state.trade_history = []

if "signal_history" not in st.session_state:
    st.session_state.signal_history = []

# Functions
def get_binance_candles(symbol, interval='1m', limit=500):
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(BINANCE_API_URL, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def compute_indicators(df):
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()

    delta = df['close'].diff()
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up).rolling(14).mean()
    roll_down = pd.Series(down).rolling(14).mean()
    rs = roll_up / roll_down
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

def detect_rsi_divergence(df):
    if len(df) < 20:
        return None

    if (df['RSI'].iloc[-1] > 50) and (df['close'].iloc[-1] > df['close'].iloc[-5]):
        return 'CALL'
    elif (df['RSI'].iloc[-1] < 50) and (df['close'].iloc[-1] < df['close'].iloc[-5]):
        return 'PUT'
    else:
        return None

def detect_ema_cross(df):
    if (df['EMA5'].iloc[-1] > df['EMA20'].iloc[-1]) and (df['EMA5'].iloc[-2] <= df['EMA20'].iloc[-2]):
        return 'CALL'
    elif (df['EMA5'].iloc[-1] < df['EMA20'].iloc[-1]) and (df['EMA5'].iloc[-2] >= df['EMA20'].iloc[-2]):
        return 'PUT'
    else:
        return None

def generate_signal(df, strategy):
    if strategy == 'EMA Cross':
        return detect_ema_cross(df)
    elif strategy == 'RSI Divergence':
        return detect_rsi_divergence(df)
    return None

def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Candles"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA5'], line=dict(color='blue', width=1), name='EMA5'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='red', width=1), name='EMA20'))
    fig.update_layout(title=asset, xaxis_rangeslider_visible=False, height=400)
    return fig

# UI
st.title("Pocket Option Signal Generator")
selected_assets = st.multiselect("Select Assets", ASSETS, default=ASSETS)
selected_strategy = st.selectbox("Select Strategy", ['EMA Cross', 'RSI Divergence'])

st.metric("Simulated Balance ($)", round(st.session_state.balance, 2))

chart_placeholders = {asset: st.empty() for asset in selected_assets}

while True:
    all_signals = []

    for asset in selected_assets:
        df = get_binance_candles(asset, interval=CANDLE_INTERVAL, limit=CANDLE_LIMIT)
        df = compute_indicators(df)

        signal = generate_signal(df, selected_strategy)

        if signal:
            all_signals.append((asset, signal))

            # Save signal history
            st.session_state.signal_history.append({
                'Asset': asset,
                'Strategy': selected_strategy,
                'Signal': signal,
                'Time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            })

            # Simulated Trading
            result = 'WIN' if signal == ('CALL' if df['close'].iloc[-1] > df['close'].iloc[-2] else 'PUT') else 'LOSS'

            if result == 'WIN':
                st.session_state.balance += TRADE_AMOUNT
            else:
                st.session_state.balance -= TRADE_AMOUNT

            st.session_state.trade_history.append({
                'Asset': asset,
                'Strategy': selected_strategy,
                'Signal': signal,
                'Result': result,
                'Balance': st.session_state.balance,
                'Time': pd.Timestamp.now()
            })

        # Live Chart
        fig = plot_chart(df, asset)
        chart_placeholders[asset].plotly_chart(fig, use_container_width=True)

    # Display signal history
    st.markdown("## Signal History")
    history_df = pd.DataFrame(st.session_state.signal_history)

    if not history_df.empty:
        history_df = history_df.sort_values(by="Time", ascending=False)
        st.dataframe(history_df[['Time', 'Asset', 'Strategy', 'Signal']], use_container_width=True, height=400)
    else:
        st.info("No signals yet.")

    time.sleep(30)  # Wait 30 seconds before next refresh
