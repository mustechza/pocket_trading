import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import plotly.graph_objs as go
import ta
import telegram

# ========== SETTINGS ==========

CANDLES = 500
REFRESH_INTERVAL = 5  # seconds
TRADE_AMOUNT = 100  # fixed manual trade amount in $

TELEGRAM_BOT_TOKEN = 'YOUR_TELEGRAM_BOT_TOKEN'
TELEGRAM_CHAT_ID = 'YOUR_TELEGRAM_CHAT_ID'

assets = ['ETH', 'SOL', 'ADA', 'BNB', 'XRP', 'LTC']

strategy_options = ['EMA Cross', 'RSI Divergence']

asset_symbol_mapping = {
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT',
    'ADA': 'ADAUSDT',
    'BNB': 'BNBUSDT',
    'XRP': 'XRPUSDT',
    'LTC': 'LTCUSDT'
}

# ========== FUNCTIONS ==========

def send_telegram_alert(message):
    try:
        bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
    except Exception as e:
        st.error(f"Telegram Error: {e}")

def fetch_binance_data(symbol):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1m&limit={CANDLES}"
    try:
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df['close'] = df['close'].astype(float)
        df['open'] = df['open'].astype(float)
        return df
    except Exception as e:
        st.warning(f"Fetching data failed, retrying... {e}")
        return None

def calculate_indicators(df):
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    return df

def detect_ema_cross(df):
    if len(df) < 21:
        return None
    if (df['EMA5'].iloc[-1] > df['EMA20'].iloc[-1]) and (df['EMA5'].iloc[-2] <= df['EMA20'].iloc[-2]):
        return "BUY"
    if (df['EMA5'].iloc[-1] < df['EMA20'].iloc[-1]) and (df['EMA5'].iloc[-2] >= df['EMA20'].iloc[-2]):
        return "SELL"
    return None

def detect_rsi_divergence(df):
    if len(df) < 20:
        return None
    if df['RSI'].iloc[-1] > 70:
        return "SELL"
    elif df['RSI'].iloc[-1] < 30:
        return "BUY"
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
        name='Candles'
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA5'], line=dict(color='blue', width=1), name='EMA5'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='red', width=1), name='EMA20'))
    fig.update_layout(title=f'{asset} - Live Chart', xaxis_rangeslider_visible=False)
    return fig

# ========== STREAMLIT APP ==========

st.set_page_config(page_title="Pocket Option Signal Scanner", layout="wide")

st.title("Pocket Option Signal Scanner (Binance Data)")
selected_asset = st.selectbox("Select Asset", assets)
selected_strategy = st.selectbox("Select Strategy", strategy_options)

placeholder = st.empty()

trade_log = []

while True:
    with placeholder.container():
        st.info("Searching for signals...")
        symbol = asset_symbol_mapping[selected_asset]
        df = fetch_binance_data(symbol)
        
        if df is not None:
            df = calculate_indicators(df)
            signal = generate_signal(df, selected_strategy)

            st.plotly_chart(plot_chart(df, selected_asset), use_container_width=True)

            if signal:
                st.success(f"Signal Found: {signal} {selected_asset}")
                send_telegram_alert(f"Signal {signal} detected for {selected_asset} on {selected_strategy} strategy!")
                
                trade_log.append({
                    'Asset': selected_asset,
                    'Strategy': selected_strategy,
                    'Signal': signal,
                    'Amount': TRADE_AMOUNT,
                    'Timestamp': pd.Timestamp.now()
                })

                st.dataframe(pd.DataFrame(trade_log))
            else:
                st.info("No Signal Yet...")

    time.sleep(REFRESH_INTERVAL)
