import streamlit as st
import pandas as pd
import numpy as np
import websocket
import threading
import json
import time
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go

# Timezone
tz = pytz.timezone("Africa/Johannesburg")

# App config
st.set_page_config(page_title="Deriv Signal App", layout="wide")

# --- SESSION STATE INIT ---
if "ws_status" not in st.session_state:
    st.session_state.ws_status = '🔴 Disconnected'

# --- SIDEBAR ---
st.sidebar.title("🔑 Deriv API & Strategy Settings")
api_key = st.sidebar.text_input("Enter your Deriv API Key", type="password")

symbol = st.sidebar.selectbox("Select Market", ["R_100", "R_75", "R_50"])
interval = st.sidebar.selectbox("Candle Interval", ["1m", "5m", "10m"])
strategy = st.sidebar.selectbox("Select Strategy", ["EMA Crossover", "RSI", "MACD", "Volume Spike", "Bollinger Bands", "Stochastic RSI", "Heikin-Ashi Reversal"])
trade_duration = st.sidebar.number_input("Trade Duration (minutes)", 1, 60, 2)
min_confidence = st.sidebar.slider("Min Confidence %", 0, 100, 70)
backtest_btn = st.sidebar.button("🔁 Run Backtest")

# Strategy params
st.sidebar.markdown("### Strategy Parameters")
fast_ema = st.sidebar.number_input("Fast EMA", 5, 50, 10)
slow_ema = st.sidebar.number_input("Slow EMA", 10, 100, 20)
rsi_period = st.sidebar.number_input("RSI Period", 5, 30, 14)
rsi_overbought = st.sidebar.slider("RSI Overbought", 70, 90, 80)
rsi_oversold = st.sidebar.slider("RSI Oversold", 10, 30, 20)
macd_fast = st.sidebar.number_input("MACD Fast", 5, 30, 12)
macd_slow = st.sidebar.number_input("MACD Slow", 10, 50, 26)
macd_signal = st.sidebar.number_input("MACD Signal", 5, 20, 9)

# --- Signal Store ---
signal_store = []
latest_df = pd.DataFrame()

# --- Strategy Logic ---
def apply_strategy(df, strategy_name):
    df = df.copy()
    if strategy_name == "EMA Crossover":
        df['EMA_Fast'] = df['close'].ewm(span=fast_ema).mean()
        df['EMA_Slow'] = df['close'].ewm(span=slow_ema).mean()
        df['Signal'] = np.where(df['EMA_Fast'] > df['EMA_Slow'], 'Buy', 'Sell')
        df['Confidence'] = (abs(df['EMA_Fast'] - df['EMA_Slow']) / df['close']) * 100

    elif strategy_name == "RSI":
        delta = df['close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(rsi_period).mean()
        avg_loss = pd.Series(loss).rolling(rsi_period).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        df['Signal'] = np.where(df['RSI'] < rsi_oversold, 'Buy',
                                np.where(df['RSI'] > rsi_overbought, 'Sell', 'Hold'))
        df['Confidence'] = 100 - abs(df['RSI'] - 50)

    elif strategy_name == "MACD":
        ema_fast = df['close'].ewm(span=macd_fast).mean()
        ema_slow = df['close'].ewm(span=macd_slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=macd_signal).mean()
        df['MACD'] = macd_line
        df['Signal_Line'] = signal_line
        df['Signal'] = np.where(macd_line > signal_line, 'Buy', 'Sell')
        df['Confidence'] = (abs(macd_line - signal_line) / df['close']) * 100

    elif strategy_name == "Volume Spike":
        df['Volume_MA'] = df['volume'].rolling(10).mean()
        df['Signal'] = np.where(df['volume'] > df['Volume_MA'] * 1.5, 'Buy', 'Hold')
        df['Confidence'] = ((df['volume'] - df['Volume_MA']) / df['Volume_MA']) * 100

    elif strategy_name == "Bollinger Bands":
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['Upper'] = df['MA20'] + 2 * df['close'].rolling(window=20).std()
        df['Lower'] = df['MA20'] - 2 * df['close'].rolling(window=20).std()
        df['Signal'] = np.where(df['close'] < df['Lower'], 'Buy', np.where(df['close'] > df['Upper'], 'Sell', 'Hold'))
        df['Confidence'] = (abs(df['close'] - df['MA20']) / df['MA20']) * 100

    elif strategy_name == "Stochastic RSI":
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(rsi_period).mean()
        avg_loss = loss.rolling(rsi_period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        stoch_rsi = ((rsi - rsi.rolling(14).min()) / (rsi.rolling(14).max() - rsi.rolling(14).min())) * 100
        df['StochRSI'] = stoch_rsi
        df['Signal'] = np.where(df['StochRSI'] < 20, 'Buy', np.where(df['StochRSI'] > 80, 'Sell', 'Hold'))
        df['Confidence'] = 100 - abs(df['StochRSI'] - 50)

    elif strategy_name == "Heikin-Ashi Reversal":
        ha_df = df.copy()
        ha_df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_open = [(df['open'][0] + df['close'][0]) / 2]
        for i in range(1, len(df)):
            ha_open.append((ha_open[i - 1] + ha_df['HA_Close'].iloc[i - 1]) / 2)
        ha_df['HA_Open'] = ha_open
        ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close', 'high']].max(axis=1)
        ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close', 'low']].min(axis=1)
        df['HA_Open'] = ha_df['HA_Open']
        df['HA_Close'] = ha_df['HA_Close']
        df['Signal'] = np.where(df['HA_Close'] > df['HA_Open'], 'Buy', 'Sell')
        df['Confidence'] = (abs(df['HA_Close'] - df['HA_Open']) / df['close']) * 100

    df['Signal_Time'] = df.index
    df.dropna(inplace=True)
    return df[(df['Signal'].isin(['Buy', 'Sell'])) & (df['Confidence'] >= min_confidence)].tail(3)
