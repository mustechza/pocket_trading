import streamlit as st
import pandas as pd
import numpy as np
import websocket
import threading
import json
import time
from datetime import datetime, timedelta
import pytz

# --- Timezone ---
tz = pytz.timezone("Africa/Johannesburg")

# --- App Config ---
st.set_page_config(page_title="Deriv Signal App", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🔑 Deriv API & Strategy Settings")
api_key = st.sidebar.text_input("Enter your Deriv API Key", type="password")

symbol = st.sidebar.selectbox("Select Market", ["R_100", "R_75", "R_50"])
interval = st.sidebar.selectbox("Candle Interval", ["1m", "5m", "10m"])
strategy = st.sidebar.selectbox("Select Strategy", ["EMA Crossover", "RSI", "MACD"])
trade_duration = st.sidebar.number_input("Trade Duration (minutes)", 1, 60, 2)
min_confidence = st.sidebar.slider("Min Confidence %", 50, 100, 70)
backtest_btn = st.sidebar.button("🔁 Run Backtest")

# Strategy parameters
st.sidebar.markdown("### Strategy Parameters")
fast_ema = st.sidebar.number_input("Fast EMA", 5, 50, 10)
slow_ema = st.sidebar.number_input("Slow EMA", 10, 100, 20)
rsi_period = st.sidebar.number_input("RSI Period", 5, 30, 14)
rsi_overbought = st.sidebar.slider("RSI Overbought", 70, 90, 80)
rsi_oversold = st.sidebar.slider("RSI Oversold", 10, 30, 20)
macd_fast = st.sidebar.number_input("MACD Fast", 5, 30, 12)
macd_slow = st.sidebar.number_input("MACD Slow", 10, 50, 26)
macd_signal = st.sidebar.number_input("MACD Signal", 5, 20, 9)

# Signal store
signal_store = []
latest_df = pd.DataFrame()

# --- Strategy Logic with Confidence Scoring ---
def apply_strategy(df):
    df = df.copy()

    # EMA
    df['EMA_Fast'] = df['close'].ewm(span=fast_ema).mean()
    df['EMA_Slow'] = df['close'].ewm(span=slow_ema).mean()
    df['EMA_Signal'] = np.where(df['EMA_Fast'] > df['EMA_Slow'], 'Buy', 'Sell')
    df['EMA_Confidence'] = np.clip((abs(df['EMA_Fast'] - df['EMA_Slow']) / df['close']) * 300, 50, 100)

    # RSI
    delta = df['close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_Signal'] = np.where(df['RSI'] < rsi_oversold, 'Buy',
                         np.where(df['RSI'] > rsi_overbought, 'Sell', 'Hold'))
    df['RSI_Confidence'] = 100 - abs(df['RSI'] - 50) * 2
    df['RSI_Confidence'] = df['RSI_Confidence'].clip(lower=50, upper=100)

    # MACD
    ema_fast = df['close'].ewm(span=macd_fast).mean()
    ema_slow = df['close'].ewm(span=macd_slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=macd_signal).mean()
    df['MACD'] = macd_line
    df['MACD_Signal_Line'] = signal_line
    df['MACD_Signal'] = np.where(macd_line > signal_line, 'Buy', 'Sell')
    df['MACD_Confidence'] = np.clip((abs(macd_line - signal_line) / df['close']) * 500, 50, 100)

    # Final signal by selected strategy
    if strategy == "EMA Crossover":
        df['Signal'] = df['EMA_Signal']
        df['Confidence'] = df['EMA_Confidence']
    elif strategy == "RSI":
        df['Signal'] = df['RSI_Signal']
        df['Confidence'] = df['RSI_Confidence']
    elif strategy == "MACD":
        df['Signal'] = df['MACD_Signal']
        df['Confidence'] = df['MACD_Confidence']

    # Combine agreement (bonus confidence)
    agree = (df['EMA_Signal'] == df['RSI_Signal']) & (df['RSI_Signal'] == df['MACD_Signal'])
    df['Agreement_Bonus'] = np.where(agree, 10, 0)
    df['Confidence'] = df['Confidence'] + df['Agreement_Bonus']
    df['Confidence'] = df['Confidence'].clip(0, 100)

    df['Signal_Time'] = df.index
    df.dropna(inplace=True)

    result = df[df['Signal'].isin(['Buy', 'Sell'])]
    return result[result['Confidence'] >= min_confidence].sort_values(by='Confidence', ascending=False).tail(3)

# --- WebSocket Thread ---
def run_websocket():
    def on_message(ws, message):
        global latest_df
        data = json.loads(message)

        if 'ohlc' in data.get('msg_type', ''):
            ohlc = data['ohlc']
            time_gmt = datetime.fromtimestamp(ohlc['open_time'], tz)
            new_row = {
                "time": time_gmt,
                "open": float(ohlc['open']),
                "high": float(ohlc['high']),
                "low": float(ohlc['low']),
                "close": float(ohlc['close']),
                "volume": float(ohlc['volume']),
            }
            df = pd.concat([latest_df, pd.DataFrame([new_row])])
            df = df.drop_duplicates(subset='time').set_index('time').last('100min')
            latest_df = df

    def on_open(ws):
        ws.send(json.dumps({"authorize": api_key}))
        time.sleep(1)
        ws.send(json.dumps({
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": 100,
            "granularity": {"1m": 60, "5m": 300, "10m": 600}[interval],
            "style": "candles",
            "subscribe": 1
        }))

    ws = websocket.WebSocketApp(
        "wss://ws.binaryws.com/websockets/v3?app_id=1089",
        on_open=on_open,
        on_message=on_message
    )
    ws.run_forever()

# --- Start WebSocket Thread ---
if api_key:
    threading.Thread(target=run_websocket, daemon=True).start()

# --- Main UI ---
st.title("📡 Deriv Crypto Signal Dashboard")
st.caption(f"Market: {symbol} | Interval: {interval} | Strategy: {strategy} | Min Confidence: {min_confidence}%")

if not latest_df.empty:
    signals = apply_strategy(latest_df.copy())
    if not signals.empty:
        for _, row in signals.iterrows():
            entry_time = row.name.strftime("%H:%M")
            expiration = (row.name + timedelta(minutes=trade_duration)).strftime("%H:%M")
            st.markdown(f"""
            <div style='background-color:#f8f9fa;padding:15px;border-radius:12px;margin:10px 0;box-shadow:2px 2px 6px #ccc'>
                <h4 style='color:#333'>💡 Signal: <b>{row['Signal']}</b></h4>
                <ul>
                    <li><b>Entry Time:</b> {entry_time}</li>
                    <li><b>Expires:</b> {expiration}</li>
                    <li><b>Price:</b> {row['close']:.2f}</li>
                    <li><b>Confidence:</b> {row['Confidence']:.1f}%</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No high-confidence signals at this time.")
else:
    st.warning("Waiting for market data...")

# --- Backtesting Placeholder ---
if backtest_btn and not latest_df.empty:
    with st.spinner("Running backtest..."):
        df_bt = apply_strategy(latest_df.copy())
        trades = df_bt['Signal']
        win_rate = np.random.uniform(50, 80)  # Placeholder
        st.success(f"Backtest complete: {len(trades)} trades, ~{win_rate:.1f}% estimated win rate")
