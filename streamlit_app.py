import streamlit as st
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from datetime import datetime
import time

# Page Config
st.set_page_config(page_title="TradeSignalPro", layout="wide", initial_sidebar_state="expanded")

# Session State for Unique Alerts
if "last_signals" not in st.session_state:
    st.session_state.last_signals = {}

# Sidebar
st.sidebar.header("Settings")
selected_assets = st.sidebar.multiselect("Select Assets", ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"], default=["AAPL"])
selected_interval = st.sidebar.selectbox("Select Interval", ["1m", "5m", "15m", "1h", "1d"], index=1)
strategy = st.sidebar.selectbox("Select Strategy", ["EMA Cross", "RSI Oversold/Overbought", "MACD Crossover", "Bollinger Band Breakout", "Stochastic Oversold/Overbought"])
trade_duration = st.sidebar.number_input("Trade Duration (minutes)", min_value=1, max_value=60, value=5)
auto_refresh = st.sidebar.checkbox("Auto Refresh Every 10 sec")

# Title
st.title("📊 TradeSignalPro")
st.markdown("Get trading signals with confidence scores in real-time 🚀")

# Browser Alert Function
def show_browser_alert(message):
    st.markdown(f"<script>alert('{message}')</script>", unsafe_allow_html=True)

# Fetch Market Data
def fetch_data(symbol, interval, lookback="100"):
    interval_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "60m", "1d": "1d"}
    df = yf.download(symbol, period="7d", interval=interval_map[interval])
    df = df.tail(int(lookback))
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df.rename(columns={"index": "timestamp"}, inplace=True)
    return df

# Add Indicators
def add_indicators(df):
    df['EMA5'] = ta.ema(df['Close'], length=5)
    df['EMA20'] = ta.ema(df['Close'], length=20)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    bb = ta.bbands(df['Close'])
    df['BB_upper'] = bb['BBU_20_2.0']
    df['BB_lower'] = bb['BBL_20_2.0']
    stoch = ta.stoch(df['High'], df['Low'], df['Close'])
    df['Stochastic'] = stoch['STOCHk_14_3_3']
    return df

# Confidence Score Function (-5 to +5)
def compute_confidence(df_row):
    score = 0
    # EMA Cross
    if df_row['EMA5'] > df_row['EMA20']:
        score += 1
    else:
        score -= 1
    # RSI
    if df_row['RSI'] < 30:
        score += 1
    elif df_row['RSI'] > 70:
        score -= 1
    # MACD
    if df_row['MACD'] > df_row['MACD_signal']:
        score += 1
    else:
        score -= 1
    # Bollinger Bands
    if df_row['Close'] < df_row['BB_lower']:
        score += 1
    elif df_row['Close'] > df_row['BB_upper']:
        score -= 1
    # Stochastic
    if df_row['Stochastic'] < 20:
        score += 1
    elif df_row['Stochastic'] > 80:
        score -= 1
    return score

# Signal Generator
def generate_signal(t, signal, price, confidence):
    return {
        "Time": t.strftime("%Y-%m-%d %H:%M"),
        "Signal": signal,
        "Price": price,
        "Trade Duration (min)": trade_duration,
        "Confidence": confidence
    }

# Detect Signals
def detect_signals(df, strategy):
    signals = []
    for i in range(1, len(df)):
        t = df['timestamp'].iloc[i]
        price = df['Close'].iloc[i]
        confidence = compute_confidence(df.iloc[i])

        if strategy == "EMA Cross":
            if df['EMA5'].iloc[i] > df['EMA20'].iloc[i] and df['EMA5'].iloc[i-1] <= df['EMA20'].iloc[i-1]:
                signals.append(generate_signal(t, "Buy (EMA Cross)", price, confidence))
            elif df['EMA5'].iloc[i] < df['EMA20'].iloc[i] and df['EMA5'].iloc[i-1] >= df['EMA20'].iloc[i-1]:
                signals.append(generate_signal(t, "Sell (EMA Cross)", price, confidence))

        elif strategy == "RSI Oversold/Overbought":
            if df['RSI'].iloc[i] < 30:
                signals.append(generate_signal(t, "Buy (RSI Oversold)", price, confidence))
            elif df['RSI'].iloc[i] > 70:
                signals.append(generate_signal(t, "Sell (RSI Overbought)", price, confidence))

        elif strategy == "MACD Crossover":
            if df['MACD'].iloc[i] > df['MACD_signal'].iloc[i] and df['MACD'].iloc[i-1] <= df['MACD_signal'].iloc[i-1]:
                signals.append(generate_signal(t, "Buy (MACD Cross)", price, confidence))
            elif df['MACD'].iloc[i] < df['MACD_signal'].iloc[i] and df['MACD'].iloc[i-1] >= df['MACD_signal'].iloc[i-1]:
                signals.append(generate_signal(t, "Sell (MACD Cross)", price, confidence))

        elif strategy == "Bollinger Band Breakout":
            if df['Close'].iloc[i] < df['BB_lower'].iloc[i]:
                signals.append(generate_signal(t, "Buy (BB Break)", price, confidence))
            elif df['Close'].iloc[i] > df['BB_upper'].iloc[i]:
                signals.append(generate_signal(t, "Sell (BB Break)", price, confidence))

        elif strategy == "Stochastic Oversold/Overbought":
            if df['Stochastic'].iloc[i] < 20:
                signals.append(generate_signal(t, "Buy (Stochastic Oversold)", price, confidence))
            elif df['Stochastic'].iloc[i] > 80:
                signals.append(generate_signal(t, "Sell (Stochastic Overbought)", price, confidence))

    return signals

# Color Confidence Column
def color_confidence(val):
    if val >= 3:
        return 'color: green; font-weight: bold'
    elif val <= -3:
        return 'color: red; font-weight: bold'
    else:
        return 'color: gray'

# Live Signal Detector
def live_signal_detector(asset):
    df = fetch_data(asset, selected_interval)
    df = add_indicators(df)
    signals = detect_signals(df, strategy)
    return signals

# Live Signals Display
st.header("🔔 Live Signals")

for asset in selected_assets:
    live_signals = live_signal_detector(asset)

    if live_signals:
        latest_signal = live_signals[-1]
        signal_id = (latest_signal['Signal'], latest_signal['Time'])

        # Show unique alerts only once per new signal
        last_signal_id = st.session_state.last_signals.get(asset)
        if signal_id != last_signal_id:
            show_browser_alert(f"New signal for {asset}: {latest_signal['Signal']} (Confidence: {latest_signal['Confidence']})")
            st.session_state.last_signals[asset] = signal_id

        # Show Table
        st.markdown(f"### {asset}")
        signals_df = pd.DataFrame(live_signals[-5:])  # last 5 signals
        styled_df = signals_df.style.applymap(color_confidence, subset=['Confidence'])
        st.dataframe(styled_df, use_container_width=True)

if auto_refresh:
    time.sleep(10)
    st.experimental_rerun()
