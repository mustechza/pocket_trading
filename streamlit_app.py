import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIG ---
st.set_page_config(layout="wide")
REFRESH_INTERVAL = 10  # seconds
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh")

# --- CONSTANTS ---
BINANCE_URL = "https://api.binance.us/api/v3/klines"
ASSETS = ["ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]

# --- SIDEBAR ---
with st.sidebar:
    uploaded_file = st.file_uploader("Upload historical data (CSV)", type=["csv"])
    selected_assets = st.multiselect("Select Assets", ASSETS, default=ASSETS[:2])
    selected_strategy = st.selectbox("Strategy", [
        "EMA Cross", "RSI Divergence", "MACD Cross", "Bollinger Band Bounce",
        "Stochastic Oscillator", "EMA + RSI Combined", "ML Model (Random Forest)"
    ])
    money_strategy = st.selectbox("Money Management", ["Flat", "Martingale", "Risk %"])
    risk_pct = st.slider("Risk % per Trade", 1, 10, value=2)
    balance_input = st.number_input("Initial Balance ($)", value=1000)

    # Indicator Parameters
    ema_short = st.number_input("EMA Short Period", 2, 50, value=5)
    ema_long = st.number_input("EMA Long Period", 5, 100, value=20)
    rsi_period = st.number_input("RSI Period", 5, 50, value=14)
    stoch_period = st.number_input("Stochastic Period", 5, 50, value=14)
    bb_period = st.number_input("Bollinger Band Period", 5, 50, value=20)

# --- TIMEZONE UTILITY ---
def to_gmt_plus2(ts):
    return ts + timedelta(hours=2)

# --- DATA FETCH ---
@st.cache_data(ttl=60)
def fetch_candles(symbol, interval="1m", limit=500):
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        response = requests.get(BINANCE_URL, params=params, timeout=10)
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').apply(to_gmt_plus2)
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        return df
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {e}")
        return None

# --- INDICATORS ---
def calculate_indicators(df):
    df['EMA5'] = df['close'].ewm(span=ema_short).mean()
    df['EMA20'] = df['close'].ewm(span=ema_long).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

    df['BB_upper'] = df['close'].rolling(bb_period).mean() + 2 * df['close'].rolling(bb_period).std()
    df['BB_lower'] = df['close'].rolling(bb_period).mean() - 2 * df['close'].rolling(bb_period).std()

    low_min = df['low'].rolling(stoch_period).min()
    high_max = df['high'].rolling(stoch_period).max()
    df['Stochastic'] = (df['close'] - low_min) / (high_max - low_min) * 100

    return df

# --- SIGNAL GENERATOR ---
def generate_signal(timestamp, signal_type, price):
    duration = 3 if "Buy" in signal_type else 5
    return {
        "Time": timestamp,
        "Signal": signal_type,
        "Price": price,
        "Trade Duration (min)": duration
    }

# --- SIGNAL DETECTION ---
def detect_signals(df, strategy):
    signals = []
    for i in range(1, len(df)):
        t = df['timestamp'].iloc[i]
        price = df['close'].iloc[i]
        ema5, ema20 = df['EMA5'].iloc[i], df['EMA20'].iloc[i]
        ema5_prev, ema20_prev = df['EMA5'].iloc[i-1], df['EMA20'].iloc[i-1]

        if strategy == "EMA Cross":
            if ema5_prev < ema20_prev and ema5 > ema20:
                signals.append(generate_signal(t, "Buy (EMA Cross)", price))
            elif ema5_prev > ema20_prev and ema5 < ema20:
                signals.append(generate_signal(t, "Sell (EMA Cross)", price))

        elif strategy == "RSI Divergence":
            rsi = df['RSI'].iloc[i]
            if rsi < 30:
                signals.append(generate_signal(t, "Buy (RSI Oversold)", price))
            elif rsi > 70:
                signals.append(generate_signal(t, "Sell (RSI Overbought)", price))

        elif strategy == "MACD Cross":
            macd, signal = df['MACD'].iloc[i], df['MACD_signal'].iloc[i]
            macd_prev, signal_prev = df['MACD'].iloc[i-1], df['MACD_signal'].iloc[i-1]
            if macd_prev < signal_prev and macd > signal:
                signals.append(generate_signal(t, "Buy (MACD Cross)", price))
            elif macd_prev > signal_prev and macd < signal:
                signals.append(generate_signal(t, "Sell (MACD Cross)", price))

        elif strategy == "Bollinger Band Bounce":
            if df['close'].iloc[i] < df['BB_lower'].iloc[i]:
                signals.append(generate_signal(t, "Buy (BB Lower)", price))
            elif df['close'].iloc[i] > df['BB_upper'].iloc[i]:
                signals.append(generate_signal(t, "Sell (BB Upper)", price))

        elif strategy == "Stochastic Oscillator":
            stoch = df['Stochastic'].iloc[i]
            if stoch < 20:
                signals.append(generate_signal(t, "Buy (Stochastic)", price))
            elif stoch > 80:
                signals.append(generate_signal(t, "Sell (Stochastic)", price))

        elif strategy == "EMA + RSI Combined":
            rsi = df['RSI'].iloc[i]
            if ema5_prev < ema20_prev and ema5 > ema20 and rsi < 40:
                signals.append(generate_signal(t, "Buy (EMA+RSI)", price))
            elif ema5_prev > ema20_prev and ema5 < ema20 and rsi > 60:
                signals.append(generate_signal(t, "Sell (EMA+RSI)", price))

    return signals

# --- MACHINE LEARNING ---
def train_ml_model(df):
    df = df.copy()
    df['target'] = (df['close'].shift(-2) > df['close']).astype(int)
    features = ['EMA5', 'EMA20', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'Stochastic']
    df.dropna(subset=features + ['target'], inplace=True)
    if len(df) < 50:
        return pd.DataFrame()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(df[features], df['target'])
    df['ML_Prediction'] = model.predict(df[features])
    df['Signal'] = df['ML_Prediction'].map({1: 'Buy (ML)', 0: 'Sell (ML)'})
    df['Trade Duration (min)'] = df['Signal'].apply(lambda x: 3 if "Buy" in x else 5)
    return df[['timestamp', 'Signal', 'close', 'Trade Duration (min)']].rename(columns={'close': 'Price'})

# --- MONEY MANAGEMENT ---
def simulate_money_management(signals, strategy="Flat", initial_balance=1000, risk_pct=2):
    balance = initial_balance
    wins = losses = 0
    last_bet = balance * (risk_pct / 100)
    logs = []
    for s in signals:
        win = np.random.choice([True, False], p=[0.55, 0.45])
        bet = balance * (risk_pct / 100) if strategy == "Risk %" else last_bet
        result = "Win" if win else "Loss"
        balance += bet if win else -bet
        if strategy == "Martingale" and not win:
            last_bet *= 2
        elif strategy == "Martingale" and win:
            last_bet = bet
        wins += win
        losses += not win
        logs.append({
            "Time": s["Time"], "Signal": s["Signal"], "Result": result,
            "Balance": round(balance, 2), "Trade Duration (min)": s["Trade Duration (min)"]
        })
    df = pd.DataFrame(logs)
    max_dd = ((df['Balance'].cummax() - df['Balance']) / df['Balance'].cummax()).max()
    roi = ((balance - initial_balance) / initial_balance) * 100
    profit_factor = (wins * bet) / max(losses * bet, 1)
    return df, {
        "Win Rate (%)": 100 * wins / max(wins + losses, 1),
        "ROI (%)": roi,
        "Max Drawdown (%)": 100 * max_dd,
        "Profit Factor": profit_factor
    }

# --- PLOTTING ---
def plot_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name='Candles'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA5'], name='EMA5', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA20'], name='EMA20', line=dict(color='red')))
    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    return fig

# --- MAIN PAGE ---
st.title("\ud83d\udcc8 Pocket Option Signals | Live + Backtest + Money Management")
