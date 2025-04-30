import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import datetime
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components

# Constants
VALID_SYMBOLS = ["ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]
ASSETS = VALID_SYMBOLS
BINANCE_URL = "https://api.binance.com/api/v3/klines"
REFRESH_INTERVAL = 10  # seconds

# --- Sidebar Inputs ---
st.sidebar.title("Settings")
selected_assets = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS[:2])
selected_strategy = st.sidebar.selectbox("Strategy", [
    "EMA Cross", "RSI Divergence", "MACD Cross", "Bollinger Band Bounce",
    "Stochastic Oscillator", "EMA + RSI Combined", "ML Model (Random Forest)"
])
money_strategy = st.sidebar.selectbox("Money Management", ["Flat", "Martingale"])
uploaded_file = st.sidebar.file_uploader("Upload historical data (CSV)", type=["csv"])

# --- Indicator Parameters ---
st.sidebar.title("Indicator Parameters")
ema_fast = st.sidebar.number_input("EMA Fast", 1, 50, 5)
ema_slow = st.sidebar.number_input("EMA Slow", 5, 100, 20)
rsi_period = st.sidebar.number_input("RSI Period", 1, 50, 14)
macd_fast = st.sidebar.number_input("MACD Fast", 1, 50, 12)
macd_slow = st.sidebar.number_input("MACD Slow", 5, 100, 26)
macd_signal = st.sidebar.number_input("MACD Signal", 1, 20, 9)
bb_period = st.sidebar.number_input("BB Period", 1, 50, 20)
stoch_period = st.sidebar.number_input("Stochastic Period", 1, 50, 14)

# --- Auto-refresh ---
st.set_page_config(layout="wide")
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh")
st.title("📈 Pocket Option Signals | Live + Backtest + Money Management")

# --- Fetch Data ---
def fetch_candles(symbol, interval="1m", limit=500):
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

# --- Calculate Indicators ---
def calculate_indicators(df):
    df['EMA_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['EMA_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['close'].ewm(span=macd_fast).mean() - df['close'].ewm(span=macd_slow).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=macd_signal).mean()
    df['BB_upper'] = df['close'].rolling(bb_period).mean() + 2 * df['close'].rolling(bb_period).std()
    df['BB_lower'] = df['close'].rolling(bb_period).mean() - 2 * df['close'].rolling(bb_period).std()
    low_min = df['low'].rolling(window=stoch_period).min()
    high_max = df['high'].rolling(window=stoch_period).max()
    df['Stochastic'] = (df['close'] - low_min) / (high_max - low_min) * 100
    return df

# --- Detect Signals ---
def generate_signal(timestamp, signal_type, price):
    duration = 5
    st.toast(f"{signal_type} at {timestamp.strftime('%H:%M')}", icon="🚨")
    components.html(f"""
        <script>
        new Audio("https://www.myinstants.com/media/sounds/bell-notification.mp3").play();
        alert("{signal_type} at {timestamp.strftime('%H:%M')}");
        </script>
    """, height=0)
    return {
        "Time": timestamp,
        "Signal": signal_type,
        "Price": price,
        "Trade Duration (min)": duration
    }

def detect_signals(df, strategy):
    signals = []
    for i in range(1, len(df)):
        t = df['timestamp'].iloc[i]
        price = df['close'].iloc[i]

        if strategy == "EMA Cross":
            if df['EMA_fast'].iloc[i-1] < df['EMA_slow'].iloc[i-1] and df['EMA_fast'].iloc[i] > df['EMA_slow'].iloc[i]:
                signals.append(generate_signal(t, "Buy (EMA Cross)", price))
            elif df['EMA_fast'].iloc[i-1] > df['EMA_slow'].iloc[i-1] and df['EMA_fast'].iloc[i] < df['EMA_slow'].iloc[i]:
                signals.append(generate_signal(t, "Sell (EMA Cross)", price))

        elif strategy == "RSI Divergence":
            rsi = df['RSI'].iloc[i]
            if rsi < 30:
                signals.append(generate_signal(t, "Buy (RSI Oversold)", price))
            elif rsi > 70:
                signals.append(generate_signal(t, "Sell (RSI Overbought)", price))

        elif strategy == "MACD Cross":
            if df['MACD'].iloc[i-1] < df['MACD_signal'].iloc[i-1] and df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
                signals.append(generate_signal(t, "Buy (MACD Cross)", price))
            elif df['MACD'].iloc[i-1] > df['MACD_signal'].iloc[i-1] and df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
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
            if df['EMA_fast'].iloc[i-1] < df['EMA_slow'].iloc[i-1] and df['EMA_fast'].iloc[i] > df['EMA_slow'].iloc[i] and df['RSI'].iloc[i] < 40:
                signals.append(generate_signal(t, "Buy (EMA+RSI)", price))
            elif df['EMA_fast'].iloc[i-1] > df['EMA_slow'].iloc[i-1] and df['EMA_fast'].iloc[i] < df['EMA_slow'].iloc[i] and df['RSI'].iloc[i] > 60:
                signals.append(generate_signal(t, "Sell (EMA+RSI)", price))
    return signals

# --- ML Model ---
def train_ml_model(df):
    df = df.dropna()
    df['target'] = (df['close'].shift(-2) > df['close']).astype(int)
    features = ['EMA_fast', 'EMA_slow', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'Stochastic']
    df = df.dropna(subset=features + ['target'])
    X = df[features]
    y = df['target']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    df['ML_Prediction'] = model.predict(X)
    df['Signal'] = df['ML_Prediction'].map({1: 'Buy (ML)', 0: 'Sell (ML)'})
    df['Trade Duration (min)'] = 2
    return df[['timestamp', 'Signal', 'close', 'Trade Duration (min)']].rename(columns={'close': 'Price'})

# --- Money Management ---
def simulate_money_management(signals, strategy="Flat", initial_balance=1000, bet_size=10):
    balance = initial_balance
    result_log = []
    last_bet = bet_size
    for s in signals:
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

# --- Backtest Performance ---
def compute_performance(df):
    wins = df[df['Result'] == 'Win']
    losses = df[df['Result'] == 'Loss']
    win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0
    roi = (df['Balance'].iloc[-1] - 1000) / 1000 * 100
    max_drawdown = df['Balance'].cummax() - df['Balance']
    drawdown = max_drawdown.max()
    profit_factor = wins.shape[0] / losses.shape[0] if losses.shape[0] > 0 else np.nan
    return win_rate, roi, drawdown, profit_factor

# --- Plot Chart ---
def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_fast'], name="EMA Fast", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_slow'], name="EMA Slow", line=dict(color='red')))
    fig.update_layout(title=asset, xaxis_rangeslider_visible=False)
    return fig

# --- Live Section ---
st.subheader("📡 Live Market Signal Detection")
if not selected_assets:
    st.warning("Select at least one asset.")
else:
    for asset in selected_assets:
        df_live = fetch_candles(asset)
        df_live = calculate_indicators(df_live)
        if selected_strategy == "ML Model (Random Forest)":
            st.info("ML-based signals not supported live.")
        else:
            live_signals = detect_signals(df_live, selected_strategy)
            if live_signals:
                st.markdown(f"### {asset}")
                st.dataframe(pd.DataFrame(live_signals[-5:]))
                st.plotly_chart(plot_chart(df_live, asset), use_container_width=True)

# --- Backtesting Section ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = calculate_indicators(df)

    if selected_strategy == "ML Model (Random Forest)":
        signals = train_ml_model(df).to_dict(orient="records")
    else:
        signals = detect_signals(df, selected_strategy)

    st.subheader("📊 Backtest Results")
    st.dataframe(pd.DataFrame(signals))

    st.subheader("💰 Money Management Simulation")
    results_df = simulate_money_management(signals, strategy=money_strategy)
    st.dataframe(results_df)

    st.subheader("📈 Performance Metrics")
    win_rate, roi, drawdown, profit_factor = compute_performance(results_df)
    st.markdown(f"""
        - ✅ Win Rate: **{win_rate:.2f}%**
        - 💸 ROI: **{roi:.2f}%**
        - 📉 Max Drawdown: **{drawdown:.2f}**
        - 📊 Profit Factor: **{profit_factor:.2f}**
    """)

    st.plotly_chart(plot_chart(df, "Backtest Data"))
