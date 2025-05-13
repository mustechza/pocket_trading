import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
import streamlit.components.v1 as components

# --- SETTINGS ---
st.set_page_config(layout="wide")
REFRESH_INTERVAL = 10  # seconds
CANDLE_LIMIT = 500
BINANCE_URL = "https://api.binance.us/api/v3/klines"
ASSETS = ["ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]

# --- AUTO REFRESH ---
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh")

# --- SIDEBAR ---
uploaded_file = st.sidebar.file_uploader("Upload historical data (CSV)", type=["csv"])
selected_assets = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS[:2])
selected_strategy = st.sidebar.selectbox("Strategy", [
    "EMA Cross", "RSI Divergence", "MACD Cross", "Bollinger Band Bounce",
    "Stochastic Oscillator", "EMA + RSI Combined", "ML Model (Random Forest)"
])
money_strategy = st.sidebar.selectbox("Money Management", ["Flat", "Martingale", "Risk %"])
risk_pct = st.sidebar.slider("Risk % per Trade", 1, 10, value=2)
balance_input = st.sidebar.number_input("Initial Balance ($)", value=1000)

ema_short = st.sidebar.number_input("EMA Short Period", 2, 50, value=5)
ema_long = st.sidebar.number_input("EMA Long Period", 5, 100, value=20)
rsi_period = st.sidebar.number_input("RSI Period", 5, 50, value=14)
stoch_period = st.sidebar.number_input("Stochastic Period", 5, 50, value=14)
bb_period = st.sidebar.number_input("Bollinger Band Period", 5, 50, value=20)

# --- FUNCTIONS ---
def to_gmt_plus2(ts):
    return ts + timedelta(hours=2)

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
        df['timestamp'] = df['timestamp'].apply(to_gmt_plus2)
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        return df
    except Exception as e:
        st.warning(f"Fetching data failed: {e}")
        return None

def calculate_indicators(df):
    df['EMA5'] = df['close'].ewm(span=ema_short, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=ema_long, adjust=False).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df['BB_upper'] = df['close'].rolling(window=bb_period).mean() + 2 * df['close'].rolling(window=bb_period).std()
    df['BB_lower'] = df['close'].rolling(window=bb_period).mean() - 2 * df['close'].rolling(window=bb_period).std()

    low_min = df['low'].rolling(window=stoch_period).min()
    high_max = df['high'].rolling(window=stoch_period).max()
    df['Stochastic'] = (df['close'] - low_min) / (high_max - low_min) * 100

    return df

def generate_signal(timestamp, signal_type, price):
    if "Buy" in signal_type:
        duration = 3
    elif "Sell" in signal_type:
        duration = 5
    else:
        duration = 4
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
            if df['EMA5'].iloc[i-1] < df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] > df['EMA20'].iloc[i]:
                signals.append(generate_signal(t, "Buy (EMA Cross)", price))
            elif df['EMA5'].iloc[i-1] > df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] < df['EMA20'].iloc[i]:
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
            if df['EMA5'].iloc[i-1] < df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] > df['EMA20'].iloc[i] and df['RSI'].iloc[i] < 40:
                signals.append(generate_signal(t, "Buy (EMA+RSI)", price))
            elif df['EMA5'].iloc[i-1] > df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] < df['EMA20'].iloc[i] and df['RSI'].iloc[i] > 60:
                signals.append(generate_signal(t, "Sell (EMA+RSI)", price))
    return signals

def train_ml_model(df):
    df = df.copy()
    df['target'] = (df['close'].shift(-2) > df['close']).astype(int)
    features = ['EMA5', 'EMA20', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'Stochastic']
    df = df.dropna(subset=features + ['target'])
    X = df[features]
    y = df['target']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    df['ML_Prediction'] = model.predict(X)
    df['Signal'] = df['ML_Prediction'].map({1: 'Buy (ML)', 0: 'Sell (ML)'})
    df['Trade Duration (min)'] = df['Signal'].apply(lambda x: 3 if "Buy" in x else 5)
    return df[['timestamp', 'Signal', 'close', 'Trade Duration (min)']].rename(columns={'close': 'Price'})

def simulate_money_management(signals, strategy="Flat", initial_balance=1000, risk_pct=2):
    balance = initial_balance
    wins, losses, pnl = 0, 0, []
    last_bet = balance * (risk_pct / 100)
    result_log = []
    for s in signals:
        win = np.random.choice([True, False], p=[0.55, 0.45])
        if strategy == "Risk %":
            bet = balance * (risk_pct / 100)
        else:
            bet = last_bet
        if win:
            balance += bet
            result = "Win"
            wins += 1
            if strategy == "Martingale":
                last_bet = bet
        else:
            balance -= bet
            result = "Loss"
            losses += 1
            if strategy == "Martingale":
                last_bet *= 2
        pnl.append(balance)
        result_log.append({
            "Time": s["Time"], "Signal": s["Signal"], "Result": result,
            "Balance": balance, "Trade Duration (min)": s["Trade Duration (min)"]
        })

    df = pd.DataFrame(result_log)
    max_drawdown = ((df['Balance'].cummax() - df['Balance']) / df['Balance'].cummax()).max()
    roi = ((balance - initial_balance) / initial_balance) * 100
    profit_factor = (wins * bet) / max(losses * bet, 1)

    return df, {
        "Win Rate (%)": 100 * wins / (wins + losses),
        "ROI (%)": roi,
        "Max Drawdown (%)": 100 * max_drawdown,
        "Profit Factor": profit_factor
    }

def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA5'], name="EMA5", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA20'], name="EMA20", line=dict(color='red')))
    fig.update_layout(title=asset, xaxis_rangeslider_visible=False)
    return fig

# --- TITLE ---
st.title("📈 Pocket Option Signals | Live + Backtest + Money Management")

# --- BACKTESTING ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp']).apply(to_gmt_plus2)
    df = calculate_indicators(df)

    if selected_strategy == "ML Model (Random Forest)":
        signals = train_ml_model(df).to_dict(orient='records')
    else:
        signals = detect_signals(df, selected_strategy)

    st.subheader("📌 Last 3 Signal Alerts")
    latest_signals = signals[-3:]
    columns = st.columns(len(latest_signals))
    for i, s in enumerate(latest_signals):
        with columns[i]:
            st.markdown(f"""
            ### 📊 Signal Alert  
            🧭 **{selected_strategy}**  
            🕒 **Entry:** {s['Time'].strftime('%H:%M')}  
            ⌛ **Duration:** {s['Trade Duration (min)']} min  
            🎯 **Price:** {s['Price']}  
            🟩 **Direction:** {s['Signal']}  
            """)

    st.subheader("💰 Money Management Simulation")
    results_df, metrics = simulate_money_management(signals, strategy=money_strategy, initial_balance=balance_input, risk_pct=risk_pct)
    st.dataframe(results_df)

    st.subheader("📈 Performance Metrics")
    st.metric("Win Rate (%)", f"{metrics['Win Rate (%)']:.2f}")
    st.metric("ROI (%)", f"{metrics['ROI (%)']:.2f}")
    st.metric("Max Drawdown (%)", f"{metrics['Max Drawdown (%)']:.2f}")
    st.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")

    st.plotly_chart(plot_chart(df, "Backtest Data"))

# --- LIVE SIGNALS ---
st.subheader("📡 Live Market Signal Detection")
for asset in selected_assets:
    df_live = fetch_candles(asset)
    if df_live is not None:
        df_live = calculate_indicators(df_live)
        if selected_strategy != "ML Model (Random Forest)":
            live_signals = detect_signals(df_live, selected_strategy)
            if live_signals:
                latest = live_signals[-3:]
                st.markdown(f"### {asset}")
                cols = st.columns(len(latest))
                for i, s in enumerate(latest):
                    with cols[i]:
                        st.markdown(f"""
                        ### 📊 Signal Alert  
                        🧭 **{selected_strategy}**  
                        🕒 **Entry:** {s['Time'].strftime('%H:%M')}  
                        ⌛ **Duration:** {s['Trade Duration (min)']} min  
                        🎯 **Price:** {s['Price']}  
                        🟩 **Direction:** {s['Signal']}  
                        """)
        st.plotly_chart(plot_chart(df_live, asset), use_container_width=True)
