import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import datetime
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import ta
import streamlit.components.v1 as components

# --- SETTINGS ---
REFRESH_INTERVAL = 10  # seconds
CANDLE_LIMIT = 500
BINANCE_URL = "https://api.binance.com/api/v3/klines"
ASSETS = ["ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]

# --- FUNCTIONS ---

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
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        return df
    except Exception as e:
        st.warning(f"Fetching data failed: {e}")
        return None

def calculate_indicators(df):
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['MACD'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

    df['BB_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['BB_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()

    low_min = df['low'].rolling(14).min()
    high_max = df['high'].rolling(14).max()
    df['Stochastic'] = (df['close'] - low_min) / (high_max - low_min) * 100

    return df

def generate_signal(timestamp, signal_type, price):
    duration = 5
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    df['ML_Prediction'] = model.predict(X)
    df['Signal'] = df['ML_Prediction'].map({1: 'Buy (ML)', 0: 'Sell (ML)'})
    df['Trade Duration (min)'] = 2

    return df[['timestamp', 'Signal', 'close', 'Trade Duration (min)']].rename(columns={'close': 'Price'})

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

def calculate_performance_metrics(df, initial_balance=1000):
    wins = df[df["Result"] == "Win"]
    losses = df[df["Result"] == "Loss"]
    win_rate = len(wins) / len(df) * 100 if len(df) > 0 else 0
    net_profit = df["Balance"].iloc[-1] - initial_balance
    max_drawdown = initial_balance - df["Balance"].min()
    profit_factor = wins["Balance"].diff().sum() / abs(losses["Balance"].diff().sum()) if not losses.empty else np.inf

    return {
        "Win Rate (%)": round(win_rate, 2),
        "Net Profit ($)": round(net_profit, 2),
        "Max Drawdown ($)": round(max_drawdown, 2),
        "Profit Factor": round(profit_factor, 2)
    }

def plot_chart(df, asset):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA5'], name="EMA5", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA20'], name="EMA20", line=dict(color='red')))
    fig.update_layout(title=asset, xaxis_rangeslider_visible=False)
    return fig

def play_browser_alert(message="New Signal!"):
    js_code = f"""
    <script>
        if (!window.alerted) {{
            alert("{message}");
            var audio = new Audio("https://www.soundjay.com/buttons/sounds/button-3.mp3");
            audio.play();
            window.alerted = true;
        }}
        setTimeout(() => {{
            window.alerted = false;
        }}, 5000);
    </script>
    """
    components.html(js_code, height=0, width=0)

# --- STREAMLIT APP ---
st.set_page_config(layout="wide")
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh")

st.title("📈 Pocket Option Signals | Live + Backtest + Money Management")

# SIDEBAR
uploaded_file = st.sidebar.file_uploader("Upload historical data (CSV)", type=["csv"])
selected_assets = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS[:2])
selected_strategy = st.sidebar.selectbox("Strategy", [
    "EMA Cross", "RSI Divergence", "MACD Cross", "Bollinger Band Bounce",
    "Stochastic Oscillator", "EMA + RSI Combined", "ML Model (Random Forest)"
])
money_strategy = st.sidebar.selectbox("Money Management", ["Flat", "Martingale"])
enable_alerts = st.sidebar.checkbox("🔔 Enable Browser Alerts", value=True)

# --- BACKTESTING ---
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = calculate_indicators(df)

    if selected_strategy == "ML Model (Random Forest)":
        signals = train_ml_model(df).to_dict(orient='records')
    else:
        signals = detect_signals(df, selected_strategy)

    st.subheader("📊 Backtest Results")
    st.dataframe(pd.DataFrame(signals))
    st.subheader("💰 Money Management Simulation")
    results_df = simulate_money_management(signals, strategy=money_strategy)
    st.dataframe(results_df)

    st.subheader("📈 Performance Metrics")
    metrics = calculate_performance_metrics(results_df)
    for k, v in metrics.items():
        st.metric(label=k, value=v)

    st.plotly_chart(plot_chart(df, "Backtest Data"))

# --- LIVE SIGNALS ---
st.subheader("📡 Live Market Signal Detection")

if not selected_assets:
    st.warning("⚠️ Please select at least one asset.")
else:
    for asset in selected_assets:
        df_live = fetch_candles(asset)
        if df_live is not None:
            df_live = calculate_indicators(df_live)
            if selected_strategy == "ML Model (Random Forest)":
                st.info("ML-based prediction is not available for live data.")
            else:
                live_signals = detect_signals(df_live, selected_strategy)
                if live_signals:
                    st.markdown(f"### {asset}")
                    st.dataframe(pd.DataFrame(live_signals[-5:]))

                    latest_signal = live_signals[-1]
                    summary_text = f"📍 **{latest_signal['Signal']}** {asset} at {latest_signal['Time'].strftime('%H:%M')} for {latest_signal['Trade Duration (min)']} min – Strategy: {selected_strategy}"
                    st.markdown(summary_text)

                    if enable_alerts:
                        play_browser_alert(f"{latest_signal['Signal']} {asset} at {latest_signal['Time'].strftime('%H:%M')}")

                    st.plotly_chart(plot_chart(df_live, asset), use_container_width=True)
