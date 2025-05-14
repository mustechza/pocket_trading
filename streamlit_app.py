
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph\_objects as go
from datetime import datetime, timedelta
from streamlit\_autorefresh import st\_autorefresh
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIG ---

st.set\_page\_config(layout="wide")
REFRESH\_INTERVAL = 10  # seconds
st\_autorefresh(interval=REFRESH\_INTERVAL \* 1000, key="refresh")

# --- CONSTANTS ---

BINANCE\_URL = "[https://api.binance.us/api/v3/klines](https://api.binance.us/api/v3/klines)"
ASSETS = \["ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]

# --- SIDEBAR ---

uploaded\_file = st.sidebar.file\_uploader("Upload historical data (CSV)", type=\["csv"])
selected\_assets = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS\[:2])
selected\_strategy = st.sidebar.selectbox("Strategy", \[
"EMA Cross", "RSI Divergence", "MACD Cross", "Bollinger Band Bounce",
"Stochastic Oscillator", "EMA + RSI Combined", "ML Model (Random Forest)"
])
money\_strategy = st.sidebar.selectbox("Money Management", \["Flat", "Martingale", "Risk %"])
risk\_pct = st.sidebar.slider("Risk % per Trade", 1, 10, value=2)
balance\_input = st.sidebar.number\_input("Initial Balance (\$)", value=1000)

ema\_short = st.sidebar.number\_input("EMA Short Period", 2, 50, value=5)
ema\_long = st.sidebar.number\_input("EMA Long Period", 5, 100, value=20)
rsi\_period = st.sidebar.number\_input("RSI Period", 5, 50, value=14)
stoch\_period = st.sidebar.number\_input("Stochastic Period", 5, 50, value=14)
bb\_period = st.sidebar.number\_input("Bollinger Band Period", 5, 50, value=20)

# --- TIMEZONE UTILITY ---

def to\_gmt\_plus2(ts):
return ts + timedelta(hours=2)

# --- DATA FETCH ---

@st.cache\_data(ttl=60)
def fetch\_candles(symbol, interval="1m", limit=500):
try:
params = {"symbol": symbol, "interval": interval, "limit": limit}
response = requests.get(BINANCE\_URL, params=params, timeout=10)
data = response.json()
df = pd.DataFrame(data, columns=\[
'timestamp', 'open', 'high', 'low', 'close', 'volume',
'close\_time', 'qav', 'num\_trades', 'taker\_base\_vol', 'taker\_quote\_vol', 'ignore'
])
df\['timestamp'] = pd.to\_datetime(df\['timestamp'], unit='ms').apply(to\_gmt\_plus2)
df\[\['open', 'high', 'low', 'close']] = df\[\['open', 'high', 'low', 'close']].astype(float)
return df
except Exception as e:
st.warning(f"Error fetching data for {symbol}: {e}")
return None

# --- INDICATORS ---

def calculate\_indicators(df):
df\['EMA5'] = df\['close'].ewm(span=ema\_short).mean()
df\['EMA20'] = df\['close'].ewm(span=ema\_long).mean()

```
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
```

# --- SIGNAL GENERATOR ---

def generate\_signal(timestamp, signal\_type, price):
duration = 3 if "Buy" in signal\_type else 5
return {
"Time": timestamp,
"Signal": signal\_type,
"Price": price,
"Trade Duration (min)": duration
}

def detect\_signals(df, strategy):
signals = \[]
for i in range(1, len(df)):
t = df\['timestamp'].iloc\[i]
price = df\['close'].iloc\[i]
ema5, ema20 = df\['EMA5'].iloc\[i], df\['EMA20'].iloc\[i]
ema5\_prev, ema20\_prev = df\['EMA5'].iloc\[i-1], df\['EMA20'].iloc\[i-1]

```
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
```

# --- ML MODEL ---

def train\_ml\_model(df):
df = df.copy()
df\['target'] = (df\['close'].shift(-2) > df\['close']).astype(int)
features = \['EMA5', 'EMA20', 'RSI', 'MACD', 'MACD\_signal', 'BB\_upper', 'BB\_lower', 'Stochastic']
df.dropna(subset=features + \['target'], inplace=True)
if len(df) < 50:
return pd.DataFrame()
model = RandomForestClassifier(n\_estimators=100, random\_state=42)
model.fit(df\[features], df\['target'])
df\['ML\_Prediction'] = model.predict(df\[features])
df\['Signal'] = df\['ML\_Prediction'].map({1: 'Buy (ML)', 0: 'Sell (ML)'})
df\['Trade Duration (min)'] = df\['Signal'].apply(lambda x: 3 if "Buy" in x else 5)
return df\[\['timestamp', 'Signal', 'close', 'Trade Duration (min)']].rename(columns={'close': 'Price'})

# --- MONEY MANAGEMENT ---

def simulate\_money\_management(signals, strategy="Flat", initial\_balance=1000, risk\_pct=2):
balance = initial\_balance
wins = losses = 0
last\_bet = balance \* (risk\_pct / 100)
logs = \[]
for s in signals:
win = np.random.choice(\[True, False], p=\[0.55, 0.45])
bet = balance \* (risk\_pct / 100) if strategy == "Risk %" else last\_bet
result = "Win" if win else "Loss"
balance += bet if win else -bet
if strategy == "Martingale" and not win:
last\_bet \*= 2
elif strategy == "Martingale" and win:
last\_bet = bet
wins += win
losses += not win
logs.append({
"Time": s\["Time"], "Signal": s\["Signal"], "Result": result,
"Balance": round(balance, 2), "Trade Duration (min)": s\["Trade Duration (min)"]
})
df = pd.DataFrame(logs)
max\_dd = ((df\['Balance'].cummax() - df\['Balance']) / df\['Balance'].cummax()).max()
roi = ((balance - initial\_balance) / initial\_balance) \* 100
profit\_factor = (wins \* bet) / max(losses \* bet, 1)
return df, {
"Win Rate (%)": 100 \* wins / max(wins + losses, 1),
"ROI (%)": roi,
"Max Drawdown (%)": 100 \* max\_dd,
"Profit Factor": profit\_factor
}

# --- CHART ---

def plot\_chart(df, title):
fig = go.Figure()
fig.add\_trace(go.Candlestick(
x=df\['timestamp'], open=df\['open'], high=df\['high'],
low=df\['low'], close=df\['close'], name='Candles'))
fig.add\_trace(go.Scatter(x=df\['timestamp'], y=df\['EMA5'], name='EMA5', line=dict(color='blue')))
fig.add\_trace(go.Scatter(x=df\['timestamp'], y=df\['EMA20'], name='EMA20', line=dict(color='red')))
fig.update\_layout(title=title, xaxis\_rangeslider\_visible=False)
return fig

# --- TITLE ---

st.title("📈 Pocket Option Signals | Live + Backtest + Money Management")

# --- BACKTEST ---

if uploaded\_file:
df = pd.read\_csv(uploaded\_file)
df\['timestamp'] = pd.to\_datetime(df\['timestamp']).apply(to\_gmt\_plus2)
df = calculate\_indicators(df)

```
signals = train_ml_model(df).to_dict('records') if selected_strategy == "ML Model (Random Forest)" else detect_signals(df, selected_strategy)

st.subheader("📌 Last 3 Signal Alerts")
latest_signals = signals[-3:]
cols = st.columns(len(latest_signals))
for i, s in enumerate(latest_signals):
    with cols[i]:
        st.markdown(f"""
        ### 📊 Signal Alert  
        🧭 **{selected_strategy}**  
        🕒 **Entry:** {s['Time'].strftime('%H:%M')}  
        ⌛ **Duration:** {s['Trade Duration (min)']} min  
        🎯 **Price:** {s['Price']:.4f}  
        🟩 **Direction:** {s['Signal']}  
        """)

st.subheader("💰 Money Management Simulation")
results_df, metrics = simulate_money_management(signals, money_strategy, balance_input, risk_pct)
st.dataframe(results_df)

st.subheader("📈 Performance Metrics")
for k, v in metrics.items():
    st.metric(k, f"{v:.2f}")

st.plotly_chart(plot_chart(df, "Backtest Data"), use_container_width=True)
```

# --- LIVE ---

st.subheader("📡 Live Signal Detection")
for asset in selected\_assets:
df\_live = fetch\_candles(asset)
if df\_live is not None:
df\_live = calculate\_indicators(df\_live)
if selected\_strategy != "ML Model (Random Forest)":
live\_signals = detect\_signals(df\_live, selected\_strategy)
if live\_signals:
st.markdown(f"### 🔔 {asset}")
latest\_live = live\_signals\[-3:]
cols = st.columns(len(latest\_live))
for i, s in enumerate(latest\_live):
with cols\[i]:
st.markdown(f"""
\### 📊 Signal Alert
🧭 **{selected\_strategy}**
🕒 **Entry:** {s\['Time'].strftime('%H:%M')}
⌛ **Duration:** {s\['Trade Duration (min)']} min
🎯 **Price:** {s\['Price']:.4f}
🟩 **Direction:** {s\['Signal']}
""")
st.plotly\_chart(plot\_chart(df\_live, asset), use\_container\_width=True)
