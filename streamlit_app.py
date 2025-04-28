import streamlit as st
import pandas as pd
import requests
import time
import plotly.graph_objects as go

# ========== Settings ==========

ASSETS = ['ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT']
CANDLE_INTERVAL = '1m'
CANDLE_LIMIT = 500
START_BALANCE = 100

# ========== Helper Functions ==========

def get_binance_candles(symbol, interval='1m', limit=500):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df.set_index('timestamp', inplace=True)
    return df[['open', 'high', 'low', 'close']]

def compute_indicators(df):
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['RSI'] = compute_rsi(df['close'], 14)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def magic_secret_strategy(df):
    if len(df) >= 20:
        if not pd.isna(df['EMA5'].iloc[-1]) and not pd.isna(df['EMA20'].iloc[-1]) and not pd.isna(df['RSI'].iloc[-1]):
            if df['EMA5'].iloc[-1] > df['EMA20'].iloc[-1] and df['RSI'].iloc[-1] > 50:
                return 'CALL'
            elif df['EMA5'].iloc[-1] < df['EMA20'].iloc[-1] and df['RSI'].iloc[-1] < 50:
                return 'PUT'
    return None

def ema_cross_only_strategy(df):
    if len(df) >= 20:
        if not pd.isna(df['EMA5'].iloc[-1]) and not pd.isna(df['EMA20'].iloc[-1]):
            if df['EMA5'].iloc[-1] > df['EMA20'].iloc[-1]:
                return 'CALL'
            elif df['EMA5'].iloc[-1] < df['EMA20'].iloc[-1]:
                return 'PUT'
    return None

def rsi_divergence_strategy(df):
    if len(df) >= 14:
        if not pd.isna(df['RSI'].iloc[-1]):
            if df['RSI'].iloc[-1] < 30:
                return 'CALL'
            elif df['RSI'].iloc[-1] > 70:
                return 'PUT'
    return None

def generate_signal(df, strategy_name):
    if strategy_name == "Magic Secret":
        return magic_secret_strategy(df)
    elif strategy_name == "EMA Cross":
        return ema_cross_only_strategy(df)
    elif strategy_name == "RSI Divergence":
        return rsi_divergence_strategy(df)
    return None

def plot_chart(df, asset_name):
    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candles'
    ))

    # EMA 5
    fig.add_trace(go.Scatter(
        x=df.index, y=df['EMA5'], mode='lines', name='EMA 5', line=dict(color='blue')
    ))

    # EMA 20
    fig.add_trace(go.Scatter(
        x=df.index, y=df['EMA20'], mode='lines', name='EMA 20', line=dict(color='orange')
    ))

    fig.update_layout(title=f"{asset_name} Live Chart", xaxis_title="Time", yaxis_title="Price")

    return fig

# ========== Streamlit App ==========

st.set_page_config(page_title="Pocket Option Signal Sender (Fast)", layout="wide")
st.title("Pocket Option Signal Sender - FAST Refresh + Live Charts")

col1, col2 = st.columns(2)

with col1:
    selected_assets = st.multiselect("Select Assets", ASSETS, default=ASSETS)

with col2:
    selected_strategy = st.selectbox("Select Strategy", ["Magic Secret", "EMA Cross", "RSI Divergence"])

trade_amount = st.number_input("Trade Amount per Signal ($)", min_value=1, value=1)
take_profit = st.number_input("Take Profit ($)", min_value=1, value=10)
stop_loss = st.number_input("Stop Loss ($)", min_value=1, value=10)

start = st.button("Start")

if 'running' not in st.session_state:
    st.session_state.running = False
if start:
    st.session_state.running = True

balance = st.session_state.get('balance', START_BALANCE)
trade_history = st.session_state.get('trade_history', [])

if st.session_state.running:
    placeholder_alert = st.empty()
    placeholder_balance = st.empty()
    placeholder_profit_chart = st.empty()
    chart_placeholders = {asset: st.empty() for asset in selected_assets}

    while st.session_state.running:
        all_signals = []

        for asset in selected_assets:
            df = get_binance_candles(asset, interval=CANDLE_INTERVAL, limit=CANDLE_LIMIT)
            df = compute_indicators(df)

            signal = generate_signal(df, selected_strategy)

            if signal:
                all_signals.append((asset, signal))

                result = 'WIN' if signal == ('CALL' if df['close'].iloc[-1] > df['close'].iloc[-2] else 'PUT') else 'LOSS'

                if result == 'WIN':
                    balance += trade_amount
                else:
                    balance -= trade_amount

                trade_history.append({
                    'Asset': asset,
                    'Strategy': selected_strategy,
                    'Signal': signal,
                    'Result': result,
                    'Balance': balance,
                    'Time': pd.Timestamp.now()
                })

                st.session_state.balance = balance
                st.session_state.trade_history = trade_history

            # Live Chart
            with chart_placeholders[asset].container():
                st.plotly_chart(plot_chart(df, asset), use_container_width=True)

        with placeholder_alert.container():
            if all_signals:
                for asset, sig in all_signals:
                    st.success(f"[POCKET OPTION] {asset}: {sig}")
            else:
                st.info("No signals at this moment.")

        placeholder_balance.metric("Balance", f"${balance:.2f}")

        if balance - START_BALANCE >= take_profit:
            st.success(f"Take Profit reached: ${balance}!")
            st.session_state.running = False
            break
        if START_BALANCE - balance >= stop_loss:
            st.error(f"Stop Loss hit: ${balance}!")
            st.session_state.running = False
            break

        if st.session_state.get('trade_history'):
            df_profit = pd.DataFrame(st.session_state.trade_history)
            placeholder_profit_chart.line_chart(df_profit['Balance'])

        time.sleep(5)

if st.session_state.get('trade_history'):
    st.subheader("Trade History")
    st.dataframe(pd.DataFrame(st.session_state.trade_history))
