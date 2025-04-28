import streamlit as st
from datetime import datetime
import pandas as pd
import requests

# Try importing st_autorefresh for auto-refresh functionality
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st.warning("streamlit_autorefresh not installed; auto-refresh disabled.")
    st_autorefresh = None

# Initialize session state for tracking signals
if 'sent_signals' not in st.session_state:
    st.session_state['sent_signals'] = set()
if 'signal_history' not in st.session_state:
    st.session_state['signal_history'] = []

st.set_page_config(page_title="Pocket Option Signal Scanner", page_icon=":satellite:", layout="wide")
st.title("Pocket Option Signal Scanner")

# Asset selection (6 assets with USDT trading pairs)
ASSETS = {
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT',
    'ADA': 'ADAUSDT',
    'BNB': 'BNBUSDT',
    'XRP': 'XRPUSDT',
    'LTC': 'LTCUSDT'
}
selected_assets = st.multiselect("Select assets to scan:", options=list(ASSETS.keys()), default=['ETH', 'SOL'])

# Telegram placeholders (optional; leave blank to disable alerts)
BOT_TOKEN = st.text_input("Telegram Bot Token (optional)", value="", type="password")
CHAT_ID = st.text_input("Telegram Chat ID (optional)", value="", type="password")

# Auto-refresh every 5 seconds using streamlit_autorefresh if available
if st_autorefresh is not None:
    st_autorefresh(interval=5000, limit=None, key="scanner_refresh")

def fetch_candles(symbol, interval='5m', limit=200):
    """Fetch candlestick (klines) data from Binance public API."""
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    res = requests.get(url, params=params, timeout=10)
    data = res.json()
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'num_trades', 
        'taker_base', 'taker_quote', 'ignore'
    ])
    # Convert data types
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df[['open_time', 'open', 'high', 'low', 'close']].copy()

def compute_indicators(df):
    """Compute EMA (12/26) and RSI (14) on the DataFrame."""
    # Exponential Moving Averages
    df['ema_fast'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=26, adjust=False).mean()
    # RSI calculation (14-period simple average)
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    window = 14
    df['avg_gain'] = gain.rolling(window=window).mean()
    df['avg_loss'] = loss.rolling(window=window).mean()
    df['rs'] = df['avg_gain'] / df['avg_loss']
    df['rsi'] = 100 - (100 / (1 + df['rs']))
    return df

# Scan each selected asset
if selected_assets:
    for asset in selected_assets:
        symbol = ASSETS[asset]
        # Fetch data and compute indicators
        with st.spinner(f"Fetching data for {symbol}..."):
            df = fetch_candles(symbol)
            df = compute_indicators(df)
        # Detect EMA cross signals (using last two points)
        if len(df) >= 2:
            prev_fast = df['ema_fast'].iloc[-2]
            prev_slow = df['ema_slow'].iloc[-2]
            last_fast = df['ema_fast'].iloc[-1]
            last_slow = df['ema_slow'].iloc[-1]
            if prev_fast < prev_slow and last_fast > last_slow:
                st.session_state['signal_history'].append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "asset": asset,
                    "strategy": "EMA Cross",
                    "signal": "Bullish"
                })
                st.session_state['sent_signals'].add(f"{asset}-EMA Cross-Bullish")
            if prev_fast > prev_slow and last_fast < last_slow:
                st.session_state['signal_history'].append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "asset": asset,
                    "strategy": "EMA Cross",
                    "signal": "Bearish"
                })
                st.session_state['sent_signals'].add(f"{asset}-EMA Cross-Bearish")
        # Detect RSI divergence (simple check on last two points)
        if len(df) >= 2 and not pd.isna(df['rsi'].iloc[-1]):
            prev_close = df['close'].iloc[-2]
            last_close = df['close'].iloc[-1]
            prev_rsi = df['rsi'].iloc[-2]
            last_rsi = df['rsi'].iloc[-1]
            if last_close < prev_close and last_rsi > prev_rsi:
                st.session_state['signal_history'].append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "asset": asset,
                    "strategy": "RSI Divergence",
                    "signal": "Bullish"
                })
                st.session_state['sent_signals'].add(f"{asset}-RSI Divergence-Bullish")
            if last_close > prev_close and last_rsi < prev_rsi:
                st.session_state['signal_history'].append({
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "asset": asset,
                    "strategy": "RSI Divergence",
                    "signal": "Bearish"
                })
                st.session_state['sent_signals'].add(f"{asset}-RSI Divergence-Bearish")
        # Plot candlestick chart for the asset
        try:
            import plotly.graph_objects as go
            fig = go.Figure(data=[go.Candlestick(
                x=df['open_time'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol
            )])
            fig.update_layout(
                title_text=f"{symbol} Candlestick Chart",
                xaxis_title="Time",
                yaxis_title="Price (USDT)"
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error plotting {symbol}: {e}")

# Send Telegram alerts for newly added signals
if 'signal_history' in st.session_state:
    # Track which signals have been notified to avoid duplicates
    notified = getattr(st.session_state, 'notified_signals', set())
    st.session_state.setdefault('notified_signals', set())
    for entry in st.session_state['signal_history']:
        key = f"{entry['asset']}-{entry['strategy']}-{entry['signal']}"
        if BOT_TOKEN and CHAT_ID and key not in notified:
            message = f"{entry['time']} - {entry['asset']} - {entry['strategy']} - {entry['signal']}"
            telegram_url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
            try:
                requests.get(telegram_url, params={"chat_id": CHAT_ID, "text": message}, timeout=5)
            except Exception as e:
                st.error(f"Telegram API error: {e}")
            notified.add(key)
    st.session_state['notified_signals'] = notified

# Display recent signal history (last 10 entries)
if st.session_state['signal_history']:
    st.subheader("Signal History (recent)")
    history_df = pd.DataFrame(st.session_state['signal_history'])
    st.table(history_df.tail(10))
