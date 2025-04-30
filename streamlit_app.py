import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import websocket
import json

# Constants
BINANCE_URL = "https://api.binance.com/api/v1/klines"
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "LTCUSDT"]

# SIDEBAR: Add Time Interval Selector and Asset Selector
selected_interval = st.sidebar.selectbox("Select Time Interval", [
    "1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"
], index=0)

selected_assets = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS[:2])

# Create session with retry logic for API requests
def create_session():
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

# Fetch candlestick data from Binance API
@st.cache(ttl=60)  # Cache for 60 seconds to reduce API calls
def fetch_candles(symbol, interval="1m", limit=500):
    session = create_session()  # Create a session with retry logic
    try:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        response = session.get(BINANCE_URL, params=params, timeout=10)
        response.raise_for_status()  # Raise error for bad responses
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

# WebSocket real-time data streaming
def on_message(ws, message):
    data = json.loads(message)
    # Extract candlestick data here from WebSocket message
    print(data)  # You can update the chart or signals here with the new data

def on_error(ws, error):
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Closed WebSocket connection")

def on_open(ws):
    print("WebSocket connection opened")

def stream_real_time_data(symbol="ethusdt"):
    socket = f"wss://stream.binance.com:9443/ws/{symbol}@kline_1m"
    ws = websocket.WebSocketApp(socket, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.on_open = on_open
    ws.run_forever()

# Plot the candlestick chart using Plotly
def plot_chart(df, symbol):
    fig = go.Figure(data=[go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=symbol
    )])
    fig.update_layout(
        title=f"{symbol} Candlestick Chart",
        xaxis_title="Time",
        yaxis_title="Price (USDT)",
        xaxis_rangeslider_visible=False
    )
    return fig

# Main logic for the Streamlit app
for asset in selected_assets:
    # Fetch the data for selected asset
    df_live = fetch_candles(asset, interval=selected_interval)
    if df_live is not None:
        # Plot candlestick chart
        st.markdown(f"### {asset}")
        st.dataframe(df_live.tail(10))  # Display latest 10 candles
        st.plotly_chart(plot_chart(df_live, asset), use_container_width=True)

        # If you want to stream live data for real-time updates, you can call the WebSocket function
        # Uncomment this line if you want to start the WebSocket stream
        # stream_real_time_data(asset.lower())
    else:
        st.warning(f"Data fetch failed for {asset}")

# Additional features like signal detection and analysis can be added below
