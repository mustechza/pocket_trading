import asyncio
import aiohttp
import pandas as pd
import time
import streamlit as st

# --- Streamlit Config ---
st.set_page_config(layout="wide")
st.title("📊 Multi-Asset Live Signals Dashboard")

ASSETS_CRYPTO = ["btcusdt", "ethusdt"]
ASSETS_FOREX = ["EURUSD", "GBPUSD"]
selected_crypto = st.sidebar.multiselect("Select Crypto", ASSETS_CRYPTO, default=ASSETS_CRYPTO)
selected_forex = st.sidebar.multiselect("Select Forex", ASSETS_FOREX, default=ASSETS_FOREX)

# Initialize session state
if "market_data" not in st.session_state:
    st.session_state.market_data = {}

# --- Binance (Crypto) ---
async def fetch_binance(symbol="btcusdt", interval="1m", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 
                'close_time', 'quote_asset_volume', 'number_of_trades', 
                'taker_buy_base', 'taker_buy_quote', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
            return df[['timestamp', 'open', 'high', 'low', 'close']]

# --- Alpha Vantage (Forex) ---
ALPHA_API_KEY = "AWKMJ8CFU7OUNCVG"

async def fetch_forex(pair="EURUSD", interval="1min", output_size="compact"):
    url = f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol={pair[:3]}&to_symbol={pair[3:]}&interval={interval}&outputsize={output_size}&apikey={ALPHA_API_KEY}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()
            if "Time Series" not in str(data):
                print("Alpha Vantage limit reached or error", data)
                return pd.DataFrame()
            key = list(data.keys())[1]
            raw = data[key]
            df = pd.DataFrame.from_dict(raw, orient='index')
            df = df.rename(columns={
                '1. open': 'open', '2. high': 'high',
                '3. low': 'low', '4. close': 'close'})
            df.index = pd.to_datetime(df.index)
            df = df.astype(float).sort_index()
            df = df.reset_index().rename(columns={'index': 'timestamp'})
            return df[['timestamp', 'open', 'high', 'low', 'close']]

# --- Async Fetch Once ---
async def fetch_once():
    tasks = []
    for sym in selected_crypto:
        tasks.append(fetch_binance(sym))
    for pair in selected_forex:
        tasks.append(fetch_forex(pair))

    results = await asyncio.gather(*tasks)

    # Save results
    for i, sym in enumerate(selected_crypto + selected_forex):
        st.session_state.market_data[sym] = results[i]

# --- Display Function ---
def show_dashboard():
    cols = st.columns(len(selected_crypto + selected_forex))
    for i, sym in enumerate(selected_crypto + selected_forex):
        df = st.session_state.market_data.get(sym, pd.DataFrame())
        if df.empty:
            cols[i].warning(f"⏳ Waiting for data for {sym.upper()}...")
            continue
        price = df['close'].iloc[-1]
        cols[i].metric(label=f"{sym.upper()} Price", value=f"{price:.5f}")
        cols[i].line_chart(df.set_index('timestamp')['close'])

# --- Async Loop ---
async def main_loop():
    # Initial load
    await fetch_once()
    while True:
        show_dashboard()
        await fetch_once()
        await asyncio.sleep(10)  # refresh every 10 sec

# --- Run ---
asyncio.run(main_loop())
