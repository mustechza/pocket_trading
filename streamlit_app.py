import streamlit as st
import pandas as pd
import numpy as np
import websocket
import threading
import json
import time
from datetime import datetime, timedelta
import pytz
import plotly.graph_objects as go

# Add Stochastic RSI, Bollinger Bands, and Heikin-Ashi
def add_indicators(df):
    df['EMA_Fast'] = df['close'].ewm(span=fast_ema).mean()
    df['EMA_Slow'] = df['close'].ewm(span=slow_ema).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['EMA12'] = df['close'].ewm(span=macd_fast).mean()
    df['EMA26'] = df['close'].ewm(span=macd_slow).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=macd_signal).mean()

    # Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(20).mean()
    df['BB_Std'] = df['close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])

    # Stochastic RSI
    min_rsi = df['RSI'].rolling(rsi_period).min()
    max_rsi = df['RSI'].rolling(rsi_period).max()
    df['Stoch_RSI'] = (df['RSI'] - min_rsi) / (max_rsi - min_rsi) * 100

    # Heikin-Ashi
    ha_df = df[['open', 'high', 'low', 'close']].copy()
    ha_df['HA_Close'] = (ha_df['open'] + ha_df['high'] + ha_df['low'] + ha_df['close']) / 4
    ha_df['HA_Open'] = (ha_df['open'].shift(1) + ha_df['close'].shift(1)) / 2
    ha_df['HA_High'] = ha_df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
    ha_df['HA_Low'] = ha_df[['low', 'HA_Open', 'HA_Close']].min(axis=1)
    df[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']] = ha_df[['HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]
    return df

def apply_strategy(df, strategy_name):
    df = add_indicators(df.copy())
    if strategy_name == "EMA Crossover":
        df['Signal'] = np.where(df['EMA_Fast'] > df['EMA_Slow'], 'Buy', 'Sell')
        df['Confidence'] = (abs(df['EMA_Fast'] - df['EMA_Slow']) / df['close']) * 100
    elif strategy_name == "RSI":
        df['Signal'] = np.where(df['RSI'] < rsi_oversold, 'Buy',
                        np.where(df['RSI'] > rsi_overbought, 'Sell', 'Hold'))
        df['Confidence'] = 100 - abs(df['RSI'] - 50)
    elif strategy_name == "MACD":
        df['Signal'] = np.where(df['MACD'] > df['Signal_Line'], 'Buy', 'Sell')
        df['Confidence'] = (abs(df['MACD'] - df['Signal_Line']) / df['close']) * 100
    elif strategy_name == "Bollinger Bands":
        df['Signal'] = np.where(df['close'] < df['BB_Lower'], 'Buy',
                        np.where(df['close'] > df['BB_Upper'], 'Sell', 'Hold'))
        df['Confidence'] = (abs(df['close'] - df['BB_Middle']) / df['BB_Middle']) * 100
    elif strategy_name == "Stochastic RSI":
        df['Signal'] = np.where(df['Stoch_RSI'] < 20, 'Buy',
                        np.where(df['Stoch_RSI'] > 80, 'Sell', 'Hold'))
        df['Confidence'] = 100 - abs(df['Stoch_RSI'] - 50)
    elif strategy_name == "Heikin-Ashi":
        df['Signal'] = np.where(df['HA_Close'] > df['HA_Open'], 'Buy', 'Sell')
        df['Confidence'] = (abs(df['HA_Close'] - df['HA_Open']) / df['close']) * 100
    df.dropna(inplace=True)
    df['Signal_Time'] = df.index
    return df[(df['Signal'].isin(['Buy', 'Sell'])) & (df['Confidence'] >= min_confidence)].tail(3)

# --- Signal Chart Visualization ---
def plot_signal_chart(df, signal_row):
    df = df.copy().last('20T')
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])
    fig.add_trace(go.Scatter(x=[signal_row.name], y=[signal_row['close']],
                             mode='markers+text',
                             text=[signal_row['Signal']],
                             textposition='top center',
                             marker=dict(size=12, color='green' if signal_row['Signal'] == 'Buy' else 'red')))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
    return fig

# --- Realistic Backtest ---
def realistic_backtest(df, strategy_name):
    df = apply_strategy(df.copy(), strategy_name)
    wins, losses = 0, 0
    for i, row in df.iterrows():
        entry_price = row['close']
        expiry_time = i + timedelta(minutes=trade_duration)
        future_prices = df[df.index > i].head(trade_duration)
        if future_prices.empty:
            continue
        close_price = future_prices.iloc[-1]['close']
        if row['Signal'] == 'Buy' and close_price > entry_price:
            wins += 1
        elif row['Signal'] == 'Sell' and close_price < entry_price:
            wins += 1
        else:
            losses += 1
    total = wins + losses
    win_rate = (wins / total * 100) if total else 0
    return total, win_rate

# --- Strategy Options ---
strategy = st.sidebar.selectbox("Select Strategy", ["EMA Crossover", "RSI", "MACD", "Bollinger Bands", "Stochastic RSI", "Heikin-Ashi"])

# --- Thread Protection ---
if api_key and "ws_thread_started" not in st.session_state:
    threading.Thread(target=run_websocket, daemon=True).start()
    st.session_state.ws_thread_started = True

# --- Live Signal Display ---
if not latest_df.empty:
    signals = apply_strategy(latest_df.copy(), strategy)
    if not signals.empty:
        for _, row in signals.iterrows():
            entry_time = row.name.strftime("%H:%M")
            expiration = (row.name + timedelta(minutes=trade_duration)).strftime("%H:%M")
            st.markdown(f"""
            <div style='background-color:#f8f9fa;padding:15px;border-radius:12px;margin:10px 0;box-shadow:2px 2px 6px #ccc'>
                <h4 style='color:#333'>💡 Signal: <b>{row['Signal']}</b></h4>
                <ul>
                    <li><b>Entry Time:</b> {entry_time}</li>
                    <li><b>Expires:</b> {expiration}</li>
                    <li><b>Price:</b> {row['close']:.2f}</li>
                    <li><b>Confidence:</b> {row['Confidence']:.1f}%</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(plot_signal_chart(latest_df.copy(), row), use_container_width=True)
    else:
        st.info("Waiting for qualifying signals...")

# --- Backtesting ---
if backtest_btn and not latest_df.empty:
    with st.spinner("Running realistic backtest..."):
        total, win_rate = realistic_backtest(latest_df.copy(), strategy)
        st.success(f"Backtest complete: {total} trades | Win rate: {win_rate:.1f}%")


