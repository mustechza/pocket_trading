import streamlit as st
import pandas as pd
import numpy as np
import datetime
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Pocket Option Signals", layout="wide")

st.title("Pocket Option Signal Generator")

uploaded_file = st.file_uploader("Upload your candlestick data (CSV)", type=["csv"])

def preprocess_data(df):
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={
        'open time': 'timestamp',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    }, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values('timestamp', inplace=True)
    df.set_index('timestamp', inplace=True)
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    macd = MACD(close=df['close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    bb = BollingerBands(close=df['close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df.dropna(inplace=True)
    return df.reset_index()

def generate_signal(timestamp, signal_type, price, confidence=0.5):
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
        "Trade Duration (min)": duration,
        "Confidence": round(confidence, 2)
    }

def detect_signals(df, strategy):
    signals = []
    for i in range(1, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        t, price = row['timestamp'], row['close']

        if strategy == "RSI Divergence":
            rsi = row['RSI']
            if rsi < 30:
                confidence = (30 - rsi) / 30
                signals.append(generate_signal(t, "Buy (RSI Oversold)", price, confidence))
            elif rsi > 70:
                confidence = (rsi - 70) / 30
                signals.append(generate_signal(t, "Sell (RSI Overbought)", price, confidence))

        elif strategy == "MACD Cross":
            if prev['MACD'] < prev['MACD_signal'] and row['MACD'] > row['MACD_signal']:
                confidence = abs(row['MACD'] - row['MACD_signal']) / max(abs(row['MACD']), 1)
                signals.append(generate_signal(t, "Buy (MACD Bullish)", price, confidence))
            elif prev['MACD'] > prev['MACD_signal'] and row['MACD'] < row['MACD_signal']:
                confidence = abs(row['MACD'] - row['MACD_signal']) / max(abs(row['MACD']), 1)
                signals.append(generate_signal(t, "Sell (MACD Bearish)", price, confidence))

        elif strategy == "Bollinger Reversal":
            if prev['close'] > prev['BB_upper'] and row['close'] < row['BB_upper']:
                confidence = abs(row['close'] - row['BB_upper']) / row['close']
                signals.append(generate_signal(t, "Sell (BB Upper Reversal)", price, confidence))
            elif prev['close'] < prev['BB_lower'] and row['close'] > row['BB_lower']:
                confidence = abs(row['close'] - row['BB_lower']) / row['close']
                signals.append(generate_signal(t, "Buy (BB Lower Reversal)", price, confidence))

    return signals[-3:]

def train_ml_model(df):
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    features = ['open', 'high', 'low', 'close', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower']
    X = df[features]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    proba = model.predict_proba(X_test_scaled)[:, 1]  # confidence for class 1
    preds = model.predict(X_test_scaled)
    df_test = df.iloc[X_test.index].copy()
    df_test['ML_Confidence'] = proba
    df_test['Signal'] = np.where(preds == 1, 'Buy (ML)', 'Sell (ML)')
    df_test['Trade Duration (min)'] = np.where(preds == 1, 3, 5)
    df_test['Confidence'] = df_test['ML_Confidence'].round(2)
    return df_test[['timestamp', 'Signal', 'close', 'Trade Duration (min)', 'Confidence']].rename(columns={"timestamp": "Time", "close": "Price"}).tail(3)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = preprocess_data(df)
    strategy = st.selectbox("Select strategy", ["RSI Divergence", "MACD Cross", "Bollinger Reversal", "Machine Learning"])
    if strategy == "Machine Learning":
        signals = train_ml_model(df)
    else:
        signals = detect_signals(df, strategy)

    st.subheader("Latest Signals")
    for i, s in enumerate(signals[::-1]):
        with st.container():
            st.markdown(f"""
            ### 📊 Signal Alert
            🧭 **{strategy}**
            🕒 **Entry:** {s['Time'].strftime('%H:%M')}  
            ⌛ **Duration:** {s['Trade Duration (min)']} min  
            🎯 **Price:** {s['Price']}  
            📈 **Confidence:** {s['Confidence']*100:.1f}%  
            🟩 **Direction:** {s['Signal']}  
            """)
                
