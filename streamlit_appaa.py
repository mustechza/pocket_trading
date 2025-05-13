#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import RandomForestClassifier
import streamlit.components.v1 as components
from typing import List, Dict, Any, Optional, Tuple

# --- CONSTANTS ---
REFRESH_INTERVAL: int = 10  # seconds
CANDLE_LIMIT: int = 500
BINANCE_URL: str = "https://api.binance.us/api/v3/klines"
ASSETS: List[str] = ["ETHUSDT", "SOLUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT"]

# Strategy Names
STRATEGY_EMA_CROSS: str = "EMA Cross"
STRATEGY_RSI_DIVERGENCE: str = "RSI Divergence"
STRATEGY_MACD_CROSS: str = "MACD Cross"
STRATEGY_BB_BOUNCE: str = "Bollinger Band Bounce"
STRATEGY_STOCH_OSCILLATOR: str = "Stochastic Oscillator"
STRATEGY_EMA_RSI_COMBINED: str = "EMA + RSI Combined"
STRATEGY_ML_MODEL: str = "ML Model (Random Forest)"

STRATEGIES_LIST: List[str] = [
    STRATEGY_EMA_CROSS, STRATEGY_RSI_DIVERGENCE, STRATEGY_MACD_CROSS, 
    STRATEGY_BB_BOUNCE, STRATEGY_STOCH_OSCILLATOR, STRATEGY_EMA_RSI_COMBINED, 
    STRATEGY_ML_MODEL
]

# Default Indicator Periods
DEFAULT_EMA_SHORT_PERIOD: int = 5
DEFAULT_EMA_LONG_PERIOD: int = 20
DEFAULT_RSI_PERIOD: int = 14
DEFAULT_STOCH_PERIOD: int = 14
DEFAULT_BB_PERIOD: int = 20
DEFAULT_MACD_SHORT_SPAN: int = 12
DEFAULT_MACD_LONG_SPAN: int = 26
DEFAULT_MACD_SIGNAL_SPAN: int = 9

# Trade Durations
DURATION_BUY: int = 3
DURATION_SELL: int = 5
DURATION_OTHER: int = 4

# --- PAGE CONFIG ---
st.set_page_config(layout="wide")

# --- AUTO REFRESH ---
st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="refresh")

# --- SIDEBAR --- 
st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Upload historical data (CSV)", type=["csv"])
selected_assets: List[str] = st.sidebar.multiselect("Select Assets", ASSETS, default=ASSETS[:2])
selected_strategy: str = st.sidebar.selectbox("Select Strategy", STRATEGIES_LIST)

money_strategy: str = st.sidebar.selectbox("Money Management", ["Flat", "Martingale", "Risk %"])
risk_pct: int = st.sidebar.slider("Risk % per Trade", 1, 10, value=2)
balance_input: int = st.sidebar.number_input("Initial Balance ($)", value=1000, min_value=0)

st.sidebar.subheader("Indicator Settings")
# Initialize period variables with defaults
ema_short_period: int = DEFAULT_EMA_SHORT_PERIOD
ema_long_period: int = DEFAULT_EMA_LONG_PERIOD
rsi_period_val: int = DEFAULT_RSI_PERIOD
stoch_period_val: int = DEFAULT_STOCH_PERIOD
bb_period_val: int = DEFAULT_BB_PERIOD

# Conditional inputs for indicator periods
if selected_strategy in [STRATEGY_EMA_CROSS, STRATEGY_EMA_RSI_COMBINED, STRATEGY_ML_MODEL]:
    ema_short_period = st.sidebar.number_input("EMA Short Period", 2, 50, value=ema_short_period, help="Used for EMA Cross, EMA+RSI, and as a feature for ML Model.")
    ema_long_period = st.sidebar.number_input("EMA Long Period", 5, 100, value=ema_long_period, help="Used for EMA Cross, EMA+RSI, and as a feature for ML Model.")

if selected_strategy in [STRATEGY_RSI_DIVERGENCE, STRATEGY_EMA_RSI_COMBINED, STRATEGY_ML_MODEL]:
    rsi_period_val = st.sidebar.number_input("RSI Period", 5, 50, value=rsi_period_val, help="Used for RSI Divergence, EMA+RSI, and as a feature for ML Model.")

if selected_strategy == STRATEGY_ML_MODEL: # MACD specific periods only shown if ML model might use them, or if we add MACD period config later
    st.sidebar.text("(MACD periods for ML features are fixed internally for now)")

if selected_strategy in [STRATEGY_BB_BOUNCE, STRATEGY_ML_MODEL]:
    bb_period_val = st.sidebar.number_input("Bollinger Band Period", 5, 50, value=bb_period_val, help="Used for Bollinger Band Bounce and as a feature for ML Model.")

if selected_strategy in [STRATEGY_STOCH_OSCILLATOR, STRATEGY_ML_MODEL]:
    stoch_period_val = st.sidebar.number_input("Stochastic Period", 5, 50, value=stoch_period_val, help="Used for Stochastic Oscillator and as a feature for ML Model.")

# Input validation for EMA periods
valid_inputs = True
if selected_strategy in [STRATEGY_EMA_CROSS, STRATEGY_EMA_RSI_COMBINED, STRATEGY_ML_MODEL]:
    if ema_short_period >= ema_long_period:
        st.sidebar.error("EMA Short Period must be less than EMA Long Period.")
        valid_inputs = False

# --- FUNCTIONS ---
# def to_gmt_plus2(ts: pd.Timestamp) -> pd.Timestamp:
#     """Converts a timestamp to GMT+2."""
#     return ts + timedelta(hours=2)

@st.cache_data(ttl=REFRESH_INTERVAL*2)
def fetch_candles(symbol: str, interval: str = "1m", limit: int = CANDLE_LIMIT) -> Optional[pd.DataFrame]:
    """Fetches candle data from Binance API."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        with st.spinner(f"Fetching {symbol} data from Binance..."):
            response = requests.get(BINANCE_URL, params=params, timeout=10)
            response.raise_for_status() 
            data = response.json()
        if not isinstance(data, list) or (data and not isinstance(data[0], list)):
            st.warning(f"Unexpected data format from Binance for {symbol}: {str(data)[:100]}...")
            return None

        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # df['timestamp'] = df['timestamp'].apply(to_gmt_plus2) # Uncomment if to_gmt_plus2 is defined and needed
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        return df.iloc[:, :6] 
    except requests.exceptions.Timeout:
        st.error(f"Fetching data for {symbol} timed out. Please check your internet connection.")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Failed to connect to Binance for {symbol} data. The service may be down or your network blocked.")
        return None
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred while fetching {symbol}: {http_err} - {response.text}. The symbol may be invalid or API endpoint changed.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while fetching {symbol}: {e}")
        return None

@st.cache_data
def calculate_indicators(df: pd.DataFrame, 
                         p_ema_short: int, p_ema_long: int, 
                         p_rsi: int, p_stoch: int, p_bb: int) -> pd.DataFrame:
    """Calculates various technical indicators."""
    df_copy = df.copy()
    if df_copy.empty:
        return df_copy

    df_copy['EMA5'] = df_copy['close'].ewm(span=p_ema_short, adjust=False).mean()
    df_copy['EMA20'] = df_copy['close'].ewm(span=p_ema_long, adjust=False).mean()

    delta = df_copy['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0).rolling(window=p_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).fillna(0).rolling(window=p_rsi).mean()
    rs = gain / loss
    df_copy['RSI'] = 100 - (100 / (1 + rs))
    df_copy['RSI'] = df_copy['RSI'].fillna(50) 

    ema12 = df_copy['close'].ewm(span=DEFAULT_MACD_SHORT_SPAN, adjust=False).mean()
    ema26 = df_copy['close'].ewm(span=DEFAULT_MACD_LONG_SPAN, adjust=False).mean()
    df_copy['MACD'] = ema12 - ema26
    df_copy['MACD_signal'] = df_copy['MACD'].ewm(span=DEFAULT_MACD_SIGNAL_SPAN, adjust=False).mean()

    sma_bb = df_copy['close'].rolling(window=p_bb).mean()
    std_bb = df_copy['close'].rolling(window=p_bb).std()
    df_copy['BB_upper'] = sma_bb + 2 * std_bb
    df_copy['BB_lower'] = sma_bb - 2 * std_bb

    low_min = df_copy['low'].rolling(window=p_stoch).min()
    high_max = df_copy['high'].rolling(window=p_stoch).max()
    df_copy['Stochastic'] = (df_copy['close'] - low_min) / (high_max - low_min) * 100
    df_copy['Stochastic'] = df_copy['Stochastic'].fillna(50)
    return df_copy

def generate_signal(timestamp: pd.Timestamp, signal_type: str, price: float) -> Dict[str, Any]:
    """Generates a signal dictionary with trade duration."""
    duration = DURATION_OTHER
    if "Buy" in signal_type:
        duration = DURATION_BUY
    elif "Sell" in signal_type:
        duration = DURATION_SELL
    return {
        "Time": timestamp,
        "Signal": signal_type,
        "Price": price,
        "Trade Duration (min)": duration
    }

# --- Modularized Signal Detection Functions ---
def detect_ema_cross_signals(df: pd.DataFrame) -> List[Dict[str, Any]]:
    signals = []
    if 'EMA5' not in df.columns or 'EMA20' not in df.columns or df.empty: return signals
    for i in range(1, len(df)):
        if df['EMA5'].iloc[i-1] < df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] > df['EMA20'].iloc[i]:
            signals.append(generate_signal(df['timestamp'].iloc[i], f"Buy ({STRATEGY_EMA_CROSS})", df['close'].iloc[i]))
        elif df['EMA5'].iloc[i-1] > df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] < df['EMA20'].iloc[i]:
            signals.append(generate_signal(df['timestamp'].iloc[i], f"Sell ({STRATEGY_EMA_CROSS})", df['close'].iloc[i]))
    return signals

def detect_rsi_divergence_signals(df: pd.DataFrame) -> List[Dict[str, Any]]:
    signals = []
    if 'RSI' not in df.columns or df.empty: return signals
    for i in range(1, len(df)):
        rsi_val = df['RSI'].iloc[i]
        if rsi_val < 30:
            signals.append(generate_signal(df['timestamp'].iloc[i], f"Buy (RSI Oversold)", df['close'].iloc[i]))
        elif rsi_val > 70:
            signals.append(generate_signal(df['timestamp'].iloc[i], f"Sell (RSI Overbought)", df['close'].iloc[i]))
    return signals

def detect_macd_cross_signals(df: pd.DataFrame) -> List[Dict[str, Any]]:
    signals = []
    if 'MACD' not in df.columns or 'MACD_signal' not in df.columns or df.empty: return signals
    for i in range(1, len(df)):
        if df['MACD'].iloc[i-1] < df['MACD_signal'].iloc[i-1] and df['MACD'].iloc[i] > df['MACD_signal'].iloc[i]:
            signals.append(generate_signal(df['timestamp'].iloc[i], f"Buy ({STRATEGY_MACD_CROSS})", df['close'].iloc[i]))
        elif df['MACD'].iloc[i-1] > df['MACD_signal'].iloc[i-1] and df['MACD'].iloc[i] < df['MACD_signal'].iloc[i]:
            signals.append(generate_signal(df['timestamp'].iloc[i], f"Sell ({STRATEGY_MACD_CROSS})", df['close'].iloc[i]))
    return signals

def detect_bb_bounce_signals(df: pd.DataFrame) -> List[Dict[str, Any]]:
    signals = []
    if 'BB_lower' not in df.columns or 'BB_upper' not in df.columns or df.empty: return signals
    for i in range(1, len(df)):
        if df['close'].iloc[i] < df['BB_lower'].iloc[i]:
            signals.append(generate_signal(df['timestamp'].iloc[i], f"Buy (BB Lower)", df['close'].iloc[i]))
        elif df['close'].iloc[i] > df['BB_upper'].iloc[i]:
            signals.append(generate_signal(df['timestamp'].iloc[i], f"Sell (BB Upper)", df['close'].iloc[i]))
    return signals

def detect_stochastic_signals(df: pd.DataFrame) -> List[Dict[str, Any]]:
    signals = []
    if 'Stochastic' not in df.columns or df.empty: return signals
    for i in range(1, len(df)):
        stoch_val = df['Stochastic'].iloc[i]
        if stoch_val < 20:
            signals.append(generate_signal(df['timestamp'].iloc[i], f"Buy (Stochastic Oversold)", df['close'].iloc[i]))
        elif stoch_val > 80:
            signals.append(generate_signal(df['timestamp'].iloc[i], f"Sell (Stochastic Overbought)", df['close'].iloc[i]))
    return signals

def detect_ema_rsi_combined_signals(df: pd.DataFrame) -> List[Dict[str, Any]]:
    signals = []
    if 'EMA5' not in df.columns or 'EMA20' not in df.columns or 'RSI' not in df.columns or df.empty: return signals
    for i in range(1, len(df)):
        if df['EMA5'].iloc[i-1] < df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] > df['EMA20'].iloc[i] and df['RSI'].iloc[i] < 40:
            signals.append(generate_signal(df['timestamp'].iloc[i], f"Buy ({STRATEGY_EMA_RSI_COMBINED})", df['close'].iloc[i]))
        elif df['EMA5'].iloc[i-1] > df['EMA20'].iloc[i-1] and df['EMA5'].iloc[i] < df['EMA20'].iloc[i] and df['RSI'].iloc[i] > 60:
            signals.append(generate_signal(df['timestamp'].iloc[i], f"Sell ({STRATEGY_EMA_RSI_COMBINED})", df['close'].iloc[i]))
    return signals


def detect_signals(df: pd.DataFrame, strategy: str) -> List[Dict[str, Any]]:
    """Detects trading signals based on the selected strategy."""
    if not valid_inputs: return [] # Prevent signal detection if inputs are invalid
    if df.empty: return []

    if strategy == STRATEGY_EMA_CROSS: return detect_ema_cross_signals(df)
    elif strategy == STRATEGY_RSI_DIVERGENCE: return detect_rsi_divergence_signals(df)
    elif strategy == STRATEGY_MACD_CROSS: return detect_macd_cross_signals(df)
    elif strategy == STRATEGY_BB_BOUNCE: return detect_bb_bounce_signals(df)
    elif strategy == STRATEGY_STOCH_OSCILLATOR: return detect_stochastic_signals(df)
    elif strategy == STRATEGY_EMA_RSI_COMBINED: return detect_ema_rsi_combined_signals(df)
    return []

@st.cache_data
def train_ml_model(df_train: pd.DataFrame) -> Tuple[Optional[RandomForestClassifier], List[str]]:
    """Trains a Random Forest Classifier model."""
    if not valid_inputs: return None, []
    df = df_train.copy()
    if df.empty: return None, []

    df['target'] = (df['close'].shift(-2) > df['close']).astype(int) 
    features = ['EMA5', 'EMA20', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'Stochastic']
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error(f"ML Model training failed: Missing feature columns: {', '.join(missing_features)}")
        return None, features
        
    df = df.dropna(subset=features + ['target'])
    if df.empty or len(df) < 10: 
        st.warning("Not enough data to train ML model after preprocessing (need at least 10 valid rows).")
        return None, features

    X = df[features]
    y = df['target']
    try:
        with st.spinner("Training ML Model (Random Forest)..."):
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X, y)
        return model, features
    except Exception as e:
        st.error(f"Error during ML model training: {e}")
        return None, features

def predict_with_ml_model(model: RandomForestClassifier, df_predict: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Generates signals using a trained ML model."""
    if not valid_inputs: return pd.DataFrame()
    df = df_predict.copy()
    if df.empty: return pd.DataFrame()

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error(f"ML Model prediction failed: Missing feature columns: {', '.join(missing_features)}")
        return pd.DataFrame() 

    df[features] = df[features].fillna(method='ffill').fillna(method='bfill')
    if df[features].isnull().any().any():
        st.warning("NaN values found in features for ML prediction. Results might be unreliable. Dropping rows with NaNs in features.")
        df = df.dropna(subset=features)

    if df.empty or df[features].empty:
        st.info("No data to make ML predictions after cleaning NaNs.")
        return pd.DataFrame()
        
    df['ML_Prediction'] = model.predict(df[features])
    df['Signal'] = df['ML_Prediction'].map({1: f'Buy ({STRATEGY_ML_MODEL})', 0: f'Sell ({STRATEGY_ML_MODEL})'})
    df['Trade Duration (min)'] = df['Signal'].apply(lambda x: DURATION_BUY if "Buy" in x else DURATION_SELL)
    return df[['timestamp', 'Signal', 'close', 'Trade Duration (min)']].rename(columns={'close': 'Price'}) 


def simulate_money_management(signals: List[Dict[str, Any]], 
                              mm_strategy: str = "Flat", 
                              initial_balance: float = 1000, 
                              risk_pct_trade: float = 2) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Simulates money management strategies based on signals."""
    balance = initial_balance
    wins, losses = 0, 0
    total_profit, total_loss = 0.0, 0.0
    current_bet_amount = balance * (risk_pct_trade / 100) 
    result_log = []

    if not signals:
        return pd.DataFrame(result_log), {
            "Win Rate (%)": 0, "ROI (%)": 0, "Max Drawdown (%)": 0, "Profit Factor": 0, "Total Trades": 0
        }

    for s in signals:
        win_probability = 0.55 
        is_win = np.random.choice([True, False], p=[win_probability, 1 - win_probability])

        if mm_strategy == "Risk %":
            bet_amount = balance * (risk_pct_trade / 100)
        elif mm_strategy == "Martingale":
            bet_amount = current_bet_amount
        else: # Flat betting
            bet_amount = initial_balance * (risk_pct_trade / 100) 
        
        bet_amount = min(bet_amount, balance)
        if balance <= 0 or bet_amount <=0 : 
            st.warning("Balance is zero or negative, or bet amount is zero. Stopping simulation.")
            break

        trade_pnl = 0
        if is_win:
            trade_pnl = bet_amount 
            balance += trade_pnl
            result = "Win"
            wins += 1
            total_profit += trade_pnl
            if mm_strategy == "Martingale":
                current_bet_amount = balance * (risk_pct_trade / 100) 
        else:
            trade_pnl = -bet_amount
            balance -= bet_amount 
            result = "Loss"
            losses += 1
            total_loss += bet_amount 
            if mm_strategy == "Martingale":
                current_bet_amount = bet_amount * 2 
        
        result_log.append({
            "Time": s["Time"], "Signal": s["Signal"], "Bet Amount": bet_amount,
            "Result": result, "Trade PnL": trade_pnl, "Balance": balance,
            "Trade Duration (min)": s["Trade Duration (min)"]
        })

    results_df = pd.DataFrame(result_log)
    max_drawdown_val = 0
    if not results_df.empty and 'Balance' in results_df.columns and initial_balance > 0:
        results_df['Cumulative Max Balance'] = results_df['Balance'].cummax()
        # Ensure cumulative max is at least initial balance for drawdown calc
        results_df['Cumulative Max Balance'] = results_df['Cumulative Max Balance'].apply(lambda x: max(x, initial_balance))
        results_df['Drawdown'] = (results_df['Cumulative Max Balance'] - results_df['Balance']) / results_df['Cumulative Max Balance']
        max_drawdown_val = results_df['Drawdown'].max()
        max_drawdown_val = max_drawdown_val if not pd.isna(max_drawdown_val) else 0

    roi_val = ((balance - initial_balance) / initial_balance) * 100 if initial_balance > 0 else 0
    win_rate_val = (100 * wins / (wins + losses)) if (wins + losses) > 0 else 0
    profit_factor_val = total_profit / total_loss if total_loss > 0 else (total_profit if total_profit > 0 else 0) 
    
    return results_df, {
        "Win Rate (%)": win_rate_val, "ROI (%)": roi_val, "Max Drawdown (%)": 100 * max_drawdown_val,
        "Profit Factor": profit_factor_val, "Total Trades": wins + losses
    }

def plot_chart(df: pd.DataFrame, asset_symbol: str) -> go.Figure:
    """Plots candlestick chart with EMAs."""
    fig = go.Figure()
    if df.empty:
        fig.update_layout(title=f"{asset_symbol} - No data to display", showlegend=True)
        return fig
        
    fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Candles'))
    if 'EMA5' in df.columns: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA5'], name="EMA Short", line=dict(color='blue')))
    if 'EMA20' in df.columns: fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA20'], name="EMA Long", line=dict(color='red')))
    if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BB_upper'], name="BB Upper", line=dict(color='rgba(152,251,152,0.5)')))
        fig.add_trace(go.Scatt
