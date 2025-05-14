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
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['BB_lower'], name="BB Lower", line=dict(color='rgba(152,251,152,0.5)'), fill='tonexty', fillcolor='rgba(152,251,152,0.1)'))

    fig.update_layout(title=f"{asset_symbol} Candlestick Chart", xaxis_rangeslider_visible=False, showlegend=True)
    return fig

# --- MAIN APPLICATION --- 
st.title("📈 Pocket Option Signals | Live + Backtest + Money Management")

if 'ml_model' not in st.session_state: st.session_state.ml_model = None
if 'ml_features' not in st.session_state: st.session_state.ml_features = []

# --- BACKTESTING SECTION ---
st.header("⚙️ Backtesting")
if uploaded_file and valid_inputs:
    df_hist_raw = pd.read_csv(uploaded_file)
    required_cols = ['timestamp', 'open', 'high', 'low', 'close']
    actual_cols = df_hist_raw.columns.str.lower().tolist()
    # Check for columns, case-insensitive
    df_hist_raw.columns = df_hist_raw.columns.str.lower()
    missing_cols = [col for col in required_cols if col not in df_hist_raw.columns]
    
    df_hist = pd.DataFrame() # Initialize as empty

    if missing_cols:
        st.error(f"Uploaded CSV is missing required columns: {', '.join(missing_cols)}. Please ensure the CSV contains at least: {', '.join(required_cols)}.")
    else:
        try:
            # Convert essential columns to numeric, coercing errors
            for col in ['open', 'high', 'low', 'close']:
                df_hist_raw[col] = pd.to_numeric(df_hist_raw[col], errors='coerce')
            if 'volume' in df_hist_raw.columns: # volume is optional but good to have
                 df_hist_raw['volume'] = pd.to_numeric(df_hist_raw['volume'], errors='coerce')
            
            df_hist_raw['timestamp'] = pd.to_datetime(df_hist_raw['timestamp'], errors='coerce')
            
            # Drop rows where essential data became NaN after conversion
            df_hist = df_hist_raw.dropna(subset=['timestamp', 'open', 'high', 'low', 'close']).copy() 
            # df_hist['timestamp'] = df_hist['timestamp'].apply(to_gmt_plus2) # Uncomment if to_gmt_plus2 is defined and needed
            
            if df_hist.empty:
                 st.warning("No valid data rows remaining after cleaning the uploaded CSV. Please check data quality (e.g., non-numeric values, date formats).")

        except Exception as e:
            st.error(f"Error processing data in uploaded file: {e}. Please check file format and content.")
            # df_hist remains empty

    if not df_hist.empty:
        df_hist_indicators = calculate_indicators(df_hist, ema_short_period, ema_long_period, rsi_period_val, stoch_period_val, bb_period_val)
        signals_hist = []

        if selected_strategy == STRATEGY_ML_MODEL:
            st.session_state.ml_model, st.session_state.ml_features = train_ml_model(df_hist_indicators)
            if st.session_state.ml_model:
                signals_df = predict_with_ml_model(st.session_state.ml_model, df_hist_indicators, st.session_state.ml_features)
                if not signals_df.empty: signals_hist = signals_df.to_dict(orient='records')
            else:
                st.warning("ML Model could not be trained with the uploaded data (or data was insufficient).")
        else:
            signals_hist = detect_signals(df_hist_indicators, selected_strategy)

        st.subheader("📌 Last 3 Signal Alerts (Backtest)")
        if signals_hist:
            latest_signals_hist = signals_hist[-3:]
            columns_hist = st.columns(len(latest_signals_hist))
            for i, s_hist in enumerate(latest_signals_hist):
                with columns_hist[i]:
                    st.markdown(f"""
                    #### 📊 Signal Alert
                    🧭 **Strategy:** {s_hist['Signal'].split('(')[1][:-1] if '(' in s_hist['Signal'] else selected_strategy}  
                    🕒 **Entry:** {s_hist['Time'].strftime('%Y-%m-%d %H:%M')}  
                    ⌛ **Duration:** {s_hist['Trade Duration (min)']} min  
                    🎯 **Price:** {s_hist['Price']:.5f}  
                    📢 **Direction:** {s_hist['Signal'].split(' ')[0]}
                    """)
        else:
            st.info("No signals generated for the backtest period with the current strategy and settings.")

        st.subheader("💰 Money Management Simulation (Backtest)")
        if signals_hist:
            results_df, metrics = simulate_money_management(signals_hist, mm_strategy=money_strategy, initial_balance=balance_input, risk_pct_trade=risk_pct)
            st.dataframe(results_df)
            st.subheader("📈 Performance Metrics (Backtest)")
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Win Rate (%)", f"{metrics['Win Rate (%)']:.2f}")
            col_m2.metric("ROI (%)", f"{metrics['ROI (%)']:.2f}")
            col_m3.metric("Max Drawdown (%)", f"{metrics['Max Drawdown (%)']:.2f}")
            col_m4.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")
            col_m5.metric("Total Trades", f"{metrics['Total Trades']}")
        else:
            st.info("No signals to simulate money management for backtest.")
        
        st.plotly_chart(plot_chart(df_hist_indicators, "Backtest Data"), use_container_width=True)
    elif uploaded_file and not df_hist.empty: # Only show if file was uploaded but processing failed to produce data
        st.warning("Uploaded CSV could not be processed successfully. Please check the file content and format.")

elif not uploaded_file:
    st.info("Upload a CSV file with historical data to run a backtest.")
elif not valid_inputs:
    st.warning("Please correct the input errors in the sidebar before running backtest or live signals.")


# --- LIVE SIGNALS SECTION ---
st.header("📡 Live Market Signal Detection")
if not valid_inputs:
    st.warning("Please correct the input errors in the sidebar before fetching live signals.")
elif not selected_assets:
    st.warning("Please select at least one asset for live signal detection.")
else:
    for asset in selected_assets:
        st.markdown(f"### {asset}")
        df_live = fetch_candles(asset, limit=CANDLE_LIMIT)
        
        if df_live is not None and not df_live.empty:
            df_live_indicators = calculate_indicators(df_live, ema_short_period, ema_long_period, rsi_period_val, stoch_period_val, bb_period_val)
            live_signals = []

            if selected_strategy == STRATEGY_ML_MODEL:
                if st.session_state.ml_model: 
                    signals_live_df = predict_with_ml_model(st.session_state.ml_model, df_live_indicators, st.session_state.ml_features)
                    if not signals_live_df.empty: live_signals = signals_live_df.to_dict(orient='records')
                else:
                    st.warning(f"ML Model ({STRATEGY_ML_MODEL}) has not been trained. Please upload historical data to train the model first for {asset}.")
            else:
                live_signals = detect_signals(df_live_indicators, selected_strategy)

            if live_signals:
                st.subheader(f"📌 Last 3 Live Signal Alerts for {asset}")
                latest_live_signals = live_signals[-3:]
                columns_live = st.columns(len(latest_live_signals))
                for i, s_live in enumerate(latest_live_signals):
                    with columns_live[i]:
                        st.markdown(f"""
                        #### 📊 Signal Alert
                        🧭 **Strategy:** {s_live['Signal'].split('(')[1][:-1] if '(' in s_live['Signal'] else selected_strategy}    
                        🕒 **Entry:** {s_live['Time'].strftime('%Y-%m-%d %H:%M')}  
                        ⌛ **Duration:** {s_live['Trade Duration (min)']} min  
                        🎯 **Price:** {s_live['Price']:.5f}  
                        📢 **Direction:** {s_live['Signal'].split(' ')[0]}
                        """)
            else:
                st.info(f"No live signals detected for {asset} with the current strategy and settings.")
            
            st.plotly_chart(plot_chart(df_live_indicators, asset), use_container_width=True)
        
        elif df_live is not None and df_live.empty:
            st.warning(f"No data returned from Binance for {asset}. The symbol might be invalid or the API might be temporarily unavailable.")
        # If df_live is None, fetch_candles already showed an error message

