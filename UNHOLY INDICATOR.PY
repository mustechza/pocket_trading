import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta


# ====== User Input Settings ======
with st.sidebar.expander("Show/Hide ALL"):
    show_all_ma = st.checkbox(
        "Moving Averages", value=True,
        help="There are 8 customizable moving averages available to plot as a ribbon."
    )

    # Bollinger Bands
    st.write("**Bollinger Bands**")
    show_bollinger = st.checkbox(
        "Enable Bollinger Bands", value=True,
        help="Enables Bollinger Bands with double standard deviation layers."
    )
    show_basis = st.checkbox(
        "Basis Line", value=False,
        help="Plots the midline (basis) of the Bollinger Band."
    )
    basis_color = st.color_picker(
        "Basis Line Color", value="#FFA500"
    )
    price_band_width = st.number_input(
        "Band Width", min_value=2.5, max_value=5.0, step=0.1, value=2.7
    )
    band_transparency = st.slider(
        "Band Transparency", min_value=0, max_value=90, step=10, value=50
    )

    # Watch Signals
    st.write("**Watch Signals**")
    show_watch_signals = st.checkbox(
        "Watch Signals", value=True,
        help="Plots signals preparing for possible entries."
    )
    show_backgrounds = st.checkbox(
        "Background", value=True,
        help="Colors background for watch signals."
    )
    watch_transparency = st.slider(
        "Watch Signal Transparency", min_value=0, max_value=100, value=80
    )

    # Buy/Sell Signals
    st.write("**Buy/Sell Signals**")
    show_signals = st.checkbox(
        "Buy/Sell Signals", value=True,
        help="Plots confirmed Buy/Sell entry markers."
    )
    show_circles = st.checkbox(
        "Circles", value=True,
        help="Adds visual circle markers to Buy/Sell signals."
    )
    show_signal_bg = st.checkbox(
        "Signal Background", value=True,
        help="Highlights the bar before a signal."
    )
    signal_transparency = st.slider(
        "Signal Transparency", min_value=0, max_value=100, value=60
    )

    # Candle Coloring
    st.write("**MA Colored Candles**")
    show_candles = st.checkbox(
        "Color Candles by MA Stages", value=True,
        help="Colors candles based on their relation to two MAs."
    )
    candle_transparency = st.slider(
        "Candle Transparency", min_value=0, max_value=100, step=10, value=20
    )

    # SuperTrend
    st.write("**Ultimate SuperTrend**")
    show_trend_band = st.checkbox(
        "SuperTrend", value=True,
        help="Plots a trend-following line with adaptive glow."
    )
    supertrend_width = st.number_input(
        "Trend Band Width", min_value=1, max_value=5, value=1
    )

# ====== Signal Sources ======
with st.sidebar.expander("Signal Sources"):
    use_rsi_signal = st.checkbox(
        "Use RSI Signals", value=True,
        help="Enables RSI-based signals when RSI crosses its moving average."
    )
    use_supertrend = st.checkbox(
        "Use SuperTrend Crosses", value=False,
        help="Adds signals when price breaks the SuperTrend in direction of trend."
    )
    use_supertrend_raw = st.checkbox(
        "All SuperTrend Signals", value=False,
        help="Signals based on SuperTrend without watch conditions."
    )

# ====== RSI Settings ======
with st.sidebar.expander("RSI Settings"):
    rsi_cycle = st.number_input("RSI Cycle", min_value=1, step=4, value=32)
    dz_size = st.number_input("Dead-Zone Size (RSI pts)", min_value=0.0, step=0.1, value=1.0)
    use_rsi_smoothing = st.checkbox(
        "RSI Signal Smoothing", value=False
    )
    smoothing_ma_type = st.selectbox(
        "Smoothing MA Type", options=["SMA", "EMA", "WMA", "RMA"], index=3
    )
    smoothing_length = st.number_input(
        "Smoothing Length", min_value=1, value=2
    )

# ====== SuperTrend Settings ======
with st.sidebar.expander("SuperTrend Settings"):
    supertrend_factor = st.number_input(
        "ATR Factor", min_value=0.1, step=0.1, value=1.5
    )
    supertrend_atr_length = st.number_input(
        "ATR Length", min_value=1, value=32
    )
    quiet_threshold = st.number_input(
        "Silence Threshold", min_value=0.0, step=0.02, value=0.1
    )

# Example usage: df should be a DataFrame with a 'close' column and datetime index
# df = pd.read_csv('data.csv', parse_dates=True, index_col='date')

def MA(series: pd.Series, length: int, ma_type: str) -> pd.Series:
    """
    Compute moving average of given type.
    """
    ma_type = ma_type.upper()
    if ma_type == 'SMA':
        return series.rolling(length).mean()
    elif ma_type == 'EMA':
        return series.ewm(span=length, adjust=False).mean()
    elif ma_type == 'WMA':
        return series.ta.wma(length)
    elif ma_type == 'RMA':
        # RMA is equivalent to Wilder's smoothing: same as EMA with alpha = 1/length
        alpha = 1 / length
        return series.ewm(alpha=alpha, adjust=False).mean()
    else:
        raise ValueError(f"Unknown MA type: {ma_type}")


def compute_indicators(
    df: pd.DataFrame,
    rsi_cycle: int = 14,
    use_rsi_smoothing: bool = False,
    smoothing_length: int = 5,
    smoothing_ma_type: str = 'EMA',
    dz_size: float = 5.0,
    supertrend_factor: float = 3.0,
    supertrend_atr_length: int = 10,
    quiet_threshold: float = 0.02,
    price_band_width: float = 2.0
) -> pd.DataFrame:
    close = df['close']

    # ROC
    previous = close.shift(10)
    roc = ((close - previous) / previous) * 100
    df['roc'] = roc
    df['roc_bullish'] = roc > 0
    df['roc_bearish'] = roc < 0

    # RSI and variants
    df['rsi'] = ta.rsi(close, length=rsi_cycle)
    half_cycle = max(1, rsi_cycle // 2)
    df['short_rsi'] = ta.rsi(close, length=half_cycle)

    # Smoothed RSI
    if use_rsi_smoothing:
        df['smoothed_rsi'] = MA(df['rsi'], smoothing_length, smoothing_ma_type)
    else:
        df['smoothed_rsi'] = df['rsi']

    # RSI moving averages and dead zone
    df['rsi_basis'] = df['rsi'].rolling(rsi_cycle).mean()
    df['rsi96'] = df['rsi'].rolling(96).apply(lambda x: ta.wma(pd.Series(x), length=96).iat[-1])
    df['rsi192'] = df['rsi'].rolling(192).apply(lambda x: ta.wma(pd.Series(x), length=192).iat[-1])

    df['upper_dz'] = df['rsi_basis'] + dz_size
    df['lower_dz'] = df['rsi_basis'] - dz_size

    # RSI Bollinger Bands
    df['rsi_dev'] = df['rsi'].rolling(rsi_cycle).std()
    df['upper_rsi_band'] = df['rsi_basis'] + 1.8 * df['rsi_dev']
    df['lower_rsi_band'] = df['rsi_basis'] - 1.8 * df['rsi_dev']

    # Supertrend
    st = ta.supertrend(df['high'], df['low'], close,
                       length=supertrend_atr_length, multiplier=supertrend_factor)
    df['supertrend'] = st[f'SUPERT_{supertrend_atr_length}_{supertrend_factor}']
    df['supertrend_dir'] = st[f'SUPERTd_{supertrend_atr_length}_{supertrend_factor}']

    # Noise Pulse
    ma5 = close.ewm(span=5, adjust=False).mean()
    ma6 = close.ewm(span=6, adjust=False).mean()
    ma7 = close.ewm(span=7, adjust=False).mean()
    noise_pulse = (ma5 - ma6).abs() + (ma6 - ma7).abs()
    df['noise_pulse'] = noise_pulse
    df['is_silent'] = noise_pulse < quiet_threshold

    # Supertrend direction flags
    df['super_up'] = df['supertrend_dir'] < 0
    df['super_down'] = df['supertrend_dir'] >= 0

    # Price Bollinger Bands
    df['price_basis'] = close.rolling(rsi_cycle).apply(lambda x: ta.wma(pd.Series(x), length=rsi_cycle).iat[-1])
    df['price_inner_dev'] = 2.4 * close.rolling(rsi_cycle).std()
    df['price_outer_dev'] = price_band_width * close.rolling(rsi_cycle).std()
    df['upper_price_inner'] = df['price_basis'] + df['price_inner_dev']
    df['lower_price_inner'] = df['price_basis'] - df['price_inner_dev']
    df['upper_price_outer'] = df['price_basis'] + df['price_outer_dev']
    df['lower_price_outer'] = df['price_basis'] - df['price_outer_dev']

    return df





