import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt

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



# --- User settings (example values) ---
band_transparency = 0.3
bull_bear_ma_length = 20
bull_bear_ma_type = "ema"        # 'sma', 'ema', etc.
show_bollinger = True
upper_bb_color_up = (0.0, 1.0, 0.0)    # matplotlib RGB tuples
upper_bb_color_down = (1.0, 0.0, 0.0)
lower_bb_color_up = (0.0, 0.0, 1.0)
lower_bb_color_down = (1.0, 1.0, 0.0)
show_basis = True
basis_color = 'gray'

# MA selections
ma_lengths = [10, 20, 50, 100, 200, 5, 7, 14]
ma_types   = ['sma','ema','sma','ema','sma','ema','sma','ema']
ma_colors  = ['#ff0000','#00ff00','#0000ff','#ff00ff','#00ffff','#ffff00','#888888','#444444']
show_all_ma = True
show_ma     = [True]*8  # toggle individual MAs

# Fast/slow MA for candle coloring
fast_ma_len, fast_ma_type = 5, 'ema'
slow_ma_len, slow_ma_type = 20, 'sma'

# --- Helper to compute a moving average of a given type ---
def MA(series, length, ma_type):
    if ma_type.lower() == 'sma':
        return series.rolling(length).mean()
    elif ma_type.lower() == 'ema':
        return series.ewm(span=length, adjust=False).mean()
    else:
        raise ValueError(f"Unsupported MA type: {ma_type}")

# --- Compute the bull/bear MA ---
df['bullBearMA'] = MA(df['close'], bull_bear_ma_length, bull_bear_ma_type)
df['maBullish']  = df['close'] >= df['bullBearMA']
df['maBearish']  = df['close'] <  df['bullBearMA']

# Upper Bollinger band fill color
def band_color(cond, up_color, down_color):
    """Return RGBA tuple based on condition and base RGB + transparency."""
    rgb = np.array(up_color) if cond else np.array(down_color)
    return (*rgb, band_transparency)

df['upperBandColor'] = df['maBullish'].map(
    lambda b: band_color(b, upper_bb_color_up, upper_bb_color_down)
) if show_bollinger else None

# Lower Bollinger band fill color (based on your rocBullish boolean)
df['lowerBandColor'] = df['rocBullish'].map(
    lambda b: band_color(b, lower_bb_color_up, lower_bb_color_down)
) if show_bollinger else None

# Basis line (middle of BB)
df['basisPlot'] = df['priceBasis'].where(show_bollinger and show_basis)

# --- Plotting ---
fig, ax = plt.subplots(figsize=(12,6))
# Invisible plots (just to align x-axis)
ax.plot(df.index, df['upperPriceInner'], alpha=0)
ax.plot(df.index, df['upperPriceOuter'], alpha=0)
ax.plot(df.index, df['lowerPriceInner'], alpha=0)
ax.plot(df.index, df['lowerPriceOuter'], alpha=0)

# Fill between upper bands
if show_bollinger:
    ax.fill_between(
        df.index,
        df['upperPriceInner'],
        df['upperPriceOuter'],
        color=list(df['upperBandColor']),
        linewidth=0
    )
    # Fill between lower bands
    ax.fill_between(
        df.index,
        df['lowerPriceInner'],
        df['lowerPriceOuter'],
        color=list(df['lowerBandColor']),
        linewidth=0
    )

# Plot basis line
if show_bollinger and show_basis:
    ax.plot(df.index, df['basisPlot'], color=basis_color, label='Price Basis')

# --- User‐selected MAs ---
for i, (length, mtype, col, show_flag) in enumerate(zip(ma_lengths, ma_types, ma_colors, show_ma), start=1):
    if show_all_ma and show_flag:
        df[f'ma{i}'] = MA(df['close'], length, mtype)
        ax.plot(df.index, df[f'ma{i}'], color=col, label=f'MA{i}')

# --- Candle coloring logic (for your own candlestick chart) ---
df['fastMa'] = MA(df['close'], fast_ma_len, fast_ma_type)
df['slowMa'] = MA(df['close'], slow_ma_len, slow_ma_type)
df['belowBoth']   = (df['close'] < df['fastMa']) & (df['close'] < df['slowMa'])
df['betweenBoth'] = ((df['close'] >= df['fastMa']) & (df['close'] < df['slowMa'])) | \
                    ((df['close'] < df['fastMa'])  & (df['close'] >= df['slowMa']))
df['aboveBoth']   = (df['close'] >= df['fastMa']) & (df['close'] >= df['slowMa'])

# --- RSI and Supertrend events (example boolean columns you already have) ---
# df['rsiCrossOverLower'], df['rsiCrossUnderUpper'], df['deadCrossOver'], etc.

# --- Alert conditions ---
df['bullishSignal']  = df['maBullish'] & df['rocBullish']
df['bearishSignal']  = df['maBearish'] & df['rocBearish']

# You can now loop over df and trigger your own alerts:
for idx, row in df.iterrows():
    if row['bullishSignal']:
        print(f"{idx}: Bullish Signal – Both Bollinger Bands are Bullish")
    if row['bearishSignal']:
        print(f"{idx}: Bearish Signal – Both Bollinger Bands are Bearish")

ax.legend(loc='best')
plt.show()


def convert_pine_to_python_signals(df: pd.DataFrame,
                                   lookback: int = 35,
                                   use_supertrend: bool = True,
                                   use_supertrend_raw: bool = True,
                                   use_rsi_signal: bool = True,
                                   enable_advanced_alerts: bool = False) -> pd.DataFrame:
    """
    Given a DataFrame `df` with the required Pine-script-derived boolean columns,
    this function computes equivalent buy/sell watch signals, final signals,
    and alert flags as new columns in the DataFrame.

    Required boolean columns in `df`:
      - priceUnderLowerInner, priceOverUpperInner
      - rsiCrossOverLower, rsiCrossUnderUpper
      - shortRsiOver30, shortRsiUnder70
      - deadCrossOver, deadCrossUnder
      - superTrendBuyTrigger, superTrendSellTrigger
      - barConfirm  # True when barstate.isconfirmed in Pine

    Returns a new DataFrame with additional columns:
      - buyWatched, sellWatched
      - buyWatchCount, sellWatchCount
      - buyWatchMet, sellWatchMet
      - superBuySignal, superSellSignal
      - rsiBuySignal, rsiSellSignal
      - buySignal, sellSignal
      - effectiveBuyWatchMet, effectiveSellWatchMet
      - plotBuyBG, plotSellBG
      - plotBuy, plotSell
      - alertBuy, alertSell
    """
    df = df.copy()

    # 1. Watch triggers
    df['buyWatched'] = (
        df['priceUnderLowerInner'] |
        df['rsiCrossOverLower'] |
        df['shortRsiOver30']
    ) & df['barConfirm']
    df['sellWatched'] = (
        df['priceOverUpperInner'] |
        df['rsiCrossUnderUpper'] |
        df['shortRsiUnder70']
    ) & df['barConfirm']

    # 2. Rolling sum over `lookback` bars
    df['buyWatchCount'] = (
        df['buyWatched']
        .astype(int)
        .rolling(window=lookback, min_periods=1)
        .sum()
    )
    df['sellWatchCount'] = (
        df['sellWatched']
        .astype(int)
        .rolling(window=lookback, min_periods=1)
        .sum()
    )

    df['buyWatchMet'] = df['buyWatchCount'] >= 1
    df['sellWatchMet'] = df['sellWatchCount'] >= 1

    # 3. Supertrend signals
    df['superBuySignal'] = use_supertrend & df['superTrendBuyTrigger']
    df['superSellSignal'] = use_supertrend & df['superTrendSellTrigger']

    # 4. RSI signals
    df['rsiBuySignal'] = use_rsi_signal & df['deadCrossOver'] & df['buyWatchMet']
    df['rsiSellSignal'] = use_rsi_signal & df['deadCrossUnder'] & df['sellWatchMet']

    # 5. Combine final triggers
    df['buySignal'] = df['superBuySignal'] | df['rsiBuySignal']
    df['sellSignal'] = df['superSellSignal'] | df['rsiSellSignal']

    # 6. Effective watch filters when raw supertrend is on
    df['effectiveBuyWatchMet'] = np.where(
        use_supertrend and use_supertrend_raw,
        True,
        df['buyWatchMet']
    )
    df['effectiveSellWatchMet'] = np.where(
        use_supertrend and use_supertrend_raw,
        True,
        df['sellWatchMet']
    )

    # 7. Background (advanced) signals
    df['plotBuyBG'] = False
    df['plotSellBG'] = False
    buy_bg_idx = df['buySignal'] & df['effectiveBuyWatchMet']
    sell_bg_idx = df['sellSignal'] & df['effectiveSellWatchMet'] & ~buy_bg_idx
    df.loc[buy_bg_idx, 'plotBuyBG'] = True
    df.loc[sell_bg_idx, 'plotSellBG'] = True

    # 8. Confirmed (final) signals at bar close
    df['plotBuy'] = False
    df['plotSell'] = False
    buy_conf_idx = buy_bg_idx & df['barConfirm']
    sell_conf_idx = sell_bg_idx & df['barConfirm'] & ~buy_conf_idx
    df.loc[buy_conf_idx, 'plotBuy'] = True
    df.loc[sell_conf_idx, 'plotSell'] = True

    # 9. Alerts based on advanced vs. confirmed
    df['alertBuy'] = np.where(
        enable_advanced_alerts,
        df['plotBuyBG'],
        df['plotBuy']
    )
    df['alertSell'] = np.where(
        enable_advanced_alerts,
        df['plotSellBG'],
        df['plotSell']
    )

    return df




