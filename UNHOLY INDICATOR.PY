import streamlit as st

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
