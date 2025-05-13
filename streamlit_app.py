import streamlit as st
import requests
import json
import os
import pandas as pd
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects

# Load API key/secret from environment or use fallback values
API_KEY = os.getenv('BINANCE_API_KEY_TEST', 'MhSFDGReh9WuilTZikVwW51OGujElIzOilRAoX7sgywPS4YMc5m0FQB67EWU0xfR')
API_SECRET = os.getenv('BINANCE_API_SECRET_TEST', 'j7BiDhZgKhaHIlPzNjv5KxQhwn3l0tWPGeVjUexNED4c3b3yEgoIwPMNgdR8nHi7')

# Binance API base URL
url = 'https://api.binance.us'

endpoint = '/api/v3/ticker/price'

headers = {
    'Content-Type': 'application/json',
    'X-MBX-APIKEY': API_KEY
}

# Title for Streamlit app
st.title("📈 Binance Ticker Prices")

try:
    response = requests.get(url + endpoint, headers=headers)
    response.raise_for_status()
    data = response.json()

    # Ensure we have a list of records
    if isinstance(data, list):
        df = pd.DataFrame.from_records(data)
    elif isinstance(data, dict):
        df = pd.DataFrame.from_records([data])
    else:
        raise ValueError("Unexpected format from Binance API")

    df['price'] = df['price'].astype(float)
    df = df.sort_values(by='symbol')

    st.dataframe(df)

except (ConnectionError, Timeout, TooManyRedirects) as e:
    st.error(f"Network-related error: {e}")
except requests.exceptions.RequestException as e:
    st.error(f"HTTP error: {e}")
except ValueError as e:
    st.error(f"Data error: {e}")
except Exception as e:
    st.error(f"Unexpected error: {e}")
