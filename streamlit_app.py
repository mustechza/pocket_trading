
#API_KEY = "MhSFDGReh9WuilTZikVwW51OGujElIzOilRAoX7sgywPS4YMc5m0FQB67EWU0xfR"
#API_SECRET = "j7BiDhZgKhaHIlPzNjv5KxQhwn3l0tWPGeVjUexNED4c3b3yEgoIwPMNgdR8nHi7"
import streamlit as st
import requests
import json
import os
import pandas as pd  # ✅ This line is required

from requests.exceptions import ConnectionError, Timeout, TooManyRedirects


api_key = os.getenv('BINANCE_API_KEY_TEST', 'MhSFDGReh9WuilTZikVwW51OGujElIzOilRAoX7sgywPS4YMc5m0FQB67EWU0xfR')
api_secret = os.getenv('BINANCE_API_SECRET_TEST', 'j7BiDhZgKhaHIlPzNjv5KxQhwn3l0tWPGeVjUexNED4c3b3yEgoIwPMNgdR8nHi7')

url = 'https://api1.binance.com'
endpoint = '/api/v3/ticker/price'

try:
    res = requests.get(url + endpoint)
    res.raise_for_status()
    data = res.json()

    # Check if the response is a list or a single dict
    if isinstance(data, list):
        df = pd.DataFrame.from_records(data)
    elif isinstance(data, dict):
        df = pd.DataFrame.from_records([data])
    else:
        raise ValueError("Unexpected data format received from Binance.")

    print(df.head())

except Exception as e:
    print("Error:", e)
