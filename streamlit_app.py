import requests
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import os
import pandas as pd

api_key = os.getenv('BINANCE_API_KEY_TEST', '')
api_secret = os.getenv('BINANCE_API_SECRET_TEST', '')

url = 'https://api1.binance.com'
api_call = '/api/v3/ticker/price'
headers = {'content-type': 'application/json', 'X-MBX-APIKEY': api_key}

try:
    response = requests.get(url + api_call, headers=headers)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame.from_records(data)
    print(df.head())
except (ConnectionError, Timeout, TooManyRedirects) as e:
    print("Network-related error occurred:", e)
except requests.exceptions.RequestException as e:
    print("HTTP error occurred:", e)
