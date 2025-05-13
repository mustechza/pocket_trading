import requests
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import os

api_key = os.environ['BINANCE_API_KEY_TEST']
api_secret = os.environ['BINANCE_API_SECRET_TEST']

url = 'https://api1.binance.com'
# url = https://api.binance.us # for US users

api_call = '/api/v3/ticker/price'

headers = {'content-type': 'application/json', 
           'X-MBX-APIKEY': api_key}

response = requests.get(url + api_call, headers=headers)

response = json.loads(response.text)
df = pd.DataFrame.from_records(response)
df.head()
