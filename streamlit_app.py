
#API_KEY = "MhSFDGReh9WuilTZikVwW51OGujElIzOilRAoX7sgywPS4YMc5m0FQB67EWU0xfR"
#API_SECRET = "j7BiDhZgKhaHIlPzNjv5KxQhwn3l0tWPGeVjUexNED4c3b3yEgoIwPMNgdR8nHi7"
import requests
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import os

api_key = os.environ['MhSFDGReh9WuilTZikVwW51OGujElIzOilRAoX7sgywPS4YMc5m0FQB67EWU0xfR']
api_secret = os.environ['j7BiDhZgKhaHIlPzNjv5KxQhwn3l0tWPGeVjUexNED4c3b3yEgoIwPMNgdR8nHi7']

url = 'https://api1.binance.com'
# url = https://api.binance.us # for US users

api_call = '/api/v3/ticker/price'

headers = {'content-type': 'application/json', 
           'X-MBX-APIKEY': api_key}

response = requests.get(url + api_call, headers=headers)

response = json.loads(response.text)
df = pd.DataFrame.from_records(response)
df.head()
