import json
import talib
import pandas as pd
from binance.client import Client

with open('./params.json') as file: 
    params = json.load(file)

# Crea una instancia del cliente de Binance
client = Client(params["Api Key"], params["Secret Key"])

# Símbolo que deseas consultar
symbol = "BTCUSDT"

# Intervalo de tiempo (5 minutos)
interval = Client.KLINE_INTERVAL_5MINUTE

# Cuanto tiempo quiero coger
timeline = "35 day ago UTC"

# Índice de la llamada a la API
columns = ["open_time", "open_price", "high_price", "low_price", "close_price", 
           "volume", "close_time", "quote_asset_volume", "number_of_trades", 
           "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]

# Columnas que quiero dropear
dropColumns = ["open_time", "close_time", "quote_asset_volume", "taker_buy_base_asset_volume", 
               "taker_buy_quote_asset_volume", "ignore"]

# Obtiene los datos necesarios
candlesticks = pd.DataFrame(client.get_historical_klines(symbol, interval, timeline), columns=columns)
candlesticks.drop(columns=dropColumns, inplace=True)

#Convertir df a float
candlesticks = candlesticks.astype(float)