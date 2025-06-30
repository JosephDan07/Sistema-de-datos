import requests
import pandas as pd
import json

headers = {
    'Content-Type': 'application/json'
}

# Obtener datos de la API de Tiingo
requestResponse = requests.get("https://api.tiingo.com/tiingo/fx/top?tickers=audusd,eurusd&token=9eaef8cdc8d497afb83a312d53be56633ec1bf51", headers=headers)

# Verificar si la respuesta es exitosa
if requestResponse.status_code == 200:
    data = requestResponse.json()
    
    # Convertir los datos a DataFrame de pandas
    df = pd.DataFrame(data)
    
    # Configurar pandas para mostrar todas las columnas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print("=== DATOS DE FOREX (AUDUSD, EURUSD) DESDE TIINGO ===")
    print(f"Número de registros: {len(df)}")
    print("\nDataFrame completo:")
    print(df)
    
    print("\nInformación del DataFrame:")
    print(df.info())
    
    print("\nEstadísticas descriptivas:")
    print(df.describe())
    
    # Procesar timestamps si existen
    if 'quoteTimestamp' in df.columns:
        df['quoteTimestamp'] = pd.to_datetime(df['quoteTimestamp'])
        print(f"\nDatos de precios por par de divisas:")
        print(df[['ticker', 'quoteTimestamp', 'bidPrice', 'askPrice', 'midPrice']])
    
else:
    print(f"Error en la solicitud: {requestResponse.status_code}")
    print(f"Respuesta: {requestResponse.text}")

# === SECCIÓN WEBSOCKET (COMENTADA TEMPORALMENTE) ===
# Descomenta esta sección si quieres recibir datos en tiempo real

"""
from websocket import create_connection
import simplejson as json
ws = create_connection("wss://api.tiingo.com/test")

subscribe = {
                'eventName':'subscribe',
                'eventData': {
                            'authToken': '9eaef8cdc8d497afb83a312d53be56633ec1bf51'
                            }
                }

ws.send(json.dumps(subscribe))
while True:
    print(ws.recv())
"""
                        
