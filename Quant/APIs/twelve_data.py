import requests
import pandas as pd
import json
from datetime import datetime, timedelta

class TwelveDataAPI:
    """
    Cliente para la API de Twelve Data
    Documentación: https://twelvedata.com/docs
    Especializado en datos de criptomonedas, acciones, forex y más
    """
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.headers = {
            'Content-Type': 'application/json'
        }
    
    def get_crypto_price(self, symbol="BTC/USD", interval="1day", outputsize=30):
        """
        Obtiene precios históricos de criptomonedas
        Intervalos: 1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 1day, 1week, 1month
        """
        url = f"{self.base_url}/time_series"
        params = {
            'symbol': symbol,
            'interval': interval,
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'values' in data:
                df = pd.DataFrame(data['values'])
                if not df.empty:
                    # Convertir columnas numéricas
                    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_columns:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Convertir fecha
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df.sort_values('datetime')
                    
                    # Información del símbolo
                    meta_info = data.get('meta', {})
                    return df, meta_info
                
            elif 'code' in data:
                print(f"Error de API: {data.get('message', 'Error desconocido')}")
                return None, None
        
        print(f"Error HTTP: {response.status_code} - {response.text}")
        return None, None
    
    def get_crypto_quote(self, symbol="BTC/USD"):
        """
        Obtiene cotización en tiempo real de criptomonedas
        """
        url = f"{self.base_url}/quote"
        params = {
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'code' not in data:  # No hay error
                df = pd.DataFrame([data])
                return df
            else:
                print(f"Error de API: {data.get('message', 'Error desconocido')}")
        else:
            print(f"Error HTTP: {response.status_code} - {response.text}")
        
        return None
    
    def get_multiple_crypto_quotes(self, symbols=["BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD"]):
        """
        Obtiene cotizaciones de múltiples criptomonedas
        """
        symbol_string = ','.join(symbols)
        url = f"{self.base_url}/quote"
        params = {
            'symbol': symbol_string,
            'apikey': self.api_key
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Si es una sola respuesta, convertir a lista
            if isinstance(data, dict) and 'symbol' in data:
                data = [data]
            
            # Filtrar errores
            valid_data = [item for item in data if 'code' not in item]
            
            if valid_data:
                df = pd.DataFrame(valid_data)
                return df
            
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
    def get_crypto_exchanges(self):
        """
        Obtiene lista de exchanges de criptomonedas disponibles
        """
        url = f"{self.base_url}/exchanges"
        params = {
            'type': 'crypto',
            'apikey': self.api_key
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                df = pd.DataFrame(data['data'])
                return df
        
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
    def get_market_state(self):
        """
        Obtiene el estado actual del mercado
        """
        url = f"{self.base_url}/market_state"
        params = {'apikey': self.api_key}
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            return df
        
        print(f"Error: {response.status_code} - {response.text}")
        return None

def analyze_bitcoin_data(df):
    """
    Análisis específico para datos de Bitcoin
    """
    if df is None or df.empty:
        return
    
    print("📊 ANÁLISIS DE BITCOIN")
    print("=" * 50)
    
    # Precio actual vs anterior
    current_price = df['close'].iloc[-1]
    previous_price = df['close'].iloc[-2] if len(df) > 1 else current_price
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price) * 100
    
    print(f"💰 Precio actual: ${current_price:,.2f}")
    print(f"📈 Cambio: ${price_change:+,.2f} ({price_change_pct:+.2f}%)")
    
    # Estadísticas del período
    max_price = df['high'].max()
    min_price = df['low'].min()
    
    print(f"🔺 Máximo del período: ${max_price:,.2f}")
    print(f"🔻 Mínimo del período: ${min_price:,.2f}")
    
    # Volumen solo si está disponible
    if 'volume' in df.columns:
        avg_volume = df['volume'].mean()
        print(f"📊 Volumen promedio: {avg_volume:,.0f}")
    else:
        print("📊 Volumen: No disponible en este endpoint")
    
    # Volatilidad
    returns = df['close'].pct_change().dropna()
    volatility = returns.std() * 100
    print(f"⚡ Volatilidad: {volatility:.2f}%")

def main():
    """
    Función principal para demostrar el uso de la API de Twelve Data con Bitcoin
    """
    # Configurar pandas
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # IMPORTANTE: Reemplaza con tu API key de Twelve Data
    # Puedes obtener una gratis en: https://twelvedata.com/
    API_KEY = "15505d35d2a44444bd8e0c80274b33cb"
    
    if API_KEY == "TU_API_KEY_AQUI":
        print("⚠️  IMPORTANTE: Por favor reemplaza 'TU_API_KEY_AQUI' con tu API key real de Twelve Data")
        print("📝 Puedes obtener una API key gratuita en: https://twelvedata.com/")
        print("🎁 Plan gratuito incluye: 800 requests/día para criptomonedas")
        return
    
    # Crear instancia de la API
    twelve_data = TwelveDataAPI(API_KEY)
    
    print("🚀 === DEMO DE TWELVE DATA API - BITCOIN FOCUS ===\n")
    
    # 1. Bitcoin - Datos históricos diarios
    print("1. 📈 BITCOIN - DATOS HISTÓRICOS (30 días)")
    print("-" * 60)
    btc_data, btc_meta = twelve_data.get_crypto_price("BTC/USD", "1day", 30)
    
    if btc_data is not None:
        print(f"Símbolo: {btc_meta.get('symbol', 'BTC/USD')}")
        print(f"Exchange: {btc_meta.get('exchange', 'N/A')}")
        print(f"Registros: {len(btc_data)}")
        
        print(f"\nColumnas disponibles: {list(btc_data.columns)}")
        
        print("\n📋 Últimos 5 días:")
        # Seleccionar solo las columnas que existen
        base_cols = ['datetime', 'open', 'high', 'low', 'close']
        available_cols = [col for col in base_cols if col in btc_data.columns]
        if 'volume' in btc_data.columns:
            available_cols.append('volume')
        
        recent_data = btc_data[available_cols].tail()
        print(recent_data.to_string(index=False))
        
        # Análisis específico de Bitcoin
        print("\n")
        analyze_bitcoin_data(btc_data)
        print()
    
    # 2. Bitcoin - Cotización en tiempo real
    print("2. ⚡ BITCOIN - COTIZACIÓN EN TIEMPO REAL")
    print("-" * 60)
    btc_quote = twelve_data.get_crypto_quote("BTC/USD")
    
    if btc_quote is not None:
        print("Datos en tiempo real:")
        print(f"Columnas disponibles: {list(btc_quote.columns)}")
        
        # Mostrar todas las columnas disponibles
        print(btc_quote.to_string(index=False))
        print()
    else:
        print("No se pudieron obtener datos en tiempo real")
        print()
    
    # 3. Top Criptomonedas
    print("3. 🏆 TOP CRIPTOMONEDAS - COTIZACIONES")
    print("-" * 60)
    top_cryptos = ["BTC/USD", "ETH/USD", "BNB/USD", "ADA/USD", "SOL/USD"]
    crypto_quotes = twelve_data.get_multiple_crypto_quotes(top_cryptos)
    
    if crypto_quotes is not None:
        print(f"Criptomonedas obtenidas: {len(crypto_quotes)}")
        print(f"Columnas disponibles: {list(crypto_quotes.columns)}")
        
        # Mostrar todas las columnas disponibles
        print(crypto_quotes.to_string(index=False))
        print()
    else:
        print("No se pudieron obtener cotizaciones múltiples")
        print()
    
    # 4. Bitcoin en diferentes intervalos
    print("4. ⏰ BITCOIN - DIFERENTES INTERVALOS DE TIEMPO")
    print("-" * 60)
    intervals = ["1h", "4h", "1day"]
    
    for interval in intervals:
        print(f"\n📊 Intervalo: {interval}")
        data, meta = twelve_data.get_crypto_price("BTC/USD", interval, 5)
        if data is not None:
            latest = data.iloc[-1]
            print(f"   Último precio: ${latest['close']:,.2f}")
            if 'volume' in data.columns:
                print(f"   Volumen: {latest['volume']:,.0f}")
            else:
                print(f"   Volumen: No disponible")
    
    print("\n✅ Demo completada!")
    print("\n💡 Consejos para Bitcoin y criptomonedas:")
    print("- Los precios cambian 24/7, a diferencia de acciones tradicionales")
    print("- Usa intervalos más cortos (1min, 5min) para trading intradía")
    print("- El volumen es crucial para entender la liquidez")
    print("- Twelve Data ofrece datos de múltiples exchanges")
    print("- Plan gratuito: 800 requests/día - ¡perfecto para análisis básico!")

if __name__ == "__main__":
    main()
