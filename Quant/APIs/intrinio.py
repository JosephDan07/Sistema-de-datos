import requests
import pandas as pd
import json
from datetime import datetime, timedelta

class IntrinioAPI:
    """
    Cliente para la API de Intrinio
    Documentación: https://docs.intrinio.com/
    """
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api-v2.intrinio.com"
        self.headers = {
            'Content-Type': 'application/json'
        }
    
    def get_company_data(self, identifier="AAPL"):
        """
        Obtiene información básica de una empresa
        """
        url = f"{self.base_url}/companies/{identifier}"
        params = {'api_key': self.api_key}
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame([data])
            return df
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    
    def get_stock_prices(self, identifier="AAPL", start_date=None, end_date=None, frequency="daily"):
        """
        Obtiene precios históricos de acciones
        frequency: daily, weekly, monthly, quarterly, yearly
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        url = f"{self.base_url}/securities/{identifier}/prices"
        params = {
            'api_key': self.api_key,
            'start_date': start_date,
            'end_date': end_date,
            'frequency': frequency,
            'page_size': 100
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'stock_prices' in data:
                df = pd.DataFrame(data['stock_prices'])
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                return df
            else:
                return pd.DataFrame()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    
    def get_realtime_quote(self, identifier="AAPL"):
        """
        Obtiene cotización en tiempo real
        """
        url = f"{self.base_url}/securities/{identifier}/prices/realtime"
        params = {'api_key': self.api_key}
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame([data])
            return df
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    
    def get_market_indices(self):
        """
        Obtiene índices de mercado
        """
        url = f"{self.base_url}/indices/stock_market"
        params = {'api_key': self.api_key}
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'indices' in data:
                df = pd.DataFrame(data['indices'])
                return df
            else:
                return pd.DataFrame()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    
    def get_economic_data(self, tag="gdp"):
        """
        Obtiene datos económicos
        Tags populares: gdp, unemployment_rate, inflation_rate, etc.
        """
        url = f"{self.base_url}/historical_data/{tag}"
        params = {
            'api_key': self.api_key,
            'page_size': 100
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'historical_data' in data:
                df = pd.DataFrame(data['historical_data'])
                if not df.empty and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                return df
            else:
                return pd.DataFrame()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

def main():
    """
    Función principal para demostrar el uso de la API de Intrinio
    """
    # Configurar pandas para mejor visualización
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # IMPORTANTE: Reemplaza con tu API key de Intrinio
    # Puedes obtener una gratis en: https://intrinio.com/
    API_KEY = "Ojc1YWM0MTBmNWUyYmI5OGQ3OGI1Y2E5NmM5ZWI0OGUx"
    
    if API_KEY == "TU_API_KEY_AQUI":
        print("⚠️  IMPORTANTE: Por favor reemplaza 'TU_API_KEY_AQUI' con tu API key real de Intrinio")
        print("📝 Puedes obtener una API key gratuita en: https://intrinio.com/")
        return
    
    # Crear instancia de la API
    intrinio = IntrinioAPI(API_KEY)
    
    print("=== DEMO DE INTRINIO API ===\n")
    
    # 1. Información de empresa
    print("1. 📊 INFORMACIÓN DE EMPRESA (AAPL)")
    print("-" * 50)
    company_data = intrinio.get_company_data("AAPL")
    if company_data is not None and not company_data.empty:
        print(f"Empresa: {company_data['name'].iloc[0] if 'name' in company_data.columns else 'N/A'}")
        print(f"Sector: {company_data['sector'].iloc[0] if 'sector' in company_data.columns else 'N/A'}")
        print(f"Ticker: {company_data['ticker'].iloc[0] if 'ticker' in company_data.columns else 'N/A'}")
        print()
    
    # 2. Precios históricos
    print("2. 📈 PRECIOS HISTÓRICOS (AAPL - Últimos 30 días)")
    print("-" * 50)
    prices_data = intrinio.get_stock_prices("AAPL")
    if prices_data is not None and not prices_data.empty:
        print(f"Registros obtenidos: {len(prices_data)}")
        print("\nÚltimos 5 precios:")
        print(prices_data[['date', 'close', 'high', 'low', 'volume']].tail())
        print(f"\nPrecio más reciente: ${prices_data['close'].iloc[-1]:.2f}")
        print()
    
    # 3. Cotización en tiempo real
    print("3. ⚡ COTIZACIÓN EN TIEMPO REAL (AAPL)")
    print("-" * 50)
    realtime_data = intrinio.get_realtime_quote("AAPL")
    if realtime_data is not None and not realtime_data.empty:
        print(realtime_data[['last_price', 'bid_price', 'ask_price', 'last_time']].to_string(index=False))
        print()
    
    # 4. Índices de mercado
    print("4. 🏛️  ÍNDICES DE MERCADO")
    print("-" * 50)
    indices_data = intrinio.get_market_indices()
    if indices_data is not None and not indices_data.empty:
        print(f"Número de índices: {len(indices_data)}")
        if len(indices_data) > 0:
            print("\nPrimeros 5 índices:")
            print(indices_data[['symbol', 'name']].head())
        print()
    
    print("✅ Demo completada!")
    print("\n💡 Consejos:")
    print("- Modifica los parámetros (símbolos, fechas) según tus necesidades")
    print("- Revisa la documentación de Intrinio para más endpoints: https://docs.intrinio.com/")
    print("- Ten en cuenta los límites de tu plan de API")

if __name__ == "__main__":
    main()
