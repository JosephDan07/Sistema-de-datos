import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import requests
import feedparser
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

class SP500Analyzer:
    """
    Analizador completo del S&P 500
    Incluye datos históricos, análisis técnico, noticias y métricas fundamentales
    """
    
    def __init__(self):
        self.sp500_ticker = "^GSPC"
        self.spy_ticker = "SPY"  # ETF del S&P 500
        
        # Configurar estilo de matplotlib
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # Configurar pandas
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.2f}'.format)
    
    def get_sp500_data(self, period="1y"):
        """
        Obtiene datos históricos del S&P 500
        Períodos: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        """
        try:
            sp500 = yf.Ticker(self.sp500_ticker)
            data = sp500.history(period=period)
            
            if data.empty:
                print("❌ No se pudieron obtener datos del S&P 500")
                return None
            
            # Calcular indicadores técnicos
            data = self._calculate_technical_indicators(data)
            
            return data
        except Exception as e:
            print(f"❌ Error obteniendo datos: {e}")
            return None
    
    def _calculate_technical_indicators(self, data):
        """
        Calcula indicadores técnicos avanzados
        """
        # Medias móviles
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Media móvil exponencial
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # RSI (Índice de Fuerza Relativa)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        rolling_mean = data['Close'].rolling(window=20).mean()
        rolling_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = rolling_mean + (rolling_std * 2)
        data['BB_Lower'] = rolling_mean - (rolling_std * 2)
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        
        # Volatilidad
        data['Volatility'] = data['Close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
        
        # Rendimientos
        data['Daily_Return'] = data['Close'].pct_change()
        data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()
        
        return data
    
    def get_sp500_info(self):
        """
        Obtiene información general del S&P 500
        """
        try:
            sp500 = yf.Ticker(self.sp500_ticker)
            info = sp500.info
            
            # Información relevante
            relevant_info = {
                'Nombre': info.get('longName', 'S&P 500'),
                'Precio_Actual': info.get('regularMarketPrice', 'N/A'),
                'Cambio_Diario': info.get('regularMarketChange', 'N/A'),
                'Cambio_Porcentual': info.get('regularMarketChangePercent', 'N/A'),
                'Precio_Apertura': info.get('regularMarketOpen', 'N/A'),
                'Precio_Anterior': info.get('regularMarketPreviousClose', 'N/A'),
                'Máximo_Día': info.get('regularMarketDayHigh', 'N/A'),
                'Mínimo_Día': info.get('regularMarketDayLow', 'N/A'),
                'Máximo_52_Semanas': info.get('fiftyTwoWeekHigh', 'N/A'),
                'Mínimo_52_Semanas': info.get('fiftyTwoWeekLow', 'N/A'),
                'Volumen': info.get('regularMarketVolume', 'N/A'),
                'Volumen_Promedio': info.get('averageVolume', 'N/A'),
            }
            
            return relevant_info
        except Exception as e:
            print(f"❌ Error obteniendo información: {e}")
            return None
    
    def get_top_sp500_companies(self, limit=10):
        """
        Obtiene las principales empresas del S&P 500 por capitalización de mercado
        """
        # Lista de las principales empresas del S&P 500 (actualizada regularmente)
        top_companies = [
            'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 
            'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO',
            'JPM', 'WMT', 'XOM', 'UNH', 'V',
            'PG', 'JNJ', 'MA', 'HD', 'CVX'
        ]
        
        companies_data = []
        
        for ticker in top_companies[:limit]:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                company_info = {
                    'Ticker': ticker,
                    'Nombre': info.get('longName', 'N/A'),
                    'Sector': info.get('sector', 'N/A'),
                    'Precio': info.get('regularMarketPrice', 0),
                    'Cambio_Pct': info.get('regularMarketChangePercent', 0),
                    'Cap_Mercado': info.get('marketCap', 0),
                    'PE_Ratio': info.get('trailingPE', 'N/A'),
                    'Dividend_Yield': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                }
                companies_data.append(company_info)
                
            except Exception as e:
                print(f"⚠️ Error con {ticker}: {e}")
                continue
        
        return pd.DataFrame(companies_data)
    
    def get_market_news(self):
        """
        Obtiene noticias recientes del mercado y S&P 500
        """
        news_data = []
        
        # Fuentes de noticias financieras
        rss_feeds = [
            ('Yahoo Finance', 'https://feeds.finance.yahoo.com/rss/2.0/headline'),
            ('MarketWatch', 'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/'),
            ('Reuters Business', 'https://feeds.reuters.com/reuters/businessNews'),
        ]
        
        for source, url in rss_feeds:
            try:
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:5]:  # Top 5 por fuente
                    news_item = {
                        'Fuente': source,
                        'Título': entry.title,
                        'Fecha': entry.published if hasattr(entry, 'published') else 'N/A',
                        'Link': entry.link,
                        'Resumen': entry.summary[:200] + '...' if hasattr(entry, 'summary') and len(entry.summary) > 200 else entry.summary if hasattr(entry, 'summary') else 'N/A'
                    }
                    news_data.append(news_item)
                    
            except Exception as e:
                print(f"⚠️ Error obteniendo noticias de {source}: {e}")
                continue
        
        return pd.DataFrame(news_data)
    
    def analyze_market_sentiment(self, data):
        """
        Analiza el sentimiento del mercado basado en indicadores técnicos
        """
        if data is None or data.empty:
            return None
        
        latest = data.iloc[-1]
        
        sentiment_score = 0
        signals = []
        
        # Análisis de tendencia (Medias móviles)
        if latest['Close'] > latest['SMA_20']:
            sentiment_score += 1
            signals.append("✅ Precio por encima de SMA 20")
        else:
            sentiment_score -= 1
            signals.append("❌ Precio por debajo de SMA 20")
        
        if latest['Close'] > latest['SMA_50']:
            sentiment_score += 1
            signals.append("✅ Precio por encima de SMA 50")
        else:
            sentiment_score -= 1
            signals.append("❌ Precio por debajo de SMA 50")
        
        if latest['SMA_20'] > latest['SMA_50']:
            sentiment_score += 1
            signals.append("✅ Tendencia alcista (SMA 20 > SMA 50)")
        else:
            sentiment_score -= 1
            signals.append("❌ Tendencia bajista (SMA 20 < SMA 50)")
        
        # RSI
        if 30 <= latest['RSI'] <= 70:
            sentiment_score += 1
            signals.append(f"✅ RSI en zona neutral ({latest['RSI']:.1f})")
        elif latest['RSI'] < 30:
            sentiment_score += 1
            signals.append(f"🟡 RSI oversold - potencial rebote ({latest['RSI']:.1f})")
        else:
            sentiment_score -= 1
            signals.append(f"🔴 RSI overbought ({latest['RSI']:.1f})")
        
        # MACD
        if latest['MACD'] > latest['MACD_Signal']:
            sentiment_score += 1
            signals.append("✅ MACD bullish")
        else:
            sentiment_score -= 1
            signals.append("❌ MACD bearish")
        
        # Bollinger Bands
        if latest['BB_Lower'] < latest['Close'] < latest['BB_Upper']:
            sentiment_score += 1
            signals.append("✅ Precio dentro de Bollinger Bands")
        
        # Determinar sentimiento general
        if sentiment_score >= 3:
            sentiment = "🟢 ALCISTA"
        elif sentiment_score >= 0:
            sentiment = "🟡 NEUTRAL"
        else:
            sentiment = "🔴 BAJISTA"
        
        return {
            'sentimiento': sentiment,
            'score': sentiment_score,
            'señales': signals,
            'rsi': latest['RSI'],
            'volatilidad': latest['Volatility']
        }
    
    def create_visualizations(self, data, save_plots=False):
        """
        Crea visualizaciones completas del análisis
        """
        if data is None or data.empty:
            print("❌ No hay datos para visualizar")
            return
        
        # Configurar el subplot
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('📊 ANÁLISIS COMPLETO DEL S&P 500', fontsize=16, fontweight='bold')
        
        # 1. Precio y Medias Móviles
        ax1 = axes[0, 0]
        ax1.plot(data.index, data['Close'], label='S&P 500', linewidth=2, color='black')
        ax1.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7, color='blue')
        ax1.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7, color='orange')
        ax1.plot(data.index, data['SMA_200'], label='SMA 200', alpha=0.7, color='red')
        ax1.fill_between(data.index, data['BB_Lower'], data['BB_Upper'], alpha=0.2, color='gray', label='Bollinger Bands')
        ax1.set_title('💹 Precio y Medias Móviles')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. RSI
        ax2 = axes[0, 1]
        ax2.plot(data.index, data['RSI'], color='purple', linewidth=2)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax2.fill_between(data.index, 30, 70, alpha=0.2, color='yellow')
        ax2.set_title('📈 RSI (Relative Strength Index)')
        ax2.set_ylabel('RSI')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. MACD
        ax3 = axes[1, 0]
        ax3.plot(data.index, data['MACD'], label='MACD', color='blue', linewidth=2)
        ax3.plot(data.index, data['MACD_Signal'], label='Signal', color='red', linewidth=2)
        ax3.bar(data.index, data['MACD_Histogram'], label='Histogram', alpha=0.3, color='green')
        ax3.set_title('📊 MACD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Volatilidad y Volumen
        ax4 = axes[1, 1]
        ax4_twin = ax4.twinx()
        
        # Volatilidad
        ax4.plot(data.index, data['Volatility'], color='red', linewidth=2, label='Volatilidad (%)')
        ax4.set_ylabel('Volatilidad (%)', color='red')
        ax4.tick_params(axis='y', labelcolor='red')
        
        # Volumen
        ax4_twin.bar(data.index, data['Volume'], alpha=0.3, color='blue', label='Volumen')
        ax4_twin.set_ylabel('Volumen', color='blue')
        ax4_twin.tick_params(axis='y', labelcolor='blue')
        
        ax4.set_title('📊 Volatilidad y Volumen')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('sp500_analysis.png', dpi=300, bbox_inches='tight')
            print("📁 Gráficos guardados como 'sp500_analysis.png'")
        
        plt.show()
    
    def generate_report(self):
        """
        Genera un reporte completo del S&P 500
        """
        print("🚀 " + "="*60)
        print("📊 REPORTE COMPLETO DEL S&P 500")
        print("🗓️  Fecha del análisis:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("="*60)
        
        # 1. Información general
        print("\n1. 📋 INFORMACIÓN GENERAL")
        print("-" * 40)
        sp500_info = self.get_sp500_info()
        if sp500_info:
            for key, value in sp500_info.items():
                if isinstance(value, (int, float)) and value > 1000000:
                    print(f"{key}: {value:,.0f}")
                elif isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value}")
        
        # 2. Análisis técnico
        print("\n2. 📈 ANÁLISIS TÉCNICO (Últimos 6 meses)")
        print("-" * 40)
        data = self.get_sp500_data("6mo")
        
        if data is not None:
            latest = data.iloc[-1]
            
            print(f"💰 Precio actual: ${latest['Close']:,.2f}")
            print(f"📊 SMA 20: ${latest['SMA_20']:,.2f}")
            print(f"📊 SMA 50: ${latest['SMA_50']:,.2f}")
            print(f"📊 RSI: {latest['RSI']:.1f}")
            print(f"⚡ Volatilidad: {latest['Volatility']:.1f}%")
            
            # Rendimiento
            total_return = ((latest['Close'] / data['Close'].iloc[0]) - 1) * 100
            print(f"📈 Rendimiento 6 meses: {total_return:+.2f}%")
            
            # Análisis de sentimiento
            sentiment = self.analyze_market_sentiment(data)
            if sentiment:
                print(f"\n🎯 SENTIMIENTO DEL MERCADO: {sentiment['sentimiento']}")
                print(f"📊 Score: {sentiment['score']}/5")
                print("\n🔍 Señales técnicas:")
                for signal in sentiment['señales']:
                    print(f"   {signal}")
        
        # 3. Top empresas
        print("\n3. 🏆 TOP 10 EMPRESAS DEL S&P 500")
        print("-" * 40)
        top_companies = self.get_top_sp500_companies(10)
        if not top_companies.empty:
            # Mostrar tabla resumida
            display_cols = ['Ticker', 'Nombre', 'Precio', 'Cambio_Pct', 'PE_Ratio']
            available_cols = [col for col in display_cols if col in top_companies.columns]
            print(top_companies[available_cols].to_string(index=False))
        
        # 4. Noticias recientes
        print("\n4. 📰 NOTICIAS RECIENTES DEL MERCADO")
        print("-" * 40)
        news = self.get_market_news()
        if not news.empty:
            for idx, row in news.head(10).iterrows():
                print(f"\n📑 {row['Fuente']} - {row['Fecha']}")
                print(f"   {row['Título']}")
                print(f"   {row['Resumen']}")
                print(f"   🔗 {row['Link']}")
        
        # 5. Generar visualizaciones
        print("\n5. 📊 GENERANDO VISUALIZACIONES...")
        print("-" * 40)
        self.create_visualizations(data, save_plots=True)
        
        print("\n✅ ¡REPORTE COMPLETADO!")
        print("💡 Tip: Las visualizaciones se han guardado como 'sp500_analysis.png'")

def main():
    """
    Función principal para ejecutar el análisis completo del S&P 500
    """
    print("🚀 Iniciando análisis completo del S&P 500...")
    
    # Crear instancia del analizador
    analyzer = SP500Analyzer()
    
    # Generar reporte completo
    analyzer.generate_report()

if __name__ == "__main__":
    main()

