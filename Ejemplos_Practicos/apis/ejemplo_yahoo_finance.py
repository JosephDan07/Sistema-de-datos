#!/usr/bin/env python3
"""
Ejemplo Práctico: API de Yahoo Finance
=====================================

Este ejemplo demuestra cómo utilizar la API de Yahoo Finance implementada
en el sistema para obtener datos financieros, realizar análisis técnico
y generar reportes profesionales.

Características:
- Conexión a Yahoo Finance API
- Análisis técnico automático
- Generación de reportes
- Visualizaciones profesionales
- Manejo de errores robusto

Autor: Sistema-de-datos
Fecha: Julio 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import sys
import os

# Configurar warnings
warnings.filterwarnings('ignore')

class YahooFinanceExample:
    """
    Ejemplo completo de uso de Yahoo Finance API
    """
    
    def __init__(self):
        self.setup_plotting()
        print("🚀 Inicializando ejemplo de Yahoo Finance API")
        
    def setup_plotting(self):
        """Configurar estilo de gráficos"""
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        sns.set_palette("husl")
        
        # Configurar pandas
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.2f}'.format)
    
    def get_stock_data(self, symbol, period="1y"):
        """
        Obtener datos históricos de una acción
        
        Args:
            symbol: Símbolo de la acción (ej: 'AAPL', 'GOOGL')
            period: Período de datos (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y)
            
        Returns:
            DataFrame con datos históricos
        """
        print(f"📊 Obteniendo datos de {symbol} para período {period}...")
        
        try:
            # Crear objeto ticker
            ticker = yf.Ticker(symbol)
            
            # Obtener datos históricos
            data = ticker.history(period=period)
            
            if data.empty:
                print(f"❌ No se pudieron obtener datos para {symbol}")
                return None
            
            # Agregar información adicional
            data['Symbol'] = symbol
            
            # Calcular indicadores técnicos básicos
            data = self.calculate_technical_indicators(data)
            
            print(f"✅ Datos obtenidos exitosamente para {symbol}")
            print(f"   📅 Período: {data.index.min().date()} a {data.index.max().date()}")
            print(f"   📊 Registros: {len(data)}")
            print(f"   💰 Precio actual: ${data['Close'].iloc[-1]:.2f}")
            
            return data
            
        except Exception as e:
            print(f"❌ Error al obtener datos de {symbol}: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, data):
        """
        Calcular indicadores técnicos
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            DataFrame con indicadores técnicos agregados
        """
        print("📈 Calculando indicadores técnicos...")
        
        # Medias móviles
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / data['BB_Width']
        
        # Volatilidad
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252) * 100
        
        # Average True Range (ATR)
        data['High_Low'] = data['High'] - data['Low']
        data['High_Close'] = abs(data['High'] - data['Close'].shift())
        data['Low_Close'] = abs(data['Low'] - data['Close'].shift())
        data['True_Range'] = data[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
        data['ATR'] = data['True_Range'].rolling(window=14).mean()
        
        # On-Balance Volume (OBV)
        data['OBV'] = (data['Volume'] * data['Returns'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)).cumsum()
        
        # VWAP (Volume Weighted Average Price)
        data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
        
        print("✅ Indicadores técnicos calculados")
        
        return data
    
    def get_stock_info(self, symbol):
        """
        Obtener información fundamental de una acción
        
        Args:
            symbol: Símbolo de la acción
            
        Returns:
            Dict con información fundamental
        """
        print(f"📋 Obteniendo información fundamental de {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extraer información relevante
            stock_info = {
                'Symbol': symbol,
                'Company_Name': info.get('longName', 'N/A'),
                'Sector': info.get('sector', 'N/A'),
                'Industry': info.get('industry', 'N/A'),
                'Market_Cap': info.get('marketCap', 0),
                'Enterprise_Value': info.get('enterpriseValue', 0),
                'PE_Ratio': info.get('trailingPE', 0),
                'Forward_PE': info.get('forwardPE', 0),
                'PEG_Ratio': info.get('pegRatio', 0),
                'Price_to_Book': info.get('priceToBook', 0),
                'Price_to_Sales': info.get('priceToSalesTrailing12Months', 0),
                'Dividend_Yield': info.get('dividendYield', 0),
                'Beta': info.get('beta', 0),
                'ROE': info.get('returnOnEquity', 0),
                'ROA': info.get('returnOnAssets', 0),
                'Profit_Margin': info.get('profitMargins', 0),
                'Debt_to_Equity': info.get('debtToEquity', 0),
                'Current_Ratio': info.get('currentRatio', 0),
                'Quick_Ratio': info.get('quickRatio', 0),
                'Revenue_Growth': info.get('revenueGrowth', 0),
                'Earnings_Growth': info.get('earningsGrowth', 0),
                'Recommendation': info.get('recommendationKey', 'N/A'),
                'Target_Price': info.get('targetMeanPrice', 0),
                '52_Week_High': info.get('fiftyTwoWeekHigh', 0),
                '52_Week_Low': info.get('fiftyTwoWeekLow', 0)
            }
            
            print(f"✅ Información fundamental obtenida para {symbol}")
            print(f"   🏢 Empresa: {stock_info['Company_Name']}")
            print(f"   🏭 Sector: {stock_info['Sector']}")
            print(f"   💰 Market Cap: ${stock_info['Market_Cap']:,.0f}")
            print(f"   📊 P/E Ratio: {stock_info['PE_Ratio']:.2f}")
            
            return stock_info
            
        except Exception as e:
            print(f"❌ Error al obtener información de {symbol}: {str(e)}")
            return None
    
    def analyze_stock(self, data, symbol):
        """
        Realizar análisis técnico completo
        
        Args:
            data: DataFrame con datos de la acción
            symbol: Símbolo de la acción
            
        Returns:
            Dict con análisis completo
        """
        print(f"🔍 Analizando {symbol}...")
        
        if data is None or data.empty:
            print(f"❌ No hay datos para analizar {symbol}")
            return None
        
        latest = data.iloc[-1]
        
        # Análisis de tendencia
        trend_analysis = {
            'price_vs_sma20': 'BULLISH' if latest['Close'] > latest['SMA_20'] else 'BEARISH',
            'price_vs_sma50': 'BULLISH' if latest['Close'] > latest['SMA_50'] else 'BEARISH',
            'price_vs_sma200': 'BULLISH' if latest['Close'] > latest['SMA_200'] else 'BEARISH',
            'sma20_vs_sma50': 'BULLISH' if latest['SMA_20'] > latest['SMA_50'] else 'BEARISH',
            'golden_cross': latest['SMA_50'] > latest['SMA_200'],
            'death_cross': latest['SMA_50'] < latest['SMA_200']\n        }\n        \n        # Análisis de momentum\n        momentum_analysis = {\n            'rsi_level': 'OVERSOLD' if latest['RSI'] < 30 else 'OVERBOUGHT' if latest['RSI'] > 70 else 'NEUTRAL',\n            'rsi_value': latest['RSI'],\n            'macd_signal': 'BULLISH' if latest['MACD'] > latest['MACD_Signal'] else 'BEARISH',\n            'macd_value': latest['MACD'],\n            'bb_position': latest['BB_Position']\n        }\n        \n        # Análisis de volatilidad\n        volatility_analysis = {\n            'current_volatility': latest['Volatility'],\n            'volatility_level': 'HIGH' if latest['Volatility'] > 30 else 'LOW' if latest['Volatility'] < 15 else 'MEDIUM',\n            'atr_value': latest['ATR'],\n            'bb_width': latest['BB_Width']\n        }\n        \n        # Análisis de volumen\n        volume_analysis = {\n            'avg_volume_20d': data['Volume'].rolling(20).mean().iloc[-1],\n            'volume_trend': 'INCREASING' if latest['Volume'] > data['Volume'].rolling(20).mean().iloc[-1] else 'DECREASING',\n            'obv_trend': 'BULLISH' if data['OBV'].iloc[-1] > data['OBV'].iloc[-20] else 'BEARISH'\n        }\n        \n        # Análisis de precios\n        price_analysis = {\n            'current_price': latest['Close'],\n            'price_change_1d': (latest['Close'] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100,\n            'price_change_1w': (latest['Close'] - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100,\n            'price_change_1m': (latest['Close'] - data['Close'].iloc[-20]) / data['Close'].iloc[-20] * 100,\n            'price_change_ytd': (latest['Close'] - data['Close'].iloc[0]) / data['Close'].iloc[0] * 100,\n            'distance_from_52w_high': (latest['Close'] - data['High'].max()) / data['High'].max() * 100,\n            'distance_from_52w_low': (latest['Close'] - data['Low'].min()) / data['Low'].min() * 100\n        }\n        \n        # Señales de trading\n        signals = {\n            'overall_trend': self.determine_overall_trend(trend_analysis),\n            'buy_signals': self.count_buy_signals(trend_analysis, momentum_analysis),\n            'sell_signals': self.count_sell_signals(trend_analysis, momentum_analysis),\n            'recommendation': self.generate_recommendation(trend_analysis, momentum_analysis, volatility_analysis)\n        }\n        \n        analysis = {\n            'symbol': symbol,\n            'timestamp': datetime.now().isoformat(),\n            'trend': trend_analysis,\n            'momentum': momentum_analysis,\n            'volatility': volatility_analysis,\n            'volume': volume_analysis,\n            'price': price_analysis,\n            'signals': signals\n        }\n        \n        print(f\"✅ Análisis completado para {symbol}\")\n        print(f\"   📈 Tendencia general: {signals['overall_trend']}\")\n        print(f\"   📊 Recomendación: {signals['recommendation']}\")\n        print(f\"   💰 Precio actual: ${price_analysis['current_price']:.2f}\")\n        print(f\"   📊 RSI: {momentum_analysis['rsi_value']:.1f}\")\n        print(f\"   📈 Volatilidad: {volatility_analysis['current_volatility']:.1f}%\")\n        \n        return analysis\n    \n    def determine_overall_trend(self, trend_analysis):\n        \"\"\"Determinar tendencia general basada en múltiples indicadores\"\"\"\n        bullish_count = sum([\n            trend_analysis['price_vs_sma20'] == 'BULLISH',\n            trend_analysis['price_vs_sma50'] == 'BULLISH',\n            trend_analysis['price_vs_sma200'] == 'BULLISH',\n            trend_analysis['sma20_vs_sma50'] == 'BULLISH',\n            trend_analysis['golden_cross']\n        ])\n        \n        if bullish_count >= 4:\n            return 'STRONG_BULLISH'\n        elif bullish_count >= 3:\n            return 'BULLISH'\n        elif bullish_count >= 2:\n            return 'NEUTRAL'\n        else:\n            return 'BEARISH'\n    \n    def count_buy_signals(self, trend_analysis, momentum_analysis):\n        \"\"\"Contar señales de compra\"\"\"\n        buy_signals = 0\n        \n        # Señales de tendencia\n        if trend_analysis['price_vs_sma20'] == 'BULLISH':\n            buy_signals += 1\n        if trend_analysis['golden_cross']:\n            buy_signals += 1\n        \n        # Señales de momentum\n        if momentum_analysis['rsi_level'] == 'OVERSOLD':\n            buy_signals += 1\n        if momentum_analysis['macd_signal'] == 'BULLISH':\n            buy_signals += 1\n        \n        return buy_signals\n    \n    def count_sell_signals(self, trend_analysis, momentum_analysis):\n        \"\"\"Contar señales de venta\"\"\"\n        sell_signals = 0\n        \n        # Señales de tendencia\n        if trend_analysis['price_vs_sma20'] == 'BEARISH':\n            sell_signals += 1\n        if trend_analysis['death_cross']:\n            sell_signals += 1\n        \n        # Señales de momentum\n        if momentum_analysis['rsi_level'] == 'OVERBOUGHT':\n            sell_signals += 1\n        if momentum_analysis['macd_signal'] == 'BEARISH':\n            sell_signals += 1\n        \n        return sell_signals\n    \n    def generate_recommendation(self, trend_analysis, momentum_analysis, volatility_analysis):\n        \"\"\"Generar recomendación de trading\"\"\"\n        buy_signals = self.count_buy_signals(trend_analysis, momentum_analysis)\n        sell_signals = self.count_sell_signals(trend_analysis, momentum_analysis)\n        \n        if buy_signals >= 3 and sell_signals <= 1:\n            return 'STRONG_BUY'\n        elif buy_signals >= 2 and sell_signals <= 1:\n            return 'BUY'\n        elif sell_signals >= 3 and buy_signals <= 1:\n            return 'STRONG_SELL'\n        elif sell_signals >= 2 and buy_signals <= 1:\n            return 'SELL'\n        else:\n            return 'HOLD'\n    \n    def create_comprehensive_chart(self, data, symbol, analysis):\n        \"\"\"Crear gráfico completo de análisis técnico\"\"\"\n        print(f\"📊 Creando gráfico completo para {symbol}...\")\n        \n        # Configurar subplots\n        fig, axes = plt.subplots(4, 1, figsize=(15, 20))\n        fig.suptitle(f'📊 Análisis Técnico Completo - {symbol}', fontsize=16, fontweight='bold')\n        \n        # 1. Precio y medias móviles\n        ax1 = axes[0]\n        ax1.plot(data.index, data['Close'], label='Precio', linewidth=2, color='black')\n        ax1.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7, color='blue')\n        ax1.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7, color='red')\n        ax1.plot(data.index, data['SMA_200'], label='SMA 200', alpha=0.7, color='green')\n        \n        # Bollinger Bands\n        ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], alpha=0.2, color='gray')\n        ax1.plot(data.index, data['BB_Upper'], alpha=0.5, color='purple', linestyle='--')\n        ax1.plot(data.index, data['BB_Lower'], alpha=0.5, color='purple', linestyle='--')\n        \n        ax1.set_title(f'💰 Precio y Medias Móviles - ${data[\"Close\"].iloc[-1]:.2f}')\n        ax1.legend()\n        ax1.grid(True, alpha=0.3)\n        \n        # 2. RSI\n        ax2 = axes[1]\n        ax2.plot(data.index, data['RSI'], linewidth=2, color='purple')\n        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)\n        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7)\n        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)\n        ax2.fill_between(data.index, 70, 100, alpha=0.2, color='red')\n        ax2.fill_between(data.index, 0, 30, alpha=0.2, color='green')\n        ax2.set_title(f'📊 RSI - {data[\"RSI\"].iloc[-1]:.1f}')\n        ax2.set_ylim(0, 100)\n        ax2.grid(True, alpha=0.3)\n        \n        # 3. MACD\n        ax3 = axes[2]\n        ax3.plot(data.index, data['MACD'], label='MACD', linewidth=2, color='blue')\n        ax3.plot(data.index, data['MACD_Signal'], label='Signal', linewidth=2, color='red')\n        ax3.bar(data.index, data['MACD_Histogram'], label='Histogram', alpha=0.3, color='green')\n        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)\n        ax3.set_title('📊 MACD')\n        ax3.legend()\n        ax3.grid(True, alpha=0.3)\n        \n        # 4. Volumen\n        ax4 = axes[3]\n        colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' for i in range(len(data))]\n        ax4.bar(data.index, data['Volume'], color=colors, alpha=0.7)\n        ax4.plot(data.index, data['Volume'].rolling(20).mean(), color='blue', linewidth=2, label='Volume MA 20')\n        ax4.set_title('📊 Volumen')\n        ax4.legend()\n        ax4.grid(True, alpha=0.3)\n        \n        plt.tight_layout()\n        plt.savefig(f'/tmp/{symbol}_technical_analysis.png', dpi=300, bbox_inches='tight')\n        plt.show()\n        \n        print(f\"✅ Gráfico guardado como {symbol}_technical_analysis.png\")\n    \n    def generate_report(self, symbol, data, info, analysis):\n        \"\"\"Generar reporte completo\"\"\"\n        print(f\"📋 Generando reporte completo para {symbol}...\")\n        \n        report = f\"\"\"\n🚀 REPORTE DE ANÁLISIS TÉCNICO\n{'=' * 50}\n\n📊 INFORMACIÓN GENERAL\n{'=' * 30}\nSímbolo: {symbol}\nEmpresa: {info['Company_Name'] if info else 'N/A'}\nSector: {info['Sector'] if info else 'N/A'}\nFecha del análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n💰 PRECIOS ACTUALES\n{'=' * 30}\nPrecio actual: ${analysis['price']['current_price']:.2f}\nCambio 1D: {analysis['price']['price_change_1d']:.2f}%\nCambio 1S: {analysis['price']['price_change_1w']:.2f}%\nCambio 1M: {analysis['price']['price_change_1m']:.2f}%\nCambio YTD: {analysis['price']['price_change_ytd']:.2f}%\n\n📈 ANÁLISIS TÉCNICO\n{'=' * 30}\nTendencia general: {analysis['signals']['overall_trend']}\nRecomendación: {analysis['signals']['recommendation']}\nSeñales de compra: {analysis['signals']['buy_signals']}/4\nSeñales de venta: {analysis['signals']['sell_signals']}/4\n\n📊 INDICADORES TÉCNICOS\n{'=' * 30}\nRSI: {analysis['momentum']['rsi_value']:.1f} ({analysis['momentum']['rsi_level']})\nMACD: {analysis['momentum']['macd_value']:.4f}\nVolatilidad: {analysis['volatility']['current_volatility']:.1f}%\nATR: {analysis['volatility']['atr_value']:.2f}\n\n🎯 NIVELES CLAVE\n{'=' * 30}\nSMA 20: ${data['SMA_20'].iloc[-1]:.2f}\nSMA 50: ${data['SMA_50'].iloc[-1]:.2f}\nSMA 200: ${data['SMA_200'].iloc[-1]:.2f}\nSoporte BB: ${data['BB_Lower'].iloc[-1]:.2f}\nResistencia BB: ${data['BB_Upper'].iloc[-1]:.2f}\n\n📊 INFORMACIÓN FUNDAMENTAL\n{'=' * 30}\"\"\"\n        \n        if info:\n            report += f\"\"\"\nMarket Cap: ${info['Market_Cap']:,.0f}\nP/E Ratio: {info['PE_Ratio']:.2f}\nBeta: {info['Beta']:.2f}\nDividend Yield: {info['Dividend_Yield']:.2%}\nROE: {info['ROE']:.2%}\n\"\"\"\n        \n        report += f\"\"\"\n\n⚠️ RIESGOS Y CONSIDERACIONES\n{'=' * 30}\n- Volatilidad actual: {analysis['volatility']['volatility_level']}\n- Nivel de RSI: {analysis['momentum']['rsi_level']}\n- Tendencia de volumen: {analysis['volume']['volume_trend']}\n\n📝 CONCLUSIONES\n{'=' * 30}\nBasado en el análisis técnico actual, la recomendación es: {analysis['signals']['recommendation']}\n\nEste análisis se basa en indicadores técnicos y no constituye asesoramiento financiero.\nSiempre realice su propia investigación antes de tomar decisiones de inversión.\n\n🗓️ Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\"\"\"\n        \n        # Guardar reporte\n        with open(f'/tmp/{symbol}_report.txt', 'w', encoding='utf-8') as f:\n            f.write(report)\n        \n        print(report)\n        print(f\"✅ Reporte guardado como {symbol}_report.txt\")\n        \n        return report\n    \n    def run_complete_analysis(self, symbol=\"AAPL\", period=\"1y\"):\n        \"\"\"Ejecutar análisis completo de una acción\"\"\"\n        print(f\"🚀 Ejecutando análisis completo de {symbol}\")\n        print(\"=\" * 60)\n        \n        # 1. Obtener datos históricos\n        data = self.get_stock_data(symbol, period)\n        if data is None:\n            return None\n        \n        # 2. Obtener información fundamental\n        info = self.get_stock_info(symbol)\n        \n        # 3. Realizar análisis técnico\n        analysis = self.analyze_stock(data, symbol)\n        \n        # 4. Crear visualizaciones\n        self.create_comprehensive_chart(data, symbol, analysis)\n        \n        # 5. Generar reporte\n        report = self.generate_report(symbol, data, info, analysis)\n        \n        # 6. Guardar datos\n        self.save_data(symbol, data, info, analysis)\n        \n        print(f\"\\n✅ Análisis completo de {symbol} terminado!\")\n        print(\"📁 Archivos generados:\")\n        print(f\"   📊 {symbol}_data.csv\")\n        print(f\"   📋 {symbol}_analysis.json\")\n        print(f\"   📈 {symbol}_technical_analysis.png\")\n        print(f\"   📄 {symbol}_report.txt\")\n        \n        return {\n            'symbol': symbol,\n            'data': data,\n            'info': info,\n            'analysis': analysis,\n            'report': report\n        }\n    \n    def save_data(self, symbol, data, info, analysis):\n        \"\"\"Guardar todos los datos del análisis\"\"\"\n        print(f\"💾 Guardando datos de {symbol}...\")\n        \n        # Guardar datos históricos\n        data.to_csv(f'/tmp/{symbol}_data.csv')\n        \n        # Guardar análisis\n        import json\n        analysis_data = {\n            'symbol': symbol,\n            'info': info,\n            'analysis': analysis,\n            'timestamp': datetime.now().isoformat()\n        }\n        \n        with open(f'/tmp/{symbol}_analysis.json', 'w') as f:\n            json.dump(analysis_data, f, indent=2, default=str)\n        \n        print(f\"✅ Datos guardados para {symbol}\")\n\ndef main():\n    \"\"\"Función principal del ejemplo\"\"\"\n    print(\"🎯 Ejemplo Práctico: Yahoo Finance API\")\n    print(\"Análisis técnico completo de acciones\")\n    print(\"=\" * 50)\n    \n    # Crear instancia del ejemplo\n    example = YahooFinanceExample()\n    \n    # Lista de acciones para analizar\n    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']\n    \n    print(f\"📊 Analizando {len(symbols)} acciones...\")\n    \n    results = {}\n    \n    for symbol in symbols:\n        print(f\"\\n🔍 Analizando {symbol}...\")\n        try:\n            result = example.run_complete_analysis(symbol, period=\"6mo\")\n            if result:\n                results[symbol] = result\n                print(f\"✅ {symbol} completado\")\n        except Exception as e:\n            print(f\"❌ Error analizando {symbol}: {str(e)}\")\n    \n    # Resumen final\n    print(\"\\n📊 RESUMEN DE ANÁLISIS\")\n    print(\"=\" * 30)\n    \n    for symbol, result in results.items():\n        if result and 'analysis' in result:\n            analysis = result['analysis']\n            print(f\"📈 {symbol}:\")\n            print(f\"   💰 Precio: ${analysis['price']['current_price']:.2f}\")\n            print(f\"   📊 Recomendación: {analysis['signals']['recommendation']}\")\n            print(f\"   📈 Tendencia: {analysis['signals']['overall_trend']}\")\n            print(f\"   📊 RSI: {analysis['momentum']['rsi_value']:.1f}\")\n    \n    print(f\"\\n🎉 ¡Análisis de {len(results)} acciones completado!\")\n    print(\"💡 Revisa los archivos generados en /tmp/ para más detalles\")\n\nif __name__ == \"__main__\":\n    main()"