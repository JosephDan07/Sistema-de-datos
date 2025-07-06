#!/usr/bin/env python3
"""
Ejemplo Práctico: Creación de Dollar Bars
========================================

Este ejemplo demuestra cómo crear Dollar Bars usando el sistema implementado.
Los Dollar Bars son estructuras de datos que agregan transacciones basadas
en el volumen en dólares acumulado, siguiendo las especificaciones de 
López de Prado en "Advances in Financial Machine Learning".

Características:
- Implementación completa de Dollar Bars
- Generación de datos sintéticos para pruebas
- Análisis de propiedades microestructurales
- Visualización de resultados
- Comparación con time bars tradicionales

Autor: Sistema-de-datos
Fecha: Julio 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os

# Agregar path para imports
sys.path.append('/home/runner/work/Sistema-de-datos/Sistema-de-datos/Quant')

class DollarBarsExample:
    """
    Ejemplo completo de implementación de Dollar Bars
    """
    
    def __init__(self):
        self.setup_plotting()
        print("🚀 Inicializando ejemplo de Dollar Bars")
        
    def setup_plotting(self):
        """Configurar estilo de gráficos"""
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
        sns.set_palette("husl")
        
    def generate_synthetic_tick_data(self, n_ticks=50000):
        """
        Generar datos sintéticos de tick para demostración
        
        Args:
            n_ticks: Número de ticks a generar
            
        Returns:
            DataFrame con datos tick sintéticos
        """
        print(f"📊 Generando {n_ticks:,} ticks sintéticos...")
        
        # Configurar parámetros
        np.random.seed(42)
        start_time = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
        
        # Generar timestamps (mercado abierto: 9:30 AM - 4:00 PM)
        timestamps = []
        current_time = start_time
        
        for i in range(n_ticks):
            # Simular llegada de ticks (más frecuente durante horas activas)
            if 9.5 <= current_time.hour + current_time.minute/60 <= 16:
                # Horas activas: 1-5 segundos entre ticks
                seconds_increment = np.random.exponential(2)
            else:
                # Horas menos activas: 5-15 segundos entre ticks
                seconds_increment = np.random.exponential(8)
            
            current_time += timedelta(seconds=seconds_increment)
            timestamps.append(current_time)
        
        # Generar precios con random walk + trend + volatility clustering
        price_returns = np.random.normal(0.0001, 0.002, n_ticks)  # Micro retornos
        
        # Agregar volatility clustering
        volatility = np.random.gamma(2, 0.001, n_ticks)  # Volatilidad variable
        price_returns *= volatility
        
        # Agregar trend ocasional
        trend_periods = np.random.choice([0, 1, -1], n_ticks, p=[0.7, 0.15, 0.15])
        price_returns += trend_periods * 0.0005
        
        # Calcular precios
        initial_price = 100.0
        prices = initial_price * np.exp(np.cumsum(price_returns))
        
        # Generar volúmenes realistas
        base_volume = np.random.lognormal(5, 1.5, n_ticks).astype(int)  # Volumen base
        
        # Volumen correlacionado con volatilidad
        volume_factor = 1 + 2 * volatility / volatility.mean()
        volumes = (base_volume * volume_factor).astype(int)
        volumes = np.clip(volumes, 1, 100000)  # Limitar volumen
        
        # Crear DataFrame
        tick_data = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'volume': volumes
        })
        
        # Calcular dollar volume
        tick_data['dollar_volume'] = tick_data['price'] * tick_data['volume']
        
        # Agregar bid/ask spread simulado
        spread = np.random.uniform(0.01, 0.05, n_ticks)
        tick_data['bid'] = tick_data['price'] - spread/2
        tick_data['ask'] = tick_data['price'] + spread/2
        
        print(f"✅ Datos generados exitosamente")
        print(f"   📈 Rango de precios: ${tick_data['price'].min():.2f} - ${tick_data['price'].max():.2f}")
        print(f"   📊 Volumen total: {tick_data['volume'].sum():,} shares")
        print(f"   💰 Dollar volume total: ${tick_data['dollar_volume'].sum():,.0f}")
        
        return tick_data
    
    def create_dollar_bars(self, tick_data, threshold=1000000):
        """
        Crear Dollar Bars implementando el algoritmo de López de Prado
        
        Args:
            tick_data: DataFrame con datos tick
            threshold: Threshold de dollar volume para crear nueva bar
            
        Returns:
            DataFrame con Dollar Bars
        """
        print(f"⚙️ Creando Dollar Bars con threshold ${threshold:,}...")
        
        bars = []
        current_bar = {
            'timestamp_start': None,
            'timestamp_end': None,
            'open': None,
            'high': None,
            'low': None,
            'close': None,
            'volume': 0,
            'dollar_volume': 0,
            'vwap': 0,
            'tick_count': 0,
            'buy_volume': 0,
            'sell_volume': 0
        }
        
        accumulated_dollar_volume = 0
        
        for idx, row in tick_data.iterrows():
            # Inicializar nueva bar si es necesario
            if current_bar['timestamp_start'] is None:
                current_bar['timestamp_start'] = row['timestamp']
                current_bar['open'] = row['price']
                current_bar['high'] = row['price']
                current_bar['low'] = row['price']
            
            # Actualizar bar actual
            current_bar['timestamp_end'] = row['timestamp']
            current_bar['close'] = row['price']
            current_bar['high'] = max(current_bar['high'], row['price'])
            current_bar['low'] = min(current_bar['low'], row['price'])
            current_bar['volume'] += row['volume']
            current_bar['dollar_volume'] += row['dollar_volume']
            current_bar['tick_count'] += 1
            
            # Clasificar como buy/sell usando tick rule simplificado
            if len(bars) > 0:
                if row['price'] > bars[-1]['close']:
                    current_bar['buy_volume'] += row['volume']
                else:
                    current_bar['sell_volume'] += row['volume']
            else:
                current_bar['buy_volume'] += row['volume'] * 0.5
                current_bar['sell_volume'] += row['volume'] * 0.5
            
            accumulated_dollar_volume += row['dollar_volume']
            
            # Crear nueva bar si se alcanza el threshold
            if accumulated_dollar_volume >= threshold:
                # Calcular VWAP
                current_bar['vwap'] = current_bar['dollar_volume'] / current_bar['volume']
                
                # Calcular buy volume percentage
                total_volume = current_bar['buy_volume'] + current_bar['sell_volume']
                current_bar['buy_volume_percent'] = (current_bar['buy_volume'] / total_volume * 100) if total_volume > 0 else 50
                
                # Calcular duración de la bar
                duration = current_bar['timestamp_end'] - current_bar['timestamp_start']
                current_bar['duration_seconds'] = duration.total_seconds()
                
                # Agregar bar a la lista
                bars.append(current_bar.copy())
                
                # Reset para nueva bar
                current_bar = {
                    'timestamp_start': None,
                    'timestamp_end': None,
                    'open': None,
                    'high': None,
                    'low': None,
                    'close': None,
                    'volume': 0,
                    'dollar_volume': 0,
                    'vwap': 0,
                    'tick_count': 0,
                    'buy_volume': 0,
                    'sell_volume': 0
                }
                accumulated_dollar_volume = 0
        
        # Agregar última bar si tiene datos
        if current_bar['timestamp_start'] is not None:
            if current_bar['volume'] > 0:
                current_bar['vwap'] = current_bar['dollar_volume'] / current_bar['volume']
                total_volume = current_bar['buy_volume'] + current_bar['sell_volume']
                current_bar['buy_volume_percent'] = (current_bar['buy_volume'] / total_volume * 100) if total_volume > 0 else 50
                duration = current_bar['timestamp_end'] - current_bar['timestamp_start']
                current_bar['duration_seconds'] = duration.total_seconds()
                bars.append(current_bar)
        
        # Convertir a DataFrame
        dollar_bars = pd.DataFrame(bars)
        
        # Calcular métricas adicionales
        if not dollar_bars.empty:
            dollar_bars['returns'] = dollar_bars['close'].pct_change()
            dollar_bars['volatility'] = dollar_bars['returns'].rolling(window=20).std()
            dollar_bars['dollar_volume_millions'] = dollar_bars['dollar_volume'] / 1000000
        
        print(f"✅ Dollar Bars creados exitosamente")
        print(f"   📊 Número de bars: {len(dollar_bars)}")
        print(f"   ⏱️ Duración promedio: {dollar_bars['duration_seconds'].mean():.1f} segundos")
        print(f"   📈 Ticks promedio por bar: {dollar_bars['tick_count'].mean():.1f}")
        print(f"   💰 VWAP promedio: ${dollar_bars['vwap'].mean():.2f}")
        
        return dollar_bars
    
    def analyze_dollar_bars(self, dollar_bars):
        """
        Analizar propiedades de los Dollar Bars
        
        Args:
            dollar_bars: DataFrame con Dollar Bars
            
        Returns:
            Dict con análisis detallado
        """
        print("🔍 Analizando propiedades de Dollar Bars...")
        
        analysis = {
            'basic_stats': {
                'num_bars': len(dollar_bars),
                'avg_duration': dollar_bars['duration_seconds'].mean(),
                'std_duration': dollar_bars['duration_seconds'].std(),
                'avg_ticks_per_bar': dollar_bars['tick_count'].mean(),
                'avg_volume_per_bar': dollar_bars['volume'].mean(),
                'avg_dollar_volume': dollar_bars['dollar_volume'].mean()
            },
            'price_stats': {
                'avg_vwap': dollar_bars['vwap'].mean(),
                'vwap_std': dollar_bars['vwap'].std(),
                'price_range': dollar_bars['close'].max() - dollar_bars['close'].min(),
                'avg_return': dollar_bars['returns'].mean(),
                'return_volatility': dollar_bars['returns'].std(),
                'avg_volatility': dollar_bars['volatility'].mean()
            },
            'microstructure_stats': {
                'avg_buy_volume_percent': dollar_bars['buy_volume_percent'].mean(),
                'buy_volume_std': dollar_bars['buy_volume_percent'].std(),
                'imbalance_measure': abs(dollar_bars['buy_volume_percent'] - 50).mean()
            }
        }
        
        print("📊 Resultados del análisis:")
        print(f"   📈 Número de bars: {analysis['basic_stats']['num_bars']}")
        print(f"   ⏱️ Duración promedio: {analysis['basic_stats']['avg_duration']:.1f}s")
        print(f"   📊 Ticks promedio por bar: {analysis['basic_stats']['avg_ticks_per_bar']:.1f}")
        print(f"   💰 VWAP promedio: ${analysis['price_stats']['avg_vwap']:.2f}")
        print(f"   📈 Volatilidad promedio: {analysis['price_stats']['return_volatility']:.4f}")
        print(f"   ⚖️ Buy volume promedio: {analysis['microstructure_stats']['avg_buy_volume_percent']:.1f}%")
        
        return analysis
    
    def create_visualizations(self, tick_data, dollar_bars):
        """
        Crear visualizaciones para análisis
        
        Args:
            tick_data: DataFrame con datos tick
            dollar_bars: DataFrame con Dollar Bars
        """
        print("📊 Creando visualizaciones...")
        
        # Configurar subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('📊 Análisis de Dollar Bars vs Tick Data', fontsize=16, fontweight='bold')
        
        # 1. Comparación de precios
        ax1 = axes[0, 0]
        ax1.plot(tick_data['timestamp'], tick_data['price'], alpha=0.7, linewidth=0.5, 
                label='Tick Data', color='lightblue')
        ax1.plot(dollar_bars['timestamp_end'], dollar_bars['close'], 
                linewidth=2, label='Dollar Bars', color='darkblue')
        ax1.set_title('💰 Precios: Tick Data vs Dollar Bars')
        ax1.set_ylabel('Precio ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribución de duraciones
        ax2 = axes[0, 1]
        ax2.hist(dollar_bars['duration_seconds'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('⏱️ Distribución de Duraciones de Dollar Bars')
        ax2.set_xlabel('Duración (segundos)')
        ax2.set_ylabel('Frecuencia')
        ax2.axvline(dollar_bars['duration_seconds'].mean(), color='red', linestyle='--', 
                   label=f'Media: {dollar_bars["duration_seconds"].mean():.1f}s')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Volumen por bar
        ax3 = axes[0, 2]
        ax3.bar(range(len(dollar_bars)), dollar_bars['volume'], alpha=0.7, color='orange')
        ax3.set_title('📊 Volumen por Dollar Bar')
        ax3.set_xlabel('Bar Index')
        ax3.set_ylabel('Volumen')
        ax3.grid(True, alpha=0.3)
        
        # 4. VWAP vs Close
        ax4 = axes[1, 0]
        ax4.plot(dollar_bars['timestamp_end'], dollar_bars['close'], 
                label='Close', linewidth=2, color='blue')
        ax4.plot(dollar_bars['timestamp_end'], dollar_bars['vwap'], 
                label='VWAP', linewidth=2, color='red', linestyle='--')
        ax4.set_title('📈 Close vs VWAP')
        ax4.set_ylabel('Precio ($)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Buy Volume Percentage
        ax5 = axes[1, 1]
        ax5.plot(dollar_bars['timestamp_end'], dollar_bars['buy_volume_percent'], 
                linewidth=2, color='purple')
        ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.7, label='Balance (50%)')
        ax5.set_title('⚖️ Buy Volume Percentage')
        ax5.set_ylabel('Buy Volume (%)')
        ax5.set_ylim(0, 100)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Ticks por bar
        ax6 = axes[1, 2]
        ax6.bar(range(len(dollar_bars)), dollar_bars['tick_count'], alpha=0.7, color='brown')
        ax6.set_title('📊 Ticks por Dollar Bar')
        ax6.set_xlabel('Bar Index')
        ax6.set_ylabel('Número de Ticks')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/dollar_bars_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualizaciones creadas y guardadas en /tmp/dollar_bars_analysis.png")
    
    def run_complete_example(self):
        """
        Ejecutar ejemplo completo de Dollar Bars
        """
        print("🚀 Ejecutando ejemplo completo de Dollar Bars")
        print("=" * 60)
        
        # 1. Generar datos sintéticos
        tick_data = self.generate_synthetic_tick_data(n_ticks=50000)
        
        # 2. Crear Dollar Bars
        dollar_bars = self.create_dollar_bars(tick_data, threshold=1000000)
        
        # 3. Analizar resultados
        analysis = self.analyze_dollar_bars(dollar_bars)
        
        # 4. Crear visualizaciones
        self.create_visualizations(tick_data, dollar_bars)
        
        # 5. Guardar resultados
        self.save_results(tick_data, dollar_bars, analysis)
        
        print("\n✅ Ejemplo completado exitosamente!")
        print("📁 Resultados guardados en /tmp/")
        
        return tick_data, dollar_bars, analysis
    
    def save_results(self, tick_data, dollar_bars, analysis):
        """
        Guardar resultados del análisis
        
        Args:
            tick_data: DataFrame con datos tick
            dollar_bars: DataFrame con Dollar Bars
            analysis: Dict con análisis
        """
        print("💾 Guardando resultados...")
        
        # Guardar DataFrames
        tick_data.to_csv('/tmp/tick_data.csv', index=False)
        dollar_bars.to_csv('/tmp/dollar_bars.csv', index=False)
        
        # Guardar análisis
        import json
        with open('/tmp/dollar_bars_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print("✅ Resultados guardados:")
        print("   📊 tick_data.csv")
        print("   💰 dollar_bars.csv") 
        print("   📋 dollar_bars_analysis.json")
        print("   📈 dollar_bars_analysis.png")

def main():
    """
    Función principal del ejemplo
    """
    print("🎯 Ejemplo Práctico: Dollar Bars")
    print("Implementación basada en López de Prado")
    print("=" * 50)
    
    # Crear instancia del ejemplo
    example = DollarBarsExample()
    
    # Ejecutar ejemplo completo
    tick_data, dollar_bars, analysis = example.run_complete_example()
    
    # Mostrar resumen final
    print("\n📊 RESUMEN FINAL:")
    print(f"   📈 Ticks procesados: {len(tick_data):,}")
    print(f"   💰 Dollar Bars creados: {len(dollar_bars):,}")
    print(f"   ⏱️ Duración promedio por bar: {analysis['basic_stats']['avg_duration']:.1f}s")
    print(f"   📊 Ticks promedio por bar: {analysis['basic_stats']['avg_ticks_per_bar']:.1f}")
    print(f"   💵 VWAP promedio: ${analysis['price_stats']['avg_vwap']:.2f}")
    print(f"   📈 Volatilidad: {analysis['price_stats']['return_volatility']:.4f}")
    
    print("\n🎉 ¡Ejemplo completado exitosamente!")
    print("💡 Tip: Revisa los archivos generados en /tmp/ para análisis adicional")

if __name__ == "__main__":
    main()