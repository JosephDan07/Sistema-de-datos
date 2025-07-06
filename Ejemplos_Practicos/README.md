# 📚 Ejemplos Prácticos - Sistema-de-datos

## 🎯 Introducción

Esta carpeta contiene ejemplos prácticos completos que demuestran cómo utilizar todas las capacidades del **Sistema-de-datos**. Cada ejemplo es un script ejecutable que muestra implementaciones reales de las funcionalidades principales.

---

## 📁 Estructura de Ejemplos

```
Ejemplos_Practicos/
├── data_structures/           # Ejemplos de estructuras de datos
│   └── ejemplo_dollar_bars.py
├── labeling/                  # Ejemplos de etiquetado
│   └── ejemplo_triple_barrier.py
├── apis/                      # Ejemplos de APIs
│   └── ejemplo_yahoo_finance.py
├── testing/                   # Ejemplos de testing
│   └── ejemplo_sistema_testing.py
├── analysis/                  # Ejemplos de análisis
│   └── ejemplo_analisis_completo.py
└── README.md                  # Este archivo
```

---

## 🚀 Ejemplos Disponibles

### 1. 📊 **Estructuras de Datos**

#### `data_structures/ejemplo_dollar_bars.py`
**Descripción**: Implementación completa de Dollar Bars siguiendo López de Prado
**Características**:
- Generación de datos sintéticos tick
- Creación de Dollar Bars con threshold configurable
- Análisis de propiedades microestructurales
- Visualizaciones profesionales
- Comparación con time bars tradicionales

**Uso**:
```bash
cd Ejemplos_Practicos/data_structures
python ejemplo_dollar_bars.py
```

**Salidas**:
- `tick_data.csv`: Datos tick sintéticos
- `dollar_bars.csv`: Dollar bars generados
- `dollar_bars_analysis.json`: Análisis detallado
- `dollar_bars_analysis.png`: Visualizaciones

### 2. 🌐 **APIs de Datos**

#### `apis/ejemplo_yahoo_finance.py`
**Descripción**: Análisis técnico completo usando Yahoo Finance API
**Características**:
- Obtención de datos históricos
- Cálculo de 20+ indicadores técnicos
- Análisis fundamental
- Generación de señales de trading
- Reportes profesionales

**Uso**:
```bash
cd Ejemplos_Practicos/apis
python ejemplo_yahoo_finance.py
```

**Salidas**:
- `{SYMBOL}_data.csv`: Datos históricos con indicadores
- `{SYMBOL}_analysis.json`: Análisis completo
- `{SYMBOL}_technical_analysis.png`: Gráficos técnicos
- `{SYMBOL}_report.txt`: Reporte detallado

### 3. 🧪 **Sistema de Testing**

#### `testing/ejemplo_sistema_testing.py`
**Descripción**: Demostración del framework de testing profesional
**Características**:
- Ejecución automatizada de tests
- Dashboard HTML interactivo
- Gráficos de rendimiento
- Reportes detallados
- Métricas de calidad

**Uso**:
```bash
cd Ejemplos_Practicos/testing
python ejemplo_sistema_testing.py
```

**Salidas**:
- `ml_testing_dashboard.html`: Dashboard interactivo
- `testing_performance_charts.png`: Gráficos de rendimiento
- `testing_detailed_report.txt`: Reporte detallado
- `testing_results.json`: Datos completos

---

## 🛠️ Requisitos

### Dependencias Básicas
```bash
pip install pandas numpy matplotlib seaborn yfinance
```

### Dependencias Adicionales
```bash
pip install scikit-learn scipy numba plotly dash
```

### Entorno Recomendado
```bash
# Desde el directorio Quant
conda env create -f environment.yml
conda activate quant_env
```

---

## 🎯 Guía de Uso

### Ejecutar Ejemplo Individual
```bash
# Navegar al directorio del ejemplo
cd Ejemplos_Practicos/data_structures

# Ejecutar ejemplo
python ejemplo_dollar_bars.py

# Ver resultados
ls /tmp/
```

### Ejecutar Todos los Ejemplos
```bash
# Script para ejecutar todos los ejemplos
python ejecutar_todos_ejemplos.py
```

### Personalizar Ejemplos
```python
# Modificar parámetros en el código
example = DollarBarsExample()

# Cambiar configuración
tick_data = example.generate_synthetic_tick_data(n_ticks=100000)  # Más datos
dollar_bars = example.create_dollar_bars(tick_data, threshold=2000000)  # Threshold mayor
```

---

## 📊 Detalles de Implementación

### 1. **Ejemplo Dollar Bars**

#### Algoritmo Implementado
```python
def create_dollar_bars(self, tick_data, threshold=1000000):
    """
    Implementación exacta del algoritmo de López de Prado
    
    Pasos:
    1. Inicializar acumulador de dollar volume
    2. Para cada tick:
       - Actualizar OHLCV de la bar actual
       - Acumular dollar volume
       - Si threshold alcanzado: crear nueva bar
    3. Calcular propiedades microestructurales
    """
```

#### Propiedades Calculadas
- **OHLCV**: Open, High, Low, Close, Volume
- **VWAP**: Volume Weighted Average Price
- **Buy/Sell Volume**: Clasificación usando tick rule
- **Volatility**: Volatilidad rolling de retornos
- **Duration**: Duración de cada bar

#### Métricas de Validación
- Consistency check: Dollar volume ≈ threshold
- Microstructure properties: VWAP, buy volume %
- Statistical properties: Distribución de duraciones

### 2. **Ejemplo Yahoo Finance**

#### Indicadores Implementados
- **Trend**: SMA(20,50,200), EMA(12,26)
- **Momentum**: RSI, MACD
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, Volume MA
- **Others**: VWAP, True Range

#### Sistema de Señales
```python
def generate_recommendation(self, trend, momentum, volatility):
    """
    Sistema de puntuación basado en múltiples indicadores
    
    Señales de compra:
    - Precio > SMA20
    - Golden cross (SMA50 > SMA200)
    - RSI oversold
    - MACD bullish
    
    Señales de venta:
    - Precio < SMA20
    - Death cross (SMA50 < SMA200)
    - RSI overbought
    - MACD bearish
    """
```

### 3. **Ejemplo Sistema Testing**

#### Arquitectura del Framework
```python
class TestingSystemExample:
    """
    Framework de testing profesional
    
    Componentes:
    - Test Runner: Ejecutor de tests
    - Dashboard Generator: Generador de visualizaciones
    - Report Generator: Generador de reportes
    - Data Manager: Gestor de datos
    """
```

#### Métricas Calculadas
- **Success Rate**: Porcentaje de tests pasados
- **Execution Time**: Tiempo total y por módulo
- **Coverage**: Cobertura de código
- **Performance**: Métricas de rendimiento

---

## 🎨 Visualizaciones Generadas

### Dollar Bars
1. **Comparación Precios**: Tick data vs Dollar bars
2. **Distribución Duraciones**: Histograma de duraciones
3. **Volumen por Bar**: Volumen de cada bar
4. **VWAP vs Close**: Comparación de precios
5. **Buy Volume %**: Porcentaje de volumen comprador
6. **Ticks por Bar**: Distribución de ticks

### Yahoo Finance
1. **Precio y Medias**: Precio con SMA y Bollinger Bands
2. **RSI**: Relative Strength Index con niveles
3. **MACD**: MACD con señal e histograma
4. **Volumen**: Volumen con media móvil

### Sistema Testing
1. **Success Rate**: Tasa de éxito por módulo
2. **Execution Time**: Tiempo de ejecución
3. **Test Distribution**: Distribución de tests
4. **Performance Summary**: Resumen de rendimiento

---

## 🔧 Personalización y Extensión

### Agregar Nuevo Ejemplo
```python
# 1. Crear nuevo archivo
touch Ejemplos_Practicos/mi_modulo/mi_ejemplo.py

# 2. Estructura básica
class MiEjemplo:
    def __init__(self):
        self.setup_environment()
    
    def run_example(self):
        # Implementar lógica
        pass
    
    def create_visualizations(self):
        # Crear gráficos
        pass
    
    def generate_report(self):
        # Generar reporte
        pass

# 3. Función main
def main():
    example = MiEjemplo()
    example.run_example()

if __name__ == "__main__":
    main()
```

### Modificar Ejemplos Existentes
```python
# Cambiar parámetros
THRESHOLD = 2000000  # Threshold más alto
N_TICKS = 100000     # Más datos
PERIOD = "2y"        # Período más largo

# Agregar nuevos indicadores
def calculate_custom_indicator(self, data):
    # Implementar nuevo indicador
    pass
```

---

## 📋 Checklist de Ejecución

### Antes de Ejecutar
- [ ] Entorno conda activado
- [ ] Dependencias instaladas
- [ ] Directorio `/tmp/` disponible para resultados
- [ ] Conexión a internet (para APIs)

### Durante la Ejecución
- [ ] Monitorear salida por consola
- [ ] Verificar creación de archivos
- [ ] Revisar gráficos generados
- [ ] Validar datos de salida

### Después de Ejecutar
- [ ] Revisar archivos generados
- [ ] Abrir dashboard HTML
- [ ] Analizar reportes
- [ ] Validar métricas

---

## 🚨 Solución de Problemas

### Problemas Comunes

#### Error: ModuleNotFoundError
```bash
# Solución: Agregar paths
export PYTHONPATH="${PYTHONPATH}:/path/to/Sistema-de-datos/Quant"
```

#### Error: API Key Missing
```python
# Solución: Configurar API keys
API_KEY = "tu_api_key_aqui"
```

#### Error: Matplotlib Backend
```python
# Solución: Configurar backend
import matplotlib
matplotlib.use('Agg')  # Para servidores sin GUI
```

#### Error: Insufficient Data
```python
# Solución: Ajustar parámetros
n_ticks = 10000      # Reducir datos
threshold = 500000   # Reducir threshold
```

---

## 📞 Soporte y Contribuciones

### Reportar Problemas
1. Crear issue en el repositorio
2. Incluir código de error completo
3. Especificar entorno (OS, Python version)
4. Proporcionar pasos para reproducir

### Contribuir Ejemplos
1. Fork del repositorio
2. Crear nueva rama: `git checkout -b nuevo-ejemplo`
3. Agregar ejemplo siguiendo la estructura
4. Crear pull request

### Mejoras Sugeridas
- Agregar más tipos de barras (Run, Imbalance)
- Implementar más APIs (Alpha Vantage, Quandl)
- Agregar ejemplos de Machine Learning
- Crear ejemplos de backtesting

---

## 📚 Referencias y Recursos

### Documentación Técnica
- [README principal](../README.md)
- [Guía práctica](../GUIA_PRACTICA.md)
- [Capacidades completas](../CAPACIDADES_COMPLETAS.md)

### Literatura Académica
- López de Prado, M. (2018). "Advances in Financial Machine Learning"
- Chan, E. (2017). "Machine Trading"
- Jansen, S. (2020). "Machine Learning for Algorithmic Trading"

### Recursos Online
- [Yahoo Finance API](https://pypi.org/project/yfinance/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)

---

## 🎉 Conclusión

Los ejemplos prácticos proporcionan una base sólida para:

✅ **Aprender** las implementaciones del sistema
✅ **Experimentar** con parámetros y configuraciones  
✅ **Validar** resultados y métricas
✅ **Extender** funcionalidades existentes
✅ **Desarrollar** nuevas características

### Próximos Pasos
1. Ejecutar ejemplos básicos
2. Personalizar parámetros
3. Crear propios ejemplos
4. Contribuir mejoras
5. Compartir resultados

---

*Ejemplos prácticos - Sistema-de-datos v1.0*
*Última actualización: Julio 2025*