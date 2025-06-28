# 📈 SISTEMA DE ANÁLISIS CUANTITATIVO FINANCIERO

## 🎯 Descripción
Sistema profesional de análisis cuantitativo financiero con módulos optimizados para microestructura de mercado y análisis de datos financieros.

## 🚀 Instalación

### Entorno Conda (Recomendado)
```bash
conda env create -f environment.yml
conda activate quant_env
```

### Pip (Alternativo)
```bash
pip install -r requirements.txt
```

## 📊 Datos Disponibles
- **WTI Crude Oil**: Datos históricos diarios
- **Bitcoin**: Datos de alta frecuencia
- **Gold/Silver**: Datos históricos
- **Índices**: DAX 40, CAC 40, Nasdaq

## 🏗️ Módulos Core

### Data Structures
- **Standard Bars**: Time, Tick, Volume, Dollar bars
- **Imbalance Bars**: En desarrollo
- **Run Bars**: En desarrollo

### Utilities
- **Fast EWMA**: Implementación optimizada con Numba
- **Volatility**: Yang-Zhang, Garman-Klass estimators
- **Volume Classifier**: BVC, Tick Rule, Lee-Ready

## � Ejemplo Básico
```python
from data_structures.standard_data_structures import get_dollar_bars
from util.volatility import calculate_yang_zhang_volatility

# Procesar datos
dollar_bars = get_dollar_bars(data, threshold=1000000)
volatility = calculate_yang_zhang_volatility(data)
```

## 📁 Estructura
```
Quant/
├── Machine Learning/     # Módulos principales
│   ├── data_structures/ # Estructuras financieras
│   └── util/           # Utilidades optimizadas
├── Datos/              # Datasets
├── environment.yml     # Configuración conda
└── requirements.txt    # Dependencias pip
```

---
**🚀 SISTEMA OPTIMIZADO PARA ANÁLISIS CUANTITATIVO**