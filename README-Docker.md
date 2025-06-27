# 🐳 Sistema de Datos - Docker Setup

## 🚀 Inicio Rápido

```bash
# Construir y levantar todos los servicios
docker-compose up --build

# Solo el analizador de patrones
docker-compose up pattern-analyzer

# En segundo plano
docker-compose up -d
```

## 📊 Servicios Disponibles

### 🔍 **Pattern Analyzer (Principal)**
- **URL**: http://localhost:8888
- **Token**: `sistema-datos-2025`
- **Descripción**: JupyterLab con todas las librerías para análisis de patrones

### 🗄️ **Base de Datos PostgreSQL**
- **Host**: localhost:5432
- **DB**: patterns_db
- **User**: patterns_user
- **Password**: patterns_secure_2025

### ⚡ **Redis Cache**
- **URL**: localhost:6379
- **Uso**: Cache de datos y resultados de patrones

### 📈 **Prometheus Monitoring**
- **URL**: http://localhost:9090
- **Uso**: Monitoreo del sistema

## 🛠️ Comandos Útiles

```bash
# Ver logs
docker-compose logs pattern-analyzer

# Ejecutar comandos en el contenedor
docker-compose exec pattern-analyzer python main.py

# Parar servicios
docker-compose down

# Limpiar todo
docker-compose down -v --rmi all
```

## 📂 Estructura de Volúmenes

```
./Quant/          → /app/Quant/     (código fuente principal)
./sql/            → /app/sql/       (scripts de base de datos)
./monitoring/     → /app/monitoring/ (configuración de monitoreo)
./data/           → /app/data/      (datasets)
./results/        → /app/results/   (resultados de análisis)
./models/         → /app/models/    (modelos ML entrenados)
./logs/           → /app/logs/      (logs del sistema)
```

## 🔧 Librerías Incluidas

### 📈 **Análisis de Patrones**
- `ta-lib` - Análisis técnico
- `stumpy` - Detección de motifs
- `prophet` - Forecasting
- `arch` - Modelos econométricos

### 🤖 **Machine Learning**
- `xgboost`, `lightgbm`, `catboost`
- `optuna` - Optimización de hiperparámetros
- `pyod` - Detección de anomalías

### 📊 **Visualización**
- `plotly`, `bokeh`, `altair`
- `dash` - Apps interactivas

### ⚡ **Big Data**
- `dask` - Computación paralela
- `vaex` - Exploración rápida de datos

## 🎯 Casos de Uso

1. **Análisis de Series Temporales**
2. **Detección de Patrones Técnicos**
3. **Identificación de Anomalías**
4. **Machine Learning Financiero**
5. **Backtesting de Estrategias**
