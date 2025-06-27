# Docker para Sistema de Identificación de Patrones en Datos Financieros
FROM python:3.11-slim

# Metadata
LABEL maintainer="Sistema-de-datos"
LABEL description="Contenedor para análisis cuantitativo e identificación de patrones"

# Variables de entorno para optimización
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV JUPYTER_ENABLE_LAB=yes

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para análisis de datos
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt primero (para cache de Docker)
COPY Quant/requirements.txt .

# Instalar dependencias Python especializadas para patrones
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Librerías adicionales específicas para identificación de patrones
RUN pip install --no-cache-dir \
    # Análisis de series temporales y patrones
    ta-lib \
    arch \
    prophet \
    stumpy \
    pykalman \
    # Análisis técnico financiero
    yfinance \
    alpha-vantage \
    ccxt \
    backtrader \
    # Machine Learning avanzado para patrones
    xgboost \
    lightgbm \
    catboost \
    optuna \
    # Visualización avanzada de patrones
    plotly \
    bokeh \
    altair \
    seaborn \
    # Procesamiento de datos masivos
    dask \
    vaex \
    # Análisis de redes y grafos (para patrones complejos)
    networkx \
    node2vec \
    # Deep Learning para patrones temporales
    torch \
    torchvision \
    keras \
    # Análisis de anomalías y detección de patrones
    pyod \
    isolation-forest \
    # Notebooks optimizados
    jupyterlab \
    ipywidgets \
    jupyter-dash

# Copiar el código del proyecto completo
COPY Quant/ ./Quant/
COPY sql/ ./sql/
COPY monitoring/ ./monitoring/
COPY README-Docker.md ./

# Configurar Jupyter para patrones
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py

# Exponer puertos
EXPOSE 8888 8050

# Crear directorios para datos y resultados
RUN mkdir -p /app/data /app/results /app/models /app/logs

# Establecer el PYTHONPATH correctamente
ENV PYTHONPATH=/app/Quant:/app

# Comando por defecto - JupyterLab para análisis interactivo
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
