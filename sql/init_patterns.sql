-- Esquema de base de datos para almacenar patrones identificados
CREATE TABLE IF NOT EXISTS detected_patterns (
    id SERIAL PRIMARY KEY,
    pattern_type VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(20) NOT NULL,
    detection_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    pattern_data JSONB,
    confidence_score DECIMAL(5,4),
    status VARCHAR(20) DEFAULT 'active',
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS pattern_performance (
    id SERIAL PRIMARY KEY,
    pattern_id INTEGER REFERENCES detected_patterns(id),
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    actual_outcome DECIMAL(10,6),
    predicted_outcome DECIMAL(10,6),
    success_rate DECIMAL(5,4),
    notes TEXT
);

-- Índices para optimizar consultas de patrones
CREATE INDEX IF NOT EXISTS idx_patterns_symbol_date ON detected_patterns(symbol, detection_date);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON detected_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_performance_pattern ON pattern_performance(pattern_id);
