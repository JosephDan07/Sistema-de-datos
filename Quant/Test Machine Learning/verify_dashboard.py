#!/usr/bin/env python3
"""
Verificador del dashboard
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dashboard_simple import SimpleDashboard
    
    print("📊 Verificando Dashboard...")
    
    # Initialize dashboard
    dashboard = SimpleDashboard()
    
    # Generate report
    html_file = dashboard.generate_comprehensive_report()
    
    print(f"\n📊 RESULTADOS DEL DASHBOARD:")
    print("=" * 40)
    print(f"Dashboard generado: ✅ {html_file}")
    
    # Check if file exists
    if Path(html_file).exists():
        file_size = Path(html_file).stat().st_size
        print(f"Tamaño del archivo: {file_size} bytes")
        print(f"Ubicación: {html_file}")
        
        # Read first few lines to verify content
        with open(html_file, 'r', encoding='utf-8') as f:
            first_lines = f.read(500)
            if "ML Testing Dashboard" in first_lines:
                print("✅ Contenido del dashboard verificado")
            else:
                print("❌ Contenido del dashboard incorrecto")
    else:
        print("❌ Archivo del dashboard no encontrado")
    
    print("\n✅ Verificación del dashboard completada!")
    
except Exception as e:
    print(f"❌ Error verificando dashboard: {e}")
    import traceback
    traceback.print_exc()
