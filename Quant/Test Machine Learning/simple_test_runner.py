#!/usr/bin/env python3
"""
Test runner simple para verificar que todo funciona
"""

import os
import sys
import subprocess
from pathlib import Path

def run_test(test_path):
    """Ejecutar un test individual"""
    print(f"\n🧪 Ejecutando: {test_path}")
    try:
        result = subprocess.run([sys.executable, test_path], 
                              capture_output=True, text=True, 
                              timeout=30)
        
        if result.returncode == 0:
            print(f"✅ {test_path} - PASSED")
            return True
        else:
            print(f"❌ {test_path} - FAILED")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_path} - TIMEOUT")
        return False
    except Exception as e:
        print(f"💥 {test_path} - EXCEPTION: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 Test Runner Simple")
    print("=" * 50)
    
    # Lista de tests
    tests = [
        "test_util/test_simple_util.py",
        "test_data_structures/test_simple_data_structures.py"
    ]
    
    # Ejecutar tests
    passed = 0
    total = len(tests)
    
    for test in tests:
        if run_test(test):
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 RESULTADOS: {passed}/{total} tests pasados")
    
    # Generar dashboard
    try:
        from dashboard_simple import SimpleDashboard
        dashboard = SimpleDashboard()
        html_file = dashboard.generate_comprehensive_report()
        print(f"📊 Dashboard: {html_file}")
    except Exception as e:
        print(f"❌ Error en dashboard: {e}")

if __name__ == "__main__":
    main()
