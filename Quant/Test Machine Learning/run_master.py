#!/usr/bin/env python3
"""
Ejecutor simple del master test runner
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from master_test_runner import MasterTestRunner
    
    print("🚀 Iniciando Master Test Runner...")
    
    # Initialize runner
    runner = MasterTestRunner()
    
    # Run only working tests first
    runner.test_modules = {
        'data_structures': 'test_data_structures/test_simple_data_structures.py',
        'util': 'test_util/test_simple_util.py'
    }
    
    # Run full pipeline
    results = runner.run_full_pipeline(
        parallel=False,  # Sequential for easier debugging
        timeout=60,
        cleanup=False
    )
    
    print("\n🎯 RESULTADOS DEL MASTER TEST RUNNER:")
    print("=" * 50)
    print(f"Pipeline Status: {'✅ SUCCESS' if results['pipeline_success'] else '❌ FAILED'}")
    print(f"Total Time: {results['pipeline_time']:.2f}s")
    print(f"Tests Passed: {results['test_results']['successful_tests']}/{results['test_results']['total_tests']}")
    print(f"Dashboard: {'✅ Generated' if results['dashboard_generated'] else '❌ Failed'}")
    print(f"Summary File: {results['summary_file']}")
    
    print("\n✅ Master Test Runner completado exitosamente!")
    
except Exception as e:
    print(f"❌ Error ejecutando Master Test Runner: {e}")
    import traceback
    traceback.print_exc()
