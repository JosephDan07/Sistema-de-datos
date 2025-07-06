#!/usr/bin/env python3
"""
Test script to verify dashboard functionality
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dashboard():
    """Test the dashboard functionality"""
    
    print("🧪 Testing Dashboard System")
    print("=" * 50)
    
    try:
        # Test imports
        from dashboard_simple import SimpleDashboard
        print("✅ Dashboard import successful")
        
        # Initialize dashboard
        dashboard = SimpleDashboard()
        print("✅ Dashboard initialized")
        
        # Create sample test results
        sample_results = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': 10,
            'tests_passed': 8,
            'execution_time': 45.6,
            'modules_tested': ['data_structures', 'util', 'labeling', 'multi_product']
        }
        
        # Save sample results
        sample_file = dashboard.results_path / 'results_data_structures_test.json'
        with open(sample_file, 'w') as f:
            json.dump(sample_results, f, indent=2)
        
        print(f"✅ Sample file created: {sample_file}")
        
        # Generate report
        html_file = dashboard.generate_comprehensive_report()
        
        if html_file and Path(html_file).exists():
            print(f"✅ Dashboard report generated: {html_file}")
            print("✅ Dashboard test PASSED")
            return True
        else:
            print("❌ Dashboard report generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Dashboard test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_master_runner():
    """Test the master test runner"""
    
    print("\n🧪 Testing Master Test Runner")
    print("=" * 50)
    
    try:
        # Test imports
        from master_test_runner import MasterTestRunner
        print("✅ Master test runner import successful")
        
        # Initialize runner
        runner = MasterTestRunner()
        print("✅ Master test runner initialized")
        
        # Test dashboard generation only
        success = runner.generate_dashboard()
        
        if success:
            print("✅ Master test runner dashboard test PASSED")
            return True
        else:
            print("❌ Master test runner dashboard test FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Master test runner test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 System Testing Suite")
    print("=" * 50)
    
    # Test dashboard
    dashboard_success = test_dashboard()
    
    # Test master runner
    runner_success = test_master_runner()
    
    print("\n" + "=" * 50)
    print("📊 FINAL TEST RESULTS")
    print("=" * 50)
    print(f"Dashboard: {'✅ PASS' if dashboard_success else '❌ FAIL'}")
    print(f"Master Runner: {'✅ PASS' if runner_success else '❌ FAIL'}")
    
    overall_success = dashboard_success and runner_success
    print(f"Overall: {'✅ PASS' if overall_success else '❌ FAIL'}")
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)
