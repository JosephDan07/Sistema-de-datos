#!/usr/bin/env python3
"""
Quick Test for System
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test the import
try:
    from test_config_manager import ConfigurationManager
    print("✅ ConfigurationManager imported successfully")
    
    # Test dashboard
    from dashboard_simple import SimpleDashboard
    print("✅ SimpleDashboard imported successfully")
    
    # Create instances
    config_manager = ConfigurationManager()
    dashboard = SimpleDashboard()
    
    print("✅ All imports and initialization successful")
    
    # Generate dashboard
    html_file = dashboard.generate_comprehensive_report()
    print(f"✅ Dashboard generated: {html_file}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
