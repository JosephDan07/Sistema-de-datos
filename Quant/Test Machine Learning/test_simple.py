#!/usr/bin/env python3
"""
Simple Test Script
"""
import sys
import os
from pathlib import Path

print("🧪 Testing ML System")
print("=" * 40)

# Check current directory
print(f"Current directory: {os.getcwd()}")

# Check if files exist
files_to_check = [
    "dashboard_simple.py",
    "master_test_runner.py",
    "test_data_structures/test_complete_data_structures.py",
    "test_util/test_complete_util.py",
    "test_labeling/test_complete_labeling.py",
    "test_multi_product/test_complete_multi_product.py"
]

print("\nChecking files:")
for file in files_to_check:
    exists = Path(file).exists()
    print(f"  {'✅' if exists else '❌'} {file}")

# Test dashboard
print("\n🧪 Testing Dashboard...")
try:
    from dashboard_simple import SimpleDashboard
    dashboard = SimpleDashboard()
    html_file = dashboard.generate_comprehensive_report()
    print(f"✅ Dashboard generated: {html_file}")
except Exception as e:
    print(f"❌ Dashboard error: {e}")

print("\n✅ Test completed")
