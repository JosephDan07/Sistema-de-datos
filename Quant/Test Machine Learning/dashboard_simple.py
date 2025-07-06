#!/usr/bin/env python3
"""
Simple Dashboard for ML Testing Results
=====================================

Simplified version of the dashboard for quick testing and visualization.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleDashboard:
    """Simple dashboard for ML testing results"""
    
    def __init__(self, results_path: str = None):
        """Initialize the dashboard"""
        if results_path is None:
            self.results_path = Path(__file__).parent.parent / "Results Machine Learning"
        else:
            self.results_path = Path(results_path)
        
        self.results_path.mkdir(exist_ok=True)
        
        logger.info(f"📊 Simple Dashboard initialized")
        logger.info(f"📁 Results path: {self.results_path}")
        
        # Test results storage
        self.test_results = {}
        
    def scan_results(self) -> Dict[str, Any]:
        """Scan for test results"""
        logger.info("🔍 Scanning for test results...")
        
        results = {}
        
        # Look for JSON result files
        for json_file in self.results_path.glob("*.json"):
            try:
                logger.info(f"  📋 Processing: {json_file.name}")
                
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract module name from filename
                filename = json_file.stem
                if filename.startswith("results_"):
                    module_name = filename.split("_")[1] if len(filename.split("_")) > 1 else "unknown"
                else:
                    module_name = filename
                
                results[module_name] = {
                    'data': data,
                    'file': str(json_file),
                    'timestamp': json_file.stat().st_mtime
                }
                
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
        
        self.test_results = results
        logger.info(f"✅ Found {len(results)} test result files")
        
        return results
    
    def generate_html_report(self) -> str:
        """Generate simple HTML report"""
        logger.info("📄 Generating HTML report...")
        
        # Basic HTML structure
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Testing Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
                .module {{ background: #ffffff; border: 1px solid #ddd; margin: 10px 0; padding: 15px; }}
                .stats {{ display: flex; gap: 20px; margin: 10px 0; }}
                .stat {{ background: #e8f5e8; padding: 10px; border-radius: 5px; }}
                .error {{ color: red; }}
                .success {{ color: green; }}
                .timestamp {{ color: #666; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 ML Testing Dashboard</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Modules: {len(self.test_results)}</p>
            </div>
        """
        
        # Add each module's results
        for module_name, result_info in self.test_results.items():
            data = result_info['data']
            timestamp = datetime.fromtimestamp(result_info['timestamp'])
            
            # Extract basic stats
            total_tests = data.get('total_tests', 0)
            passed_tests = data.get('tests_passed', 0)
            execution_time = data.get('execution_time', 0)
            
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            status_class = "success" if success_rate >= 80 else "error"
            
            html += f"""
            <div class="module">
                <h2>{module_name.replace('_', ' ').title()}</h2>
                <div class="stats">
                    <div class="stat">
                        <strong>Tests:</strong> {passed_tests}/{total_tests}
                    </div>
                    <div class="stat">
                        <strong>Success Rate:</strong> 
                        <span class="{status_class}">{success_rate:.1f}%</span>
                    </div>
                    <div class="stat">
                        <strong>Execution Time:</strong> {execution_time:.2f}s
                    </div>
                </div>
                <div class="timestamp">
                    Last updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        # Save HTML file
        html_file = self.results_path / "dashboard.html"
        
        try:
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html)
            
            logger.info(f"✅ HTML report saved: {html_file}")
            return str(html_file)
            
        except Exception as e:
            logger.error(f"❌ Error saving HTML report: {e}")
            return ""
    
    def generate_summary_json(self) -> str:
        """Generate JSON summary"""
        logger.info("📋 Generating JSON summary...")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_modules': len(self.test_results),
            'modules': {}
        }
        
        for module_name, result_info in self.test_results.items():
            data = result_info['data']
            
            summary['modules'][module_name] = {
                'total_tests': data.get('total_tests', 0),
                'passed_tests': data.get('tests_passed', 0),
                'execution_time': data.get('execution_time', 0),
                'success_rate': (data.get('tests_passed', 0) / data.get('total_tests', 1) * 100) if data.get('total_tests', 0) > 0 else 0,
                'last_updated': datetime.fromtimestamp(result_info['timestamp']).isoformat()
            }
        
        # Save summary
        summary_file = self.results_path / "dashboard_summary.json"
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"✅ JSON summary saved: {summary_file}")
            return str(summary_file)
            
        except Exception as e:
            logger.error(f"❌ Error saving JSON summary: {e}")
            return ""
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive report"""
        logger.info("📊 Generating comprehensive report...")
        
        # Scan results first
        self.scan_results()
        
        # Generate HTML report
        html_file = self.generate_html_report()
        
        # Generate JSON summary
        json_file = self.generate_summary_json()
        
        logger.info("✅ Comprehensive report generated")
        return html_file
    
    def generate_html_report(self) -> str:
        """Generate HTML report"""
        logger.info("🔧 Generating HTML report...")
        
        # Get test results
        results = self.scan_results()
        
        # Calculate overall stats
        total_tests = sum(r.get('total_tests', 0) for r in results.values())
        passed_tests = sum(r.get('tests_passed', 0) for r in results.values())
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Generate simple HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Testing Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
                .stat {{ background: #e8f4f8; padding: 15px; border-radius: 8px; flex: 1; }}
                .modules {{ margin-top: 20px; }}
                .module {{ background: white; border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 8px; }}
                .success {{ color: green; }}
                .error {{ color: red; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 ML Testing Dashboard</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="stats">
                <div class="stat">
                    <h3>Total Tests</h3>
                    <p>{total_tests}</p>
                </div>
                <div class="stat">
                    <h3>Passed Tests</h3>
                    <p>{passed_tests}</p>
                </div>
                <div class="stat">
                    <h3>Success Rate</h3>
                    <p>{success_rate:.1f}%</p>
                </div>
            </div>
            
            <div class="modules">
                <h2>Module Results</h2>
                {self._generate_modules_html(results)}
            </div>
        </body>
        </html>
        """
        
        # Save HTML file
        html_file = self.results_path / "ml_testing_dashboard.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ HTML report saved: {html_file}")
        return str(html_file)
    
    def _generate_modules_html(self, results: Dict[str, Any]) -> str:
        """Generate HTML for modules"""
        if not results:
            return "<p>No test results found</p>"
        
        html = ""
        for module_name, module_results in results.items():
            total = module_results.get('total_tests', 0)
            passed = module_results.get('tests_passed', 0)
            status = "success" if passed == total else "error"
            
            html += f"""
            <div class="module">
                <h3>{module_name.replace('_', ' ').title()}</h3>
                <p class="{status}">Tests: {passed}/{total}</p>
                <p>Time: {module_results.get('execution_time', 0):.1f}s</p>
            </div>
            """
        
        return html
    
    def generate_summary_json(self) -> str:
        """Generate JSON summary"""
        results = self.scan_results()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_modules': len(results),
            'results': results
        }
        
        json_file = self.results_path / "dashboard_summary.json"
        with open(json_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(json_file)

def main():
    """Test the simple dashboard"""
    print("🧪 Testing Simple Dashboard")
    
    dashboard = SimpleDashboard()
    
    # Create sample test results for testing
    sample_results = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': 10,
        'tests_passed': 8,
        'execution_time': 45.6,
        'modules_tested': ['data_structures', 'util', 'labeling', 'multi_product']
    }
    
    # Save sample results
    sample_file = dashboard.results_path / "results_sample_test.json"
    with open(sample_file, 'w') as f:
        json.dump(sample_results, f, indent=2)
    
    # Generate report
    html_file = dashboard.generate_comprehensive_report()
    
    print(f"✅ Dashboard report generated: {html_file}")
    
    return html_file

if __name__ == "__main__":
    main()
