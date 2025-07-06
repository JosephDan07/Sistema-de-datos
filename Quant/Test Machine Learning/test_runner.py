#!/usr/bin/env python3
"""
Test Runner for Advanced Machine Learning Testing Suite
======================================================

This script orchestrates the execution of all advanced tests across the ML modules.
It generates comprehensive reports, dashboards, and visualizations.

Features:
- Automated test discovery and execution
- Performance benchmarking across modules
- Statistical analysis and validation
- Interactive dashboards and reports
- Stress testing and edge case validation
- Comparative analysis between modules

Author: Advanced ML Finance Team
Date: July 2025
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
import multiprocessing as mp
from pathlib import Path

# Core libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import plot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedTestRunner:
    """
    Advanced test runner for comprehensive ML module testing
    """
    
    def __init__(self, 
                 test_dir: str = "./",
                 output_dir: str = "./test_results",
                 parallel: bool = True,
                 generate_reports: bool = True):
        """
        Initialize the test runner
        
        :param test_dir: Directory containing test modules
        :param output_dir: Directory for test results and reports
        :param parallel: Whether to run tests in parallel
        :param generate_reports: Whether to generate comprehensive reports
        """
        self.test_dir = Path(test_dir)
        self.output_dir = Path(output_dir)
        self.parallel = parallel
        self.generate_reports = generate_reports
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Test results storage
        self.test_results = {}
        self.module_reports = {}
        self.performance_summary = {}
        self.error_log = []
        
        # Test modules to run
        self.test_modules = self._discover_test_modules()
        
        logger.info(f"Advanced Test Runner initialized")
        logger.info(f"Test directory: {self.test_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Discovered {len(self.test_modules)} test modules")
    
    def _discover_test_modules(self) -> List[str]:
        """
        Discover all test modules in the test directory
        
        :return: List of test module names
        """
        test_modules = []
        
        # Look for test_*.py files
        for test_file in self.test_dir.rglob("test_*.py"):
            if test_file.is_file() and test_file.name != "__init__.py":
                # Convert path to module name
                relative_path = test_file.relative_to(self.test_dir)
                module_name = str(relative_path.with_suffix(''))
                module_name = module_name.replace(os.sep, '.')
                test_modules.append(module_name)
        
        return sorted(test_modules)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all discovered tests
        
        :return: Summary of all test results
        """
        logger.info("Starting comprehensive test suite execution")
        start_time = time.time()
        
        # Run tests (parallel or sequential)
        if self.parallel and len(self.test_modules) > 1:
            self._run_tests_parallel()
        else:
            self._run_tests_sequential()
        
        # Generate summary
        total_time = time.time() - start_time
        summary = self._generate_execution_summary(total_time)
        
        # Generate reports if requested
        if self.generate_reports:
            self._generate_comprehensive_reports()
        
        logger.info(f"Test suite completed in {total_time:.2f} seconds")
        return summary
    
    def _run_tests_sequential(self):
        """Run tests sequentially"""
        logger.info("Running tests sequentially")
        
        for module_name in self.test_modules:
            try:
                logger.info(f"Running tests for module: {module_name}")
                result = self._run_single_module_test(module_name)
                self.test_results[module_name] = result
                
            except Exception as e:
                error_msg = f"Error running tests for {module_name}: {str(e)}"
                logger.error(error_msg)
                self.error_log.append(error_msg)
                self.test_results[module_name] = {"error": str(e), "status": "failed"}
    
    def _run_tests_parallel(self):
        """Run tests in parallel"""
        logger.info("Running tests in parallel")
        
        # Use multiprocessing to run tests in parallel
        with mp.Pool(processes=min(len(self.test_modules), mp.cpu_count())) as pool:
            results = pool.map(self._run_single_module_test, self.test_modules)
        
        # Collect results
        for module_name, result in zip(self.test_modules, results):
            self.test_results[module_name] = result
    
    def _run_single_module_test(self, module_name: str) -> Dict[str, Any]:
        """
        Run tests for a single module
        
        :param module_name: Name of the module to test
        :return: Test results for the module
        """
        test_start_time = time.time()
        
        try:
            # Import the module dynamically
            module = __import__(module_name, fromlist=[''])
            
            # Look for test functions or classes
            test_functions = [attr for attr in dir(module) 
                            if (callable(getattr(module, attr)) and 
                                (attr.startswith('test_') or attr.startswith('run_')))]
            
            module_results = {
                'module_name': module_name,
                'status': 'passed',
                'execution_time': 0,
                'tests_run': 0,
                'tests_passed': 0,
                'tests_failed': 0,
                'function_results': {},
                'error': None
            }
            
            # Run each test function
            for func_name in test_functions:
                try:
                    func = getattr(module, func_name)
                    func_start = time.time()
                    
                    # Execute the function
                    result = func()
                    
                    func_time = time.time() - func_start
                    module_results['function_results'][func_name] = {
                        'status': 'passed',
                        'execution_time': func_time,
                        'result': result
                    }
                    module_results['tests_run'] += 1
                    module_results['tests_passed'] += 1
                    
                except Exception as e:
                    func_time = time.time() - func_start
                    module_results['function_results'][func_name] = {
                        'status': 'failed',
                        'execution_time': func_time,
                        'error': str(e)
                    }
                    module_results['tests_run'] += 1
                    module_results['tests_failed'] += 1
            
            module_results['execution_time'] = time.time() - test_start_time
            
            # If any test failed, mark module as failed
            if module_results['tests_failed'] > 0:
                module_results['status'] = 'failed'
            
            return module_results
            
        except Exception as e:
            return {
                'module_name': module_name,
                'status': 'error',
                'execution_time': time.time() - test_start_time,
                'error': str(e),
                'tests_run': 0,
                'tests_passed': 0,
                'tests_failed': 1
            }
    
    def _generate_execution_summary(self, total_time: float) -> Dict[str, Any]:
        """
        Generate execution summary
        
        :param total_time: Total execution time
        :return: Summary dictionary
        """
        summary = {
            'execution_date': datetime.now().isoformat(),
            'total_execution_time': total_time,
            'total_modules': len(self.test_modules),
            'modules_passed': sum(1 for r in self.test_results.values() if r.get('status') == 'passed'),
            'modules_failed': sum(1 for r in self.test_results.values() if r.get('status') in ['failed', 'error']),
            'total_tests': sum(r.get('tests_run', 0) for r in self.test_results.values()),
            'total_passed': sum(r.get('tests_passed', 0) for r in self.test_results.values()),
            'total_failed': sum(r.get('tests_failed', 0) for r in self.test_results.values()),
            'average_execution_time': np.mean([r.get('execution_time', 0) for r in self.test_results.values()]),
            'module_results': self.test_results,
            'errors': self.error_log
        }
        
        return summary
    
    def _generate_comprehensive_reports(self):
        """Generate comprehensive HTML and JSON reports"""
        logger.info("Generating comprehensive reports")
        
        # Generate HTML dashboard
        html_report = self._generate_html_dashboard()
        html_path = self.output_dir / "comprehensive_test_dashboard.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # Generate JSON report
        json_path = self.output_dir / "test_results_complete.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Generate performance analysis
        self._generate_performance_analysis()
        
        # Generate error analysis
        self._generate_error_analysis()
        
        logger.info(f"Reports generated:")
        logger.info(f"  - HTML Dashboard: {html_path}")
        logger.info(f"  - JSON Results: {json_path}")
    
    def _generate_html_dashboard(self) -> str:
        """Generate HTML dashboard"""
        # Calculate statistics
        total_modules = len(self.test_results)
        passed_modules = sum(1 for r in self.test_results.values() if r.get('status') == 'passed')
        failed_modules = total_modules - passed_modules
        
        total_tests = sum(r.get('tests_run', 0) for r in self.test_results.values())
        passed_tests = sum(r.get('tests_passed', 0) for r in self.test_results.values())
        failed_tests = sum(r.get('tests_failed', 0) for r in self.test_results.values())
        
        # Generate module results table
        module_table = self._generate_module_results_table()
        
        # Generate performance charts
        performance_chart = self._generate_performance_chart()
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Advanced ML Testing Dashboard</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .dashboard-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .dashboard-header h1 {{ margin: 0; font-size: 2.5em; }}
                .dashboard-header p {{ margin: 5px 0; opacity: 0.9; }}
                .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                .stat-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .stat-card h3 {{ margin: 0 0 10px 0; color: #333; }}
                .stat-number {{ font-size: 2em; font-weight: bold; color: #667eea; }}
                .stat-success {{ color: #28a745; }}
                .stat-error {{ color: #dc3545; }}
                .section {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; }}
                .section h2 {{ margin: 0 0 20px 0; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f8f9fa; font-weight: bold; }}
                .status-passed {{ color: #28a745; font-weight: bold; }}
                .status-failed {{ color: #dc3545; font-weight: bold; }}
                .status-error {{ color: #fd7e14; font-weight: bold; }}
                .chart-container {{ height: 400px; margin: 20px 0; }}
                .error-log {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #dc3545; }}
                .recommendations {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; border-left: 4px solid #2196f3; }}
            </style>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <div class="dashboard-header">
                <h1>🧪 Advanced ML Testing Dashboard</h1>
                <p>Comprehensive testing results for Machine Learning modules</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <h3>📦 Total Modules</h3>
                    <div class="stat-number">{total_modules}</div>
                </div>
                <div class="stat-card">
                    <h3>✅ Passed Modules</h3>
                    <div class="stat-number stat-success">{passed_modules}</div>
                </div>
                <div class="stat-card">
                    <h3>❌ Failed Modules</h3>
                    <div class="stat-number stat-error">{failed_modules}</div>
                </div>
                <div class="stat-card">
                    <h3>🧪 Total Tests</h3>
                    <div class="stat-number">{total_tests}</div>
                </div>
                <div class="stat-card">
                    <h3>✅ Passed Tests</h3>
                    <div class="stat-number stat-success">{passed_tests}</div>
                </div>
                <div class="stat-card">
                    <h3>❌ Failed Tests</h3>
                    <div class="stat-number stat-error">{failed_tests}</div>
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Module Results</h2>
                {module_table}
            </div>
            
            <div class="section">
                <h2>⚡ Performance Analysis</h2>
                <div class="chart-container" id="performance-chart"></div>
            </div>
            
            <div class="section">
                <h2>🎯 Recommendations</h2>
                <div class="recommendations">
                    {self._generate_recommendations()}
                </div>
            </div>
            
            <div class="section">
                <h2>📋 Error Log</h2>
                <div class="error-log">
                    {self._format_error_log()}
                </div>
            </div>
            
            <script>
                {performance_chart}
            </script>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_module_results_table(self) -> str:
        """Generate HTML table for module results"""
        if not self.test_results:
            return "<p>No test results available</p>"
        
        html = "<table><tr><th>Module</th><th>Status</th><th>Tests Run</th><th>Passed</th><th>Failed</th><th>Execution Time</th></tr>"
        
        for module_name, result in self.test_results.items():
            status = result.get('status', 'unknown')
            status_class = f"status-{status}"
            
            tests_run = result.get('tests_run', 0)
            tests_passed = result.get('tests_passed', 0)
            tests_failed = result.get('tests_failed', 0)
            exec_time = result.get('execution_time', 0)
            
            html += f"""
            <tr>
                <td>{module_name}</td>
                <td class="{status_class}">{status.upper()}</td>
                <td>{tests_run}</td>
                <td>{tests_passed}</td>
                <td>{tests_failed}</td>
                <td>{exec_time:.3f}s</td>
            </tr>
            """
        
        html += "</table>"
        return html
    
    def _generate_performance_chart(self) -> str:
        """Generate JavaScript code for performance chart"""
        if not self.test_results:
            return ""
        
        # Extract data for chart
        modules = list(self.test_results.keys())
        exec_times = [self.test_results[m].get('execution_time', 0) for m in modules]
        test_counts = [self.test_results[m].get('tests_run', 0) for m in modules]
        
        # Shorten module names for display
        short_names = [m.split('.')[-1] for m in modules]
        
        js_code = f"""
        var trace1 = {{
            x: {short_names},
            y: {exec_times},
            type: 'bar',
            name: 'Execution Time (s)',
            marker: {{color: 'rgba(102, 126, 234, 0.6)'}}
        }};
        
        var trace2 = {{
            x: {short_names},
            y: {test_counts},
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Test Count',
            yaxis: 'y2',
            marker: {{color: 'rgba(255, 99, 132, 0.8)'}}
        }};
        
        var layout = {{
            title: 'Module Performance Analysis',
            xaxis: {{title: 'Module'}},
            yaxis: {{title: 'Execution Time (seconds)'}},
            yaxis2: {{
                title: 'Number of Tests',
                overlaying: 'y',
                side: 'right'
            }},
            hovermode: 'closest'
        }};
        
        Plotly.newPlot('performance-chart', [trace1, trace2], layout);
        """
        
        return js_code
    
    def _generate_performance_analysis(self):
        """Generate detailed performance analysis"""
        if not self.test_results:
            return
        
        # Create performance DataFrame
        perf_data = []
        for module_name, result in self.test_results.items():
            perf_data.append({
                'module': module_name,
                'execution_time': result.get('execution_time', 0),
                'tests_run': result.get('tests_run', 0),
                'tests_passed': result.get('tests_passed', 0),
                'tests_failed': result.get('tests_failed', 0),
                'success_rate': result.get('tests_passed', 0) / max(result.get('tests_run', 1), 1) * 100
            })
        
        df = pd.DataFrame(perf_data)
        
        # Save detailed performance analysis
        perf_path = self.output_dir / "performance_analysis.csv"
        df.to_csv(perf_path, index=False)
        
        # Generate performance plots
        self._create_performance_plots(df)
    
    def _create_performance_plots(self, df: pd.DataFrame):
        """Create detailed performance plots"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Execution Time by Module', 'Success Rate by Module',
                          'Tests vs Execution Time', 'Module Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Execution time
        fig.add_trace(
            go.Bar(x=df['module'], y=df['execution_time'], name='Execution Time'),
            row=1, col=1
        )
        
        # Success rate
        fig.add_trace(
            go.Bar(x=df['module'], y=df['success_rate'], name='Success Rate'),
            row=1, col=2
        )
        
        # Tests vs execution time scatter
        fig.add_trace(
            go.Scatter(x=df['tests_run'], y=df['execution_time'], 
                      mode='markers', name='Tests vs Time'),
            row=2, col=1
        )
        
        # Module comparison radar-like
        fig.add_trace(
            go.Scatter(x=df['success_rate'], y=df['execution_time'], 
                      mode='markers+text', text=df['module'], name='Module Performance'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Detailed Performance Analysis",
            height=800,
            showlegend=False
        )
        
        # Save plot
        plot_path = self.output_dir / "performance_detailed_analysis.html"
        plot(fig, filename=str(plot_path), auto_open=False)
    
    def _generate_error_analysis(self):
        """Generate error analysis report"""
        errors = []
        
        for module_name, result in self.test_results.items():
            if result.get('status') in ['failed', 'error']:
                errors.append({
                    'module': module_name,
                    'error': result.get('error', 'Unknown error'),
                    'status': result.get('status', 'unknown')
                })
        
        if errors:
            error_df = pd.DataFrame(errors)
            error_path = self.output_dir / "error_analysis.csv"
            error_df.to_csv(error_path, index=False)
    
    def _generate_recommendations(self) -> str:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Performance recommendations
        exec_times = [r.get('execution_time', 0) for r in self.test_results.values()]
        if exec_times:
            avg_time = np.mean(exec_times)
            if avg_time > 10:
                recommendations.append("⚠️ High execution times detected. Consider optimizing test performance.")
        
        # Error rate recommendations
        total_tests = sum(r.get('tests_run', 0) for r in self.test_results.values())
        failed_tests = sum(r.get('tests_failed', 0) for r in self.test_results.values())
        
        if total_tests > 0:
            error_rate = failed_tests / total_tests * 100
            if error_rate > 10:
                recommendations.append("🔧 High error rate detected. Review failed tests and improve code quality.")
            elif error_rate > 5:
                recommendations.append("⚡ Moderate error rate. Consider additional testing and validation.")
        
        # Coverage recommendations
        if len(self.test_results) < 10:
            recommendations.append("📈 Consider adding more comprehensive tests for better coverage.")
        
        # General recommendations
        recommendations.extend([
            "✅ Regularly run comprehensive tests as part of CI/CD pipeline",
            "📊 Monitor performance trends over time",
            "🔍 Investigate any performance regressions",
            "📝 Document test failures and resolutions"
        ])
        
        return "<ul>" + "".join(f"<li>{rec}</li>" for rec in recommendations) + "</ul>"
    
    def _format_error_log(self) -> str:
        """Format error log for HTML display"""
        if not self.error_log:
            return "<p>No errors recorded during test execution ✅</p>"
        
        html = "<ul>"
        for error in self.error_log:
            html += f"<li>{error}</li>"
        html += "</ul>"
        
        return html


def main():
    """Main function to run the test suite"""
    parser = argparse.ArgumentParser(description='Advanced ML Testing Suite')
    parser.add_argument('--test-dir', default='./', help='Directory containing test modules')
    parser.add_argument('--output-dir', default='./test_results', help='Output directory for results')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--no-reports', action='store_true', help='Skip report generation')
    
    args = parser.parse_args()
    
    # Initialize and run tests
    runner = AdvancedTestRunner(
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        parallel=args.parallel,
        generate_reports=not args.no_reports
    )
    
    # Run all tests
    summary = runner.run_all_tests()
    
    # Print summary
    print(f"\n{'='*80}")
    print("ADVANCED ML TESTING SUITE - EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Modules Tested: {summary['total_modules']}")
    print(f"Modules Passed: {summary['modules_passed']}")
    print(f"Modules Failed: {summary['modules_failed']}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Tests Passed: {summary['total_passed']}")
    print(f"Tests Failed: {summary['total_failed']}")
    print(f"Total Execution Time: {summary['total_execution_time']:.2f} seconds")
    print(f"Average Module Time: {summary['average_execution_time']:.2f} seconds")
    
    if summary['total_failed'] > 0:
        print(f"\n⚠️  {summary['total_failed']} tests failed. Check the detailed reports.")
    else:
        print(f"\n✅ All tests passed successfully!")
    
    print(f"\nReports generated in: {args.output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
