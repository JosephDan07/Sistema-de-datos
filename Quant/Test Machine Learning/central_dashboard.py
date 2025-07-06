"""
Central Testing Dashboard - Sistema de Análisis ML Finance
=========================================================

Dashboard HTML centralizado que agrega y muestra resultados de todos los módulos de testing.
Diseñado para consumir los archivos test_summary.json de cada módulo y crear visualizaciones unificadas.

Features:
- Dashboard HTML interactivo centralizado
- Agregación de resultados de múltiples módulos
- Visualizaciones comparativas entre módulos
- Sistema de limpieza automática de resultados antiguos
- Navegación entre reportes individuales

Author: Advanced ML Finance Team
Date: July 2025
"""

import os
import json
import glob
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Visualization libraries
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
    logger.info("✅ Plotly imported successfully")
except ImportError as e:
    PLOTLY_AVAILABLE = False
    logger.warning(f"⚠️ Plotly not available: {e}")

try:
    import pandas as pd
    import numpy as np
    logger.info("✅ Data libraries imported successfully")
except ImportError as e:
    logger.error(f"❌ Required libraries not available: {e}")


class CentralDashboard:
    """
    Central dashboard system for aggregating and displaying test results from all modules
    """
    
    def __init__(self, 
                 test_machine_learning_path: str = "./",
                 results_path: str = "./Dashboard_Results/",
                 max_results_age_days: int = 7):
        """
        Initialize the central dashboard
        
        Args:
            test_machine_learning_path: Path to Test Machine Learning directory
            results_path: Path to store centralized results
            max_results_age_days: Maximum age for keeping results
        """
        self.base_path = test_machine_learning_path
        self.results_path = results_path
        self.max_age_days = max_results_age_days
        
        # Ensure results directory exists
        os.makedirs(results_path, exist_ok=True)
        
        # Initialize dashboard data
        self.modules_data = {}
        self.aggregated_stats = {}
        self.dashboard_timestamp = datetime.now()
        
        logger.info(f"✅ Central Dashboard initialized")
        logger.info(f"📁 Base path: {os.path.abspath(test_machine_learning_path)}")
        logger.info(f"📁 Results path: {os.path.abspath(results_path)}")
        
    def scan_module_results(self) -> Dict[str, Any]:
        """
        Scan all subdirectories for test_summary.json files and collect results
        
        Returns:
            Dictionary with aggregated module results
        """
        logger.info("🔍 Scanning for module test results...")
        
        # Search pattern for test_summary.json files
        search_pattern = os.path.join(self.base_path, "**/test_summary.json")
        summary_files = glob.glob(search_pattern, recursive=True)
        
        logger.info(f"📊 Found {len(summary_files)} module summary files")
        
        modules_data = {}
        
        for summary_file in summary_files:
            try:
                # Extract module directory name
                module_dir = os.path.dirname(summary_file)
                module_name = os.path.basename(module_dir)
                
                # Load summary data
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                
                # Add file paths and metadata
                summary_data['summary_file_path'] = summary_file
                summary_data['module_directory'] = module_dir
                summary_data['scan_timestamp'] = datetime.now().isoformat()
                
                # Check if results directory exists and count files
                results_dir = os.path.join(module_dir, "test_results")
                if os.path.exists(results_dir):
                    result_files = os.listdir(results_dir)
                    summary_data['result_files_count'] = len(result_files)
                    summary_data['result_files'] = result_files
                else:
                    summary_data['result_files_count'] = 0
                    summary_data['result_files'] = []
                
                modules_data[module_name] = summary_data
                logger.info(f"  ✅ Loaded: {module_name} ({summary_data.get('tests_run', 0)} tests)")
                
            except Exception as e:
                logger.error(f"  ❌ Error loading {summary_file}: {str(e)}")
        
        self.modules_data = modules_data
        return modules_data
    
    def generate_aggregated_statistics(self) -> Dict[str, Any]:
        """
        Generate aggregated statistics from all modules
        
        Returns:
            Dictionary with aggregated statistics
        """
        logger.info("📊 Generating aggregated statistics...")
        
        if not self.modules_data:
            logger.warning("⚠️ No module data available for aggregation")
            return {}
        
        # Initialize aggregated stats
        stats = {
            'total_modules': len(self.modules_data),
            'total_tests_run': 0,
            'total_tests_passed': 0,
            'total_tests_failed': 0,
            'total_plots_generated': 0,
            'modules_success_rates': {},
            'modules_performance': {},
            'overall_success_rate': 0,
            'latest_test_date': None,
            'oldest_test_date': None,
            'modules_summary': []
        }
        
        test_dates = []
        
        for module_name, module_data in self.modules_data.items():
            # Aggregate test counts
            tests_run = module_data.get('tests_run', 0)
            tests_passed = module_data.get('tests_passed', 0)
            tests_failed = module_data.get('tests_failed', 0)
            plots_count = module_data.get('plots_generated', 0)
            success_rate = module_data.get('success_rate', 0)
            
            stats['total_tests_run'] += tests_run
            stats['total_tests_passed'] += tests_passed
            stats['total_tests_failed'] += tests_failed
            stats['total_plots_generated'] += plots_count
            
            # Store individual module stats
            stats['modules_success_rates'][module_name] = success_rate
            stats['modules_performance'][module_name] = {
                'tests_run': tests_run,
                'tests_passed': tests_passed,
                'tests_failed': tests_failed,
                'success_rate': success_rate,
                'plots_generated': plots_count,
                'result_files_count': module_data.get('result_files_count', 0)
            }
            
            # Track test dates
            test_date = module_data.get('test_date')
            if test_date:
                test_dates.append(test_date)
            
            # Module summary for table
            stats['modules_summary'].append({
                'module_name': module_name,
                'tests_run': tests_run,
                'tests_passed': tests_passed,
                'tests_failed': tests_failed,
                'success_rate': success_rate,
                'plots_generated': plots_count,
                'test_date': test_date,
                'html_report': module_data.get('html_report', ''),
                'json_report': module_data.get('json_report', '')
            })
        
        # Calculate overall statistics
        if stats['total_tests_run'] > 0:
            stats['overall_success_rate'] = (stats['total_tests_passed'] / stats['total_tests_run']) * 100
        
        if test_dates:
            stats['latest_test_date'] = max(test_dates)
            stats['oldest_test_date'] = min(test_dates)
        
        self.aggregated_stats = stats
        
        logger.info(f"📊 Aggregated Statistics:")
        logger.info(f"  📦 Total Modules: {stats['total_modules']}")
        logger.info(f"  🧪 Total Tests: {stats['total_tests_run']}")
        logger.info(f"  ✅ Tests Passed: {stats['total_tests_passed']}")
        logger.info(f"  ❌ Tests Failed: {stats['total_tests_failed']}")
        logger.info(f"  🎯 Overall Success Rate: {stats['overall_success_rate']:.1f}%")
        logger.info(f"  📈 Total Plots: {stats['total_plots_generated']}")
        
        return stats
    
    def create_interactive_dashboard(self) -> str:
        """
        Create comprehensive interactive dashboard with Plotly
        
        Returns:
            Path to generated dashboard HTML file
        """
        logger.info("🎯 Creating interactive central dashboard...")
        
        if not PLOTLY_AVAILABLE:
            logger.error("❌ Plotly not available - cannot create interactive dashboard")
            return ""
        
        if not self.aggregated_stats:
            logger.warning("⚠️ No aggregated statistics available")
            return ""
        
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=(
                    'Module Success Rates', 'Tests Distribution', 'Module Performance',
                    'Test Results Overview', 'Plots Generated', 'Timeline Analysis',
                    'Module Comparison', 'Success Rate Trend', 'System Health'
                ),
                specs=[
                    [{"type": "bar"}, {"type": "pie"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}],
                    [{"type": "bar"}, {"type": "scatter"}, {"type": "indicator"}]
                ]
            )
            
            # 1. Module Success Rates (Bar Chart)
            modules = list(self.aggregated_stats['modules_success_rates'].keys())
            success_rates = list(self.aggregated_stats['modules_success_rates'].values())
            
            fig.add_trace(
                go.Bar(
                    x=modules,
                    y=success_rates,
                    name='Success Rate (%)',
                    marker_color='lightgreen',
                    hovertemplate='Module: %{x}<br>Success Rate: %{y:.1f}%<extra></extra>'
                ),
                row=1, col=1
            )
            
            # 2. Tests Distribution (Pie Chart)
            fig.add_trace(
                go.Pie(
                    labels=['Passed', 'Failed'],
                    values=[self.aggregated_stats['total_tests_passed'], 
                           self.aggregated_stats['total_tests_failed']],
                    marker_colors=['lightgreen', 'lightcoral'],
                    hovertemplate='%{label}: %{value}<br>%{percent}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # 3. Module Performance (Scatter Plot)
            perf_modules = []
            tests_run = []
            plots_generated = []
            
            for module, perf in self.aggregated_stats['modules_performance'].items():
                perf_modules.append(module)
                tests_run.append(perf['tests_run'])
                plots_generated.append(perf['plots_generated'])
            
            fig.add_trace(
                go.Scatter(
                    x=tests_run,
                    y=plots_generated,
                    mode='markers+text',
                    text=perf_modules,
                    textposition="top center",
                    name='Module Performance',
                    marker=dict(size=10, color='blue'),
                    hovertemplate='Module: %{text}<br>Tests: %{x}<br>Plots: %{y}<extra></extra>'
                ),
                row=1, col=3
            )
            
            # 4. Test Results Overview (Bar Chart)
            categories = ['Tests Run', 'Tests Passed', 'Tests Failed', 'Plots Generated']
            values = [
                self.aggregated_stats['total_tests_run'],
                self.aggregated_stats['total_tests_passed'],
                self.aggregated_stats['total_tests_failed'],
                self.aggregated_stats['total_plots_generated']
            ]
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
            
            fig.add_trace(
                go.Bar(
                    x=categories,
                    y=values,
                    name='Overview',
                    marker_color=colors,
                    hovertemplate='Category: %{x}<br>Count: %{y}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # 5. Plots Generated by Module (Bar Chart)
            plot_counts = [self.aggregated_stats['modules_performance'][mod]['plots_generated'] 
                          for mod in modules]
            
            fig.add_trace(
                go.Bar(
                    x=modules,
                    y=plot_counts,
                    name='Plots per Module',
                    marker_color='lightyellow',
                    hovertemplate='Module: %{x}<br>Plots: %{y}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # 6. Timeline Analysis (if dates available)
            if self.aggregated_stats.get('latest_test_date'):
                # Simple timeline representation
                fig.add_trace(
                    go.Scatter(
                        x=[self.aggregated_stats['oldest_test_date'], 
                           self.aggregated_stats['latest_test_date']],
                        y=[1, 1],
                        mode='markers+lines',
                        name='Test Timeline',
                        marker=dict(size=15, color='purple'),
                        line=dict(color='purple', width=3),
                        hovertemplate='Date: %{x}<extra></extra>'
                    ),
                    row=2, col=3
                )
            
            # 7. Module Comparison (Tests Run vs Success Rate)
            fig.add_trace(
                go.Bar(
                    x=modules,
                    y=[self.aggregated_stats['modules_performance'][mod]['tests_run'] 
                       for mod in modules],
                    name='Tests Run per Module',
                    marker_color='lightsteelblue',
                    hovertemplate='Module: %{x}<br>Tests Run: %{y}<extra></extra>'
                ),
                row=3, col=1
            )
            
            # 8. Success Rate Trend (simplified)
            fig.add_trace(
                go.Scatter(
                    x=modules,
                    y=success_rates,
                    mode='lines+markers',
                    name='Success Rate Trend',
                    line=dict(color='green', width=3),
                    marker=dict(size=8, color='green'),
                    hovertemplate='Module: %{x}<br>Success Rate: %{y:.1f}%<extra></extra>'
                ),
                row=3, col=2
            )
            
            # 9. System Health Indicator
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=self.aggregated_stats['overall_success_rate'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Overall Success Rate (%)"},
                    delta={'reference': 90},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ),
                row=3, col=3
            )
            
            # Update layout
            fig.update_layout(
                title={
                    'text': f"ML Finance Testing Dashboard - Central Overview<br><sub>Generated: {self.dashboard_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</sub>",
                    'x': 0.5,
                    'font': {'size': 20}
                },
                showlegend=False,
                height=1200,
                template='plotly_white'
            )
            
            # Update axis labels
            fig.update_xaxes(title_text="Modules", row=1, col=1)
            fig.update_yaxes(title_text="Success Rate (%)", row=1, col=1)
            
            fig.update_xaxes(title_text="Tests Run", row=1, col=3)
            fig.update_yaxes(title_text="Plots Generated", row=1, col=3)
            
            fig.update_xaxes(title_text="Categories", row=2, col=1)
            fig.update_yaxes(title_text="Count", row=2, col=1)
            
            fig.update_xaxes(title_text="Modules", row=2, col=2)
            fig.update_yaxes(title_text="Plots Generated", row=2, col=2)
            
            fig.update_xaxes(title_text="Time", row=2, col=3)
            fig.update_yaxes(title_text="Timeline", row=2, col=3)
            
            fig.update_xaxes(title_text="Modules", row=3, col=1)
            fig.update_yaxes(title_text="Tests Run", row=3, col=1)
            
            fig.update_xaxes(title_text="Modules", row=3, col=2)
            fig.update_yaxes(title_text="Success Rate (%)", row=3, col=2)
            
            # Save dashboard
            dashboard_path = os.path.join(self.results_path, 'central_dashboard.html')
            fig.write_html(dashboard_path)
            
            logger.info(f"  🎯 Interactive dashboard saved to: {dashboard_path}")
            return dashboard_path
            
        except Exception as e:
            logger.error(f"  ❌ Error creating interactive dashboard: {str(e)}")
            return ""
    
    def generate_html_report(self) -> str:
        """
        Generate comprehensive HTML report with module links
        
        Returns:
            Path to generated HTML report
        """
        logger.info("📄 Generating central HTML report...")
        
        try:
            # Generate modules table
            modules_table_rows = ""
            for module_summary in self.aggregated_stats.get('modules_summary', []):
                status_class = "success" if module_summary['success_rate'] > 80 else "warning" if module_summary['success_rate'] > 60 else "error"
                
                html_link = f"<a href='{module_summary['html_report']}' target='_blank'>View Report</a>" if module_summary['html_report'] else "N/A"
                json_link = f"<a href='{module_summary['json_report']}' target='_blank'>View JSON</a>" if module_summary['json_report'] else "N/A"
                
                modules_table_rows += f"""
                <tr>
                    <td><strong>{module_summary['module_name']}</strong></td>
                    <td>{module_summary['tests_run']}</td>
                    <td class="success">{module_summary['tests_passed']}</td>
                    <td class="error">{module_summary['tests_failed']}</td>
                    <td class="{status_class}">{module_summary['success_rate']:.1f}%</td>
                    <td>{module_summary['plots_generated']}</td>
                    <td>{module_summary['test_date']}</td>
                    <td>{html_link}</td>
                    <td>{json_link}</td>
                </tr>
                """
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>ML Finance Testing - Central Dashboard</title>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body {{ 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                        margin: 0; 
                        padding: 20px; 
                        background-color: #f5f5f5;
                    }}
                    .container {{
                        max-width: 1400px;
                        margin: 0 auto;
                        background-color: white;
                        padding: 30px;
                        border-radius: 10px;
                        box-shadow: 0 0 20px rgba(0,0,0,0.1);
                    }}
                    .header {{ 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 30px; 
                        border-radius: 10px; 
                        margin-bottom: 30px;
                        text-align: center;
                    }}
                    .header h1 {{ margin: 0; font-size: 2.5em; }}
                    .header p {{ margin: 10px 0 0 0; font-size: 1.2em; opacity: 0.9; }}
                    .stats-grid {{ 
                        display: grid; 
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                        gap: 20px; 
                        margin-bottom: 30px; 
                    }}
                    .stat-card {{ 
                        background: linear-gradient(145deg, #ffffff, #f0f0f0);
                        padding: 20px; 
                        border-radius: 10px; 
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                        text-align: center;
                        border-left: 4px solid #667eea;
                    }}
                    .stat-card h3 {{ margin: 0 0 10px 0; color: #333; font-size: 1.1em; }}
                    .stat-card .value {{ font-size: 2.5em; font-weight: bold; color: #667eea; margin: 0; }}
                    .section {{ 
                        margin: 30px 0; 
                        padding: 25px; 
                        background: white;
                        border-radius: 10px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                        border-left: 4px solid #667eea; 
                    }}
                    .section h2 {{ margin-top: 0; color: #333; font-size: 1.5em; }}
                    .success {{ color: #28a745; font-weight: bold; }}
                    .error {{ color: #dc3545; font-weight: bold; }}
                    .warning {{ color: #ffc107; font-weight: bold; }}
                    table {{ 
                        border-collapse: collapse; 
                        width: 100%; 
                        margin-top: 15px;
                        background: white;
                        border-radius: 10px;
                        overflow: hidden;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                    }}
                    th, td {{ 
                        border: 1px solid #e0e0e0; 
                        padding: 12px; 
                        text-align: left; 
                    }}
                    th {{ 
                        background: linear-gradient(145deg, #667eea, #764ba2);
                        color: white;
                        font-weight: bold;
                        text-transform: uppercase;
                        font-size: 0.9em;
                        letter-spacing: 0.5px;
                    }}
                    tr:nth-child(even) {{ background-color: #f8f9fa; }}
                    tr:hover {{ background-color: #e3f2fd; }}
                    a {{ 
                        color: #667eea; 
                        text-decoration: none; 
                        font-weight: bold;
                        padding: 5px 10px;
                        border-radius: 5px;
                        background-color: #f0f4ff;
                        transition: all 0.3s ease;
                    }}
                    a:hover {{ 
                        background-color: #667eea; 
                        color: white; 
                    }}
                    .dashboard-link {{
                        display: inline-block;
                        background: linear-gradient(145deg, #28a745, #20c997);
                        color: white;
                        padding: 15px 30px;
                        border-radius: 50px;
                        text-decoration: none;
                        font-weight: bold;
                        margin: 20px 10px;
                        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
                        transition: all 0.3s ease;
                    }}
                    .dashboard-link:hover {{
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
                        color: white;
                    }}
                    .footer {{
                        text-align: center;
                        margin-top: 50px;
                        padding: 20px;
                        color: #666;
                        border-top: 1px solid #e0e0e0;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🚀 ML Finance Testing Dashboard</h1>
                        <p>Central Overview - Generated on {self.dashboard_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    
                    <div class="stats-grid">
                        <div class="stat-card">
                            <h3>📦 Total Modules</h3>
                            <p class="value">{self.aggregated_stats.get('total_modules', 0)}</p>
                        </div>
                        <div class="stat-card">
                            <h3>🧪 Total Tests</h3>
                            <p class="value">{self.aggregated_stats.get('total_tests_run', 0)}</p>
                        </div>
                        <div class="stat-card">
                            <h3>✅ Tests Passed</h3>
                            <p class="value">{self.aggregated_stats.get('total_tests_passed', 0)}</p>
                        </div>
                        <div class="stat-card">
                            <h3>❌ Tests Failed</h3>
                            <p class="value">{self.aggregated_stats.get('total_tests_failed', 0)}</p>
                        </div>
                        <div class="stat-card">
                            <h3>🎯 Success Rate</h3>
                            <p class="value">{self.aggregated_stats.get('overall_success_rate', 0):.1f}%</p>
                        </div>
                        <div class="stat-card">
                            <h3>📈 Total Plots</h3>
                            <p class="value">{self.aggregated_stats.get('total_plots_generated', 0)}</p>
                        </div>
                    </div>
                    
                    <div style="text-align: center; margin: 30px 0;">
                        <a href="central_dashboard.html" class="dashboard-link">📊 View Interactive Dashboard</a>
                        <a href="#modules-detail" class="dashboard-link">📋 Module Details</a>
                    </div>
                    
                    <div class="section" id="modules-detail">
                        <h2>📋 Module Testing Results</h2>
                        <p>Detailed results from all testing modules in the ML Finance system.</p>
                        <table>
                            <thead>
                                <tr>
                                    <th>Module Name</th>
                                    <th>Tests Run</th>
                                    <th>Passed</th>
                                    <th>Failed</th>
                                    <th>Success Rate</th>
                                    <th>Plots Generated</th>
                                    <th>Test Date</th>
                                    <th>HTML Report</th>
                                    <th>JSON Data</th>
                                </tr>
                            </thead>
                            <tbody>
                                {modules_table_rows}
                            </tbody>
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>📊 System Health Summary</h2>
                        <ul>
                            <li><strong>Overall System Health:</strong> 
                                <span class="{'success' if self.aggregated_stats.get('overall_success_rate', 0) > 80 else 'warning' if self.aggregated_stats.get('overall_success_rate', 0) > 60 else 'error'}">
                                    {self.aggregated_stats.get('overall_success_rate', 0):.1f}% Success Rate
                                </span>
                            </li>
                            <li><strong>Test Coverage:</strong> {self.aggregated_stats.get('total_modules', 0)} modules tested</li>
                            <li><strong>Quality Assurance:</strong> {self.aggregated_stats.get('total_plots_generated', 0)} visualizations generated</li>
                            <li><strong>Latest Update:</strong> {self.aggregated_stats.get('latest_test_date', 'N/A')}</li>
                        </ul>
                    </div>
                    
                    <div class="footer">
                        <p>🤖 Advanced ML Finance Testing System | Generated automatically | 
                        Next scan: {(self.dashboard_timestamp + timedelta(hours=1)).strftime('%H:%M')}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Save HTML report
            html_path = os.path.join(self.results_path, 'index.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"  📄 HTML report saved to: {html_path}")
            return html_path
            
        except Exception as e:
            logger.error(f"  ❌ Error generating HTML report: {str(e)}")
            return ""
    
    def cleanup_old_results(self):
        """
        Clean up old result files to prevent system saturation
        """
        logger.info("🧹 Cleaning up old test results...")
        
        cleanup_count = 0
        cutoff_date = datetime.now() - timedelta(days=self.max_age_days)
        
        # Clean up individual module results
        for module_name, module_data in self.modules_data.items():
            module_dir = module_data.get('module_directory', '')
            results_dir = os.path.join(module_dir, 'test_results')
            
            if os.path.exists(results_dir):
                for file_name in os.listdir(results_dir):
                    file_path = os.path.join(results_dir, file_name)
                    
                    # Check file age
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_mtime < cutoff_date:
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                                cleanup_count += 1
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                                cleanup_count += 1
                        except Exception as e:
                            logger.warning(f"⚠️ Could not remove {file_path}: {str(e)}")
        
        # Clean up central dashboard results (keep only latest)
        dashboard_files = glob.glob(os.path.join(self.results_path, '*.html'))
        dashboard_files.extend(glob.glob(os.path.join(self.results_path, '*.json')))
        
        # Sort by modification time and keep only the 3 most recent
        dashboard_files.sort(key=os.path.getmtime, reverse=True)
        
        for old_file in dashboard_files[3:]:  # Keep 3 most recent
            try:
                os.remove(old_file)
                cleanup_count += 1
            except Exception as e:
                logger.warning(f"⚠️ Could not remove {old_file}: {str(e)}")
        
        logger.info(f"  🧹 Cleaned up {cleanup_count} old files")
    
    def save_aggregated_data(self):
        """
        Save aggregated data for future reference
        """
        logger.info("💾 Saving aggregated dashboard data...")
        
        try:
            # Save complete aggregated data
            data_path = os.path.join(self.results_path, 'dashboard_data.json')
            dashboard_data = {
                'timestamp': self.dashboard_timestamp.isoformat(),
                'modules_data': self.modules_data,
                'aggregated_stats': self.aggregated_stats,
                'scan_info': {
                    'base_path': self.base_path,
                    'results_path': self.results_path,
                    'modules_found': len(self.modules_data)
                }
            }
            
            with open(data_path, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            logger.info(f"  💾 Dashboard data saved to: {data_path}")
            
        except Exception as e:
            logger.error(f"  ❌ Error saving dashboard data: {str(e)}")
    
    def generate_complete_dashboard(self) -> Dict[str, str]:
        """
        Complete dashboard generation workflow
        
        Returns:
            Dictionary with paths to generated files
        """
        logger.info("🚀 Starting complete dashboard generation...")
        
        try:
            # Step 1: Scan module results
            self.scan_module_results()
            
            # Step 2: Generate aggregated statistics
            self.generate_aggregated_statistics()
            
            # Step 3: Clean up old results
            self.cleanup_old_results()
            
            # Step 4: Create interactive dashboard
            interactive_path = self.create_interactive_dashboard()
            
            # Step 5: Generate HTML report
            html_path = self.generate_html_report()
            
            # Step 6: Save aggregated data
            self.save_aggregated_data()
            
            result_paths = {
                'interactive_dashboard': interactive_path,
                'html_report': html_path,
                'results_directory': self.results_path
            }
            
            logger.info("✅ Dashboard generation completed successfully!")
            logger.info(f"📁 Results available in: {os.path.abspath(self.results_path)}")
            
            return result_paths
            
        except Exception as e:
            logger.error(f"❌ Error in dashboard generation: {str(e)}")
            return {}


def main():
    """
    Main function to generate the central dashboard
    """
    print("🚀 Starting Central Dashboard Generation")
    print("=" * 60)
    
    # Initialize dashboard
    dashboard = CentralDashboard(
        test_machine_learning_path="./",
        results_path="./Dashboard_Results/",
        max_results_age_days=7
    )
    
    # Generate complete dashboard
    results = dashboard.generate_complete_dashboard()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 DASHBOARD GENERATION SUMMARY")
    print("=" * 60)
    
    if results:
        print(f"✅ Interactive Dashboard: {results.get('interactive_dashboard', 'Not generated')}")
        print(f"✅ HTML Report: {results.get('html_report', 'Not generated')}")
        print(f"📁 Results Directory: {results.get('results_directory', 'Not available')}")
        
        print(f"\n📊 Statistics:")
        print(f"  📦 Modules Scanned: {dashboard.aggregated_stats.get('total_modules', 0)}")
        print(f"  🧪 Total Tests: {dashboard.aggregated_stats.get('total_tests_run', 0)}")
        print(f"  🎯 Overall Success Rate: {dashboard.aggregated_stats.get('overall_success_rate', 0):.1f}%")
        print(f"  📈 Total Plots: {dashboard.aggregated_stats.get('total_plots_generated', 0)}")
        
        print(f"\n🌐 Access your dashboard:")
        print(f"  Main Dashboard: file://{os.path.abspath(results.get('html_report', ''))}")
        print(f"  Interactive View: file://{os.path.abspath(results.get('interactive_dashboard', ''))}")
    else:
        print("❌ Dashboard generation failed")
    
    print("\n✅ Central Dashboard completed!")
    return results


if __name__ == "__main__":
    main()
