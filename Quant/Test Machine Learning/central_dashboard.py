"""
Central Dashboard for Machine Learning Testing Results
=====================================================

Professional dashboard system that aggregates and displays testing results
from all ML modules in a unified, interactive interface.

Features:
- Automatic discovery of test results from Results Machine Learning
- Interactive visualizations with Plotly
- Performance metrics aggregation
- Module comparison and analysis
- Professional HTML reports
- Efficient caching and optimization

Author: Advanced ML Finance Team
Date: July 2025
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available - using basic HTML reports")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("Pandas not available - using basic data structures")

class CentralDashboard:
    """
    Central Dashboard System for ML Testing Results
    
    This class provides a unified interface for viewing and analyzing
    test results from all ML modules in the system.
    """
    
    def __init__(self, results_base_path: str = None):
        """
        Initialize the central dashboard
        
        Args:
            results_base_path: Base path for Results Machine Learning directory
        """
        # Set up paths
        if results_base_path is None:
            current_dir = Path(__file__).parent
            self.results_base_path = current_dir.parent / "Results Machine Learning"
        else:
            self.results_base_path = Path(results_base_path)
        
        self.dashboard_path = Path(__file__).parent / "Dashboard_Results"
        self.dashboard_path.mkdir(exist_ok=True)
        
        # Initialize data structures
        self.modules_data = {}
        self.summary_stats = {}
        self.last_update = None
        
        logger.info(f"📊 Central Dashboard initialized")
        logger.info(f"📁 Results path: {self.results_base_path}")
        logger.info(f"📁 Dashboard path: {self.dashboard_path}")
        
    def scan_results(self) -> Dict[str, Any]:
        """
        Scan the Results Machine Learning directory for test results
        
        Returns:
            Dictionary containing all discovered test results
        """
        logger.info("🔍 Scanning for test results...")
        
        if not self.results_base_path.exists():
            logger.warning(f"⚠️ Results directory not found: {self.results_base_path}")
            return {}
        
        discovered_modules = {}
        
        # Scan for JSON result files directly in results directory
        for result_file in self.results_base_path.glob("*.json"):
            try:
                # Parse filename to extract module info
                filename = result_file.stem
                if filename.startswith("results_"):
                    parts = filename.split("_")
                    if len(parts) >= 3:
                        module_name = parts[1]  # data_structures, util, labeling, multi_product
                        test_type = parts[2]    # complete, specific test name
                        
                        logger.info(f"  📋 Found results: {module_name}/{test_type}")
                        
                        # Load and parse result file
                        with open(result_file, 'r') as f:
                            result_data = json.load(f)
                        
                        # Initialize module if not exists
                        if module_name not in discovered_modules:
                            discovered_modules[module_name] = {
                                'name': module_name,
                                'tests': {},
                                'summary': {},
                                'last_updated': None
                            }
                        
                        # Add test results
                        discovered_modules[module_name]['tests'][test_type] = {
                            'file': str(result_file),
                            'data': result_data,
                            'timestamp': result_file.stat().st_mtime
                        }
                        
                        # Update last updated timestamp
                        file_time = datetime.fromtimestamp(result_file.stat().st_mtime)
                        if (discovered_modules[module_name]['last_updated'] is None or 
                            file_time > discovered_modules[module_name]['last_updated']):
                            discovered_modules[module_name]['last_updated'] = file_time
                        
            except Exception as e:
                logger.warning(f"Error processing result file {result_file}: {e}")
        
        # Scan all subdirectories for legacy format
        for category_dir in self.results_base_path.iterdir():
            if category_dir.is_dir():
                logger.info(f"  📂 Scanning category: {category_dir.name}")
                
                # Scan for test result directories
                for module_dir in category_dir.iterdir():
                    if module_dir.is_dir() and module_dir.name.startswith("test_results_"):
                        module_name = module_dir.name.replace("test_results_", "")
                        logger.info(f"    📋 Found module: {module_name}")
                        
                        # Load module data
                        module_data = self._load_module_data(module_dir)
                        if module_data:
                            discovered_modules[f"{category_dir.name}/{module_name}"] = module_data
        
        self.modules_data = discovered_modules
        self.last_update = datetime.now()
        
        logger.info(f"✅ Discovered {len(discovered_modules)} test modules")
        return discovered_modules
    
    def _load_module_data(self, module_dir: Path) -> Optional[Dict[str, Any]]:
        """
        Load data from a specific module directory
        
        Args:
            module_dir: Path to the module results directory
            
        Returns:
            Dictionary containing module data or None if loading fails
        """
        try:
            module_data = {
                'name': module_dir.name.replace("test_results_", ""),
                'path': str(module_dir),
                'files': {},
                'plots': [],
                'reports': {},
                'summary': {}
            }
            
            # Load summary file
            summary_file = module_dir / "test_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    module_data['summary'] = json.load(f)
            
            # Load detailed report
            report_file = module_dir / "test_report.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    module_data['reports']['detailed'] = json.load(f)
            
            # Find plot files
            for file_path in module_dir.glob("*.png"):
                module_data['plots'].append(str(file_path))
            
            # Find HTML files
            for file_path in module_dir.glob("*.html"):
                module_data['files'][file_path.stem] = str(file_path)
            
            # Get file timestamps
            module_data['last_modified'] = max(
                [f.stat().st_mtime for f in module_dir.iterdir() if f.is_file()],
                default=0
            )
            
            return module_data
            
        except Exception as e:
            logger.error(f"❌ Error loading module data from {module_dir}: {e}")
            return None
    
    def generate_summary_stats(self) -> Dict[str, Any]:
        """
        Generate summary statistics from all modules
        
        Returns:
            Dictionary containing aggregated statistics
        """
        logger.info("📊 Generating summary statistics...")
        
        if not self.modules_data:
            logger.warning("⚠️ No module data available for statistics")
            return {}
        
        stats = {
            'total_modules': len(self.modules_data),
            'total_tests': 0,
            'total_passed': 0,
            'total_failed': 0,
            'total_plots': 0,
            'modules_by_category': {},
            'performance_summary': {},
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
        
        # Aggregate data from all modules
        for module_key, module_data in self.modules_data.items():
            category = module_key.split('/')[0]
            
            # Count by category
            if category not in stats['modules_by_category']:
                stats['modules_by_category'][category] = 0
            stats['modules_by_category'][category] += 1
            
            # Aggregate test results
            if 'summary' in module_data and module_data['summary']:
                summary = module_data['summary']
                stats['total_tests'] += summary.get('tests_run', 0)
                stats['total_passed'] += summary.get('tests_passed', 0)
                stats['total_failed'] += summary.get('tests_failed', 0)
                stats['total_plots'] += summary.get('plots_generated', 0)
        
        # Calculate success rate
        if stats['total_tests'] > 0:
            stats['success_rate'] = (stats['total_passed'] / stats['total_tests']) * 100
        else:
            stats['success_rate'] = 0
        
        self.summary_stats = stats
        logger.info(f"✅ Generated statistics for {stats['total_modules']} modules")
        
        return stats
    
    def create_professional_dashboard(self) -> str:
        """
        Create a professional HTML dashboard with modern design
        
        Returns:
            Path to the generated dashboard HTML file
        """
        logger.info("🎨 Creating professional dashboard...")
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ML Testing Dashboard - Sistema de Datos</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                
                .dashboard-container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 20px;
                    backdrop-filter: blur(10px);
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                    position: relative;
                }}
                
                .header::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 100" fill="white" opacity="0.1"><path d="M0,50 Q250,0 500,50 T1000,50 V100 H0 Z"/></svg>');
                    background-size: cover;
                }}
                
                .header-content {{
                    position: relative;
                    z-index: 2;
                }}
                
                .main-title {{
                    font-size: 3em;
                    font-weight: 300;
                    margin-bottom: 10px;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                }}
                
                .subtitle {{
                    font-size: 1.2em;
                    opacity: 0.9;
                    margin-bottom: 20px;
                }}
                
                .update-time {{
                    font-size: 0.9em;
                    opacity: 0.8;
                    background: rgba(255,255,255,0.2);
                    padding: 8px 16px;
                    border-radius: 20px;
                    display: inline-block;
                }}
                
                .content {{
                    padding: 40px;
                }}
                
                .stats-overview {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 30px;
                    margin-bottom: 50px;
                }}
                
                .stat-card {{
                    background: white;
                    padding: 30px;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                    text-align: center;
                    position: relative;
                    overflow: hidden;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}
                
                .stat-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 20px 40px rgba(0,0,0,0.15);
                }}
                
                .stat-card::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 4px;
                    background: linear-gradient(90deg, #667eea, #764ba2);
                }}
                
                .stat-icon {{
                    font-size: 3em;
                    margin-bottom: 15px;
                    opacity: 0.8;
                }}
                
                .stat-number {{
                    font-size: 3em;
                    font-weight: bold;
                    color: #333;
                    margin: 15px 0;
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }}
                
                .stat-label {{
                    color: #666;
                    font-size: 1.1em;
                    font-weight: 500;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                
                .modules-section {{
                    margin-bottom: 50px;
                }}
                
                .section-title {{
                    font-size: 2.5em;
                    color: #333;
                    margin-bottom: 30px;
                    text-align: center;
                    position: relative;
                }}
                
                .section-title::after {{
                    content: '';
                    position: absolute;
                    bottom: -10px;
                    left: 50%;
                    transform: translateX(-50%);
                    width: 100px;
                    height: 4px;
                    background: linear-gradient(90deg, #667eea, #764ba2);
                    border-radius: 2px;
                }}
                
                .modules-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                    gap: 30px;
                }}
                
                .module-card {{
                    background: white;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                    overflow: hidden;
                    position: relative;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}
                
                .module-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 20px 40px rgba(0,0,0,0.15);
                }}
                
                .module-header {{
                    background: linear-gradient(135deg, #28a745, #20c997);
                    color: white;
                    padding: 25px;
                    position: relative;
                }}
                
                .module-name {{
                    font-size: 1.5em;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                
                .module-category {{
                    opacity: 0.9;
                    font-size: 0.9em;
                }}
                
                .module-body {{
                    padding: 25px;
                }}
                
                .module-stats {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 20px;
                    margin-bottom: 25px;
                }}
                
                .module-stat {{
                    text-align: center;
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 10px;
                }}
                
                .module-stat-number {{
                    font-size: 1.8em;
                    font-weight: bold;
                    color: #28a745;
                    margin-bottom: 5px;
                }}
                
                .module-stat-label {{
                    color: #666;
                    font-size: 0.8em;
                    text-transform: uppercase;
                }}
                
                .success-badge {{
                    background: linear-gradient(135deg, #28a745, #20c997);
                    color: white;
                    padding: 8px 16px;
                    border-radius: 25px;
                    font-size: 0.9em;
                    font-weight: bold;
                    display: inline-block;
                    margin-bottom: 20px;
                }}
                
                .plots-section {{
                    margin-top: 20px;
                }}
                
                .plots-title {{
                    font-weight: bold;
                    color: #333;
                    margin-bottom: 15px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                
                .plot-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 10px;
                }}
                
                .plot-link {{
                    display: block;
                    padding: 12px 16px;
                    background: linear-gradient(135deg, #007bff, #0056b3);
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                    font-size: 0.85em;
                    text-align: center;
                    transition: all 0.3s ease;
                    font-weight: 500;
                }}
                
                .plot-link:hover {{
                    background: linear-gradient(135deg, #0056b3, #004085);
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0,123,255,0.3);
                }}
                
                .report-link {{
                    background: linear-gradient(135deg, #6f42c1, #5a2d8f);
                }}
                
                .report-link:hover {{
                    background: linear-gradient(135deg, #5a2d8f, #4a246b);
                    box-shadow: 0 5px 15px rgba(111,66,193,0.3);
                }}
                
                .footer {{
                    background: #2c3e50;
                    color: white;
                    padding: 40px;
                    text-align: center;
                }}
                
                .footer-content {{
                    max-width: 800px;
                    margin: 0 auto;
                }}
                
                .footer-title {{
                    font-size: 1.5em;
                    margin-bottom: 15px;
                }}
                
                .footer-description {{
                    opacity: 0.9;
                    line-height: 1.6;
                }}
                
                @media (max-width: 768px) {{
                    .main-title {{
                        font-size: 2em;
                    }}
                    
                    .stats-overview {{
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                    }}
                    
                    .modules-grid {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .content {{
                        padding: 20px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <div class="header">
                    <div class="header-content">
                        <h1 class="main-title">🚀 ML Testing Dashboard</h1>
                        <p class="subtitle">Sistema de Análisis y Monitoreo de Machine Learning</p>
                        <div class="update-time">
                            📅 Última actualización: {self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else 'Nunca'}
                        </div>
                    </div>
                </div>
                
                <div class="content">
                    <div class="stats-overview">
                        <div class="stat-card">
                            <div class="stat-icon">📊</div>
                            <div class="stat-number">{self.summary_stats.get('total_modules', 0)}</div>
                            <div class="stat-label">Módulos Analizados</div>
                        </div>
                        
                        <div class="stat-card">
                            <div class="stat-icon">🧪</div>
                            <div class="stat-number">{self.summary_stats.get('total_tests', 0)}</div>
                            <div class="stat-label">Tests Ejecutados</div>
                        </div>
                        
                        <div class="stat-card">
                            <div class="stat-icon">✅</div>
                            <div class="stat-number">{self.summary_stats.get('success_rate', 0):.1f}%</div>
                            <div class="stat-label">Tasa de Éxito</div>
                        </div>
                        
                        <div class="stat-card">
                            <div class="stat-icon">📈</div>
                            <div class="stat-number">{self.summary_stats.get('total_plots', 0)}</div>
                            <div class="stat-label">Gráficos Generados</div>
                        </div>
                    </div>
                    
                    <div class="modules-section">
                        <h2 class="section-title">📋 Análisis por Módulo</h2>
                        <div class="modules-grid">
        """
        
        # Add module cards
        for module_key, module_data in self.modules_data.items():
            category, module_name = module_key.split('/', 1)
            summary = module_data.get('summary', {})
            tests_run = summary.get('tests_run', 0)
            tests_passed = summary.get('tests_passed', 0)
            tests_failed = summary.get('tests_failed', 0)
            success_rate = (tests_passed / tests_run * 100) if tests_run > 0 else 0
            plots_count = len(module_data.get('plots', []))
            
            html_content += f"""
                            <div class="module-card">
                                <div class="module-header">
                                    <div class="module-name">🔬 {module_name.replace('_', ' ').title()}</div>
                                    <div class="module-category">📂 {category}</div>
                                </div>
                                <div class="module-body">
                                    <div class="success-badge">
                                        ✨ {success_rate:.1f}% Éxito
                                    </div>
                                    
                                    <div class="module-stats">
                                        <div class="module-stat">
                                            <div class="module-stat-number">{tests_run}</div>
                                            <div class="module-stat-label">Tests</div>
                                        </div>
                                        <div class="module-stat">
                                            <div class="module-stat-number">{tests_passed}</div>
                                            <div class="module-stat-label">Pasados</div>
                                        </div>
                                        <div class="module-stat">
                                            <div class="module-stat-number">{plots_count}</div>
                                            <div class="module-stat-label">Gráficos</div>
                                        </div>
                                    </div>
                                    
                                    <div class="plots-section">
                                        <div class="plots-title">
                                            📊 Visualizaciones y Reportes
                                        </div>
                                        <div class="plot-grid">
            """
            
            # Add plot links
            for plot_path in module_data.get('plots', []):
                plot_name = os.path.basename(plot_path).replace('_', ' ').replace('.png', '').title()
                relative_path = os.path.relpath(plot_path, self.dashboard_path)
                html_content += f'<a href="{relative_path}" class="plot-link" target="_blank">📈 {plot_name}</a>'
            
            # Add report links
            for report_name, report_path in module_data.get('files', {}).items():
                if 'report' in report_name or 'dashboard' in report_name:
                    relative_path = os.path.relpath(report_path, self.dashboard_path)
                    display_name = report_name.replace('_', ' ').title()
                    html_content += f'<a href="{relative_path}" class="plot-link report-link" target="_blank">📄 {display_name}</a>'
            
            html_content += """
                                        </div>
                                    </div>
                                </div>
                            </div>
            """
        
        html_content += f"""
                        </div>
                    </div>
                </div>
                
                <div class="footer">
                    <div class="footer-content">
                        <h3 class="footer-title">🔧 Sistema de Testing Avanzado</h3>
                        <p class="footer-description">
                            Dashboard profesional para el monitoreo y análisis de resultados de testing 
                            en sistemas de Machine Learning financiero. Generado automáticamente para 
                            proporcionar insights en tiempo real sobre la calidad y rendimiento del código.
                        </p>
                        <p style="margin-top: 20px; opacity: 0.8;">
                            📊 {len(self.modules_data)} módulos • 
                            🧪 {self.summary_stats.get('total_tests', 0)} tests • 
                            📈 {self.summary_stats.get('total_plots', 0)} visualizaciones
                        </p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Save dashboard
        dashboard_file = self.dashboard_path / "professional_dashboard.html"
        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✅ Professional dashboard saved to: {dashboard_file}")
        return str(dashboard_file)
    
    def generate_full_report(self) -> Dict[str, str]:
        """
        Generate a comprehensive report including all dashboards and summaries
        
        Returns:
            Dictionary containing paths to all generated files
        """
        logger.info("📊 Generating comprehensive dashboard report...")
        
        start_time = time.time()
        
        # Scan for results
        self.scan_results()
        
        # Generate statistics
        self.generate_summary_stats()
        
        # Create dashboards
        generated_files = {}
        
        # Professional dashboard (always available)
        professional_path = self.create_professional_dashboard()
        generated_files['professional_dashboard'] = professional_path
        
        # Save summary statistics
        summary_file = self.dashboard_path / "summary_statistics.json"
        with open(summary_file, 'w') as f:
            json.dump(self.summary_stats, f, indent=2, default=str)
        generated_files['summary_stats'] = str(summary_file)
        
        # Save modules data
        modules_file = self.dashboard_path / "modules_data.json"
        with open(modules_file, 'w') as f:
            json.dump(self.modules_data, f, indent=2, default=str)
        generated_files['modules_data'] = str(modules_file)
        
        execution_time = time.time() - start_time
        
        logger.info(f"✅ Dashboard generation completed in {execution_time:.2f} seconds")
        logger.info(f"📁 Generated files: {len(generated_files)}")
        
        for file_type, file_path in generated_files.items():
            logger.info(f"  📄 {file_type}: {file_path}")
        
        return generated_files
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate comprehensive HTML report with all test results
        Designed to work with master_test_runner.py
        
        :return: Path to generated HTML file
        """
        logger.info("📊 Generating comprehensive dashboard report...")
        
        # Load test results from master test runner
        self._load_master_test_results()
        
        # Check if we have any results
        if not self.test_results:
            logger.warning("⚠️  No test results found")
            return self._generate_empty_report()
        
        # Generate HTML content
        html_content = self._generate_master_html_report()
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_path / f"comprehensive_dashboard_{timestamp}.html"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"✅ Dashboard report generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"❌ Error saving dashboard report: {e}")
            return ""
    
    def _load_master_test_results(self):
        """Load test results from master test runner execution"""
        try:
            # Look for execution summary files
            summary_files = list(self.results_path.glob("master_execution_summary_*.json"))
            
            if not summary_files:
                logger.warning("⚠️  No master execution summary files found")
                return
            
            # Get the most recent summary
            latest_summary = max(summary_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_summary, 'r') as f:
                execution_data = json.load(f)
            
            # Extract test results
            self.test_results = execution_data.get('results', {})
            
            logger.info(f"✅ Loaded test results from: {latest_summary}")
            
        except Exception as e:
            logger.error(f"❌ Error loading master test results: {e}")
            self.test_results = {}
    
    def _generate_master_html_report(self) -> str:
        """Generate the complete HTML report for master test runner"""
        
        # Calculate overall statistics
        total_modules = len(self.test_results)
        successful_modules = len([r for r in self.test_results.values() if r.get('success', False)])
        total_execution_time = sum([r.get('execution_time', 0) for r in self.test_results.values()])
        
        overall_success_rate = (successful_modules / total_modules * 100) if total_modules > 0 else 0
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🚀 ML Testing Dashboard</title>
            <style>
                {self._get_master_dashboard_css()}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <header class="dashboard-header">
                    <h1>🚀 Machine Learning Testing Dashboard</h1>
                    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p class="subtitle">Professional ML Module Testing & Validation System</p>
                </header>
                
                <div class="summary-section">
                    <h2>📊 Overall Summary</h2>
                    <div class="summary-stats">
                        <div class="summary-card">
                            <div class="card-icon">🎯</div>
                            <h3>Modules Tested</h3>
                            <div class="summary-value">{total_modules}</div>
                        </div>
                        <div class="summary-card">
                            <div class="card-icon">✅</div>
                            <h3>Successful Modules</h3>
                            <div class="summary-value">{successful_modules}</div>
                        </div>
                        <div class="summary-card">
                            <div class="card-icon">⏱️</div>
                            <h3>Total Execution Time</h3>
                            <div class="summary-value">{total_execution_time:.2f}s</div>
                        </div>
                        <div class="summary-card">
                            <div class="card-icon">📈</div>
                            <h3>Success Rate</h3>
                            <div class="summary-value">{overall_success_rate:.1f}%</div>
                        </div>
                    </div>
                </div>
                
                <div class="modules-section">
                    <h2>🔧 Module Details</h2>
                    {''.join([self._generate_master_module_section(module, results) for module, results in self.test_results.items()])}
                </div>
                
                <footer class="dashboard-footer">
                    <p>🤖 Dashboard generated by Advanced ML Testing System</p>
                    <p>Sistema de Datos - Quant Analysis Team</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _generate_master_module_section(self, module_name: str, results: Dict[str, Any]) -> str:
        """Generate HTML section for a specific module from master test runner"""
        
        success = results.get('success', False)
        execution_time = results.get('execution_time', 0)
        message = results.get('message', 'No message')
        output = results.get('output', '')
        error = results.get('error', '')
        
        # Status color and emoji
        if success:
            status_color = "green"
            status_emoji = "🟢"
            status_text = "PASSED"
        else:
            status_color = "red"
            status_emoji = "🔴"
            status_text = "FAILED"
        
        # Module icon
        module_icons = {
            'data_structures': '📊',
            'util': '🔧',
            'labeling': '🏷️',
            'multi_product': '📈'
        }
        module_icon = module_icons.get(module_name, '📦')
        
        # Performance rating
        if success and execution_time < 10:
            performance_rating = "⭐⭐⭐⭐⭐"
        elif success and execution_time < 30:
            performance_rating = "⭐⭐⭐⭐"
        elif success and execution_time < 60:
            performance_rating = "⭐⭐⭐"
        elif success:
            performance_rating = "⭐⭐"
        else:
            performance_rating = "⭐"
        
        html = f"""
        <div class="module-section">
            <div class="module-header">
                <h3 class="module-title">{module_icon} {module_name.replace('_', ' ').title()}</h3>
                <div class="module-status status-{status_color}">
                    {status_emoji} {status_text}
                </div>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-icon">⏱️</div>
                    <h4>Execution Time</h4>
                    <div class="stat-value">{execution_time:.2f}s</div>
                    <div class="stat-label">Total</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-icon">📊</div>
                    <h4>Status</h4>
                    <div class="stat-value status-{status_color}">{status_text}</div>
                    <div class="stat-label">Result</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-icon">🎯</div>
                    <h4>Performance</h4>
                    <div class="stat-value">{performance_rating}</div>
                    <div class="stat-label">Rating</div>
                </div>
                
                <div class="stat-card">
                    <div class="stat-icon">📝</div>
                    <h4>Return Code</h4>
                    <div class="stat-value">{results.get('return_code', 'N/A')}</div>
                    <div class="stat-label">Exit Code</div>
                </div>
            </div>
            
            <div class="details-section">
                <h4>📋 Execution Details</h4>
                <div class="execution-details">
                    <p><strong>Message:</strong> {message}</p>
                    
                    {f'''
                    <div class="output-section">
                        <h5>📤 Output</h5>
                        <pre class="output-text">{output[:1000]}{"..." if len(output) > 1000 else ""}</pre>
                    </div>
                    ''' if output else ''}
                    
                    {f'''
                    <div class="error-section">
                        <h5>❌ Error</h5>
                        <pre class="error-text">{error[:1000]}{"..." if len(error) > 1000 else ""}</pre>
                    </div>
                    ''' if error else ''}
                </div>
            </div>
            
            <div class="charts-section">
                <h4>📈 Performance Charts</h4>
                <div class="chart-container">
                    {self._generate_performance_charts(module_name, results)}
                </div>
            </div>
        </div>
        """
        
        return html
    
    def _generate_empty_report(self) -> str:
        """Generate empty dashboard when no results are available"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_path / f"empty_dashboard_{timestamp}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🚀 ML Testing Dashboard - No Results</title>
            <style>
                {self._get_master_dashboard_css()}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <header class="dashboard-header">
                    <h1>🚀 Machine Learning Testing Dashboard</h1>
                    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </header>
                
                <div class="empty-state">
                    <div class="empty-icon">📊</div>
                    <h2>No Test Results Available</h2>
                    <p>Run the master test runner to generate results and populate this dashboard.</p>
                    <div class="empty-actions">
                        <code>python master_test_runner.py</code>
                    </div>
                </div>
                
                <footer class="dashboard-footer">
                    <p>🤖 Dashboard generated by Advanced ML Testing System</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"📊 Empty dashboard generated: {report_file}")
            return str(report_file)
            
        except Exception as e:
            logger.error(f"❌ Error saving empty dashboard: {e}")
            return ""
    
    def _get_master_dashboard_css(self) -> str:
        """Get CSS styles for the master dashboard"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
            line-height: 1.6;
        }
        
        .dashboard-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .dashboard-header {
            text-align: center;
            background: rgba(255,255,255,0.95);
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .dashboard-header h1 {
            font-size: 2.8em;
            color: #2c3e50;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .timestamp {
            color: #7f8c8d;
            font-size: 1.2em;
            margin-bottom: 5px;
        }
        
        .subtitle {
            color: #34495e;
            font-size: 1.1em;
            font-weight: 500;
        }
        
        .summary-section {
            background: rgba(255,255,255,0.95);
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .summary-section h2 {
            color: #2c3e50;
            margin-bottom: 25px;
            font-size: 2em;
            font-weight: 600;
        }
        
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }
        
        .summary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            position: relative;
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .summary-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }
        
        .card-icon {
            font-size: 3em;
            margin-bottom: 15px;
            opacity: 0.9;
        }
        
        .summary-card h3 {
            font-size: 1.2em;
            margin-bottom: 10px;
            opacity: 0.9;
        }
        
        .summary-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-top: 10px;
        }
        
        .modules-section {
            background: rgba(255,255,255,0.95);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .modules-section h2 {
            color: #2c3e50;
            margin-bottom: 30px;
            font-size: 2em;
            font-weight: 600;
        }
        
        .module-section {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            border: 1px solid #e9ecef;
        }
        
        .module-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e9ecef;
        }
        
        .module-title {
            color: #2c3e50;
            font-size: 1.8em;
            font-weight: 600;
        }
        
        .module-status {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 1.1em;
        }
        
        .status-green { background: #d4edda; color: #155724; }
        .status-red { background: #f8d7da; color: #721c24; }
        .status-orange { background: #fff3cd; color: #856404; }
        .status-blue { background: #cce5ff; color: #004085; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #dee2e6;
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .stat-icon {
            font-size: 2.5em;
            margin-bottom: 15px;
            opacity: 0.7;
        }
        
        .stat-card h4 {
            color: #6c757d;
            margin-bottom: 10px;
            font-size: 1em;
        }
        
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .stat-label {
            color: #6c757d;
            font-size: 0.9em;
        }
        
        .details-section {
            margin-bottom: 30px;
        }
        
        .details-section h4 {
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .execution-details {
            background: white;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #dee2e6;
        }
        
        .output-section, .error-section {
            margin-top: 20px;
        }
        
        .output-section h5 {
            color: #28a745;
            margin-bottom: 10px;
        }
        
        .error-section h5 {
            color: #dc3545;
            margin-bottom: 10px;
        }
        
        .output-text, .error-text {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            font-size: 0.9em;
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid #dee2e6;
        }
        
        .charts-section h4 {
            color: #495057;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .chart-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .chart-item {
            text-align: center;
            background: white;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #dee2e6;
        }
        
        .chart-image {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        
        .chart-caption {
            margin-top: 10px;
            color: #6c757d;
            font-size: 0.9em;
        }
        
        .no-charts {
            text-align: center;
            padding: 40px;
            color: #6c757d;
        }
        
        .chart-info {
            font-size: 0.9em;
            margin-top: 10px;
        }
        
        .empty-state {
            text-align: center;
            padding: 80px 40px;
            background: rgba(255,255,255,0.95);
            border-radius: 20px;
            margin: 40px 0;
        }
        
        .empty-icon {
            font-size: 5em;
            margin-bottom: 20px;
            opacity: 0.5;
        }
        
        .empty-state h2 {
            color: #495057;
            margin-bottom: 15px;
        }
        
        .empty-actions {
            margin-top: 30px;
        }
        
        .empty-actions code {
            background: #f8f9fa;
            padding: 10px 20px;
            border-radius: 5px;
            font-size: 1.1em;
            border: 1px solid #dee2e6;
        }
        
        .dashboard-footer {
            text-align: center;
            padding: 30px;
            color: rgba(255,255,255,0.8);
            margin-top: 40px;
        }
        
        .dashboard-footer p {
            margin: 5px 0;
        }
        
        @media (max-width: 768px) {
            .dashboard-container {
                padding: 10px;
            }
            
            .dashboard-header {
                padding: 20px;
            }
            
            .dashboard-header h1 {
                font-size: 2em;
            }
            
            .summary-stats {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: 1fr;
            }
            
            .module-header {
                flex-direction: column;
                gap: 15px;
            }
        }
        """


def main():
    """
    Main function to generate the central dashboard
    """
    print("🚀 Starting Central Dashboard Generation")
    print("=" * 60)
    
    # Initialize dashboard
    dashboard = CentralDashboard()
    
    # Generate comprehensive report
    generated_files = dashboard.generate_full_report()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 DASHBOARD SUMMARY")
    print("=" * 60)
    print(f"Módulos Analizados: {dashboard.summary_stats.get('total_modules', 0)}")
    print(f"Total Tests: {dashboard.summary_stats.get('total_tests', 0)}")
    print(f"Tasa de Éxito: {dashboard.summary_stats.get('success_rate', 0):.1f}%")
    print(f"Gráficos Generados: {dashboard.summary_stats.get('total_plots', 0)}")
    
    print(f"\n📁 Archivos Generados:")
    for file_type, file_path in generated_files.items():
        print(f"  📄 {file_type}: {file_path}")
    
    print(f"\n🌐 Abrir el dashboard:")
    print(f"  📊 Dashboard Profesional: {generated_files.get('professional_dashboard', 'No generado')}")
    
    print("\n✅ Generación del dashboard completada exitosamente!")
    
    return generated_files


if __name__ == "__main__":
    main()
