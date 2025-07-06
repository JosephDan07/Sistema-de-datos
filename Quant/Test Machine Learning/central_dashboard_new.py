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
        
        # Scan all subdirectories
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
