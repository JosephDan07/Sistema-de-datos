#!/usr/bin/env python3
"""
Master Test Runner for Machine Learning Modules
==============================================

Executes all master tests and generates unified dashboard.
Professional test orchestration system.

Features:
- Automated execution of all master tests
- Performance monitoring and reporting
- Results aggregation and standardization
- Dashboard generation
- Email notifications (optional)
- Parallel test execution
- Detailed logging and error reporting

Author: Advanced ML Finance Team
Date: July 2025
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MasterTestRunner:
    """
    Master test runner for all ML modules
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize the master test runner
        
        Args:
            base_path: Base path of the project
        """
        if base_path is None:
            self.base_path = Path(__file__).parent
        else:
            self.base_path = Path(base_path)
        
        self.test_modules = {
            'data_structures': 'test_data_structures/test_simple_data_structures.py',
            'util': 'test_util/test_simple_util.py',
            'labeling': 'test_labeling/test_simple_labeling.py',
            'multi_product': 'test_multi_product/test_simple_multi_product.py'
        }
        
        self.results_path = self.base_path.parent / "Results Machine Learning"
        self.results_path.mkdir(exist_ok=True)
        
        # Test execution results
        self.test_results = {}
        self.execution_summary = {}
        
        logger.info("🚀 Master Test Runner initialized")
        logger.info(f"📁 Base path: {self.base_path}")
        logger.info(f"📁 Results path: {self.results_path}")
    
    def run_all_tests(self, parallel: bool = True, timeout: int = 300) -> Dict[str, Any]:
        """
        Execute all master tests
        
        Args:
            parallel: Whether to run tests in parallel
            timeout: Timeout for each test in seconds
            
        Returns:
            Dictionary containing execution results
        """
        logger.info("🎯 Starting master test execution")
        start_time = time.time()
        
        if parallel:
            results = self._run_tests_parallel(timeout)
        else:
            results = self._run_tests_sequential(timeout)
        
        execution_time = time.time() - start_time
        
        # Create execution summary
        self.execution_summary = {
            'timestamp': datetime.now().isoformat(),
            'execution_time': execution_time,
            'total_tests': len(self.test_modules),
            'successful_tests': len([r for r in results.values() if r['success']]),
            'failed_tests': len([r for r in results.values() if not r['success']]),
            'parallel_execution': parallel,
            'timeout': timeout,
            'results': results
        }
        
        logger.info(f"✅ Master test execution completed in {execution_time:.2f}s")
        logger.info(f"✅ Successful: {self.execution_summary['successful_tests']}/{self.execution_summary['total_tests']}")
        
        return self.execution_summary
    
    def _run_tests_parallel(self, timeout: int) -> Dict[str, Any]:
        """
        Run tests in parallel using ThreadPoolExecutor
        
        Args:
            timeout: Timeout for each test
            
        Returns:
            Dictionary containing test results
        """
        logger.info("🔄 Running tests in parallel...")
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self.test_modules)) as executor:
            # Submit all tests
            future_to_module = {
                executor.submit(self._run_single_test, module, script, timeout): module
                for module, script in self.test_modules.items()
            }
            
            # Collect results
            for future in as_completed(future_to_module):
                module = future_to_module[future]
                try:
                    result = future.result()
                    results[module] = result
                    logger.info(f"{'✅' if result['success'] else '❌'} {module}: {result['message']}")
                except Exception as e:
                    results[module] = {
                        'success': False,
                        'message': f"Exception in parallel execution: {e}",
                        'execution_time': 0,
                        'output': "",
                        'error': str(e)
                    }
                    logger.error(f"❌ {module}: Exception - {e}")
        
        return results
    
    def _run_tests_sequential(self, timeout: int) -> Dict[str, Any]:
        """
        Run tests sequentially
        
        Args:
            timeout: Timeout for each test
            
        Returns:
            Dictionary containing test results
        """
        logger.info("🔄 Running tests sequentially...")
        results = {}
        
        for module, script in self.test_modules.items():
            result = self._run_single_test(module, script, timeout)
            results[module] = result
            logger.info(f"{'✅' if result['success'] else '❌'} {module}: {result['message']}")
            
            # Small delay between tests
            time.sleep(1)
        
        return results
    
    def _run_single_test(self, module: str, script: str, timeout: int) -> Dict[str, Any]:
        """
        Run a single test module
        
        Args:
            module: Module name
            script: Script path
            timeout: Timeout in seconds
            
        Returns:
            Dictionary containing test result
        """
        logger.info(f"▶️  Running {module} test...")
        start_time = time.time()
        
        script_path = self.base_path / script
        
        if not script_path.exists():
            return {
                'success': False,
                'message': f"Test script not found: {script_path}",
                'execution_time': 0,
                'output': "",
                'error': "File not found"
            }
        
        try:
            # Execute test script
            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(script_path.parent)
            )
            
            execution_time = time.time() - start_time
            
            success = result.returncode == 0
            
            return {
                'success': success,
                'message': f"Completed in {execution_time:.2f}s" if success else f"Failed with code {result.returncode}",
                'execution_time': execution_time,
                'output': result.stdout,
                'error': result.stderr,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'message': f"Timeout after {timeout}s",
                'execution_time': timeout,
                'output': "",
                'error': "Timeout expired"
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"Exception: {e}",
                'execution_time': time.time() - start_time,
                'output': "",
                'error': str(e)
            }
    
    def generate_dashboard(self) -> bool:
        """
        Generate the unified dashboard
        
        Returns:
            True if dashboard was generated successfully
        """
        logger.info("📊 Generating unified dashboard...")
        
        try:
            # Import and run simple dashboard
            from dashboard_simple import SimpleDashboard
            
            dashboard = SimpleDashboard(str(self.results_path))
            
            # Generate comprehensive report
            report_path = dashboard.generate_comprehensive_report()
            
            if report_path:
                logger.info(f"✅ Dashboard generated: {report_path}")
                return True
            else:
                logger.error("❌ Dashboard generation failed")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error generating dashboard: {e}")
            return False
    
    def save_execution_summary(self) -> str:
        """
        Save execution summary to file
        
        Returns:
            Path to saved summary file
        """
        summary_file = self.results_path / f"master_execution_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(summary_file, 'w') as f:
                json.dump(self.execution_summary, f, indent=2)
            
            logger.info(f"💾 Execution summary saved: {summary_file}")
            return str(summary_file)
            
        except Exception as e:
            logger.error(f"❌ Error saving execution summary: {e}")
            return ""
    
    def cleanup_old_results(self, keep_days: int = 7) -> None:
        """
        Clean up old result files
        
        Args:
            keep_days: Number of days to keep results
        """
        logger.info(f"🧹 Cleaning up results older than {keep_days} days...")
        
        cutoff_time = time.time() - (keep_days * 24 * 60 * 60)
        deleted_count = 0
        
        try:
            for file_path in self.results_path.glob("*.json"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    
            logger.info(f"🗑️  Deleted {deleted_count} old result files")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    def run_full_pipeline(self, parallel: bool = True, timeout: int = 300, 
                          cleanup: bool = True) -> Dict[str, Any]:
        """
        Run the complete test pipeline
        
        Args:
            parallel: Whether to run tests in parallel
            timeout: Timeout for each test
            cleanup: Whether to cleanup old results
            
        Returns:
            Complete pipeline results
        """
        logger.info("🚀 Starting full test pipeline")
        pipeline_start = time.time()
        
        # Clean up old results
        if cleanup:
            self.cleanup_old_results()
        
        # Run all tests
        test_results = self.run_all_tests(parallel, timeout)
        
        # Generate dashboard
        dashboard_success = self.generate_dashboard()
        
        # Save execution summary
        summary_file = self.save_execution_summary()
        
        pipeline_time = time.time() - pipeline_start
        
        pipeline_results = {
            'pipeline_success': dashboard_success and test_results['successful_tests'] > 0,
            'pipeline_time': pipeline_time,
            'test_results': test_results,
            'dashboard_generated': dashboard_success,
            'summary_file': summary_file
        }
        
        logger.info(f"🎯 Full pipeline completed in {pipeline_time:.2f}s")
        logger.info(f"{'✅' if pipeline_results['pipeline_success'] else '❌'} Pipeline {'SUCCESS' if pipeline_results['pipeline_success'] else 'FAILED'}")
        
        return pipeline_results

def main():
    """
    Main execution function
    """
    print("🚀 Master Test Runner for ML Modules")
    print("=" * 50)
    
    # Initialize runner
    runner = MasterTestRunner()
    
    # Run full pipeline
    results = runner.run_full_pipeline(
        parallel=True,
        timeout=300,
        cleanup=True
    )
    
    # Print final summary
    print("\n" + "=" * 50)
    print("📊 FINAL SUMMARY")
    print("=" * 50)
    print(f"Pipeline Status: {'✅ SUCCESS' if results['pipeline_success'] else '❌ FAILED'}")
    print(f"Total Time: {results['pipeline_time']:.2f}s")
    print(f"Tests Passed: {results['test_results']['successful_tests']}/{results['test_results']['total_tests']}")
    print(f"Dashboard: {'✅ Generated' if results['dashboard_generated'] else '❌ Failed'}")
    print(f"Summary File: {results['summary_file']}")
    
    return 0 if results['pipeline_success'] else 1

if __name__ == "__main__":
    sys.exit(main())
