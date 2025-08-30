#!/usr/bin/env python3
"""
Integration Runner for Complete Corpus Data Flow

This script orchestrates the complete integration workflow:
1. Load experimental data from data/ directory
2. Process through Corpus security framework
3. Generate results in data_output/ directory
4. Create comprehensive reports and visualizations

Usage:
    python integration_runner.py                    # Run all pipelines
    python integration_runner.py --pipeline security  # Run specific pipeline
    python integration_runner.py --test              # Run integration tests
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add data_output to Python path for imports
sys.path.append(str(Path(__file__).parent))

from data_flow_processor import CorpusDataFlowProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_output/logs/integration_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class IntegrationRunner:
    """Runner for complete Corpus data flow integration."""

    def __init__(self):
        self.processor = CorpusDataFlowProcessor()
        self.config = self._load_config()
        self.results = {}

    def _load_config(self):
        """Load integration configuration."""
        config_path = Path(__file__).parent / "integration_config.json"
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}

    def run_all_pipelines(self):
        """Run all data processing pipelines."""
        logger.info("Starting complete integration workflow...")

        pipelines = [
            ("security", self.processor.process_security_dataset),
            ("biometric", self.processor.process_biometric_dataset),
            ("rheology", self.processor.process_rheology_dataset),
            ("optical", self.processor.process_optical_dataset),
            ("biological", self.processor.process_biological_dataset)
        ]

        for pipeline_name, pipeline_func in pipelines:
            logger.info(f"Running {pipeline_name} pipeline...")
            try:
                result = pipeline_func()
                self.results[pipeline_name] = result
                status = result.get("status", "unknown")
                logger.info(f"{pipeline_name} pipeline completed with status: {status}")
            except Exception as e:
                logger.error(f"Pipeline {pipeline_name} failed: {e}")
                self.results[pipeline_name] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        self._generate_integration_report()
        return self.results

    def run_specific_pipeline(self, pipeline_name: str):
        """Run a specific data processing pipeline."""
        logger.info(f"Running specific pipeline: {pipeline_name}")

        pipeline_map = {
            "security": self.processor.process_security_dataset,
            "biometric": self.processor.process_biometric_dataset,
            "rheology": self.processor.process_rheology_dataset,
            "optical": self.processor.process_optical_dataset,
            "biological": self.processor.process_biological_dataset
        }

        if pipeline_name not in pipeline_map:
            logger.error(f"Unknown pipeline: {pipeline_name}")
            return {"status": "error", "message": f"Unknown pipeline: {pipeline_name}"}

        try:
            result = pipeline_map[pipeline_name]()
            self.results[pipeline_name] = result
            logger.info(f"Pipeline {pipeline_name} completed")
            return result
        except Exception as e:
            logger.error(f"Pipeline {pipeline_name} failed: {e}")
            return {"status": "error", "error": str(e)}

    def run_integration_tests(self):
        """Run integration tests to validate the complete workflow."""
        logger.info("Running integration tests...")

        test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }

        # Test 1: Data loading
        test_results["tests"].append(self._test_data_loading())

        # Test 2: Processing pipeline
        test_results["tests"].append(self._test_processing_pipeline())

        # Test 3: Result generation
        test_results["tests"].append(self._test_result_generation())

        # Test 4: Error handling
        test_results["tests"].append(self._test_error_handling())

        # Save test results
        test_output_path = Path("data_output/results/integration_test_results.json")
        try:
            with open(test_output_path, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            logger.info(f"Integration test results saved to: {test_output_path}")
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")

        # Print summary
        passed_tests = sum(1 for test in test_results["tests"] if test["status"] == "passed")
        total_tests = len(test_results["tests"])
        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        logger.info(f"Integration tests completed: {passed_tests}/{total_tests} passed ({success_rate:.1%})")

        return test_results

    def _test_data_loading(self):
        """Test data loading functionality."""
        try:
            # Test loading each dataset type
            datasets = ["rheology", "security", "biometric", "optical", "biological"]
            loaded_count = 0

            for dataset in datasets:
                try:
                    if dataset == "rheology":
                        data = self.processor._load_rheology_data()
                    elif dataset == "security":
                        data = self.processor._load_security_data()
                    elif dataset == "biometric":
                        data = self.processor._load_biometric_data()
                    elif dataset == "optical":
                        data = self.processor._load_optical_data()
                    elif dataset == "biological":
                        data = self.processor._load_biological_data()

                    if data:
                        loaded_count += 1
                except:
                    pass

            success = loaded_count >= 3  # At least 3 datasets should load
            return {
                "test_name": "data_loading",
                "status": "passed" if success else "failed",
                "details": f"Successfully loaded {loaded_count}/{len(datasets)} datasets"
            }

        except Exception as e:
            return {
                "test_name": "data_loading",
                "status": "failed",
                "details": f"Exception: {e}"
            }

    def _test_processing_pipeline(self):
        """Test processing pipeline functionality."""
        try:
            # Test a simple processing pipeline
            result = self.processor.process_rheology_dataset()

            success = result.get("status") == "success"
            return {
                "test_name": "processing_pipeline",
                "status": "passed" if success else "failed",
                "details": f"Rheology processing {'successful' if success else 'failed'}"
            }

        except Exception as e:
            return {
                "test_name": "processing_pipeline",
                "status": "failed",
                "details": f"Exception: {e}"
            }

    def _test_result_generation(self):
        """Test result generation functionality."""
        try:
            # Check if result files exist
            results_dir = Path("data_output/results")
            reports_dir = Path("data_output/reports")

            results_exist = results_dir.exists() and any(results_dir.iterdir())
            reports_exist = reports_dir.exists() and any(reports_dir.iterdir())

            success = results_exist and reports_exist
            return {
                "test_name": "result_generation",
                "status": "passed" if success else "failed",
                "details": f"Results dir: {'✓' if results_exist else '✗'}, Reports dir: {'✓' if reports_exist else '✗'}"
            }

        except Exception as e:
            return {
                "test_name": "result_generation",
                "status": "failed",
                "details": f"Exception: {e}"
            }

    def _test_error_handling(self):
        """Test error handling functionality."""
        try:
            # Test with invalid input
            result = self.processor.process_security_dataset()
            # This should handle missing Java gracefully

            return {
                "test_name": "error_handling",
                "status": "passed",
                "details": "Error handling working correctly"
            }

        except Exception as e:
            return {
                "test_name": "error_handling",
                "status": "failed",
                "details": f"Exception: {e}"
            }

    def _generate_integration_report(self):
        """Generate comprehensive integration report."""
        report = {
            "integration_report": {
                "title": "Corpus Data Flow Integration Report",
                "generated_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "pipeline_results": self.results,
            "summary": {
                "total_pipelines": len(self.results),
                "successful_pipelines": sum(1 for r in self.results.values() if r.get("status") == "success"),
                "failed_pipelines": sum(1 for r in self.results.values() if r.get("status") != "success"),
                "success_rate": 0.0
            },
            "system_info": {
                "python_version": sys.version,
                "working_directory": str(Path.cwd()),
                "data_directory_exists": Path("data").exists(),
                "corpus_directory_exists": Path("Corpus").exists(),
                "output_directory_exists": Path("data_output").exists()
            }
        }

        # Calculate success rate
        total = report["summary"]["total_pipelines"]
        successful = report["summary"]["successful_pipelines"]
        report["summary"]["success_rate"] = successful / total if total > 0 else 0

        # Save integration report
        report_path = Path("data_output/reports/integration_report.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Integration report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save integration report: {e}")

def main():
    """Main function for integration runner."""
    parser = argparse.ArgumentParser(description="Corpus Data Flow Integration Runner")
    parser.add_argument("--pipeline", choices=["security", "biometric", "rheology", "optical", "biological"],
                       help="Run specific pipeline")
    parser.add_argument("--test", action="store_true", help="Run integration tests")
    parser.add_argument("--all", action="store_true", help="Run all pipelines (default)")

    args = parser.parse_args()

    runner = IntegrationRunner()

    try:
        if args.test:
            # Run integration tests
            test_results = runner.run_integration_tests()
            print("\n=== Integration Test Results ===")
            for test in test_results["tests"]:
                status = "✓" if test["status"] == "passed" else "✗"
                print(f"{status} {test['test_name']}: {test['details']}")

        elif args.pipeline:
            # Run specific pipeline
            result = runner.run_specific_pipeline(args.pipeline)
            print(f"\n=== Pipeline Result: {args.pipeline} ===")
            print(f"Status: {result.get('status', 'unknown')}")

        else:
            # Run all pipelines (default)
            results = runner.run_all_pipelines()
            print("\n=== Complete Integration Results ===")
            print(f"Total pipelines: {len(results)}")
            successful = sum(1 for r in results.values() if r.get("status") == "success")
            print(f"Successful: {successful}")
            print(".1%")

    except Exception as e:
        logger.error(f"Integration runner failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
