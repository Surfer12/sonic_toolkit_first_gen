#!/usr/bin/env python3
"""
Complete Integration Test for All Five Directory Components

This script validates the complete integration across:
1. data/ - Experimental datasets
2. Corpus/ - Security processing framework
3. data_output/ - Results and reports
4. Farmer/ - iOS Swift implementations
5. docs/ - Documentation and research

Usage:
    python complete_integration_test.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompleteIntegrationTest:
    """Test complete integration across all directory components."""

    def __init__(self):
        self.test_results = {
            "test_timestamp": datetime.now().isoformat(),
            "components_tested": [],
            "overall_status": "unknown",
            "summary": {}
        }

    def run_complete_test(self):
        """Run complete integration test suite."""
        logger.info("Starting complete integration test...")

        # Test 1: Directory Structure
        self.test_directory_structure()

        # Test 2: Data Directory
        self.test_data_directory()

        # Test 3: Corpus Directory
        self.test_corpus_directory()

        # Test 4: Farmer Directory
        self.test_farmer_directory()

        # Test 5: Docs Directory
        self.test_docs_directory()

        # Test 6: Data Output Directory
        self.test_data_output_directory()

        # Test 7: Integration Workflow
        self.test_integration_workflow()

        # Generate final report
        self.generate_final_report()

        return self.test_results

    def test_directory_structure(self):
        """Test that all required directories exist."""
        logger.info("Testing directory structure...")

        required_dirs = [
            "data",
            "Corpus",
            "data_output",
            "Farmer",
            "docs"
        ]

        results = []
        for dir_name in required_dirs:
            exists = Path(dir_name).exists()
            results.append({
                "component": dir_name,
                "test": "directory_exists",
                "status": "passed" if exists else "failed",
                "details": f"Directory {'exists' if exists else 'does not exist'}"
            })

        self.test_results["components_tested"].append({
            "component": "directory_structure",
            "tests": results,
            "status": "passed" if all(r["status"] == "passed" for r in results) else "failed"
        })

    def test_data_directory(self):
        """Test data directory contents and structure."""
        logger.info("Testing data directory...")

        data_dir = Path("data")
        if not data_dir.exists():
            self.test_results["components_tested"].append({
                "component": "data_directory",
                "status": "failed",
                "details": "Data directory does not exist"
            })
            return

        expected_subdirs = ["rheology", "security", "biometric", "optical", "biological"]
        expected_files = [
            "README.md",
            "process_experimental_data.py"
        ]

        results = []

        # Check subdirectories
        for subdir in expected_subdirs:
            exists = (data_dir / subdir).exists()
            results.append({
                "test": f"subdir_{subdir}",
                "status": "passed" if exists else "failed",
                "details": f"Subdirectory {subdir} {'exists' if exists else 'does not exist'}"
            })

        # Check files
        for file in expected_files:
            exists = (data_dir / file).exists()
            results.append({
                "test": f"file_{file}",
                "status": "passed" if exists else "failed",
                "details": f"File {file} {'exists' if exists else 'does not exist'}"
            })

        # Check for dataset files
        dataset_checks = [
            ("rheology", "herschel_bulkley_experimental_data.json"),
            ("security", "java_application_vulnerability_data.json"),
            ("biometric", "iris_recognition_dataset.json"),
            ("optical", "optical_depth_measurement_data.json"),
            ("biological", "biological_transport_experimental_data.json")
        ]

        for subdir, filename in dataset_checks:
            exists = (data_dir / subdir / filename).exists()
            results.append({
                "test": f"dataset_{subdir}",
                "status": "passed" if exists else "failed",
                "details": f"Dataset {filename} {'exists' if exists else 'does not exist'}"
            })

        status = "passed" if all(r["status"] == "passed" for r in results) else "failed"
        self.test_results["components_tested"].append({
            "component": "data_directory",
            "tests": results,
            "status": status
        })

    def test_corpus_directory(self):
        """Test Corpus directory contents and Java framework."""
        logger.info("Testing Corpus directory...")

        corpus_dir = Path("Corpus")
        if not corpus_dir.exists():
            self.test_results["components_tested"].append({
                "component": "corpus_directory",
                "status": "failed",
                "details": "Corpus directory does not exist"
            })
            return

        qualia_dir = corpus_dir / "qualia"
        if not qualia_dir.exists():
            self.test_results["components_tested"].append({
                "component": "corpus_directory",
                "status": "failed",
                "details": "Corpus/qualia directory does not exist"
            })
            return

        expected_java_files = [
            "JavaPenetrationTesting.java",
            "ReverseKoopmanOperator.java",
            "README.md",
            "build.sh"
        ]

        results = []
        for filename in expected_java_files:
            exists = (qualia_dir / filename).exists()
            results.append({
                "test": f"java_file_{filename}",
                "status": "passed" if exists else "failed",
                "details": f"Java file {filename} {'exists' if exists else 'does not exist'}"
            })

        # Test Java compilation (if Java is available)
        try:
            result = subprocess.run(
                ["javac", "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            java_available = result.returncode == 0
            results.append({
                "test": "java_compilation_available",
                "status": "passed" if java_available else "warning",
                "details": f"Java compilation {'available' if java_available else 'not available'}"
            })
        except:
            results.append({
                "test": "java_compilation_available",
                "status": "warning",
                "details": "Java compilation check failed"
            })

        status = "passed" if all(r["status"] in ["passed", "warning"] for r in results) else "failed"
        self.test_results["components_tested"].append({
            "component": "corpus_directory",
            "tests": results,
            "status": status
        })

    def test_farmer_directory(self):
        """Test Farmer directory contents and Swift framework."""
        logger.info("Testing Farmer directory...")

        farmer_dir = Path("Farmer")
        if not farmer_dir.exists():
            self.test_results["components_tested"].append({
                "component": "farmer_directory",
                "status": "failed",
                "details": "Farmer directory does not exist"
            })
            return

        expected_files = [
            "Package.swift",
            "Sources/UOIFCore/iOSPenetrationTesting.swift",
            "Sources/UOIFCore/ReverseKoopmanOperator.swift",
            "Tests/UOIFCoreTests/iOSPenetrationTestingTests.swift"
        ]

        results = []
        for filename in expected_files:
            exists = (farmer_dir / filename).exists()
            results.append({
                "test": f"swift_file_{filename}",
                "status": "passed" if exists else "failed",
                "details": f"Swift file {filename} {'exists' if exists else 'does not exist'}"
            })

        # Test Swift compilation (if Swift is available)
        try:
            result = subprocess.run(
                ["swift", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            swift_available = result.returncode == 0
            results.append({
                "test": "swift_compilation_available",
                "status": "passed" if swift_available else "warning",
                "details": f"Swift compilation {'available' if swift_available else 'not available'}"
            })
        except:
            results.append({
                "test": "swift_compilation_available",
                "status": "warning",
                "details": "Swift compilation check failed"
            })

        status = "passed" if all(r["status"] in ["passed", "warning"] for r in results) else "failed"
        self.test_results["components_tested"].append({
            "component": "farmer_directory",
            "tests": results,
            "status": status
        })

    def test_docs_directory(self):
        """Test docs directory contents and documentation."""
        logger.info("Testing docs directory...")

        docs_dir = Path("docs")
        if not docs_dir.exists():
            self.test_results["components_tested"].append({
                "component": "docs_directory",
                "status": "failed",
                "details": "docs directory does not exist"
            })
            return

        expected_files = [
            "index.md",
            "achievements-showcase.md",
            "frameworks/inverse-precision.md",
            "_config.yml"
        ]

        results = []
        for filename in expected_files:
            exists = (docs_dir / filename).exists()
            results.append({
                "test": f"doc_file_{filename}",
                "status": "passed" if exists else "failed",
                "details": f"Documentation file {filename} {'exists' if exists else 'does not exist'}"
            })

        status = "passed" if all(r["status"] == "passed" for r in results) else "failed"
        self.test_results["components_tested"].append({
            "component": "docs_directory",
            "tests": results,
            "status": status
        })

    def test_data_output_directory(self):
        """Test data_output directory contents and integration."""
        logger.info("Testing data_output directory...")

        output_dir = Path("data_output")
        if not output_dir.exists():
            self.test_results["components_tested"].append({
                "component": "data_output_directory",
                "status": "failed",
                "details": "data_output directory does not exist"
            })
            return

        expected_subdirs = ["results", "reports", "visualizations", "logs"]
        expected_files = [
            "data_flow_processor.py",
            "integration_runner.py",
            "integration_config.json",
            "README.md"
        ]

        results = []

        # Check subdirectories
        for subdir in expected_subdirs:
            exists = (output_dir / subdir).exists()
            results.append({
                "test": f"subdir_{subdir}",
                "status": "passed" if exists else "failed",
                "details": f"Subdirectory {subdir} {'exists' if exists else 'does not exist'}"
            })

        # Check files
        for filename in expected_files:
            exists = (output_dir / filename).exists()
            results.append({
                "test": f"file_{filename}",
                "status": "passed" if exists else "failed",
                "details": f"File {filename} {'exists' if exists else 'does not exist'}"
            })

        status = "passed" if all(r["status"] == "passed" for r in results) else "failed"
        self.test_results["components_tested"].append({
            "component": "data_output_directory",
            "tests": results,
            "status": status
        })

    def test_integration_workflow(self):
        """Test the complete integration workflow."""
        logger.info("Testing integration workflow...")

        results = []

        # Test data flow processor import
        try:
            sys.path.append("data_output")
            from data_flow_processor import CorpusDataFlowProcessor
            results.append({
                "test": "data_flow_processor_import",
                "status": "passed",
                "details": "Data flow processor imported successfully"
            })
        except ImportError as e:
            results.append({
                "test": "data_flow_processor_import",
                "status": "failed",
                "details": f"Import failed: {e}"
            })

        # Test processor initialization
        try:
            if 'CorpusDataFlowProcessor' in locals():
                processor = CorpusDataFlowProcessor()
                results.append({
                    "test": "processor_initialization",
                    "status": "passed",
                    "details": "Processor initialized successfully"
                })
            else:
                results.append({
                    "test": "processor_initialization",
                    "status": "failed",
                    "details": "Processor class not available"
                })
        except Exception as e:
            results.append({
                "test": "processor_initialization",
                "status": "failed",
                "details": f"Initialization failed: {e}"
            })

        # Test configuration loading
        try:
            with open("data_output/integration_config.json", 'r') as f:
                config = json.load(f)
            results.append({
                "test": "configuration_loading",
                "status": "passed",
                "details": "Integration configuration loaded successfully"
            })
        except Exception as e:
            results.append({
                "test": "configuration_loading",
                "status": "failed",
                "details": f"Configuration loading failed: {e}"
            })

        status = "passed" if all(r["status"] == "passed" for r in results) else "failed"
        self.test_results["components_tested"].append({
            "component": "integration_workflow",
            "tests": results,
            "status": status
        })

    def generate_final_report(self):
        """Generate comprehensive test report."""
        # Calculate summary statistics
        components = self.test_results["components_tested"]
        total_components = len(components)
        passed_components = sum(1 for c in components if c["status"] == "passed")
        failed_components = sum(1 for c in components if c["status"] == "failed")

        success_rate = passed_components / total_components if total_components > 0 else 0

        self.test_results["summary"] = {
            "total_components": total_components,
            "passed_components": passed_components,
            "failed_components": failed_components,
            "success_rate": success_rate,
            "overall_status": "passed" if success_rate >= 0.8 else "failed"
        }

        # Save detailed report
        report_path = Path("integration_test_report.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            logger.info(f"Integration test report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save test report: {e}")

        # Print summary to console
        print("\n" + "="*60)
        print("COMPLETE INTEGRATION TEST RESULTS")
        print("="*60)
        print(f"Test Timestamp: {self.test_results['test_timestamp']}")
        print(f"Total Components Tested: {total_components}")
        print(f"Passed: {passed_components}")
        print(f"Failed: {failed_components}")
        print(".1%")
        print(f"Overall Status: {'✅ PASSED' if success_rate >= 0.8 else '❌ FAILED'}")
        print("="*60)

        for component in components:
            status_icon = "✅" if component["status"] == "passed" else "❌"
            print(f"{status_icon} {component['component']}: {component['status'].upper()}")

            if "tests" in component:
                for test in component["tests"]:
                    if test["status"] != "passed":
                        test_icon = "⚠️" if test["status"] == "warning" else "❌"
                        print(f"   {test_icon} {test['test']}: {test['details']}")

        print("="*60)

def main():
    """Main function to run complete integration test."""
    tester = CompleteIntegrationTest()

    try:
        results = tester.run_complete_test()
        overall_status = results["summary"]["overall_status"]

        if overall_status == "passed":
            logger.info("Integration test completed successfully!")
            sys.exit(0)
        else:
            logger.error("Integration test failed!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Integration test failed with exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
