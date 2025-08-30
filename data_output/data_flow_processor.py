#!/usr/bin/env python3
"""
Data Flow Processor for Corpus Framework Integration

This module establishes the data flow between:
1. data/ directory (input datasets)
2. Corpus/qualia framework (processing)
3. data_output/ directory (results and reports)

The processor orchestrates the complete workflow from experimental data
loading through security analysis to result generation and reporting.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import subprocess

# Add Corpus directory to Python path for imports
sys.path.append(str(Path(__file__).parent.parent / "Corpus"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_output/logs/data_flow_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CorpusDataFlowProcessor:
    """Main processor for Corpus data flow integration."""

    def __init__(self, data_dir: str = "data", output_dir: str = "data_output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.corpus_dir = Path("Corpus")

        # Ensure output directories exist
        self._ensure_output_directories()

    def _ensure_output_directories(self):
        """Create necessary output directories if they don't exist."""
        directories = [
            self.output_dir / "results",
            self.output_dir / "reports",
            self.output_dir / "visualizations",
            self.output_dir / "logs"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")

    def process_security_dataset(self) -> Dict[str, Any]:
        """Process security dataset through Corpus framework."""
        logger.info("Processing security dataset...")

        # Load security data
        security_data = self._load_security_data()
        if not security_data:
            logger.error("Failed to load security data")
            return {"status": "error", "message": "Security data loading failed"}

        # Run Java penetration testing
        java_results = self._run_java_penetration_testing(security_data)
        if not java_results:
            logger.warning("Java penetration testing failed or not available")

        # Run GPTOSS AI testing if applicable
        gptoss_results = self._run_gptoss_testing(security_data)

        # Generate integrated report
        integrated_report = self._generate_integrated_security_report(
            security_data, java_results, gptoss_results
        )

        # Save results
        self._save_security_results(integrated_report)

        logger.info("Security dataset processing completed")
        return {
            "status": "success",
            "input_data": security_data,
            "java_results": java_results,
            "gptoss_results": gptoss_results,
            "integrated_report": integrated_report
        }

    def process_biometric_dataset(self) -> Dict[str, Any]:
        """Process biometric dataset through relevant analysis."""
        logger.info("Processing biometric dataset...")

        # Load biometric data
        biometric_data = self._load_biometric_data()
        if not biometric_data:
            logger.error("Failed to load biometric data")
            return {"status": "error", "message": "Biometric data loading failed"}

        # Perform biometric analysis
        analysis_results = self._analyze_biometric_data(biometric_data)

        # Generate biometric report
        biometric_report = self._generate_biometric_report(biometric_data, analysis_results)

        # Save results
        self._save_biometric_results(biometric_report)

        logger.info("Biometric dataset processing completed")
        return {
            "status": "success",
            "input_data": biometric_data,
            "analysis_results": analysis_results,
            "biometric_report": biometric_report
        }

    def process_rheology_dataset(self) -> Dict[str, Any]:
        """Process rheological dataset through analysis frameworks."""
        logger.info("Processing rheology dataset...")

        # Load rheology data
        rheology_data = self._load_rheology_data()
        if not rheology_data:
            logger.error("Failed to load rheology data")
            return {"status": "error", "message": "Rheology data loading failed"}

        # Perform rheological analysis
        analysis_results = self._analyze_rheology_data(rheology_data)

        # Generate rheology report
        rheology_report = self._generate_rheology_report(rheology_data, analysis_results)

        # Save results
        self._save_rheology_results(rheology_report)

        logger.info("Rheology dataset processing completed")
        return {
            "status": "success",
            "input_data": rheology_data,
            "analysis_results": analysis_results,
            "rheology_report": rheology_report
        }

    def process_optical_dataset(self) -> Dict[str, Any]:
        """Process optical dataset through analysis frameworks."""
        logger.info("Processing optical dataset...")

        # Load optical data
        optical_data = self._load_optical_data()
        if not optical_data:
            logger.error("Failed to load optical data")
            return {"status": "error", "message": "Optical data loading failed"}

        # Perform optical analysis
        analysis_results = self._analyze_optical_data(optical_data)

        # Generate optical report
        optical_report = self._generate_optical_report(optical_data, analysis_results)

        # Save results
        self._save_optical_results(optical_report)

        logger.info("Optical dataset processing completed")
        return {
            "status": "success",
            "input_data": optical_data,
            "analysis_results": analysis_results,
            "optical_report": optical_report
        }

    def process_biological_dataset(self) -> Dict[str, Any]:
        """Process biological dataset through analysis frameworks."""
        logger.info("Processing biological dataset...")

        # Load biological data
        biological_data = self._load_biological_data()
        if not biological_data:
            logger.error("Failed to load biological data")
            return {"status": "error", "message": "Biological data loading failed"}

        # Perform biological analysis
        analysis_results = self._analyze_biological_data(biological_data)

        # Generate biological report
        biological_report = self._generate_biological_report(biological_data, analysis_results)

        # Save results
        self._save_biological_results(biological_report)

        logger.info("Biological dataset processing completed")
        return {
            "status": "success",
            "input_data": biological_data,
            "analysis_results": analysis_results,
            "biological_report": biological_report
        }

    def run_complete_data_flow(self) -> Dict[str, Any]:
        """Run complete data flow processing across all datasets."""
        logger.info("Starting complete data flow processing...")

        results = {
            "timestamp": datetime.now().isoformat(),
            "processing_results": {},
            "summary": {}
        }

        # Process each dataset type
        dataset_processors = [
            ("security", self.process_security_dataset),
            ("biometric", self.process_biometric_dataset),
            ("rheology", self.process_rheology_dataset),
            ("optical", self.process_optical_dataset),
            ("biological", self.process_biological_dataset)
        ]

        successful_processes = 0
        total_processes = len(dataset_processors)

        for dataset_name, processor in dataset_processors:
            try:
                logger.info(f"Processing {dataset_name} dataset...")
                result = processor()
                results["processing_results"][dataset_name] = result

                if result.get("status") == "success":
                    successful_processes += 1
                    logger.info(f"Successfully processed {dataset_name} dataset")
                else:
                    logger.error(f"Failed to process {dataset_name} dataset: {result.get('message', 'Unknown error')}")

            except Exception as e:
                logger.error(f"Exception processing {dataset_name} dataset: {e}")
                results["processing_results"][dataset_name] = {
                    "status": "error",
                    "message": str(e)
                }

        # Generate summary
        results["summary"] = {
            "total_datasets": total_processes,
            "successful_processes": successful_processes,
            "success_rate": successful_processes / total_processes if total_processes > 0 else 0,
            "processing_timestamp": datetime.now().isoformat()
        }

        # Save complete results
        self._save_complete_results(results)

        logger.info(f"Complete data flow processing finished. Success rate: {results['summary']['success_rate']:.2%}")
        return results

    # MARK: - Data Loading Methods

    def _load_security_data(self) -> Optional[Dict[str, Any]]:
        """Load security dataset from data directory."""
        try:
            with open(self.data_dir / "security" / "java_application_vulnerability_data.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load security data: {e}")
            return None

    def _load_biometric_data(self) -> Optional[Dict[str, Any]]:
        """Load biometric dataset from data directory."""
        try:
            with open(self.data_dir / "biometric" / "iris_recognition_dataset.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load biometric data: {e}")
            return None

    def _load_rheology_data(self) -> Optional[Dict[str, Any]]:
        """Load rheology dataset from data directory."""
        try:
            with open(self.data_dir / "rheology" / "herschel_bulkley_experimental_data.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load rheology data: {e}")
            return None

    def _load_optical_data(self) -> Optional[Dict[str, Any]]:
        """Load optical dataset from data directory."""
        try:
            with open(self.data_dir / "optical" / "optical_depth_measurement_data.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load optical data: {e}")
            return None

    def _load_biological_data(self) -> Optional[Dict[str, Any]]:
        """Load biological dataset from data directory."""
        try:
            with open(self.data_dir / "biological" / "biological_transport_experimental_data.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load biological data: {e}")
            return None

    # MARK: - Analysis Methods

    def _run_java_penetration_testing(self, security_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run Java penetration testing using Corpus framework."""
        try:
            # Check if Java is available
            result = subprocess.run(
                ["java", "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                logger.warning("Java not available for penetration testing")
                return None

            # Run Java penetration testing demo
            java_cmd = [
                "java",
                "-cp", "Corpus/out",
                "qualia.JavaPenetrationTestingDemo"
            ]

            result = subprocess.run(
                java_cmd,
                cwd=self.data_dir.parent,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                return {
                    "status": "success",
                    "output": result.stdout,
                    "execution_time": "N/A"
                }
            else:
                logger.warning(f"Java penetration testing failed: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"Error running Java penetration testing: {e}")
            return None

    def _run_gptoss_testing(self, security_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run GPTOSS AI testing (placeholder for future implementation)."""
        logger.info("GPTOSS testing not yet implemented - placeholder")
        return {
            "status": "not_implemented",
            "message": "GPTOSS AI testing framework not yet available"
        }

    def _analyze_biometric_data(self, biometric_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze biometric dataset."""
        # Placeholder analysis - in real implementation, this would use
        # actual biometric analysis algorithms
        return {
            "identification_accuracy": biometric_data.get("validation_results", {}).get("identification_accuracy", 0),
            "quality_metrics": biometric_data.get("dataset_statistics", {}).get("iris_quality_distribution", {}),
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _analyze_rheology_data(self, rheology_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze rheological dataset."""
        return {
            "material": rheology_data.get("experiment_metadata", {}).get("material", "Unknown"),
            "temperature": rheology_data.get("experiment_metadata", {}).get("temperature", 0),
            "expected_parameters": rheology_data.get("expected_parameters", {}),
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _analyze_optical_data(self, optical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optical dataset."""
        return {
            "enhancement_factor": optical_data.get("enhancement_processing", {}).get("enhancement_factor", 0),
            "precision_improvement": optical_data.get("enhancement_processing", {}).get("original_precision", {}),
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _analyze_biological_data(self, biological_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze biological dataset."""
        return {
            "tissue_type": biological_data.get("biological_experiment_metadata", {}).get("tissue_type", "Unknown"),
            "transport_analysis": biological_data.get("transport_measurements", {}),
            "validation_targets": biological_data.get("validation_targets", {}),
            "analysis_timestamp": datetime.now().isoformat()
        }

    # MARK: - Report Generation Methods

    def _generate_integrated_security_report(self, security_data: Dict[str, Any],
                                           java_results: Optional[Dict[str, Any]],
                                           gptoss_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate integrated security report."""
        return {
            "report_title": "Integrated Security Analysis Report",
            "generated_at": datetime.now().isoformat(),
            "input_data_summary": {
                "application_name": security_data.get("security_assessment_metadata", {}).get("application_name"),
                "vulnerabilities_found": security_data.get("security_metrics", {}).get("total_vulnerabilities", 0),
                "risk_score": security_data.get("security_metrics", {}).get("overall_risk_score", 0)
            },
            "java_testing_results": java_results,
            "gptoss_testing_results": gptoss_results,
            "recommendations": security_data.get("assessment_summary", {}).get("recommendations", [])
        }

    def _generate_biometric_report(self, biometric_data: Dict[str, Any],
                                 analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate biometric analysis report."""
        return {
            "report_title": "Biometric Analysis Report",
            "generated_at": datetime.now().isoformat(),
            "dataset_info": biometric_data.get("biometric_dataset_metadata", {}),
            "validation_results": biometric_data.get("validation_results", {}),
            "analysis_results": analysis_results
        }

    def _generate_rheology_report(self, rheology_data: Dict[str, Any],
                                analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate rheological analysis report."""
        return {
            "report_title": "Rheological Analysis Report",
            "generated_at": datetime.now().isoformat(),
            "material_info": rheology_data.get("experiment_metadata", {}),
            "expected_parameters": rheology_data.get("expected_parameters", {}),
            "analysis_results": analysis_results
        }

    def _generate_optical_report(self, optical_data: Dict[str, Any],
                               analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optical analysis report."""
        return {
            "report_title": "Optical Analysis Report",
            "generated_at": datetime.now().isoformat(),
            "experiment_info": optical_data.get("optical_experiment_metadata", {}),
            "enhancement_results": optical_data.get("enhancement_processing", {}),
            "analysis_results": analysis_results
        }

    def _generate_biological_report(self, biological_data: Dict[str, Any],
                                  analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate biological analysis report."""
        return {
            "report_title": "Biological Analysis Report",
            "generated_at": datetime.now().isoformat(),
            "experiment_info": biological_data.get("biological_experiment_metadata", {}),
            "transport_data": biological_data.get("transport_measurements", {}),
            "analysis_results": analysis_results
        }

    # MARK: - Result Saving Methods

    def _save_security_results(self, report: Dict[str, Any]):
        """Save security analysis results."""
        self._save_json_result("security_analysis_report.json", report, "reports")

    def _save_biometric_results(self, report: Dict[str, Any]):
        """Save biometric analysis results."""
        self._save_json_result("biometric_analysis_report.json", report, "reports")

    def _save_rheology_results(self, report: Dict[str, Any]):
        """Save rheological analysis results."""
        self._save_json_result("rheology_analysis_report.json", report, "reports")

    def _save_optical_results(self, report: Dict[str, Any]):
        """Save optical analysis results."""
        self._save_json_result("optical_analysis_report.json", report, "reports")

    def _save_biological_results(self, report: Dict[str, Any]):
        """Save biological analysis results."""
        self._save_json_result("biological_analysis_report.json", report, "reports")

    def _save_complete_results(self, results: Dict[str, Any]):
        """Save complete processing results."""
        self._save_json_result("complete_data_flow_results.json", results, "results")

    def _save_json_result(self, filename: str, data: Dict[str, Any], subdirectory: str):
        """Save JSON result to specified subdirectory."""
        output_path = self.output_dir / subdirectory / filename
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved results to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_path}: {e}")

def main():
    """Main function to run data flow processor."""
    logger.info("Starting Corpus Data Flow Processor...")

    processor = CorpusDataFlowProcessor()

    try:
        results = processor.run_complete_data_flow()

        logger.info("Data flow processing completed successfully")
        logger.info(f"Success rate: {results['summary']['success_rate']:.2%}")

        # Print summary
        print("\n=== Data Flow Processing Summary ===")
        print(f"Total datasets processed: {results['summary']['total_datasets']}")
        print(f"Successful processes: {results['summary']['successful_processes']}")
        print(".2%")
        print(f"Results saved to: {processor.output_dir}")

    except Exception as e:
        logger.error(f"Data flow processing failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
