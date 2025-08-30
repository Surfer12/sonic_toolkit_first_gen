#!/usr/bin/env python3
"""
Experimental Data Processing Module

This module provides utilities for loading, processing, and validating
experimental datasets used in the scientific computing toolkit.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentalDataProcessor:
    """Processor for experimental datasets used in scientific computing validation."""

    def __init__(self, data_directory: str = "data"):
        self.data_directory = Path(data_directory)
        self.datasets = {}

    def load_dataset(self, category: str, filename: str) -> Dict[str, Any]:
        """Load a dataset from the specified category and filename."""
        file_path = self.data_directory / category / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        with open(file_path, 'r') as f:
            data = json.load(f)

        logger.info(f"Loaded dataset: {category}/{filename}")
        return data

    def load_rheology_data(self) -> Dict[str, Any]:
        """Load rheological experimental data."""
        return self.load_dataset("rheology", "herschel_bulkley_experimental_data.json")

    def load_security_data(self) -> Dict[str, Any]:
        """Load security testing data."""
        return self.load_dataset("security", "java_application_vulnerability_data.json")

    def load_biometric_data(self) -> Dict[str, Any]:
        """Load biometric identification data."""
        return self.load_dataset("biometric", "iris_recognition_dataset.json")

    def load_optical_data(self) -> Dict[str, Any]:
        """Load optical measurement data."""
        return self.load_dataset("optical", "optical_depth_measurement_data.json")

    def load_biological_data(self) -> Dict[str, Any]:
        """Load biological transport data."""
        return self.load_dataset("biological", "biological_transport_experimental_data.json")

    def extract_measurement_arrays(self, data: Dict[str, Any],
                                 measurement_key: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract measurement arrays from dataset."""
        measurement_data = data.get("measurement_data", {})
        if measurement_key not in measurement_data:
            raise KeyError(f"Measurement key '{measurement_key}' not found in dataset")

        x_data = np.array(measurement_data[measurement_key])
        y_data = np.array(measurement_data.get("viscosity", measurement_data.get("concentration_profile", [])))

        return x_data, y_data

    def validate_dataset_structure(self, data: Dict[str, Any]) -> bool:
        """Validate that dataset has required structure."""
        required_keys = ["experiment_metadata", "measurement_data", "validation_targets"]

        for key in required_keys:
            if key not in data:
                logger.error(f"Missing required key: {key}")
                return False

        # Validate measurement data structure
        measurement_data = data.get("measurement_data", {})
        if not isinstance(measurement_data, dict):
            logger.error("measurement_data must be a dictionary")
            return False

        return True

    def get_validation_targets(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract validation targets from dataset."""
        return data.get("validation_targets", {})

    def create_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the dataset for reporting."""
        summary = {
            "experiment_name": data.get("experiment_metadata", {}).get("experiment_name", "Unknown"),
            "data_points": 0,
            "measurement_types": [],
            "validation_targets": self.get_validation_targets(data),
            "quality_metrics": {}
        }

        # Count data points and identify measurement types
        measurement_data = data.get("measurement_data", {})
        for key, value in measurement_data.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (int, float)):
                    summary["measurement_types"].append(key)
                    summary["data_points"] = max(summary["data_points"], len(value))
                elif isinstance(value[0], list):
                    summary["measurement_types"].append(key)
                    summary["data_points"] = max(summary["data_points"], len(value))

        return summary

    def export_to_csv(self, data: Dict[str, Any], output_path: str,
                     measurement_keys: List[str] = None) -> None:
        """Export dataset measurements to CSV format."""
        measurement_data = data.get("measurement_data", {})

        if measurement_keys is None:
            measurement_keys = [k for k, v in measurement_data.items()
                              if isinstance(v, list) and len(v) > 0]

        # Create DataFrame from measurements
        df_data = {}
        for key in measurement_keys:
            if key in measurement_data:
                df_data[key] = measurement_data[key]

        if df_data:
            df = pd.DataFrame(df_data)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported data to: {output_path}")
        else:
            logger.warning("No measurement data found to export")

def main():
    """Main function for testing data processing capabilities."""
    processor = ExperimentalDataProcessor()

    # Test loading different datasets
    datasets_to_test = [
        ("Rheology", processor.load_rheology_data),
        ("Security", processor.load_security_data),
        ("Biometric", processor.load_biometric_data),
        ("Optical", processor.load_optical_data),
        ("Biological", processor.load_biological_data)
    ]

    for name, loader in datasets_to_test:
        try:
            data = loader()
            if processor.validate_dataset_structure(data):
                summary = processor.create_data_summary(data)
                logger.info(f"{name} Dataset Summary: {summary}")
            else:
                logger.error(f"{name} dataset structure validation failed")
        except Exception as e:
            logger.error(f"Error loading {name} dataset: {e}")

if __name__ == "__main__":
    main()
