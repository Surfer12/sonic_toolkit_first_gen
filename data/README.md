# Experimental Datasets Directory

This directory contains experimental datasets for validating the scientific computing toolkit across multiple domains.

## Directory Structure

```
data/
├── rheology/              # Rheological characterization data
├── security/              # Security testing datasets
├── biometric/             # Biometric identification data
├── optical/               # Optical measurement datasets
├── biological/            # Biological transport data
└── README.md             # This file
```

## Dataset Categories

### Rheology Data
- **herschel_bulkley_experimental_data.json**: Polymer melt viscosity measurements
- **Purpose**: Parameter extraction validation for Herschel-Bulkley constitutive models
- **Applications**: Polymer processing, food rheology, complex fluid characterization
- **Key Metrics**: Shear rate, viscosity, yield stress, flow behavior index

### Security Data
- **java_application_vulnerability_data.json**: Comprehensive security assessment results
- **Purpose**: Validation of penetration testing frameworks
- **Applications**: Security testing, vulnerability assessment, risk analysis
- **Key Metrics**: CVSS scores, severity levels, remediation status

### Biometric Data
- **iris_recognition_dataset.json**: 3D iris recognition dataset
- **Purpose**: Validation of biometric identification systems
- **Applications**: Access control, identity verification, security systems
- **Key Metrics**: Identification accuracy, false acceptance/rejection rates

### Optical Data
- **optical_depth_measurement_data.json**: Sub-nanometer precision measurements
- **Purpose**: Validation of optical depth enhancement algorithms
- **Applications**: Semiconductor metrology, surface characterization
- **Key Metrics**: Depth precision, enhancement factor, measurement uncertainty

### Biological Data
- **biological_transport_experimental_data.json**: Nutrient transport in tissue scaffolds
- **Purpose**: Validation of biological transport modeling
- **Applications**: Tissue engineering, drug delivery, organ preservation
- **Key Metrics**: Concentration profiles, transport rates, tissue properties

## Data Format Standards

All datasets follow consistent JSON format with:

### Required Sections
- **Metadata**: Experiment details, conditions, researcher information
- **Measurement Data**: Raw experimental measurements with units
- **Validation Targets**: Expected performance metrics and thresholds
- **Quality Metrics**: Data quality indicators and uncertainty estimates

### Data Quality Assurance
- **Units Specification**: All measurements include explicit units
- **Uncertainty Quantification**: Error bounds and confidence intervals
- **Metadata Completeness**: Full experimental context and conditions
- **Validation Targets**: Clear success criteria for model validation

## Usage Guidelines

### For Model Validation
```python
import json

# Load experimental dataset
with open('data/rheology/herschel_bulkley_experimental_data.json', 'r') as f:
    data = json.load(f)

# Extract validation targets
targets = data['validation_targets']
convergence_threshold = targets['convergence_threshold']  # 0.9987

# Use for model validation
# ... validation code ...
```

### For Research Applications
```python
# Load biological transport data
with open('data/biological/biological_transport_experimental_data.json', 'r') as f:
    bio_data = json.load(f)

# Extract tissue properties for modeling
tissue_props = bio_data['tissue_properties']
diffusivity = tissue_props['diffusivity_coefficients']

# Use in transport modeling
# ... modeling code ...
```

## Dataset Maintenance

### Adding New Datasets
1. Follow established JSON format structure
2. Include comprehensive metadata
3. Specify validation targets and quality metrics
4. Document experimental conditions and procedures
5. Update this README with new dataset information

### Quality Control
- Regular validation against known analytical solutions
- Cross-verification with published experimental data
- Uncertainty quantification and error analysis
- Documentation of data collection procedures

## Integration with Scientific Computing Toolkit

### Framework Connections
- **Corpus/qualia/**: Security data for penetration testing validation
- **docs/frameworks/**: Datasets referenced in framework documentation
- **data_output/**: Processed results from these experimental datasets

### Workflow Integration
1. **Data Loading**: Load experimental datasets from this directory
2. **Model Application**: Apply scientific computing algorithms
3. **Validation**: Compare results against validation targets
4. **Output Generation**: Store results in data_output/ directory

## Contributing

### Dataset Standards
- Use descriptive, consistent naming conventions
- Include comprehensive documentation
- Provide uncertainty estimates
- Specify experimental conditions
- Include validation criteria

### Review Process
- Peer review of dataset quality and completeness
- Validation of experimental procedures
- Verification of data integrity
- Documentation review and approval
