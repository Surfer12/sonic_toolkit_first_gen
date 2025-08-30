# Scientific Computing Toolkit - Getting Started Tutorial

## Welcome to the Scientific Computing Toolkit! 🎉

This comprehensive tutorial will guide you through your first steps with the Scientific Computing Toolkit, from initial setup to running your first scientific analysis. By the end of this tutorial, you'll be able to:

- ✅ Set up your development environment
- ✅ Run basic scientific computations
- ✅ Understand the toolkit's core components
- ✅ Execute integrated workflows
- ✅ Generate publication-ready results

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Your First Analysis](#your-first-analysis)
4. [Understanding the Framework](#understanding-the-framework)
5. [Advanced Workflows](#advanced-workflows)
6. [Troubleshooting](#troubleshooting)
7. [Next Steps](#next-steps)

---

## Prerequisites

### System Requirements
Before we begin, ensure your system meets these requirements:

- **Operating System**: macOS 12+, Ubuntu 20.04+, or Windows 10+
- **Python**: Version 3.8 or higher
- **Memory**: At least 4GB RAM (8GB recommended)
- **Storage**: 2GB free space for installation and data

### Required Software
```bash
# Check Python version
python3 --version

# Check pip version
pip --version
```

---

## Environment Setup

### Step 1: Clone the Repository

```bash
# Clone the toolkit repository
git clone https://github.com/your-org/scientific-computing-toolkit.git
cd scientific-computing-toolkit

# Verify the contents
ls -la
```

You should see directories like `data/`, `Corpus/`, `docs/`, and various Python files.

### Step 2: Install Dependencies

#### Option A: Using pip (Recommended)
```bash
# Install core dependencies
pip install numpy scipy matplotlib

# Install additional scientific libraries
pip install pandas sympy jupyter

# Verify installation
python3 -c "import numpy as np; import scipy; import matplotlib.pyplot as plt; print('✅ All dependencies installed successfully!')"
```

#### Option B: Using pixi (Advanced)
```bash
# If you have pixi installed
pixi install

# Or install specific dependencies
pixi run pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
# Run the integration test
python3 complete_integration_test.py
```

You should see output like:
```
Starting complete integration test...
Testing directory structure...
✅ PASSED
Testing data directory...
✅ PASSED
...
Overall Status: ✅ PASSED
```

---

## Your First Analysis

### Step 1: Run a Simple Rheological Analysis

Let's start with a basic Herschel-Bulkley fluid analysis:

```bash
# Navigate to the scientific computing tools directory
cd scientific-computing-tools

# Run the Herschel-Bulkley demonstration
python3 hb_demo.py
```

This will generate several output files including:
- `hb_flow_curve.png` - Flow curve visualization
- `hb_rheogram_newtonian.png` - Newtonian comparison
- `hb_rheogram_hb_(shear-thinning).png` - Shear-thinning behavior
- `hb_rheogram_bingham.png` - Bingham plastic comparison

### Step 2: Examine the Results

```bash
# List generated files
ls *.png

# Open the main flow curve (if on macOS)
open hb_flow_curve.png
```

The analysis will show:
- **Yield stress** determination
- **Flow behavior index** calculation
- **Consistency index** estimation
- **Correlation coefficients** validation

### Step 3: Run Interactive Analysis

```bash
# Run the interactive showcase
python3 interactive_showcase.py
```

This provides an interactive environment where you can:
- Adjust rheological parameters
- Visualize different fluid behaviors
- Compare Newtonian vs. non-Newtonian fluids
- Export results for publications

---

## Understanding the Framework

### Core Components Overview

The toolkit consists of several integrated frameworks:

#### 1. Rheological Framework (`hbflow/`)
```python
# Basic usage
from hbflow.models import hb_tau_from_gamma, fit_herschel_bulkley

# Calculate shear stress from shear rate
tau = hb_tau_from_gamma(gamma_dot=10.0, tau_y=5.0, K=2.0, n=0.8)

# Fit Herschel-Bulkley model to experimental data
params = fit_herschel_bulkley(stress_data, shear_rate_data)
```

#### 2. Optical Analysis (`optical_depth_enhancement.py`)
```python
# Basic optical depth enhancement
from optical_depth_enhancement import OpticalDepthAnalyzer

analyzer = OpticalDepthAnalyzer()
enhanced_depth = analyzer.enhance_depth(raw_depth_data)
```

#### 3. Biological Transport (`biological_transport_modeling.py`)
```python
# Biological nutrient transport analysis
from biological_transport_modeling import BiologicalNutrientTransport

transport = BiologicalNutrientTransport()
results = transport.simulate_tissue_nutrition('cartilage', 'glucose', conditions)
```

#### 4. Cryptographic Analysis (`cryptographic_analysis.py`)
```python
# Post-quantum cryptographic analysis
from cryptographic_analysis import PostQuantumAnalyzer

analyzer = PostQuantumAnalyzer()
security_metrics = analyzer.analyze_key_strength(key_data)
```

### Integration Architecture

The toolkit uses a modular integration architecture:

```
Data Sources → Processing Frameworks → Integration Layer → Results
     ↓              ↓                        ↓              ↓
  data/        Corpus/qualia/          data_output/      docs/
```

---

## Advanced Workflows

### Workflow 1: Complete Rheological Study

```bash
# 1. Run comprehensive rheological analysis
cd scientific-computing-tools

# Execute all rheological demonstrations
python3 complex_fluids_demo.py
python3 advanced_rheology_demo.py
python3 thixotropic_integration_demo.py

# 2. Generate publication-ready visualizations
python3 image_gallery.py

# 3. Validate results
python3 -m pytest test_herschel_bulkley.py -v
```

### Workflow 2: Multi-Framework Analysis

```bash
# Run integrated analysis across multiple domains
cd data_output

# Execute the complete integration pipeline
python3 integration_runner.py --all

# Check results
ls results/
ls reports/
ls visualizations/
```

### Workflow 3: Research Publication Pipeline

```bash
# 1. Generate research data
cd scientific-computing-tools
python3 inverse_precision_framework.py

# 2. Create publication figures
python3 advanced_rheology_demo.py

# 3. Export results for LaTeX
python3 data_export.py --format latex --output ../docs/

# 4. Compile research paper
cd ..
./compile_paper.sh
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# If you get import errors, check Python path
python3 -c "import sys; print(sys.path)"

# Install missing dependencies
pip install numpy scipy matplotlib

# Or reinstall the toolkit
pip install -e .
```

#### 2. File Not Found Errors
```bash
# Check current directory
pwd

# Navigate to correct location
cd scientific-computing-toolkit/scientific-computing-tools

# Verify files exist
ls *.py
```

#### 3. Memory Issues
```bash
# Check available memory
free -h  # On Linux
vm_stat   # On macOS

# Run with reduced memory usage
export PYTHONOPTIMIZE=1
python3 your_script.py
```

#### 4. Permission Errors
```bash
# Fix file permissions
chmod +x *.sh
chmod +x *.py

# Or run with sudo if necessary
sudo python3 your_script.py
```

### Getting Help

#### Documentation Resources
- **Main Documentation**: `docs/index.md`
- **API Reference**: `docs/api/`
- **Troubleshooting Guide**: `docs/troubleshooting.md`
- **Examples**: `docs/examples/`

#### Community Support
- **GitHub Issues**: Report bugs and request features
- **Discussion Forum**: Community Q&A and best practices
- **Email Support**: enterprise@scientific-computing-toolkit.com

---

## Next Steps

### Continue Your Learning Journey

#### 1. Explore Advanced Topics
```bash
# Advanced rheological modeling
cd scientific-computing-tools
python3 thixotropic_structure_demo.py

# Optical precision analysis
python3 optical_depth_enhancement.py

# Biological transport modeling
python3 plant_biology_model.py
```

#### 2. Customize the Toolkit
```python
# Create your own analysis module
from scientific_computing_tools.base import BaseAnalysis

class MyAnalysis(BaseAnalysis):
    def __init__(self):
        super().__init__()

    def analyze(self, data):
        # Your custom analysis logic
        return self.process_data(data)
```

#### 3. Integrate with Your Research
```python
# Example: Integrate with existing research workflow
import scientific_computing_tools as sct

# Load your experimental data
data = sct.load_data('my_experiment.json')

# Apply toolkit analysis
results = sct.analyze_rheology(data)

# Generate publication figures
sct.create_publication_figures(results, 'my_paper_figures/')
```

#### 4. Contribute to the Community
```bash
# Fork the repository
git fork https://github.com/your-org/scientific-computing-toolkit.git

# Create a feature branch
git checkout -b feature/my-awesome-contribution

# Make your changes and submit a pull request
```

### Advanced Tutorials

1. **Rheological Parameter Estimation**: Deep dive into HB model fitting
2. **Optical Depth Enhancement**: 3500x precision improvement techniques
3. **Biological Transport Modeling**: Multi-scale nutrient analysis
4. **Cryptographic Security Analysis**: Post-quantum algorithm evaluation
5. **Integration Pipeline Development**: Custom workflow creation

### Professional Development

1. **Research Integration**: Connect with existing research pipelines
2. **Industry Applications**: Pharmaceutical, materials science, AI use cases
3. **Performance Optimization**: Scaling for large datasets
4. **Custom Framework Development**: Extend toolkit capabilities

---

## Congratulations! 🎉

You've successfully completed the Getting Started tutorial for the Scientific Computing Toolkit! You now have:

- ✅ A working development environment
- ✅ Experience with rheological analysis
- ✅ Understanding of the framework architecture
- ✅ Ability to run integrated workflows
- ✅ Knowledge of troubleshooting common issues
- ✅ Pathways for continued learning and contribution

### What's Next?

Choose your learning path:

**For Researchers**: Dive into [Advanced Rheological Modeling](advanced_rheology_tutorial.md)
**For Developers**: Explore [Framework Integration](integration_tutorial.md)
**For Industry Users**: Check out [Industry Applications](industry_applications.md)
**For Contributors**: Read [Contributing Guide](CONTRIBUTING.md)

---

**Happy computing! 🔬⚗️🧪**

*Scientific Computing Toolkit Team*
