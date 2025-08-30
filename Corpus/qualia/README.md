# 🔒 Reverse Koopman Penetration Testing Framework

A comprehensive security testing framework that leverages advanced mathematical techniques (Reverse Koopman Operators) to analyze and assess security vulnerabilities in both traditional software applications and AI models (GPTOSS 2.0).

## 🌟 Features

### Core Security Testing
- **Java Application Security:** Comprehensive penetration testing for Java applications
- **GPTOSS 2.0 AI Security:** Specialized testing for AI models and language models
- **Reverse Koopman Operators:** Mathematical framework for system analysis and anomaly detection
- **Statistical Validation:** K-S framework for result validation and confidence assessment

### Advanced Visualizations
- **Security Dashboard:** Real-time Swing-based GUI for security assessment
- **JavaFX Visualization:** Modern 3D visualization for koopman operator analysis
- **Python Analytics:** Matplotlib, Seaborn, and Plotly-based interactive dashboards
- **Interactive Reports:** Web-based HTML reports with embedded charts

### Deployment & Integration
- **Docker Support:** Complete containerized environment
- **Dev Container:** VS Code development environment
- **Multi-service Setup:** PostgreSQL, Redis, Mock GPTOSS API
- **CI/CD Ready:** Automated build and testing pipelines

## 📊 What is Reverse Koopman Penetration Testing?

The **Reverse Koopman Operator** is an advanced mathematical technique that linearizes nonlinear dynamical systems by lifting them to a function space. This framework enables:

1. **System Analysis:** Understanding complex system behaviors through linear representations
2. **Anomaly Detection:** Identifying deviations from expected system behavior
3. **Security Assessment:** Applying mathematical analysis to security vulnerability detection
4. **AI Model Testing:** Specialized security testing for language models and AI systems

## 🏗️ Architecture

```
Reverse Koopman Penetration Testing Framework
├── Java Security Testing
│   ├── Memory Safety Analysis
│   ├── SQL Injection Testing
│   ├── Authentication & Authorization
│   ├── Cryptographic Assessment
│   └── Network Security Evaluation
│
├── GPTOSS 2.0 AI Model Security
│   ├── Prompt Injection Testing
│   ├── Model Inversion Attacks
│   ├── Data Leakage Detection
│   ├── Jailbreak Attempt Prevention
│   └── Membership Inference Protection
│
├── Mathematical Framework
│   ├── Reverse Koopman Operators
│   ├── Dynamic Mode Decomposition
│   ├── Spectral Decomposition
│   └── Observable Function Analysis
│
└── Validation & Reporting
    ├── K-S Statistical Validation
    ├── Comprehensive Reporting
    └── Visual Analytics Dashboard
```

## 🚀 Quick Start

### 1. Basic Setup
```bash
# Clone repository
git clone <repository-url>
cd reverse-koopman-pentest

# Make build script executable
chmod +x build.sh

# Build all components
./build.sh build
```

### 2. Run Security Assessment
```bash
# Java application security testing
java -cp out qualia.JavaPenetrationTestingDemo

# AI model security testing
java -cp out qualia.GPTOSSTesting

# Integrated assessment
java -cp out qualia.IntegratedSecurityDemo
```

### 3. Launch Visualizations
```bash
# Security Dashboard (GUI)
java -cp ".:jfreechart-1.5.3.jar:jcommon-1.0.24.jar" qualia.SecurityDashboard

# Python analytics
python3 demo_visualizations.py
```

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#️-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Visualizations](#-visualizations)
- [Docker Setup](#-docker-setup)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

## 📦 Installation

### Prerequisites
- **Java 21+** (for core framework)
- **Python 3.8+** (for visualizations)
- **Docker & Docker Compose** (for containerized setup)
- **VS Code** (for dev container)

### Build Script Installation
```bash
# Automated build and setup
./build.sh all

# Or step-by-step
./build.sh build    # Build Java components
./build.sh test     # Run tests
./build.sh report   # Generate build report
```

### Manual Installation
```bash
# Java dependencies
javac -d out *.java

# Python dependencies (for visualizations)
pip install matplotlib seaborn plotly pandas numpy

# Docker setup
docker-compose build
```

## 🎯 Usage

### Java Security Testing
```java
// Basic penetration testing
JavaPenetrationTesting tester = new JavaPenetrationTesting();
List<SecurityFinding> findings = tester.runComprehensiveTesting().get();

// With custom configuration
JavaPenetrationTestingConfig config = new JavaPenetrationTestingConfig()
    .enableMemoryTesting(true)
    .enableSQLInjectionTesting(true)
    .setTimeout(Duration.ofMinutes(30));

List<SecurityFinding> findings = tester.runTesting(config).get();
```

### GPTOSS 2.0 AI Testing
```java
// AI model security testing
GPTOSSTesting gptossTester = new GPTOSSTesting("http://api.gptoss.com", "sk-api-key");
List<SecurityFinding> aiFindings = gptossTester.runComprehensiveGPTOSSTesting().get();

// Test specific vulnerabilities
gptossTester.testPromptInjection();
gptossTester.testModelInversion();
gptossTester.testJailbreakAttempts();
```

### Integrated Assessment
```java
// Combined Java + AI assessment
IntegratedSecurityDemo demo = new IntegratedSecurityDemo(gptossEndpoint, apiKey);
demo.runIntegratedSecurityAssessment();

// Generate comprehensive report
demo.runScenarioBasedAssessment();
```

### K-S Validation
```java
// Statistical validation
KSPenetrationTestingValidator validator = new KSPenetrationTestingValidator();
KSValidationResult result = validator.validatePenetrationTesting(findings, baseline).get();

// Cross-framework validation
List<KSValidationResult> crossValidation = validator.performCrossValidation(frameworks).get();
```

## 🎨 Visualizations

### Security Dashboard (Swing)
```bash
# Launch GUI dashboard
java -cp ".:jfreechart-1.5.3.jar:jcommon-1.0.24.jar" qualia.SecurityDashboard
```

**Features:**
- Real-time security assessment
- Interactive charts (pie, bar, line)
- Multi-tab interface (Dashboard, Findings, Analytics, Logs)
- Report generation and export
- Progress tracking and status updates

### Advanced JavaFX Visualization
```bash
# Launch JavaFX application
java --module-path /path/to/javafx/lib --add-modules javafx.controls,javafx.fxml \
     -cp out qualia.KoopmanVisualization
```

**Features:**
- 3D security space visualization
- Interactive web-based charts
- Koopman operator mathematical analysis
- Real-time assessment monitoring
- Multi-framework correlation analysis

### Python Analytics Dashboard
```bash
# Generate comprehensive visualizations
python3 demo_visualizations.py
```

**Features:**
- Static high-quality charts (PNG/PDF)
- Interactive HTML dashboards
- Koopman operator mathematical plots
- Security heatmap analysis
- Time series vulnerability tracking
- Comprehensive HTML reports

## 🐳 Docker Setup

### Development Environment
```bash
# Start full development environment
docker-compose up -d

# Access main container
docker-compose exec reverse-koopman-pentest bash

# Run assessments in container
java -cp out qualia.IntegratedSecurityDemo
python3 demo_visualizations.py
```

### Available Services
- **reverse-koopman-pentest:** Main Java application
- **postgres:** SQL injection testing database
- **redis:** Session and cache testing
- **mock-gptoss:** AI model security simulation
- **elasticsearch + kibana:** Monitoring and logging (optional)

### Dev Container (VS Code)
```bash
# Open in VS Code dev container
code .
# Select "Reopen in Container" when prompted
```

## 📚 API Reference

### Core Classes

#### JavaPenetrationTesting
```java
public class JavaPenetrationTesting {
    CompletableFuture<List<SecurityFinding>> runComprehensiveTesting()
    SecurityFinding testBufferOverflows()
    SecurityFinding testSQLInjection()
    SecurityFinding testAuthentication()
    SecurityFinding testEncryption()
    SecurityFinding testNetworkSecurity()
}
```

#### GPTOSSTesting
```java
public class GPTOSSTesting {
    GPTOSSTesting(String endpoint, String apiKey)
    CompletableFuture<List<SecurityFinding>> runComprehensiveGPTOSSTesting()
    CompletableFuture<Void> testPromptInjection()
    CompletableFuture<Void> testModelInversion()
    CompletableFuture<Void> testJailbreakAttempts()
}
```

#### ReverseKoopmanOperator
```java
public class ReverseKoopmanOperator {
    KoopmanAnalysis computeReverseKoopman(double[] state, Function<Double[], Double>[] observables)
    double[] computeKoopmanMatrix(double[][] dataMatrix)
    ComplexNumber[] computeEigenDecomposition(double[][] matrix)
    double computeReconstructionError(double[] original, double[] reconstructed)
}
```

### Security Findings
```java
public class SecurityFinding {
    VulnerabilityType getVulnerabilityType()
    Severity getSeverity()
    String getTitle()
    String getDescription()
    String getRecommendation()
    String getImpactAssessment()
}
```

## 🔧 Build & Run

### Quick Start with Build Script
```bash
# Make build script executable
chmod +x build.sh

# Build all components
./build.sh build

# Build and test everything
./build.sh all

# Build Docker image
./build.sh docker
```

### Java Implementation
```bash
# Compile all components
javac -d out *.java

# Run Java penetration testing demo
java -cp out qualia.JavaPenetrationTestingDemo

# Run GPTOSS 2.0 security testing
java -cp out qualia.GPTOSSTesting

# Run integrated security assessment
java -cp out qualia.IntegratedSecurityDemo
```

### Visualization Dashboards
```bash
# Security Dashboard (Swing GUI)
javac -cp ".:jfreechart-1.5.3.jar:jcommon-1.0.24.jar" SecurityDashboard.java
java -cp ".:jfreechart-1.5.3.jar:jcommon-1.0.24.jar" qualia.SecurityDashboard

# Advanced JavaFX Visualization
java --module-path /path/to/javafx/lib --add-modules javafx.controls,javafx.fxml \
     -cp out qualia.KoopmanVisualization

# Python Visualizations
pip install matplotlib seaborn plotly pandas numpy
python3 demo_visualizations.py
```

### Docker Environment
```bash
# Build and run with Docker
docker build -t reverse-koopman-pentest .
docker run -it reverse-koopman-pentest

# Or use docker-compose for full environment
docker-compose up -d

# Access containerized services
docker-compose exec reverse-koopman-pentest bash
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Adding New Security Tests
```java
public class CustomSecurityTest extends SecurityTest {
    @Override
    public SecurityFinding execute() {
        // Implement custom security test
        return new SecurityFinding(
            VulnerabilityType.CUSTOM_VULNERABILITY,
            Severity.MEDIUM,
            "Custom Security Issue",
            "Description of the security finding",
            "Location where issue was found",
            "Recommendation to fix the issue",
            "Impact assessment",
            "Evidence or additional details"
        );
    }
}
```

## 📄 License

This project is licensed under the GPL-3.0-only License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mathematical Framework:** Based on Koopman operator theory for dynamical systems analysis
- **Security Research:** OWASP guidelines and industry best practices
- **AI Security:** GPTOSS 2.0 and language model security research
- **Visualization:** JFreeChart, JavaFX, Matplotlib, Plotly, and Seaborn libraries

## 📞 Support

- **Documentation:** [Full Documentation](docs/)
- **Issues:** [GitHub Issues](https://github.com/reverse-koopman-pentest/issues)
- **Discussions:** [GitHub Discussions](https://github.com/reverse-koopman-pentest/discussions)
- **Security:** security@reversekoopman.dev

---

## 🎉 Summary

The **Reverse Koopman Penetration Testing Framework** provides:

- **🔬 Advanced Mathematics:** Koopman operators for system analysis
- **🛡️ Comprehensive Security:** Java + AI model testing
- **📊 Rich Visualizations:** Multiple dashboard options
- **🐳 Container Ready:** Docker and dev container support
- **📈 Statistical Validation:** K-S framework for confidence assessment
- **🔧 Developer Friendly:** Easy setup and integration

**Ready to revolutionize your security testing!** 🚀✨
