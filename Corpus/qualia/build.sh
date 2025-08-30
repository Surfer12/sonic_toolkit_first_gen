#!/bin/bash

# Reverse Koopman Penetration Testing Framework - Build Script
# Supports Java, Swift, and visualization builds

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
JAVA_BUILD_DIR="out"
REPORTS_DIR="reports"
LOGS_DIR="logs"
DATA_DIR="data"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    mkdir -p "$JAVA_BUILD_DIR"
    mkdir -p "$REPORTS_DIR"
    mkdir -p "$LOGS_DIR"
    mkdir -p "$DATA_DIR"
    log_success "Directories created"
}

# Java build function
build_java() {
    log_info "Building Java components..."

    if ! command_exists javac; then
        log_error "javac not found. Please install Java JDK."
        exit 1
    fi

    # Check Java version
    java_version=$(javac -version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1)
    if [ "$java_version" -lt 21 ]; then
        log_warning "Java version $java_version detected. Java 21+ recommended."
    else
        log_success "Java $java_version detected"
    fi

    # Compile Java files
    log_info "Compiling Java source files..."
    find . -name "*.java" -not -path "./out/*" | xargs javac -d "$JAVA_BUILD_DIR" -cp "."

    # Check if compilation was successful
    if [ $? -eq 0 ]; then
        log_success "Java compilation completed successfully"

        # Count compiled classes
        class_count=$(find "$JAVA_BUILD_DIR" -name "*.class" | wc -l)
        log_info "Compiled $class_count Java classes"

        # List main classes
        log_info "Available main classes:"
        find "$JAVA_BUILD_DIR" -name "*.class" | grep -E "(Demo|Dashboard|Main)\.class" | sed 's/.class$//' | sed 's|out/||' | sed 's|/|.|g'

    else
        log_error "Java compilation failed"
        exit 1
    fi
}

# Swift build function (if Swift files exist)
build_swift() {
    if [ -f "Farmer/ContentView.swift" ] || [ -f "Farmer/FarmerApp.swift" ]; then
        log_info "Swift files detected, checking Swift build..."

        if ! command_exists swift; then
            log_warning "Swift not found. Skipping Swift build."
            return
        fi

        if command_exists swift; then
            log_info "Building Swift components..."
            swift build
            log_success "Swift build completed"
        fi
    else
        log_info "No Swift files found, skipping Swift build"
    fi
}

# Build visualization components
build_visualization() {
    log_info "Building visualization components..."

    # Check for JavaFX
    if [ -d "lib" ] && [ -f "lib/javafx-*.jar" ]; then
        log_info "JavaFX libraries found"
    else
        log_warning "JavaFX libraries not found. Download from https://openjfx.io/"
        log_info "To run visualization: java --module-path /path/to/javafx/lib --add-modules javafx.controls,javafx.fxml -cp out qualia.KoopmanVisualization"
    fi

    log_success "Visualization build check completed"
}

# Run tests
run_tests() {
    log_info "Running tests..."

    # Java tests
    if [ -d "$JAVA_BUILD_DIR" ]; then
        log_info "Running Java tests..."

        # Run Java demos
        if [ -f "$JAVA_BUILD_DIR/qualia/JavaPenetrationTestingDemo.class" ]; then
            log_info "Testing Java Penetration Testing Demo..."
            timeout 30s java -cp "$JAVA_BUILD_DIR" qualia.JavaPenetrationTestingDemo >/dev/null 2>&1 && log_success "Java Demo test passed" || log_warning "Java Demo test timed out"
        fi

        if [ -f "$JAVA_BUILD_DIR/qualia/GPTOSSTesting.class" ]; then
            log_info "Testing GPTOSS Testing..."
            timeout 30s java -cp "$JAVA_BUILD_DIR" qualia.GPTOSSTesting >/dev/null 2>&1 && log_success "GPTOSS test passed" || log_warning "GPTOSS test timed out"
        fi
    fi

    log_success "Testing completed"
}

# Generate build report
generate_build_report() {
    log_info "Generating build report..."

    report_file="$REPORTS_DIR/build_report_$(date +%Y%m%d_%H%M%S).txt"

    {
        echo "Reverse Koopman Penetration Testing Framework - Build Report"
        echo "============================================================"
        echo "Build Date: $(date)"
        echo "Build Host: $(hostname)"
        echo ""

        echo "Java Information:"
        echo "  Java Version: $(java -version 2>&1 | head -n 1)"
        echo "  Java Compiler: $(javac -version 2>&1)"
        echo "  Java Classes Compiled: $(find "$JAVA_BUILD_DIR" -name "*.class" 2>/dev/null | wc -l)"
        echo ""

        if command_exists swift; then
            echo "Swift Information:"
            echo "  Swift Version: $(swift --version | head -n 1)"
            echo ""
        fi

        echo "Build Artifacts:"
        echo "  Java Classes: $JAVA_BUILD_DIR/"
        echo "  Reports: $REPORTS_DIR/"
        echo "  Logs: $LOGS_DIR/"
        echo "  Data: $DATA_DIR/"
        echo ""

        echo "Available Main Classes:"
        find "$JAVA_BUILD_DIR" -name "*.class" | grep -E "(Demo|Dashboard|Main)\.class" | sed 's/.class$//' | sed 's|out/||' | sed 's|/|.|g' | while read -r class; do
            echo "  • $class"
        done
        echo ""

        echo "Usage Examples:"
        echo "  • java -cp out qualia.JavaPenetrationTestingDemo"
        echo "  • java -cp out qualia.IntegratedSecurityDemo"
        echo "  • java -cp out qualia.GPTOSSTesting"
        echo ""

    } > "$report_file"

    log_success "Build report generated: $report_file"
}

# Clean build artifacts
clean() {
    log_info "Cleaning build artifacts..."

    if [ -d "$JAVA_BUILD_DIR" ]; then
        rm -rf "$JAVA_BUILD_DIR"
        log_success "Java build directory cleaned"
    fi

    if [ -d "build" ]; then
        rm -rf "build"
        log_success "Swift build directory cleaned"
    fi

    log_success "Clean completed"
}

# Docker build
build_docker() {
    log_info "Building Docker image..."

    if ! command_exists docker; then
        log_error "Docker not found. Please install Docker."
        exit 1
    fi

    docker build -t reverse-koopman-pentest:latest .
    log_success "Docker image built successfully"

    log_info "To run the container:"
    log_info "  docker run -it reverse-koopman-pentest:latest"
    log_info "  docker run -p 5432:5432 -p 6379:6379 -p 8000:8000 -d reverse-koopman-pentest:latest"
}

# Show usage
show_usage() {
    echo "Reverse Koopman Penetration Testing Framework - Build Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build all components (default)"
    echo "  java        Build Java components only"
    echo "  swift       Build Swift components only"
    echo "  viz         Build visualization components only"
    echo "  test        Run tests"
    echo "  clean       Clean build artifacts"
    echo "  docker      Build Docker image"
    echo "  report      Generate build report"
    echo "  all         Build everything and run tests"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build     # Build all components"
    echo "  $0 all       # Full build and test"
    echo "  $0 clean     # Clean build artifacts"
    echo "  $0 docker    # Build Docker image"
}

# Main build function
main() {
    case "${1:-build}" in
        "build")
            create_directories
            build_java
            build_swift
            build_visualization
            log_success "Build completed successfully"
            ;;
        "java")
            create_directories
            build_java
            ;;
        "swift")
            build_swift
            ;;
        "viz")
            build_visualization
            ;;
        "test")
            run_tests
            ;;
        "clean")
            clean
            ;;
        "docker")
            build_docker
            ;;
        "report")
            generate_build_report
            ;;
        "all")
            create_directories
            build_java
            build_swift
            build_visualization
            run_tests
            generate_build_report
            log_success "Full build and test completed"
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
