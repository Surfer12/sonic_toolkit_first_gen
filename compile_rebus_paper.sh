#!/bin/bash

# Compile Rebus Interpretation Paper Script
# Usage: ./compile_rebus_paper.sh

PAPER_NAME="rebus_interpretation_paper"
OUTPUT_DIR="rebus_paper_output"

echo "🧩 Compiling Rebus Interpretation Paper"
echo "======================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check if LaTeX is installed
if ! command -v pdflatex &> /dev/null; then
    echo "❌ Error: pdflatex not found. Please install LaTeX (TeX Live recommended)"
    echo "   Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "   macOS: brew install mactex"
    echo "   Windows: Install MiKTeX or TeX Live"
    exit 1
fi

# Check if input file exists
if [ ! -f "${PAPER_NAME}.tex" ]; then
    echo "❌ Error: ${PAPER_NAME}.tex not found in current directory"
    exit 1
fi

echo "📄 Compiling ${PAPER_NAME}.tex..."

# Multiple compilation passes for cross-references
for i in {1..3}; do
    echo "🔄 Pass $i/3..."
    pdflatex -output-directory="$OUTPUT_DIR" "${PAPER_NAME}.tex" > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "❌ Compilation failed on pass $i"
        echo "🔍 Check ${OUTPUT_DIR}/${PAPER_NAME}.log for details"
        exit 1
    fi
done

# Check if PDF was generated successfully
if [ -f "${OUTPUT_DIR}/${PAPER_NAME}.pdf" ]; then
    PDF_SIZE=$(stat -f%z "${OUTPUT_DIR}/${PAPER_NAME}.pdf" 2>/dev/null || stat -c%s "${OUTPUT_DIR}/${PAPER_NAME}.pdf" 2>/dev/null || echo "unknown")
    echo "✅ Compilation successful!"
    echo "📊 Output: ${OUTPUT_DIR}/${PAPER_NAME}.pdf"
    echo "📏 Size: ${PDF_SIZE} bytes"
    echo ""
    echo "📁 Generated files in ${OUTPUT_DIR}/:"
    ls -la "${OUTPUT_DIR}/" | grep "${PAPER_NAME}"
    echo ""
    echo "🎯 Publication-ready PDF generated successfully!"
    echo "   This paper presents a novel application of HB models and LSTM"
    echo "   convergence theorems to visual rebus puzzle interpretation"
else
    echo "❌ PDF generation failed"
    echo "🔍 Check ${OUTPUT_DIR}/${PAPER_NAME}.log for compilation errors"
    exit 1
fi

echo ""
echo "🧩 Paper Summary:"
echo "   Title: Hierarchical Bayesian Model and Oates' LSTM Hidden State"
echo "          Convergence Theorem for Rebus Puzzle Interpretation"
echo "   Focus: Applying advanced mathematical frameworks to interpret"
echo "          visual puzzles (ECG-like jagged lines as 'life has ups/downs')"
echo "   Methods: HB probabilistic modeling + LSTM convergence analysis"
echo "   Innovation: Cross-domain application of rigorous math to creative puzzles"

echo ""
echo "📖 This paper demonstrates the universal applicability of inverse"
echo "   problem-solving techniques across diverse domains, from fluid"
echo "   dynamics to visual puzzle interpretation."
