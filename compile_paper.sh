#!/bin/bash

# Compile LaTeX Paper Script
# Usage: ./compile_paper.sh [paper_name.tex]

PAPER_NAME=${1:-"SCIENTIFIC_COMPUTING_TOOLKIT_ANALYSIS"}
OUTPUT_DIR="publication_output"

echo "🔬 Compiling Scientific Computing Toolkit Analysis Paper"
echo "=================================================="

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

# Generate bibliography if .bib file exists
if [ -f "references.bib" ]; then
    echo "📚 Processing bibliography..."
    bibtex "$OUTPUT_DIR/${PAPER_NAME}" > /dev/null 2>&1

    # Additional compilation passes for bibliography
    for i in {1..2}; do
        pdflatex -output-directory="$OUTPUT_DIR" "${PAPER_NAME}.tex" > /dev/null 2>&1
    done
fi

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
    echo "   Ready for submission to journals, conferences, or technical reports"
else
    echo "❌ PDF generation failed"
    echo "🔍 Check ${OUTPUT_DIR}/${PAPER_NAME}.log for compilation errors"
    exit 1
fi

echo ""
echo "📖 Next steps:"
echo "   1. Review PDF for formatting issues"
echo "   2. Add any missing references to references.bib"
echo "   3. Submit to target publication venue"
echo "   4. Consider creating supplementary materials"
