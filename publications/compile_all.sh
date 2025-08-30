#!/bin/bash

# 📚 Independent Mathematical Framework Publications - Batch Compilation Script
# This script compiles all publication-ready LaTeX documents showcasing the remarkable
# independent design and serendipitous Blackwell convergence achievement

echo "🎯 Independent Mathematical Framework Publications Compiler"
echo "=========================================================="
echo ""
echo "📖 Compiling publications that demonstrate:"
echo "   • Independent framework design (developed without Blackwell constraints)"
echo "   • 0.9987 precision criterion as original mathematical achievement"
echo "   • Serendipitous Blackwell convergence as unexpected validation"
echo "   • Fundamental computational principles transcending hardware implementations"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to compile a single document
compile_document() {
    local filename="$1"
    local basename="${filename%.tex}"

    echo -e "${BLUE}Compiling: ${basename}.tex${NC}"

    # First pass
    if pdflatex -interaction=nonstopmode "$filename" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ First pass successful${NC}"

        # Bibliography compilation if .bib file exists
        if [ -f "${basename}.bib" ]; then
            echo -e "${YELLOW}→ Processing bibliography...${NC}"
            bibtex "${basename}" > /dev/null 2>&1
            pdflatex -interaction=nonstopmode "$filename" > /dev/null 2>&1
        fi

        # Second pass
        if pdflatex -interaction=nonstopmode "$filename" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Compilation successful: ${basename}.pdf${NC}"
            return 0
        else
            echo -e "${RED}✗ Second pass failed${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ First pass failed${NC}"
        return 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -a, --all          Compile all .tex files (default)"
    echo "  -f, --file FILE    Compile specific file (without .tex extension)"
    echo "  -l, --list         List all available .tex files"
    echo "  -c, --clean        Clean auxiliary files"
    echo ""
    echo "Examples:"
    echo "  $0                    # Compile all documents"
    echo "  $0 -f core_algorithms # Compile specific document"
    echo "  $0 -l                 # List all documents"
    echo "  $0 -c                 # Clean auxiliary files"
}

# Function to list available documents
list_documents() {
    echo "📚 Available Publication Documents:"
    echo "=================================="
    echo ""

    local count=1
    for file in *.tex; do
        if [ -f "$file" ]; then
            local basename="${file%.tex}"
            echo -e "${BLUE}${count}.${NC} ${basename}.tex"

            # Try to extract title from first few lines
            local title=$(head -20 "$file" | grep -i '\\title' | head -1 | sed 's/.*{//' | sed 's/}.*//')
            if [ ! -z "$title" ]; then
                echo -e "   └─ ${YELLOW}${title}${NC}"
            fi
            echo ""
            ((count++))
        fi
    done

    echo -e "${GREEN}Total: $(ls *.tex 2>/dev/null | wc -l) publication documents${NC}"
}

# Function to clean auxiliary files
clean_auxiliary() {
    echo "🧹 Cleaning auxiliary files..."
    rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fdb_latexmk *.fls *.synctex.gz
    echo -e "${GREEN}✓ Auxiliary files cleaned${NC}"
}

# Main script
case "${1:-}" in
    -h|--help)
        show_usage
        exit 0
        ;;
    -l|--list)
        list_documents
        exit 0
        ;;
    -c|--clean)
        clean_auxiliary
        exit 0
        ;;
    -f|--file)
        if [ -z "$2" ]; then
            echo -e "${RED}Error: Please specify a filename${NC}"
            show_usage
            exit 1
        fi

        filename="$2.tex"
        if [ ! -f "$filename" ]; then
            echo -e "${RED}Error: File '$filename' not found${NC}"
            echo "Available files:"
            ls *.tex 2>/dev/null || echo "No .tex files found"
            exit 1
        fi

        echo "Compiling single document: $filename"
        compile_document "$filename"
        exit $?
        ;;
    -a|--all|"")
        # Compile all documents
        echo "📄 Compiling All Publication Documents"
        echo "====================================="
        echo ""

        success_count=0
        total_count=0

        for file in *.tex; do
            if [ -f "$file" ]; then
                ((total_count++))
                echo ""
                if compile_document "$file"; then
                    ((success_count++))
                fi
            fi
        done

        echo ""
        echo "📊 Compilation Summary:"
        echo "======================"
        echo -e "${GREEN}Successful: ${success_count}/${total_count}${NC}"

        if [ $success_count -eq $total_count ]; then
            echo -e "${GREEN}🎉 All documents compiled successfully!${NC}"
        else
            echo -e "${YELLOW}⚠️  Some documents had compilation issues${NC}"
        fi

        exit 0
        ;;
    *)
        echo -e "${RED}Error: Unknown option '$1'${NC}"
        show_usage
        exit 1
        ;;
esac
