#!/bin/bash
# Corretto 24 Setup Verification Script

echo "🔧 Corretto 24 Setup Verification"
echo "================================="

# Check Java installation
echo "1. Java Version Check:"
java -version
if [ $? -ne 0 ]; then
    echo "❌ Java not found!"
    exit 1
fi

echo ""
echo "2. Java Compiler Check:"
javac -version
if [ $? -ne 0 ]; then
    echo "❌ Java compiler not found!"
    exit 1
fi

# Check JAVA_HOME
echo ""
echo "3. JAVA_HOME Check:"
if [ -z "$JAVA_HOME" ]; then
    echo "⚠️  JAVA_HOME not set"
else
    echo "✅ JAVA_HOME: $JAVA_HOME"
fi

# Check for Corretto specifically
echo ""
echo "4. Corretto Detection:"
if java -version 2>&1 | grep -q "Corretto"; then
    echo "✅ Amazon Corretto detected!"
else
    echo "⚠️  Not detecting Corretto in version string"
    echo "   (This might be normal if version string differs)"
fi

# Test Gradle
echo ""
echo "5. Gradle Check:"
if command -v gradle &> /dev/null; then
    gradle --version | head -3
    echo "✅ Gradle ready"
else
    echo "❌ Gradle not found"
fi

# Test pixi integration
echo ""
echo "6. Pixi Java Tasks:"
cd /Users/ryan_david_oates/archive08262025202ampstRDOHomeMax
echo "Available Java tasks:"
pixi task list | grep -E "(java|gradle|corretto)"

echo ""
echo "🎉 Setup verification complete!"
echo "Run 'pixi run corretto-info' to test pixi integration"
