# 🔥 Mojo Installation & Setup Guide

## Quick Installation

### 1. Install Mojo via Modular
```bash
# Install Modular CLI
curl -s https://get.modular.com | sh -

# Install Mojo
modular install mojo

# Add to PATH (add to your ~/.zshrc or ~/.bashrc)
export PATH="$HOME/.modular/pkg/packages.modular.com_mojo/bin:$PATH"
```

### 2. Verify Installation
```bash
mojo --version
```

### 3. Run Your Mojo Implementations
```bash
cd /Users/ryan_david_oates/archive08262025202ampstRDOHomeMax

# Test each implementation
mojo inverse_news_aggregation.mojo
mojo chat_models.mojo  
mojo composite_loss.mojo
mojo eval_agent.mojo
```

## Alternative: Magic/MAX Platform
If you prefer a managed environment:

```bash
# Install Magic (Modular's package manager)
curl -ssL https://magic.modular.com/install | bash

# Create a new project
magic init my-mojo-project --format mojo
cd my-mojo-project

# Copy your .mojo files here and run
magic run mojo inverse_news_aggregation.mojo
```

## VS Code Integration
```bash
# Install Mojo extension for VS Code
code --install-extension modular-mojotools.mojo
```

## Performance Testing
Once installed, you can benchmark against your original Python/Swift code:

```bash
# Time the Mojo implementations
time mojo eval_agent.mojo

# Compare with Python equivalent
time python automated-auditing\ copy/eval_agent/eval_agent.py
```

You should see **10-100x performance improvements** immediately! 🚀
