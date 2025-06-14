# ReRAM Crossbar Simulator - Visualization & Monitoring Guide

This document describes the enhanced visualization and monitoring capabilities of the ReRAM Crossbar Simulator.

## 🎯 Overview

The simulator now provides three levels of visualization and analysis:

1. **Live Visualization** - Real-time monitoring during execution
2. **Interactive Architecture Explorer** - Menu-driven hardware exploration  
3. **Comprehensive Text Reports** - Detailed post-execution analysis

## 🔄 Live Visualization

### Purpose
Monitor your ReRAM crossbar execution in real-time to:
- Track layer-by-layer execution progress
- Monitor hardware utilization (crossbars, memory, compute units)
- Identify performance bottlenecks as they occur
- View live metrics and operation counts

### Usage
```bash
# Basic live visualization (recommended starting point)
python main.py --model tiny_cnn --execute --live-viz

# Live visualization with cycle-accurate simulation (slower but more detailed)
python main.py --model tiny_cnn --execute --live-viz --cycle-accurate

# Live visualization with comprehensive analysis
python main.py --model sample_cnn --execute --live-viz --visualize --verbose
```

### What You'll See
```
🔄 LIVE RERAM SIMULATOR MONITORING
================================================================================
⏱️  Elapsed Time: 5.2s | Current Layer: 1
================================================================================

🧠 LAYER EXECUTION PROGRESS
----------------------------------------
   🔄 Layer 0 (conv2d   ): [████████████████████] 100.0%
   ✅ Layer 1 (pooling ): [██████████░░░░░░░░░░]  50.0%
   ⏸️ Layer 2 (dense   ): [░░░░░░░░░░░░░░░░░░░░]   0.0%

🔧 HARDWARE ACTIVITY
----------------------------------------
   🔥 ST0_T0_XB0: 1569 ops (95.2%)
   🔸 ST0_T1_XB0:  425 ops (45.1%)
   📊 Active Crossbars: 2 | Total Ops: 1,994

💾 MEMORY ACTIVITY
----------------------------------------
   💥 global_buffer   :   12 ops |  13.0 cyc |  15.2%
   📦 shared_buffers  :    5 ops |   4.0 cyc |   8.1%

📈 LIVE METRICS
----------------------------------------
   🚀 Crossbar Ops/sec: 384.2
   💾 Memory Operations: 17
   ⚡ Total Crossbar Ops: 1,994

📊 Press Ctrl+C to stop monitoring...
```

### Features
- **Real-time updates** every 500ms
- **Progress tracking** for each DNN layer
- **Hardware utilization** with visual indicators:
  - 🔥 Very active (updated <1s ago)
  - 🔸 Active (updated <5s ago)  
  - ⚪ Idle (updated >5s ago)
- **Live performance metrics** including operations per second
- **Memory system monitoring** with latency and utilization

## 🏗️ Interactive Architecture Explorer

### Purpose
Interactively explore your ReRAM chip architecture to:
- Understand the hierarchical organization (Chip → SuperTiles → Tiles → Crossbars)
- Inspect individual component statistics
- Analyze memory hierarchy details
- View comprehensive system statistics

### Usage
```bash
# Interactive architecture exploration (requires interactive terminal)
python main.py --explore-arch

# Note: This feature requires an interactive terminal environment
# and may not work in automated scripts or some IDEs
```

### Navigation Menu
```
🔍 ARCHITECTURE EXPLORATION MENU
============================================================
1. 🎯 Chip Overview
2. 🏢 SuperTile Details
3. 🏠 Tile Details
4. ⚡ Crossbar Arrays
5. 💾 Memory Hierarchy
6. 📊 Detailed Statistics
q. 🚪 Quit Explorer
```

### How to Navigate
- **Enter a number (1-6)** and press Enter to select menu options
- **Enter 'q'** and press Enter to quit any menu level
- **Use arrow keys** for scrolling through longer displays
- **Press Ctrl+C** to force exit if needed

### Example Exploration Session
1. **Chip Overview**: See total crossbars, ReRAM cells, current activity
2. **SuperTile Details**: Drill down into each SuperTile's performance
3. **Crossbar Arrays**: View individual crossbar operations and resistance patterns
4. **Memory Hierarchy**: Analyze buffer utilization and access patterns

## 📊 Comprehensive Text Reports

### Purpose
Generate detailed post-execution analysis including:
- Architecture diagrams and dataflow visualization
- Performance breakdowns and bottleneck analysis
- Energy consumption analysis
- Hardware utilization charts
- Optimization recommendations

### Usage
```bash
# Generate comprehensive report (start with small model)
python main.py --model tiny_cnn --execute --visualize

# Verbose output with detailed logging
python main.py --model sample_cnn --execute --visualize --verbose

# Save results to JSON file
python main.py --model tiny_cnn --execute --visualize --output results.json
```

### Report Sections

#### 1. Architecture Diagram
```
🔹 Chip Overview:
   ├── SuperTiles: 2
   ├── Total Tiles: 8  
   ├── Total Crossbars: 32
   └── Total ReRAM Cells: 524,288

🔹 Hierarchy Structure:
   SuperTile_0
   ├── Shared Buffer: 512 KB eDRAM
   ├── Tile_0
   │   ├── Local Buffer: 64 KB SRAM
   │   └── Crossbar_0: 128x128
```

#### 2. Dataflow Diagram
Shows how data flows through your DNN layers and hardware mapping.

#### 3. Layer Execution Timeline
```
⏱️  LAYER EXECUTION TIMELINE
────────────────────────────────────────────────────────────
Layer 0 (convolution_2d):
  [████████████████████████░░░░░░] 1,681 cycles (80.8%)
  Input: (16, 16, 1) → Output: (14, 14, 8)
```

#### 4. Detailed Hardware Analysis
```
🔧 DETAILED HARDWARE ANALYSIS
🔹 Crossbar Array Analysis:
   ├── ST0_T0_XB0: 1,569 operations
   └── Total: 32 crossbars, 1,569 operations

🔹 Peripheral Circuit Analysis:
   ├── Total ADCs: 256 units, 0 conversions
   └── Total DACs: 4096 units, 0 conversions

🔹 Memory Operation Analysis:
   ├── Global Buffer: 1 ops, 6.40e+01 J, 13.0 cyc latency
```

#### 5. Bottleneck Analysis
```
🚨 BOTTLENECK ANALYSIS
🔹 Execution Time Breakdown:
   ├── Slowest Layer: Layer 0 (convolution_2d)
   │   └── 1,681 cycles (80.8%)

🔹 Optimization Recommendations:
   ⚠️  Low IPC (0.000) - microcontroller underutilized
```

## 🛠️ Advanced Usage Examples

### Example 1: Debug Performance Issues
```bash
# Run with live monitoring to identify bottlenecks (note: slower execution)
python main.py --model tiny_cnn --execute --live-viz --cycle-accurate --verbose
```

### Example 2: Comprehensive System Analysis  
```bash
# Generate complete analysis with all visualizations
python main.py --model sample_cnn --execute --live-viz --visualize --output analysis.json
```

### Example 3: Hardware Exploration
```bash
# Explore architecture without running inference (interactive terminal required)
python main.py --explore-arch
```

## 📈 Key Metrics Tracked

### Hardware Utilization
- **Crossbar Operations**: Matrix-vector multiply operations per crossbar
- **ADC/DAC Conversions**: Analog-to-digital and digital-to-analog conversions
- **Memory Operations**: Read/write operations across the memory hierarchy
- **MCU Instructions**: Microcontroller instruction execution

### Performance Metrics
- **Execution Cycles**: Cycle-accurate timing for each layer
- **Operations Per Second**: Real-time throughput measurement
- **Energy Consumption**: Component-wise energy breakdown
- **Utilization Rates**: Hardware resource utilization percentages

### Bottleneck Identification
- **Layer Timing**: Which layers consume the most execution time
- **Memory Latency**: Memory access latencies and conflicts
- **Resource Underutilization**: Idle hardware components

## 🎯 Use Cases

### Research & Development
- **Algorithm Analysis**: Understand how different DNN architectures map to ReRAM hardware
- **Hardware Optimization**: Identify optimal chip configurations for specific workloads
- **Bottleneck Discovery**: Find performance limitations in the memory hierarchy or compute units

### Education & Learning
- **Architecture Visualization**: Learn ReRAM crossbar organization through interactive exploration
- **Performance Understanding**: See real-time execution of neural network inference
- **System Integration**: Understand the interplay between crossbars, memory, and control units

### System Design
- **Capacity Planning**: Determine hardware requirements for specific DNN models
- **Power Analysis**: Analyze energy consumption patterns
- **Performance Prediction**: Estimate execution times and resource requirements

## 💡 Tips for Effective Usage

1. **Start with Live Visualization** to get an intuitive feel for execution
2. **Use Interactive Explorer** to understand your hardware configuration
3. **Generate Comprehensive Reports** for detailed analysis and documentation
4. **Combine Multiple Tools** for complete system understanding
5. **Save Results to JSON** for further analysis and comparison

## 🔧 Troubleshooting

### Live Visualization Issues

**Live Visualization Not Updating**
```bash
# Ensure your terminal supports ANSI escape sequences
echo -e "\033[31mRed text\033[0m"  # Should display red text

# If not supported, use static visualization instead:
python main.py --model tiny_cnn --execute --visualize  # Remove --live-viz
```

**Progress Appears Stuck**
```bash
# This is normal for small models that execute quickly
# Try a larger model to see more gradual progress:
python main.py --model lenet --execute --live-viz
```

### Architecture Explorer Issues

**EOF Error or Input Issues**
```bash
# Ensure you're running in an interactive terminal
# Architecture explorer doesn't work in:
# - Non-interactive environments
# - Some IDEs
# - Automated scripts

# Solution: Use a proper terminal:
# Linux/Mac: bash/zsh terminal
# Windows: WSL, PowerShell, or Command Prompt
```

**Menu Navigation Problems**
```bash
# Correct navigation:
# 1. Type a number (1-6) and press Enter
# 2. Type 'q' and press Enter to quit
# 3. Use Ctrl+C to force exit if stuck

# Incorrect: Don't just press numbers without Enter
```

### Performance Issues

**Slow Visualization Performance**
```bash
# Cycle-accurate simulation is much slower but more detailed
# For faster testing, avoid --cycle-accurate:
python main.py --model tiny_cnn --execute --live-viz  # Faster

# For detailed analysis, use cycle-accurate with smaller models:
python main.py --model tiny_cnn --execute --live-viz --cycle-accurate  # Slower but detailed
```

**High Memory Usage**
```bash
# Use smaller models for testing:
python main.py --model tiny_cnn --execute --live-viz    # ~1-3 seconds
python main.py --model sample_cnn --execute --live-viz  # ~10-30 seconds  
python main.py --model lenet --execute --live-viz       # ~1-5 minutes

# Reduce max execution cycles if needed:
python main.py --model sample_cnn --execute --live-viz --max-cycles 10000
```

### Output Issues

**Missing JSON Output Files**
```bash
# Ensure output directory exists:
mkdir -p results
python main.py --model tiny_cnn --execute --visualize --output results/analysis.json

# Check file permissions:
ls -la results/
```

**Incomplete Visualization Reports**
```bash
# Some visualizations require completed execution:
# Ensure simulation completes successfully before expecting full reports

# Check for errors in verbose mode:
python main.py --model tiny_cnn --execute --visualize --verbose
```

### Terminal Compatibility

**ANSI Color Issues**
```bash
# Test terminal color support:
python -c "print('\033[31mRed\033[32mGreen\033[33mYellow\033[0mNormal')"

# If colors don't appear, your terminal may not support ANSI
# Use static visualization instead:
python main.py --model tiny_cnn --execute --visualize
```

**WSL-Specific Issues**
```bash
# Ensure proper WSL terminal setup:
# Use Windows Terminal or WSL terminal, not Windows Command Prompt
# Install unicode font support if symbols appear as boxes
```

### Recent Fixes & Updates

**ADC Units Error (FIXED)**
The "PeripheralManager lacks sufficient ADC units" error has been resolved in recent updates. If you still encounter this:
```bash
# Update to latest version:
git pull origin main
python main.py --model tiny_cnn --execute --live-viz  # Should work now
```

**Import Path Issues**
If you encounter module import errors when running examples:
```bash
# Add src to Python path:
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
# Or use the main.py interface which handles paths automatically
```

This visualization system provides unprecedented insight into ReRAM crossbar neural network accelerator operation, making it an invaluable tool for research, development, and education.