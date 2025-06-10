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
# Basic live visualization
python main.py --execute --live-viz

# Live visualization with cycle-accurate simulation
python main.py --execute --live-viz --cycle-accurate

# Live visualization with comprehensive analysis
python main.py --execute --live-viz --visualize --verbose
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
python main.py --explore-arch
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
# Generate comprehensive report
python main.py --execute --visualize

# Verbose output with detailed logging
python main.py --execute --visualize --verbose

# Save results to JSON file
python main.py --execute --visualize --output results.json
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
# Run with live monitoring to identify bottlenecks
python main.py --execute --live-viz --cycle-accurate --verbose
```

### Example 2: Comprehensive System Analysis  
```bash
# Generate complete analysis with all visualizations
python main.py --execute --live-viz --visualize --explore-arch --output analysis.json
```

### Example 3: Hardware Exploration
```bash
# Explore architecture without running inference
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

### Live Visualization Not Working
- Ensure your terminal supports ANSI escape sequences
- Try running with `--visualize` instead for static reports

### Architecture Explorer Input Issues
- Use single characters (1, 2, 3, etc.) for menu navigation
- Press 'q' to quit any menu level

### Performance Issues
- Use `--cycle-accurate` only when needed (slower but more accurate)
- Reduce `--max-cycles` for faster execution during development

This visualization system provides unprecedented insight into ReRAM crossbar neural network accelerator operation, making it an invaluable tool for research, development, and education.