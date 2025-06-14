# ReRAM Crossbar Simulator - Visualization & Monitoring Guide

This document describes the comprehensive visualization and monitoring capabilities of the ReRAM Crossbar Simulator.

## 🎯 Overview

The simulator provides four levels of visualization and analysis:

1. **🌐 Web-Based Dashboard** - Real-time browser visualization with 3D graphics (RECOMMENDED)
2. **🎓 Educational Web Tool** - Interactive CNN mapping tutorial for students  
3. **🔄 Live Terminal Visualization** - Real-time monitoring in terminal
4. **🏗️ Interactive Architecture Explorer** - Menu-driven hardware exploration  
5. **📊 Comprehensive Text Reports** - Detailed post-execution analysis

## 🌐 Web-Based Visualization (NEW & RECOMMENDED)

### Purpose
Experience the most advanced ReRAM simulation visualization through your web browser:
- **Real-time 3D chip visualization** with Three.js
- **Interactive crossbar heatmaps** with D3.js  
- **Live performance monitoring** with WebSocket updates
- **Educational CNN mapping tutorials** for students
- **Multi-device access** - view simulations from any device
- **Publication-quality visualizations** for research presentations

### Installation
```bash
# Install required dependencies
pip install flask>=2.3.0 flask-socketio>=5.3.0

# Verify installation
python main.py --model tiny_cnn --execute --web-viz
# Should show: "🌐 Web Dashboard: http://localhost:8080/"
```

### Usage
```bash
# Real-time monitoring dashboard (researchers)
python main.py --model sample_cnn --execute --web-viz
# Open: http://localhost:8080/

# Educational mapping tool (students)  
python main.py --model tiny_cnn --execute --web-viz
# Open: http://localhost:8080/educational

# Custom port for multi-user access
python main.py --model lenet --execute --web-viz --web-port 9090

# Combined with terminal monitoring
python main.py --model sample_cnn --execute --web-viz --live-viz --verbose
```

### 🎓 Educational Dashboard (`/educational`)

**Purpose**: Interactive step-by-step learning tool for understanding CNN-to-crossbar mapping

**Features**:
- **Step-by-Step Tutorial**: 5-stage guided learning process
- **CNN Architecture Visualization**: Interactive layer structure exploration
- **Weight Matrix Decomposition**: See how 4D tensors become 2D crossbar arrays
- **Real-time Allocation Animation**: Watch crossbars being allocated to layers
- **Interactive Tooltips**: Detailed explanations of mapping concepts
- **Progress Tracking**: Monitor learning completion across tutorial steps

**Tutorial Steps**:
1. **CNN Architecture Analysis** - Understand layer structure and parameters
2. **Weight Reshaping** - Learn 4D to 2D tensor transformations  
3. **Crossbar Allocation** - See hardware mapping in action
4. **Data Flow Visualization** - Understand execution pathways
5. **Live Execution Monitoring** - Real-time performance observation

```bash
# Start educational tool with different models
python main.py --model tiny_cnn --execute --web-viz     # Beginner friendly
python main.py --model sample_cnn --execute --web-viz   # Intermediate
python main.py --model lenet --execute --web-viz        # Advanced
```

### ⚡ Real-Time Dashboard (`/`)

**Purpose**: Professional monitoring interface for researchers and developers

**Key Panels**:

#### 🔥 Crossbar Activity Heatmap
- **32 crossbar grid** with real-time utilization coloring
- **Hover tooltips** showing operations, energy, utilization percentage
- **Activity indicators**: Red (high), Yellow (medium), Green (low), Blue (idle)
- **Smooth animations** for activity transitions

#### 🧠 3D Chip Visualization  
- **Interactive Three.js scene** with zoom, pan, rotate controls
- **Hierarchical rendering**: SuperTiles → Tiles → Crossbars
- **Real-time activity pulsing** based on utilization levels
- **Color-coded components** with utilization-based coloring
- **Lighting effects** and realistic material rendering

#### 📊 Performance Metrics Panel
- **Live throughput** (operations per second)
- **Total operations counter** with real-time updates
- **Energy consumption tracking** (mJ)
- **Execution cycle monitoring** 
- **Historical performance graphs**

#### 💾 Memory Hierarchy Monitoring
- **Global Buffer utilization** with animated progress bars
- **Shared Buffer monitoring** across SuperTiles
- **Local Buffer tracking** for individual Tiles
- **Memory access latency visualization**

#### ⚙️ Peripheral Activity Tracking
- **ADC utilization percentage** with real-time updates
- **DAC conversion monitoring** 
- **Analog-digital conversion statistics**
- **Peripheral efficiency metrics**

#### 📈 Execution Progress
- **Current layer indicator** with progress percentage
- **Layer-by-layer completion tracking**
- **Execution timeline visualization**
- **Performance predictions** and completion estimates

### 🌐 Web Interface Benefits

#### For Researchers
- **Real-time Insights**: Identify bottlenecks immediately during execution
- **Publication Quality**: High-resolution screenshots and visualizations for papers
- **Comparative Analysis**: Multiple browser tabs for simultaneous comparison
- **Remote Monitoring**: Monitor long-running simulations from anywhere
- **Data Export**: JSON export of all visualization data

#### For Educators  
- **Classroom Demonstrations**: Large screen visualization for teaching
- **Student Engagement**: Interactive tutorials keep students involved
- **Progress Tracking**: Monitor student completion of tutorial steps
- **Customizable Scenarios**: Easily switch between different CNN models
- **Assessment Integration**: Built-in understanding checkpoints

#### For Students
- **Visual Learning**: See abstract concepts in interactive 3D
- **Self-Paced Learning**: Complete tutorials at individual speed  
- **Immediate Feedback**: Real-time visualization of mapping concepts
- **Hands-On Experience**: Interactive exploration builds understanding
- **Multi-Device Access**: Study from laptop, tablet, or phone

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

### Web Visualization Troubleshooting

**Web Server Not Starting**
```bash
# Check if dependencies are installed
pip install flask flask-socketio

# Verify Python version (requires 3.8+)
python --version

# Check if port is available
python main.py --model tiny_cnn --execute --web-viz --web-port 9090
```

**Dashboard Shows "Disconnected"**
```bash
# Check browser console (F12) for WebSocket errors
# Common issues:
# 1. Firewall blocking WebSocket connections
# 2. Browser security settings
# 3. Network proxy issues

# Solution: Try different port and check firewall
python main.py --model tiny_cnn --execute --web-viz --web-port 8081
```

**3D Visualization Not Loading**
```bash
# Check browser compatibility:
# Supported: Chrome 60+, Firefox 55+, Safari 12+, Edge 79+

# Check browser console for Three.js errors
# Ensure WebGL is enabled:
# Chrome: Go to chrome://flags/ and enable WebGL
# Firefox: Go to about:config and set webgl.force-enabled to true
```

**Educational Tool Not Responding**
```bash
# Ensure simulation is running with web visualization enabled
python main.py --model sample_cnn --execute --web-viz

# Check if educational.js is loading properly
# Open browser console (F12) and look for JavaScript errors
```

**Performance Issues with Web Interface**
```bash
# Reduce update frequency by using larger models (slower execution)
python main.py --model lenet --execute --web-viz

# For faster web updates, use smaller models
python main.py --model tiny_cnn --execute --web-viz

# Limit browser tabs to reduce resource usage
```

### Browser Compatibility

**Recommended Browsers**:
- **Chrome 60+**: Full feature support, best performance
- **Firefox 55+**: Full feature support, good performance  
- **Safari 12+**: Full feature support, moderate performance
- **Edge 79+**: Full feature support, good performance

**Not Supported**:
- Internet Explorer (any version)
- Chrome < 60, Firefox < 55, Safari < 12

**Mobile Devices**:
- iOS Safari 12+: Educational tool works, 3D visualization limited
- Android Chrome 60+: Full support on newer devices
- Performance depends on device capabilities

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

**Web Visualization Dependencies**
Web visualization requires additional dependencies that are installed separately:
```bash
# Install web dependencies
pip install flask>=2.3.0 flask-socketio>=5.3.0

# Verify installation
python -c "import flask, flask_socketio; print('Web dependencies installed successfully')"
```

This comprehensive visualization system provides unprecedented insight into ReRAM crossbar neural network accelerator operation, making it an invaluable tool for research, development, and education. The new web-based interface brings professional-grade visualization capabilities to any device with a modern web browser.