# ReRAM Crossbar Simulator (xBARSimV1) 🚀

A comprehensive, cycle-accurate simulator for ReRAM crossbar-based neural network accelerators with WSL-friendly text-based visualization. The current focus is on CNN-style networks, with planned LLM support.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Working](https://img.shields.io/badge/Status-Working-green.svg)]()

## 🎯 Quick Start

```bash
# Clone and setup
git clone https://github.com/cyanmishra92/xBARSimV1.git
cd xBARSimV1

# Create virtual environment (recommended)
python -m venv xbarsim_env
source xbarsim_env/bin/activate  # Linux/Mac
# or: xbarsim_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For web visualization (optional but recommended)
pip install flask flask-socketio

# Verify installation
python main.py --model tiny_cnn --execute

# Run with web-based visualization 🆕
python main.py --model sample_cnn --execute --web-viz
# Then open: http://localhost:8080/

# Run with traditional text visualization
python main.py --model sample_cnn --execute --visualize

# Try different models
python examples/demo_models.py
```

## 📋 Table of Contents

- [Features](#-features)
- [Architecture Overview](#-architecture-overview)
- [Installation](#-installation)
- [Quick Examples](#-quick-examples)
- [Detailed Documentation](#-detailed-documentation)
- [Hardware Modeling](#-hardware-modeling)
- [Neural Network Support](#-neural-network-support)
- [Configuration Guide](#-configuration-guide)
- [Performance Analysis](#-performance-analysis)
- [Visualization](#-visualization)
- [API Reference](#-api-reference)
- [Examples Library](#-examples-library)
- [Troubleshooting](#-troubleshooting)

## 🌟 Features

### ✅ **Complete Hardware Stack**
- **ReRAM Crossbar Arrays**: Individual cell physics, analog computation, device variability
- **Hierarchical Architecture**: Crossbars → Tiles → SuperTiles → Chips
- **Peripheral Circuits**: ADCs, DACs, sense amplifiers, drivers with realistic timing
- **Memory Hierarchy**: Multi-level buffers (SRAM/eDRAM/DRAM) with controllers
- **Interconnect Networks**: Mesh/Bus/Crossbar topologies with congestion modeling
- **Microcontroller**: Instruction scheduling, pipeline, control flow

### ✅ **Neural Network Support**
- **CNN Layers**: Convolution, Pooling, Batch Normalization, Activation
- **Dense Layers**: Fully connected with matrix-vector multiplication
- **Multi-bit Precision**: Configurable precision with shift-and-add units
- **Automatic Mapping**: DNN-to-hardware mapping with capacity validation
- **Performance Optimization**: Hardware recommendations for given DNNs

### ✅ **Advanced Visualization & Monitoring** 🆕
- **🌐 Web-Based Dashboard**: Real-time monitoring via browser interface with `--web-viz`
- **🎓 Educational CNN Mapping Tool**: Interactive step-by-step visualization for learning
- **🔥 3D Hardware Visualization**: Interactive Three.js chip architecture exploration
- **📊 Real-time Heatmaps**: Live crossbar activity and memory utilization tracking
- **⚡ Live Performance Metrics**: Operations/sec, energy consumption, execution cycles
- **🔄 Live Visualization**: Terminal-based real-time monitoring with `--live-viz`
- **🏗️ Interactive Architecture Explorer**: Menu-driven hardware exploration with `--explore-arch`
- **📈 Comprehensive Text Reports**: Detailed hardware analysis, bottleneck identification, energy breakdown

### ✅ **Simulation Features**
- **Cycle-Accurate Timing**: Precise modeling of all hardware components
- **Energy Modeling**: Power consumption tracking for all operations
- **Functional Execution**: Actually runs CNN inference with correct results
- **WSL-Friendly Visualization**: Text-based diagrams and performance charts
- **Comprehensive Metrics**: Latency, throughput, utilization, accuracy analysis

## 🏗️ Architecture Overview

### Hardware Hierarchy
```
ReRAM Chip
├── SuperTile 0
│   ├── Shared Buffer (eDRAM) 
│   ├── Tile 0
│   │   ├── Local Buffer (SRAM)
│   │   ├── Crossbar 0 (128×128 ReRAM)
│   │   ├── Crossbar 1 (128×128 ReRAM)
│   │   └── ...
│   ├── Tile 1
│   └── ...
├── SuperTile 1
├── Global Buffer (DRAM)
├── Microcontroller
└── Interconnect Network
```

### Key Components
1. **ReRAM Crossbar**: Stores weights, performs analog matrix-vector multiplication
2. **Peripheral Circuits**: ADCs/DACs for analog-digital conversion
3. **Compute Units**: Shift-add, activation, pooling, normalization
4. **Memory System**: Hierarchical buffers with realistic latencies
5. **Microcontroller**: Operation scheduling and control
6. **Interconnect**: Data routing with timing and energy modeling

## 💻 Installation

### Requirements
- **Python**: 3.8+ (tested on 3.8-3.11)
- **Operating System**: Linux, macOS, Windows (WSL recommended for Windows)
- **System Dependencies**: Build tools for scientific computing (see below)

### System Setup (Ubuntu/WSL)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install build-essential python3-dev python3-venv
```

### Python Environment Setup
```bash
# 1. Create virtual environment (strongly recommended)
python -m venv xbarsim_env
source xbarsim_env/bin/activate  # Linux/Mac
# Windows: xbarsim_env\Scripts\activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Verify installation
python main.py --model tiny_cnn --execute
```

### Web Visualization Installation 🆕
```bash
# Install web visualization dependencies (recommended)
pip install flask>=2.3.0 flask-socketio>=5.3.0

# Verify web visualization works
python main.py --model tiny_cnn --execute --web-viz
# Open browser: http://localhost:8080/
```

### Manual Installation (Core Dependencies Only)
```bash
# Minimal installation for basic functionality
pip install numpy>=1.21.0 matplotlib>=3.5.0 scipy>=1.7.0 pandas>=1.3.0
```

### Development Installation
```bash
# For contributors and developers
pip install -r requirements.txt
# Includes testing (pytest), linting (black, flake8), and documentation tools
```

## 🚀 Quick Examples

### Example 1: Basic Simulation (Start Here!)
```bash
# Run default model with basic output
python main.py --model tiny_cnn --execute

# Run with detailed visualization
python main.py --model sample_cnn --execute --visualize
```

### Example 2: Web-Based Visualization 🌐 (Recommended)
```bash
# Interactive web dashboard (prompts for browser setup)
python main.py --model lenet --execute --web-viz --cycle-accurate
# 1. Server starts and displays clickable links
# 2. Open http://localhost:8080/ in browser
# 3. Press ENTER to start simulation with real-time monitoring

# Auto-start without prompt (for scripts)
python main.py --model sample_cnn --execute --web-viz --auto-start

# Custom port for multi-user access
python main.py --model lenet --execute --web-viz --web-port 9090
```

### Example 3: Live Terminal Visualization 🆕
```bash
# Basic live visualization during execution
python main.py --model tiny_cnn --execute --live-viz

# Live visualization with comprehensive analysis
python main.py --model sample_cnn --execute --live-viz --visualize --verbose
```

### Example 4: Interactive Architecture Explorer 🆕
```bash
# Explore chip architecture interactively (no execution required)
python main.py --explore-arch

# Combined: execution + live monitoring + architecture exploration
python main.py --model tiny_cnn --execute --live-viz --explore-arch
```

### Example 5: Comprehensive Analysis & Reporting 🆕
```bash
# Generate detailed hardware analysis report
python main.py --model sample_cnn --execute --visualize --output results.json

# Full analysis with all visualization features
python main.py --model lenet --execute --live-viz --visualize --verbose --output analysis.json
```

### Example 5: Custom Hardware (Programmatic Usage)
```python
#!/usr/bin/env python3
import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.hierarchy import ReRAMChip, ChipConfig, SuperTileConfig, TileConfig
from core.crossbar import CrossbarConfig
from core.dnn_manager import DNNManager, DNNConfig, LayerConfig, LayerType
from core.execution_engine import ExecutionEngine

# Create custom chip configuration
crossbar_config = CrossbarConfig(rows=64, cols=64)
tile_config = TileConfig(crossbars_per_tile=4, crossbar_config=crossbar_config)
supertile_config = SuperTileConfig(tiles_per_supertile=2, tile_config=tile_config)
chip_config = ChipConfig(supertiles_per_chip=1, supertile_config=supertile_config)

# Create chip and DNN
chip = ReRAMChip(chip_config)
layers = [
    LayerConfig(layer_type=LayerType.CONV2D, input_shape=(8, 8, 1), 
                output_shape=(6, 6, 4), kernel_size=(3, 3), weights_shape=(3, 3, 1, 4)),
    LayerConfig(layer_type=LayerType.DENSE, input_shape=(144,), 
                output_shape=(3,), weights_shape=(144, 3))
]
dnn_config = DNNConfig("CustomCNN", layers, (8, 8, 1), (3,), 8)

# Run simulation
dnn_manager = DNNManager(dnn_config, chip)
weight_data = {"layer_0": np.random.randn(3, 3, 1, 4), "layer_1": np.random.randn(144, 3)}
dnn_manager.map_dnn_to_hardware(weight_data)

engine = ExecutionEngine(chip, dnn_manager)
result = engine.execute_inference(np.random.randn(8, 8, 1))
print(f"Predicted class: {result['inference_result']['predicted_class']}")
```

### Example 6: Multiple Models Demo
```bash
# Run all example models with comparisons
python examples/demo_models.py

# Run specific examples
python examples/simple_test.py
```

## 🧠 Available Neural Network Models

The simulator includes three predefined models accessible via the `--model` flag:

### Model Specifications
| Model | Input Size | Classes | Parameters | Layers | Use Case |
|-------|------------|---------|------------|---------|----------|
| `tiny_cnn` | 8×8×1 | 3 | 144 | Conv→Pool→Dense | Quick testing |
| `sample_cnn` | 16×16×1 | 10 | 5,292 | Conv→Pool→Dense | Medium complexity |
| `lenet` | 28×28×1 | 10 | ~61K | LeNet-5 architecture | Full CNN demo |

### Running Different Models
```bash
# Quick test model (fastest execution)
python main.py --model tiny_cnn --execute

# Default model (good balance of complexity and speed)
python main.py --model sample_cnn --execute --visualize

# Full LeNet-5 model (most comprehensive)
python main.py --model lenet --execute --visualize --verbose
```

**Note**: If no `--model` is specified, `sample_cnn` is used by default.

## 🌐 Web-Based Simulation Tools

### 🎓 Educational Dashboard
Interactive learning platform for understanding CNN-to-crossbar mapping:
```bash
# Start educational tool
python main.py --model sample_cnn --execute --web-viz
# Open: http://localhost:8080/educational
```

**Features:**
- **Step-by-Step Tutorial**: Interactive walkthrough of CNN mapping process
- **Layer-by-Layer Analysis**: Visualize how each CNN layer maps to crossbars
- **Weight Matrix Decomposition**: See how 4D tensors become 2D crossbar arrays
- **Real-time Allocation**: Watch crossbars being allocated during mapping
- **Student-Friendly Interface**: Designed for learning and teaching

### ⚡ Real-Time Monitoring Dashboard
Live hardware monitoring during simulation execution:
```bash
# Start monitoring dashboard
python main.py --model tiny_cnn --execute --web-viz
# Open: http://localhost:8080/
```

**Features:**
- **🔥 Crossbar Heatmaps**: Real-time activity visualization with D3.js
- **🧠 3D Chip Visualization**: Interactive Three.js hardware exploration
- **📊 Performance Metrics**: Live throughput, energy, and utilization tracking
- **💾 Memory Monitoring**: Buffer utilization across all hierarchy levels
- **⚙️ Peripheral Activity**: ADC/DAC conversion tracking
- **📈 Execution Progress**: Layer-by-layer execution monitoring

### 🎯 Key Web Visualization Benefits
- **Immediate Feedback**: See simulation results in real-time via browser
- **Multi-User Access**: Multiple researchers can view same simulation
- **Educational Value**: Perfect for classroom demonstrations
- **Publication Quality**: High-resolution exports for papers
- **Cross-Platform**: Works on any device with a web browser

### 📱 Web Interface Components

#### Real-Time Dashboard (`http://localhost:8080/`)
- **Crossbar Activity Grid**: 32 crossbars with color-coded utilization
- **Memory Hierarchy Bars**: Global, shared, and local buffer utilization
- **Performance Metrics Panel**: Operations/sec, energy consumption, cycles
- **3D Chip Visualization**: Interactive hardware exploration
- **Execution Progress**: Current layer and completion percentage

#### Educational Tool (`http://localhost:8080/educational`)
- **Tutorial Navigation**: 5-step interactive learning process
- **CNN Model Viewer**: Layer structure and parameter visualization
- **Crossbar Mapping Viewer**: Real-time allocation demonstration
- **Weight Matrix Analysis**: Tensor decomposition explanations
- **Progress Tracking**: Step completion and learning analytics

## 🎯 Traditional Visualization Features

### 🔄 Live Terminal Visualization
Monitor your ReRAM execution in real-time via terminal:
- **Layer Progress**: See execution progress for each DNN layer
- **Hardware Activity**: Live crossbar operations and utilization
- **Memory Monitoring**: Real-time buffer access patterns and latencies
- **Performance Metrics**: Operations per second, energy consumption

```bash
python main.py --execute --live-viz
```

### 🏗️ Interactive Architecture Explorer
Explore your hardware configuration via terminal interface:
- **Chip Overview**: Total crossbars, memory hierarchy, capacity
- **Component Details**: Drill down into SuperTiles, Tiles, Crossbars
- **Live Statistics**: Real-time hardware utilization and operation counts
- **Memory Analysis**: Buffer utilization and access patterns

```bash
python main.py --explore-arch
```

### 📊 Comprehensive Hardware Analysis
Get detailed insights into:
- **Crossbar Operations**: Individual crossbar activity tracking
- **ADC/DAC Usage**: Analog-digital conversion statistics  
- **Memory Operations**: Buffer access patterns and latencies
- **Bottleneck Analysis**: Performance optimization recommendations
- **Energy Breakdown**: Component-wise power consumption

See [VISUALIZATION.md](VISUALIZATION.md) for complete documentation.

## 📚 Detailed Documentation

### Core Classes

#### `ReRAMChip` - Main Chip Class
```python
# Create chip configuration
chip_config = ChipConfig(
    supertiles_per_chip=2,
    supertile_config=SuperTileConfig(
        tiles_per_supertile=4,
        tile_config=TileConfig(
            crossbars_per_tile=4,
            crossbar_config=CrossbarConfig(rows=128, cols=128)
        )
    )
)

chip = ReRAMChip(chip_config)
chip.print_architecture_summary()
```

#### `DNNManager` - Neural Network Manager
```python
# Define CNN layers
layers = [
    LayerConfig(
        layer_type=LayerType.CONV2D,
        input_shape=(32, 32, 3),
        output_shape=(30, 30, 32),
        kernel_size=(3, 3),
        activation="relu",
        weights_shape=(3, 3, 3, 32)
    ),
    # ... more layers
]

dnn_config = DNNConfig(model_name="MyCNN", layers=layers)
dnn_manager = DNNManager(dnn_config, chip)

# Map to hardware
layer_mappings = dnn_manager.map_dnn_to_hardware(weight_data)
```

#### `ExecutionEngine` - Inference Engine
```python
execution_engine = ExecutionEngine(chip, dnn_manager)
result = execution_engine.execute_inference(input_data)

print(f"Predicted class: {result['inference_result']['predicted_class']}")
print(f"Execution cycles: {result['total_execution_cycles']}")
```

## 🔧 Hardware Modeling

### ReRAM Crossbar Configuration
```python
crossbar_config = CrossbarConfig(
    rows=128, cols=128,           # Array dimensions
    r_on=1e3, r_off=1e6,          # Resistance states (Ohms)
    v_read=0.2, v_write=3.0,      # Operating voltages (V)
    device_variability=0.1,       # Device-to-device variation
    endurance=1e6,                # Write/erase cycles
    retention_time=1e6            # Data retention (seconds)
)
```

### Memory Hierarchy
```python
buffer_configs = {
    'global_buffer': MemoryConfig(
        memory_type=MemoryType.DRAM,
        size_kb=16384,               # 16 MB
        read_latency=10,             # cycles
        write_latency=12,
        banks=8
    ),
    'local_buffers': MemoryConfig(
        memory_type=MemoryType.SRAM,
        size_kb=64,                  # 64 KB
        read_latency=1,
        write_latency=1,
        banks=2
    )
}
```

### Peripheral Circuits
```python
adc_config = ADCConfig(
    type=ADCType.SAR,
    resolution=8,                    # bits
    sampling_rate=1e6,               # Hz
    power_consumption=1e-3,          # W
    conversion_time=1e-6             # seconds
)

dac_config = DACConfig(
    type=DACType.CURRENT_STEERING,
    resolution=8,
    settling_time=1e-7,
    power_consumption=5e-4
)
```

## 🧠 Neural Network Support

### Supported Layer Types
```python
# Convolution Layer
LayerConfig(
    layer_type=LayerType.CONV2D,
    input_shape=(32, 32, 3),
    output_shape=(30, 30, 32),
    kernel_size=(3, 3),
    stride=(1, 1),
    padding="valid",
    activation="relu"
)

# Pooling Layer
LayerConfig(
    layer_type=LayerType.POOLING,
    input_shape=(30, 30, 32),
    output_shape=(15, 15, 32),
    kernel_size=(2, 2),
    stride=(2, 2)
)

# Dense Layer
LayerConfig(
    layer_type=LayerType.DENSE,
    input_shape=(7200,),
    output_shape=(10,),
    activation="softmax"
)
```

### Activation Functions
- ReLU, Sigmoid, Tanh
- Leaky ReLU, Swish, GELU
- Softmax for classification

### Precision Control
```python
dnn_config = DNNConfig(
    model_name="MyModel",
    layers=layers,
    precision=8  # 8-bit quantization
)
```

## ⚙️ Configuration Guide

### Hardware Sizing
```python
# Small configuration (testing)
small_config = ChipConfig(
    supertiles_per_chip=1,
    supertile_config=SuperTileConfig(
        tiles_per_supertile=2,
        tile_config=TileConfig(crossbars_per_tile=2)
    )
)

# Large configuration (production)
large_config = ChipConfig(
    supertiles_per_chip=4,
    supertile_config=SuperTileConfig(
        tiles_per_supertile=8,
        tile_config=TileConfig(crossbars_per_tile=8)
    )
)
```

### Execution Configuration
```python
# Fast simulation
fast_config = ExecutionConfig(
    enable_cycle_accurate_simulation=False,
    enable_energy_modeling=True,
    max_execution_cycles=10000
)

# Detailed simulation
detailed_config = ExecutionConfig(
    enable_cycle_accurate_simulation=True,
    enable_energy_modeling=True,
    enable_detailed_logging=True,
    max_execution_cycles=100000
)
```

## 📊 Performance Analysis

### Metrics Collection
```python
# Get comprehensive statistics
stats = chip.get_total_statistics()
memory_stats = buffer_manager.get_all_statistics()
compute_stats = compute_manager.get_all_statistics()

# Performance metrics
print(f"Total operations: {stats['performance']['total_operations']}")
print(f"Energy consumption: {stats['energy']['total_energy']} J")
print(f"Average latency: {memory_stats['global_buffer']['memory_stats']['average_latency']} cycles")
```

### Hardware Utilization
```python
validation = dnn_manager.validate_hardware_capacity()
print(f"Crossbar utilization: {validation['utilization']['crossbar_utilization']:.1%}")
print(f"Memory utilization: {validation['utilization']['memory_utilization']:.1%}")
```

## 🎨 Visualization

### Text-Based Architecture Diagrams
```python
from visualization.text_viz import create_complete_text_report

# Generate comprehensive report
create_complete_text_report(chip, dnn_manager, execution_result)

# Individual visualizations
print_architecture_diagram(chip)
print_crossbar_heatmap(crossbar)
print_performance_summary(statistics)
```

### Sample Output
```
════════════════════════════════════════════════════════════════════════════════
📊 CHIP ARCHITECTURE DIAGRAM
════════════════════════════════════════════════════════════════════════════════
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
   │   ├── Crossbar_0: 128x128
   │   └── ...
```

## 📖 API Reference

### Command Line Interface
```bash
# Basic simulation
python main.py --model tiny_cnn --execute

# Web-based visualization (recommended)
python main.py --model sample_cnn --execute --web-viz --web-port 8080

# Terminal-based live monitoring
python main.py --model lenet --execute --live-viz --visualize

# Architecture exploration
python main.py --explore-arch

# Full analysis with output
python main.py --model sample_cnn --execute --web-viz --visualize --output results.json
```

### Core Functions
```python
# Quick setup
chip = create_default_chip(crossbar_size=(128, 128), num_crossbars=16)
dnn_config = create_simple_cnn(input_shape=(32, 32, 3), num_classes=10)
result = quick_simulation(execute_inference=True)

# Detailed setup
chip = ReRAMChip(chip_config)
dnn_manager = DNNManager(dnn_config, chip)
execution_engine = ExecutionEngine(chip, dnn_manager)
result = execution_engine.execute_inference(input_data)

# Web visualization setup
from visualization.web_viz import create_web_visualization_server
web_server = create_web_visualization_server(port=8080)
web_server.connect_execution_engine(execution_engine)
web_server.start_monitoring()
```

### Key Methods
- `chip.print_architecture_summary()` - Display hardware details
- `dnn_manager.map_dnn_to_hardware(weights)` - Map neural network
- `execution_engine.execute_inference(input)` - Run inference
- `metrics_collector.get_summary_metrics()` - Get performance data
- `web_server.start_monitoring()` - Begin real-time web visualization
- `web_server.get_real_time_stats()` - Get current hardware statistics

## 📁 Examples Library

### Basic Examples
```bash
# Simple functionality test
python examples/simple_test.py

# Multiple model demos
python examples/demo_models.py

# Main interface (runs default SampleCNN)
python main.py --execute --visualize

# Run TinyCNN with main.py
python main.py --model tiny_cnn --execute --visualize

# Run LeNet with main.py
python main.py --model lenet --execute --visualize
```

### Web Visualization Examples
```bash
# Interactive setup (recommended for first-time users)
python main.py --model lenet --execute --web-viz --cycle-accurate
# → Displays links, waits for browser setup, then starts monitoring

# Quick auto-start for experienced users
python main.py --model sample_cnn --execute --web-viz --auto-start

# Educational tool with cycle-accurate timing
python main.py --model tiny_cnn --execute --web-viz --cycle-accurate
# → Open educational tool at http://localhost:8080/educational

# Combined web + terminal visualization
python main.py --model lenet --execute --web-viz --live-viz --verbose --auto-start

# Custom port for multi-user access
python main.py --model sample_cnn --execute --web-viz --web-port 9090
```

### Model Examples
1. **Tiny CNN** (8×8×1 → 3 classes): Quick testing
2. **MNIST CNN** (28×28×1 → 10 classes): Handwritten digits  
3. **CIFAR CNN** (32×32×3 → 10 classes): Object classification

### Hardware Examples
1. **Minimal**: 1 SuperTile, 1 Tile, 4 Crossbars
2. **Medium**: 2 SuperTiles, 4 Tiles each, 4 Crossbars each
3. **Large**: 4 SuperTiles, 8 Tiles each, 8 Crossbars each

## 🧪 Testing & Validation

### Quick Installation Verification
```bash
# Verify basic functionality after installation
python main.py --model tiny_cnn --execute
# Expected: Should complete without errors and show "Simulation completed successfully!"
```

### Running the Test Suite

#### Option 1: Direct Python Execution (Recommended)
Due to pytest compatibility issues in some environments, individual test execution is more reliable:

```bash
# Test buffer operations
python3 -c "
import sys; sys.path.insert(0, 'src')
from tests.test_buffer_write_size import test_write_request_size_matches_data_length
test_write_request_size_matches_data_length()
print('✓ Buffer write size test passed')
"

# Test cycle-accurate microcontroller
python3 -c "
import sys; sys.path.insert(0, 'src')
from tests.test_cycle_accurate_mcu import test_cycle_accurate_mcu
test_cycle_accurate_mcu()
print('✓ Cycle-accurate MCU test passed')
"

# Test memory management
python3 -c "
import sys; sys.path.insert(0, 'src')
from tests.test_microcontroller_buffer import test_microcontroller_buffer_integration
test_microcontroller_buffer_integration()
print('✓ Microcontroller buffer test passed')
"
```

#### Option 2: pytest (if working in your environment)
```bash
# Install pytest if not already installed
pip install pytest

# Run all tests
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_buffer_write_size.py -v
python -m pytest tests/test_cycle_accurate_mcu.py -v

# Run with coverage (requires pytest-cov)
python -m pytest tests/ --cov=src --cov-report=html
```

#### Option 3: Run All Tests with Script
Create a test runner script:

```bash
# Create test_runner.py
cat > test_runner.py << 'EOF'
#!/usr/bin/env python3
import sys
import os
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_test(test_module, test_function, test_name):
    try:
        module = __import__(f'tests.{test_module}', fromlist=[test_function])
        test_func = getattr(module, test_function)
        test_func()
        print(f'✓ {test_name} PASSED')
        return True
    except Exception as e:
        print(f'✗ {test_name} FAILED: {e}')
        traceback.print_exc()
        return False

def main():
    tests = [
        ('test_buffer_write_size', 'test_write_request_size_matches_data_length', 'Buffer Write Size'),
        ('test_buffer_write_size', 'test_np_array_write_records_expected_size', 'NumPy Array Write'),
        ('test_cycle_accurate_mcu', 'test_cycle_accurate_mcu', 'Cycle Accurate MCU'),
        ('test_microcontroller_buffer', 'test_microcontroller_buffer_integration', 'MCU Buffer Integration'),
    ]
    
    passed = 0
    total = len(tests)
    
    print("Running xBARSimV1 Test Suite")
    print("=" * 40)
    
    for test_module, test_function, test_name in tests:
        if run_test(test_module, test_function, test_name):
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
EOF

# Run the test suite
python test_runner.py
```

### Performance Benchmarks
```bash
# Basic performance test with tiny model
time python main.py --model tiny_cnn --execute
# Expected: ~1-3 seconds

# Medium complexity test
time python main.py --model sample_cnn --execute --visualize
# Expected: ~10-30 seconds

# Full complexity test
time python main.py --model lenet --execute --visualize
# Expected: ~1-5 minutes
```

### Validation Tests
```bash
# Test all models work
python main.py --model tiny_cnn --execute && echo "✓ TinyCNN works"
python main.py --model sample_cnn --execute && echo "✓ SampleCNN works"
python main.py --model lenet --execute && echo "✓ LeNet works"

# Test visualization features
python main.py --model tiny_cnn --execute --visualize && echo "✓ Visualization works"
python main.py --explore-arch && echo "✓ Architecture explorer works"
python main.py --model tiny_cnn --execute --live-viz && echo "✓ Live visualization works"

# Test web visualization (requires Flask)
python main.py --model tiny_cnn --execute --web-viz --auto-start && echo "✓ Web visualization works"
# Manual check: Open http://localhost:8080/ and verify dashboard loads

# Test output generation
python main.py --model tiny_cnn --execute --output test_output.json && echo "✓ JSON output works"
```

### Expected Output Verification
A successful test run should show:
```
============================================================
ReRAM Crossbar Simulator v1.0
============================================================
...
5. Running inference...
   ✓ Inference completed successfully!
   Predicted class: [0-9]
   Execution cycles: [number]
============================================================
Simulation completed successfully!
============================================================
```

## 🐛 Troubleshooting

### Common Issues & Solutions

#### **Installation & Environment Issues**

**Q: Import errors when running examples**
```bash
# Error: ModuleNotFoundError: No module named 'src'
# Solution: Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
# Or in Python scripts:
import sys; sys.path.insert(0, 'src')
```

**Q: pytest segmentation fault**
```bash
# Problem: pytest crashes with segfault
# Solution: Use direct Python execution instead
python3 -c "import sys; sys.path.insert(0, 'src'); from tests.test_name import test_function; test_function()"
```

**Q: Package dependency conflicts**
```bash
# Solution: Use fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install -r requirements.txt
```

#### **Simulation Runtime Issues**

**Q: "PeripheralManager lacks sufficient ADC units" (FIXED in latest version)**
```bash
# This error has been fixed in the latest version
# If you still see it, update to the latest code:
git pull origin main
```

**Q: "Not enough crossbars" error**
```bash
# Solution 1: Use smaller model
python main.py --model tiny_cnn --execute

# Solution 2: Increase hardware size in code
# Edit main.py create_default_chip_config() to increase:
# - supertiles_per_chip = 4
# - tiles_per_supertile = 8  
# - crossbars_per_tile = 8
```

**Q: Memory allocation failures**
```bash
# Solution: Use models with fewer parameters
python main.py --model tiny_cnn --execute  # Smallest model
# Or increase buffer sizes in chip configuration
```

**Q: Slow simulation performance**
```bash
# Solution 1: Disable cycle-accurate simulation (faster but less precise)
python main.py --model tiny_cnn --execute  # No --cycle-accurate flag

# Solution 2: Use smaller models for testing
python main.py --model tiny_cnn --execute  # Instead of lenet

# Solution 3: Reduce max cycles
python main.py --model sample_cnn --execute --max-cycles 10000
```

#### **Visualization Issues**

**Q: Web visualization not working**
```bash
# Check if dependencies are installed
pip install flask flask-socketio

# Test web server manually
python main.py --model tiny_cnn --execute --web-viz
# Should show: "🌐 Web Dashboard: http://localhost:8080/"

# If port is occupied, try different port
python main.py --model tiny_cnn --execute --web-viz --web-port 9090
```

**Q: Web dashboard shows "Disconnected"**
```bash
# Check if WebSocket connection is working
# Open browser console (F12) and look for connection errors
# Try restarting the simulation with web visualization enabled
python main.py --model tiny_cnn --execute --web-viz
```

**Q: 3D visualization not loading**
```bash
# Check browser console for Three.js errors
# Ensure you have a modern browser (Chrome, Firefox, Safari, Edge)
# Try refreshing the page or restarting the simulation
```

**Q: Live terminal visualization not working**
```bash
# Check if your terminal supports ANSI escape sequences
echo -e "\033[31mRed text\033[0m"  # Should show red text
# If not supported, use static visualization:
python main.py --model tiny_cnn --execute --visualize  # Remove --live-viz
```

**Q: Architecture explorer input issues**
```bash
# Use single character inputs only: 1, 2, 3, q
# Press Enter after each selection
# Use 'q' to quit any menu level
```

#### **Output & Results Issues**

**Q: JSON output file not created**
```bash
# Ensure directory exists and is writable
mkdir -p results
python main.py --model tiny_cnn --execute --output results/test.json
```

**Q: Unexpected predicted class results**
```bash
# This is normal - models use random weights for simulation
# The focus is on hardware simulation, not ML accuracy
```

### Debugging & Logging

#### Enable Debug Mode
```bash
# Verbose output with debug information
python main.py --model tiny_cnn --execute --verbose --log-level DEBUG

# Save debug output to file
python main.py --model tiny_cnn --execute --verbose --log-file debug.log
```

#### Performance Monitoring
```bash
# Time execution
time python main.py --model tiny_cnn --execute

# Monitor memory usage (Linux/Mac)
/usr/bin/time -v python main.py --model tiny_cnn --execute
```

### Environment-Specific Issues

#### **WSL (Windows Subsystem for Linux)**
```bash
# Install required system packages
sudo apt-get update
sudo apt-get install build-essential python3-dev

# Fix locale warnings (optional)
sudo locale-gen en_US.UTF-8
export LC_ALL=en_US.UTF-8
```

#### **macOS**
```bash
# Install Xcode command line tools
xcode-select --install

# Use Homebrew Python if needed
brew install python@3.11
```

#### **Conda Environments**
```bash
# Create clean conda environment
conda create -n xbarsim python=3.11
conda activate xbarsim
pip install -r requirements.txt
```

### Performance Tips & Best Practices

1. **Start Small**: Always test with `tiny_cnn` first
2. **Use Virtual Environments**: Isolate dependencies to avoid conflicts  
3. **Monitor Resources**: Large models can consume significant memory
4. **Gradual Complexity**: tiny_cnn → sample_cnn → lenet
5. **Save Outputs**: Use `--output` flag to save results for analysis
6. **Check System Requirements**: Ensure adequate RAM for larger simulations

### Getting Help

If you encounter issues not covered here:

1. **Check Error Messages**: Most errors include helpful information
2. **Test with Tiny Model**: `python main.py --model tiny_cnn --execute`
3. **Enable Verbose Output**: Add `--verbose --log-level DEBUG`
4. **Check System Resources**: Monitor CPU and memory usage
5. **Update Code**: `git pull origin main` for latest fixes

## 🎓 Learning Resources

### Understanding ReRAM Crossbars
1. **Basic Concept**: ReRAM cells store weights as resistance values
2. **Analog Computation**: Kirchhoff's law performs matrix-vector multiplication
3. **Peripheral Circuits**: ADCs/DACs convert between analog/digital domains
4. **Energy Efficiency**: Massive parallelism with low energy per operation

### Hierarchy Design
1. **Crossbars**: Fundamental compute units (e.g., 128×128)
2. **Tiles**: Multiple crossbars sharing local resources
3. **SuperTiles**: Multiple tiles sharing regional resources  
4. **Chip**: Complete system with global memory and control

### Neural Network Mapping
1. **Weight Mapping**: Distribute CNN weights across crossbars
2. **Data Flow**: Route activations through memory hierarchy
3. **Scheduling**: Microcontroller coordinates operations
4. **Optimization**: Balance parallelism vs. resource utilization

## 🤝 Contributing

### Development Setup
```bash
git clone <repository>
cd xBARSimV1
pip install -r requirements.txt
python -m pytest tests/  # Run tests
```

### Adding New Features
1. **New Layer Types**: Extend `LayerType` enum and execution functions
2. **Hardware Components**: Add to hierarchy or peripherals modules
3. **Metrics**: Extend metrics collection classes
4. **Visualization**: Add new text-based visualization functions

### Code Style
- Follow PEP 8 conventions
- Add docstrings to all functions
- Include type hints where possible
- Add unit tests for new features

## 📄 Citation

```bibtex
@software{xbarsim2024,
  title={xBARSimV1: A Comprehensive ReRAM Crossbar Simulator for Neural Networks},
  author={Research Team},
  year={2024},
  url={https://github.com/your-repo/xBARSimV1}
}
```

## 📞 Support

- **Documentation**: This README and inline code documentation
- **Examples**: Check `examples/` directory for usage patterns
- **Issues**: Report bugs and feature requests via GitHub issues
- **Questions**: Use GitHub Discussions for questions and help

## 📜 License

MIT License - see LICENSE file for details.

---

**🎉 Happy Simulating! 🎉**

The ReRAM Crossbar Simulator provides a complete research platform for exploring novel neural network accelerator architectures. From individual device physics to full system performance, every aspect is modeled to help advance the field of in-memory computing.
