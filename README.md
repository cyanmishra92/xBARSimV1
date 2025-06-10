# ReRAM Crossbar Simulator (xBARSimV1) 🚀

A comprehensive, cycle-accurate simulator for ReRAM crossbar-based neural network accelerators supporting CNN and LLM workloads with WSL-friendly text-based visualization.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Working](https://img.shields.io/badge/Status-Working-green.svg)]()

## 🎯 Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd xBARSimV1
pip install -r requirements.txt

# Run basic simulation
python main.py --execute --visualize

# Run comprehensive test
python examples/simple_test.py

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
- Python 3.8+
- NumPy, Matplotlib (optional), SciPy
- Works on Linux, macOS, Windows (WSL recommended)

### Install
```bash
pip install numpy matplotlib seaborn scipy pandas

# Or use requirements file
pip install -r requirements.txt
```

## 🚀 Quick Examples

### Example 1: Basic Simulation
```bash
python main.py --execute --visualize --output results.json
```

### Example 2: Custom Hardware
```python
from src import *

# Create custom chip
crossbar_config = CrossbarConfig(rows=64, cols=64)
chip = create_default_chip(crossbar_size=(64, 64), num_crossbars=8)

# Create simple CNN
dnn_config = create_simple_cnn(input_shape=(16, 16, 1), num_classes=5)

# Run simulation
result = quick_simulation(execute_inference=True)
```

### Example 3: Multiple Models
```bash
python examples/demo_models.py
```

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
```

### Key Methods
- `chip.print_architecture_summary()` - Display hardware details
- `dnn_manager.map_dnn_to_hardware(weights)` - Map neural network
- `execution_engine.execute_inference(input)` - Run inference
- `metrics_collector.get_summary_metrics()` - Get performance data

## 📁 Examples Library

### Basic Examples
```bash
# Simple functionality test
python examples/simple_test.py

# Multiple model demos
python examples/demo_models.py

# Main interface
python main.py --execute --visualize
```

### Model Examples
1. **Tiny CNN** (8×8×1 → 3 classes): Quick testing
2. **MNIST CNN** (28×28×1 → 10 classes): Handwritten digits  
3. **CIFAR CNN** (32×32×3 → 10 classes): Object classification

### Hardware Examples
1. **Minimal**: 1 SuperTile, 1 Tile, 4 Crossbars
2. **Medium**: 2 SuperTiles, 4 Tiles each, 4 Crossbars each
3. **Large**: 4 SuperTiles, 8 Tiles each, 8 Crossbars each

## 🐛 Troubleshooting

### Common Issues

**Q: "Not enough crossbars" error**
```python
# Solution: Increase hardware size or reduce DNN
chip_config.supertiles_per_chip = 4  # More hardware
# OR
dnn_config.precision = 4  # Smaller DNN
```

**Q: Memory allocation failures**
```python
# Solution: Increase buffer sizes
tile_config.local_buffer_size = 128  # Increase from 64
```

**Q: Slow simulation**
```python
# Solution: Disable cycle-accurate simulation
execution_config = ExecutionConfig(
    enable_cycle_accurate_simulation=False
)
```

### Debug Mode
```bash
python main.py --execute --verbose --log-level DEBUG
```

### Performance Tips
1. Start with small models for testing
2. Disable cycle-accurate simulation for faster results
3. Use `quick_simulation()` for rapid prototyping
4. Enable detailed logging only when needed

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