"""
xBARSimV1: A Comprehensive ReRAM Crossbar Simulator for Neural Networks

This package provides a detailed, cycle-accurate simulator for ReRAM crossbar-based 
neural network accelerators. It models the complete hardware stack from individual 
ReRAM devices up to full chip architectures, with support for CNN and LLM workloads.

Main Components:
- Core hardware modeling (crossbars, tiles, chips)
- Neural network mapping and execution
- Cycle-accurate timing and energy modeling  
- Performance metrics and visualization
- Microcontroller and memory hierarchy simulation

Author: [Your Name]
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@domain.com"

# Core imports for easy access
from .core.crossbar import CrossbarArray, CrossbarConfig, ReRAMCell
from .core.hierarchy import ReRAMChip, ChipConfig, SuperTileConfig, TileConfig, ProcessingTile, SuperTile
from .core.dnn_manager import DNNManager, DNNConfig, LayerConfig, LayerType, HardwareRequirement
from .core.execution_engine import ExecutionEngine, ExecutionConfig, SystemIntegrator
from .core.metrics import MetricsCollector, PerformanceProfiler
from .core.microcontroller import Microcontroller, MCUConfig, Instruction, InstructionType
from .core.memory_system import BufferManager, MemoryConfig, MemoryType, PartialSumBuffer
from .core.compute_units import ComputeUnitManager, ActivationType, PoolingType, NormalizationType
from .core.interconnect import InterconnectNetwork, InterconnectConfig, InterconnectTopology, TimingModel
from .core.peripherals import PeripheralManager, ADCConfig, DACConfig, SenseAmplifierConfig, DriverConfig

# Visualization imports
from .visualization.architecture_viz import ArchitectureVisualizer, DataflowVisualizer, create_comprehensive_visualization

# Version information
__all__ = [
    # Core classes
    'CrossbarArray', 'CrossbarConfig', 'ReRAMCell',
    'ReRAMChip', 'ChipConfig', 'SuperTileConfig', 'TileConfig', 'ProcessingTile', 'SuperTile',
    'DNNManager', 'DNNConfig', 'LayerConfig', 'LayerType', 'HardwareRequirement',
    'ExecutionEngine', 'ExecutionConfig', 'SystemIntegrator',
    'MetricsCollector', 'PerformanceProfiler',
    'Microcontroller', 'MCUConfig', 'Instruction', 'InstructionType',
    'BufferManager', 'MemoryConfig', 'MemoryType', 'PartialSumBuffer',
    'ComputeUnitManager', 'ActivationType', 'PoolingType', 'NormalizationType',
    'InterconnectNetwork', 'InterconnectConfig', 'InterconnectTopology', 'TimingModel',
    'PeripheralManager', 'ADCConfig', 'DACConfig', 'SenseAmplifierConfig', 'DriverConfig',
    
    # Visualization
    'ArchitectureVisualizer', 'DataflowVisualizer', 'create_comprehensive_visualization',
    
    # Package info
    '__version__', '__author__', '__email__'
]

def get_version():
    """Get the package version."""
    return __version__

def print_info():
    """Print package information."""
    print(f"xBARSimV1 v{__version__}")
    print(f"ReRAM Crossbar Simulator for Neural Networks")
    print(f"Author: {__author__}")
    print(f"Email: {__email__}")
    print()
    print("Key Features:")
    print("- Hierarchical ReRAM crossbar architecture modeling")
    print("- Cycle-accurate timing and energy simulation")
    print("- CNN/LLM workload support with automatic mapping")
    print("- Comprehensive performance metrics and visualization")
    print("- Modular and configurable design")

# Utility functions for quick setup
def create_default_chip(crossbar_size=(128, 128), num_crossbars=16):
    """Create a default chip configuration for quick testing."""
    from .core.crossbar import CrossbarConfig
    from .core.hierarchy import ChipConfig, SuperTileConfig, TileConfig
    
    crossbar_config = CrossbarConfig(rows=crossbar_size[0], cols=crossbar_size[1])
    
    # Calculate hierarchy to accommodate requested crossbars
    crossbars_per_tile = 4
    tiles_per_supertile = 4
    supertiles_needed = max(1, (num_crossbars + 15) // 16)  # 16 crossbars per supertile
    
    tile_config = TileConfig(
        crossbars_per_tile=crossbars_per_tile,
        crossbar_config=crossbar_config
    )
    
    supertile_config = SuperTileConfig(
        tiles_per_supertile=tiles_per_supertile,
        tile_config=tile_config
    )
    
    chip_config = ChipConfig(
        supertiles_per_chip=supertiles_needed,
        supertile_config=supertile_config
    )
    
    return ReRAMChip(chip_config)

def create_simple_cnn(input_shape=(32, 32, 3), num_classes=10):
    """Create a simple CNN configuration for quick testing."""
    layers = [
        LayerConfig(
            layer_type=LayerType.CONV2D,
            input_shape=input_shape,
            output_shape=(input_shape[0]-2, input_shape[1]-2, 32),
            kernel_size=(3, 3),
            activation="relu",
            weights_shape=(3, 3, input_shape[2], 32)
        ),
        LayerConfig(
            layer_type=LayerType.POOLING,
            input_shape=(input_shape[0]-2, input_shape[1]-2, 32),
            output_shape=((input_shape[0]-2)//2, (input_shape[1]-2)//2, 32),
            kernel_size=(2, 2)
        ),
        LayerConfig(
            layer_type=LayerType.DENSE,
            input_shape=(((input_shape[0]-2)//2) * ((input_shape[1]-2)//2) * 32,),
            output_shape=(num_classes,),
            activation="softmax",
            weights_shape=(((input_shape[0]-2)//2) * ((input_shape[1]-2)//2) * 32, num_classes)
        )
    ]
    
    return DNNConfig(
        model_name="SimpleCNN",
        layers=layers,
        input_shape=input_shape,
        output_shape=(num_classes,)
    )

def quick_simulation(input_shape=(32, 32, 3), num_classes=10, execute_inference=True):
    """Run a quick simulation with default settings."""
    import numpy as np
    
    print("Setting up quick simulation...")
    
    # Create default configurations
    dnn_config = create_simple_cnn(input_shape, num_classes)
    chip = create_default_chip(num_crossbars=20)  # Ensure enough crossbars
    
    # Create DNN manager
    dnn_manager = DNNManager(dnn_config, chip)
    
    print(f"Created chip with {chip.get_chip_configuration()['compute_capacity']['total_crossbars']} crossbars")
    print(f"DNN has {dnn_config.total_parameters:,} parameters")
    
    if execute_inference:
        # Create execution engine
        execution_engine = ExecutionEngine(chip, dnn_manager)
        
        # Generate test input
        test_input = np.random.randn(*input_shape)
        
        print("Running inference...")
        result = execution_engine.execute_inference(test_input)
        
        if result['success']:
            print(f"✓ Inference successful!")
            print(f"Predicted class: {result['inference_result']['predicted_class']}")
            print(f"Execution cycles: {result['total_execution_cycles']:,}")
            return result
        else:
            print(f"✗ Inference failed: {result.get('error', 'Unknown error')}")
            return None
    else:
        # Just validate mapping
        validation = dnn_manager.validate_hardware_capacity()
        print(f"Hardware validation: {validation['overall_sufficient']}")
        return validation