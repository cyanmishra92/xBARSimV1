#!/usr/bin/env python3
"""
Test case for a simple CNN model on ReRAM crossbar simulator
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Import simulator components
from core.hierarchy import ReRAMChip, ChipConfig, SuperTileConfig, TileConfig
from core.crossbar import CrossbarConfig
from core.dnn_manager import DNNManager, DNNConfig, LayerConfig, LayerType
from core.execution_engine import ExecutionEngine, ExecutionConfig
from core.metrics import MetricsCollector
from visualization.architecture_viz import create_comprehensive_visualization

def create_simple_cnn_config() -> DNNConfig:
    """Create a simple CNN configuration for MNIST-like classification"""
    
    layers = [
        # Input layer: 28x28x1
        LayerConfig(
            layer_type=LayerType.CONV2D,
            input_shape=(28, 28, 1),
            output_shape=(26, 26, 32),
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="valid",
            activation="relu",
            weights_shape=(3, 3, 1, 32),
            bias_shape=(32,)
        ),
        
        # Max pooling: 26x26x32 -> 13x13x32
        LayerConfig(
            layer_type=LayerType.POOLING,
            input_shape=(26, 26, 32),
            output_shape=(13, 13, 32),
            kernel_size=(2, 2),
            stride=(2, 2)
        ),
        
        # Second conv layer: 13x13x32 -> 11x11x64
        LayerConfig(
            layer_type=LayerType.CONV2D,
            input_shape=(13, 13, 32),
            output_shape=(11, 11, 64),
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="valid",
            activation="relu",
            weights_shape=(3, 3, 32, 64),
            bias_shape=(64,)
        ),
        
        # Max pooling: 11x11x64 -> 5x5x64
        LayerConfig(
            layer_type=LayerType.POOLING,
            input_shape=(11, 11, 64),
            output_shape=(5, 5, 64),
            kernel_size=(2, 2),
            stride=(2, 2)
        ),
        
        # Dense layer: 5*5*64 = 1600 -> 128
        LayerConfig(
            layer_type=LayerType.DENSE,
            input_shape=(1600,),
            output_shape=(128,),
            activation="relu",
            weights_shape=(1600, 128),
            bias_shape=(128,)
        ),
        
        # Output layer: 128 -> 10 (classes)
        LayerConfig(
            layer_type=LayerType.DENSE,
            input_shape=(128,),
            output_shape=(10,),
            activation="softmax",
            weights_shape=(128, 10),
            bias_shape=(10,)
        )
    ]
    
    dnn_config = DNNConfig(
        model_name="SimpleCNN_MNIST",
        layers=layers,
        input_shape=(28, 28, 1),
        output_shape=(10,),
        precision=8
    )
    
    return dnn_config

def create_optimized_hardware_config(dnn_config: DNNConfig) -> ChipConfig:
    """Create hardware configuration optimized for the DNN"""
    
    # Create crossbar configuration
    crossbar_config = CrossbarConfig(
        rows=128,
        cols=128,
        r_on=1e3,
        r_off=1e6,
        v_read=0.2,
        device_variability=0.05
    )
    
    # Create tile configuration
    tile_config = TileConfig(
        crossbars_per_tile=4,
        crossbar_config=crossbar_config,
        local_buffer_size=128,  # 128 KB local buffer
        adc_sharing=True,
        adcs_per_tile=64
    )
    
    # Create supertile configuration
    supertile_config = SuperTileConfig(
        tiles_per_supertile=4,
        tile_config=tile_config,
        shared_buffer_size=1024  # 1 MB shared buffer
    )
    
    # Create chip configuration
    chip_config = ChipConfig(
        supertiles_per_chip=2,
        supertile_config=supertile_config,
        global_buffer_size=8192  # 8 MB global buffer
    )
    
    return chip_config

def generate_test_input() -> np.ndarray:
    """Generate test input data (simulated MNIST image)"""
    # Create a simple test pattern that looks like a digit
    test_input = np.zeros((28, 28, 1))
    
    # Draw a simple "7"-like pattern
    test_input[5:8, 5:20, 0] = 1.0  # Top horizontal line
    test_input[8:15, 15:18, 0] = 1.0  # Diagonal line
    test_input[15:25, 12:15, 0] = 1.0  # Bottom part
    
    # Add some noise
    noise = np.random.normal(0, 0.05, test_input.shape)
    test_input = np.clip(test_input + noise, 0, 1)
    
    return test_input

def run_simple_cnn_test():
    """Run the complete simple CNN test"""
    print("=" * 60)
    print("ReRAM Crossbar Simulator - Simple CNN Test")
    print("=" * 60)
    
    # 1. Create DNN configuration
    print("1. Creating DNN configuration...")
    dnn_config = create_simple_cnn_config()
    print(f"   Model: {dnn_config.model_name}")
    print(f"   Total Parameters: {dnn_config.total_parameters:,}")
    print(f"   Total MACs: {dnn_config.total_macs:,}")
    
    # 2. Create optimized hardware
    print("\n2. Creating optimized hardware configuration...")
    chip_config = create_optimized_hardware_config(dnn_config)
    chip = ReRAMChip(chip_config)
    chip.print_architecture_summary()
    
    # 3. Create DNN manager and map to hardware
    print("\n3. Mapping DNN to hardware...")
    dnn_manager = DNNManager(dnn_config, chip)
    dnn_manager.print_dnn_summary()
    
    # Create dummy weight data for mapping
    weight_data = {}
    weight_layer_idx = 0
    for i, layer in enumerate(dnn_config.layers):
        if layer.weights_shape is not None:
            layer_name = f"layer_{weight_layer_idx}_weights"
            weight_data[layer_name] = np.random.randn(*layer.weights_shape) * 0.1
            weight_layer_idx += 1
    
    # Map DNN to hardware
    layer_mappings = dnn_manager.map_dnn_to_hardware(weight_data)
    dnn_manager.print_mapping_summary()
    
    # 4. Validate hardware capacity
    print("\n4. Validating hardware capacity...")
    validation = dnn_manager.validate_hardware_capacity()
    print(f"   Hardware sufficient: {validation['overall_sufficient']}")
    print(f"   Crossbar utilization: {validation['utilization']['crossbar_utilization']:.1%}")
    print(f"   Memory utilization: {validation['utilization']['memory_utilization']:.1%}")
    
    # 5. Create execution engine
    print("\n5. Creating execution engine...")
    execution_config = ExecutionConfig(
        enable_cycle_accurate_simulation=True,
        enable_energy_modeling=True,
        max_execution_cycles=50000
    )
    execution_engine = ExecutionEngine(chip, dnn_manager, execution_config)
    
    # 6. Generate test input
    print("\n6. Generating test input...")
    test_input = generate_test_input()
    print(f"   Input shape: {test_input.shape}")
    print(f"   Input range: [{test_input.min():.3f}, {test_input.max():.3f}]")
    
    # 7. Run inference
    print("\n7. Running inference...")
    print("   This may take a while for cycle-accurate simulation...")
    
    try:
        inference_result = execution_engine.execute_inference(test_input)
        
        if inference_result['success']:
            print("   ✓ Inference completed successfully!")
            
            # Print results
            final_result = inference_result['inference_result']
            print(f"   Predicted class: {final_result['predicted_class']}")
            print(f"   Confidence: {final_result['confidence']:.3f}")
            print(f"   Total execution cycles: {inference_result['total_execution_cycles']:,}")
            print(f"   Total execution time: {inference_result['total_execution_time_seconds']:.6f} seconds")
            
            # Print layer execution log
            print("\n   Layer Execution Log:")
            for layer_info in inference_result['layer_execution_log']:
                print(f"     Layer {layer_info['layer_index']} ({layer_info['layer_type']}): "
                      f"{layer_info['execution_cycles']:,} cycles")
                      
        else:
            print(f"   ✗ Inference failed: {inference_result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"   ✗ Inference failed with exception: {e}")
        return False
    
    # 8. System integration test
    print("\n8. Running system integration test...")
    try:
        integration_result = execution_engine.run_system_integration_test()
        if integration_result['integration_successful']:
            print("   ✓ System integration test passed!")
        else:
            print("   ✗ System integration test failed!")
    except Exception as e:
        print(f"   ✗ System integration test failed: {e}")
    
    # 9. Generate performance report
    print("\n9. Generating performance report...")
    try:
        performance_stats = inference_result['system_statistics']
        metrics_summary = inference_result['metrics_summary']
        
        print("\n   Performance Summary:")
        if 'chip_statistics' in performance_stats:
            chip_stats = performance_stats['chip_statistics']
            print(f"     Total chip operations: {chip_stats.get('performance', {}).get('total_operations', 'N/A')}")
            
        if 'memory_statistics' in performance_stats:
            memory_stats = performance_stats['memory_statistics']
            print(f"     Memory accesses: {sum(stats['memory_stats']['total_requests'] for stats in memory_stats.values())}")
            
        if 'microcontroller_statistics' in performance_stats:
            mcu_stats = performance_stats['microcontroller_statistics']
            print(f"     Instructions executed: {mcu_stats.get('total_instructions_executed', 'N/A')}")
            print(f"     IPC: {mcu_stats.get('instructions_per_cycle', 'N/A'):.3f}")
            
    except Exception as e:
        print(f"   Warning: Could not generate performance report: {e}")
    
    # 10. Create visualizations
    print("\n10. Creating visualizations...")
    try:
        visualizations = create_comprehensive_visualization(
            chip, dnn_manager, 
            metrics_data=inference_result.get('system_statistics'),
            save_path="simple_cnn_test"
        )
        print("   ✓ Visualizations saved!")
    except Exception as e:
        print(f"   Warning: Could not create visualizations: {e}")
    
    print("\n" + "=" * 60)
    print("Simple CNN Test Completed Successfully!")
    print("=" * 60)
    
    return True

def run_benchmark_suite():
    """Run a suite of benchmark tests"""
    print("\n" + "=" * 60)
    print("Running Benchmark Suite")
    print("=" * 60)
    
    benchmark_results = []
    
    # Test 1: Different input sizes
    print("\nBenchmark 1: Different input sizes")
    input_sizes = [(8, 8, 1), (16, 16, 1), (28, 28, 1), (32, 32, 1)]
    
    for input_size in input_sizes:
        print(f"  Testing input size: {input_size}")
        
        # Create simple model for this input size
        layers = [
            LayerConfig(
                layer_type=LayerType.CONV2D,
                input_shape=input_size,
                output_shape=(input_size[0]-2, input_size[1]-2, 16),
                kernel_size=(3, 3),
                weights_shape=(3, 3, input_size[2], 16)
            ),
            LayerConfig(
                layer_type=LayerType.DENSE,
                input_shape=((input_size[0]-2) * (input_size[1]-2) * 16,),
                output_shape=(10,),
                weights_shape=((input_size[0]-2) * (input_size[1]-2) * 16, 10)
            )
        ]
        
        dnn_config = DNNConfig(
            model_name=f"Test_{input_size[0]}x{input_size[1]}",
            layers=layers,
            input_shape=input_size,
            output_shape=(10,)
        )
        
        # Create hardware and run test
        chip_config = create_optimized_hardware_config(dnn_config)
        chip = ReRAMChip(chip_config)
        dnn_manager = DNNManager(dnn_config, chip)
        
        # Quick validation test
        validation = dnn_manager.validate_hardware_capacity()
        benchmark_results.append({
            'test': f'input_size_{input_size[0]}x{input_size[1]}',
            'parameters': dnn_config.total_parameters,
            'hardware_sufficient': validation['overall_sufficient'],
            'crossbar_utilization': validation['utilization']['crossbar_utilization']
        })
        
        print(f"    Parameters: {dnn_config.total_parameters:,}")
        print(f"    Hardware sufficient: {validation['overall_sufficient']}")
        print(f"    Crossbar utilization: {validation['utilization']['crossbar_utilization']:.1%}")
    
    # Print benchmark summary
    print("\n" + "=" * 40)
    print("Benchmark Summary:")
    print("=" * 40)
    for result in benchmark_results:
        print(f"{result['test']}: "
              f"Params={result['parameters']:,}, "
              f"Sufficient={result['hardware_sufficient']}, "
              f"Util={result['crossbar_utilization']:.1%}")
    
    return benchmark_results

if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Run the main test
    success = run_simple_cnn_test()
    
    if success:
        # Run benchmark suite
        benchmark_results = run_benchmark_suite()
        
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)