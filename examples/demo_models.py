#!/usr/bin/env python3
"""
Demo script with different model examples for the ReRAM crossbar simulator
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from core.hierarchy import ReRAMChip, ChipConfig, SuperTileConfig, TileConfig
from core.crossbar import CrossbarConfig
from core.dnn_manager import DNNManager, DNNConfig, LayerConfig, LayerType
from core.execution_engine import ExecutionEngine, ExecutionConfig
from visualization.text_viz import create_complete_text_report

def create_tiny_cnn():
    """Create a tiny CNN for quick testing"""
    layers = [
        LayerConfig(
            layer_type=LayerType.CONV2D,
            input_shape=(8, 8, 1),
            output_shape=(6, 6, 4),
            kernel_size=(3, 3),
            activation="relu",
            weights_shape=(3, 3, 1, 4)
        ),
        LayerConfig(
            layer_type=LayerType.POOLING,
            input_shape=(6, 6, 4),
            output_shape=(3, 3, 4),
            kernel_size=(2, 2)
        ),
        LayerConfig(
            layer_type=LayerType.DENSE,
            input_shape=(36,),
            output_shape=(3,),
            activation="softmax",
            weights_shape=(36, 3)
        )
    ]
    
    return DNNConfig(
        model_name="TinyCNN",
        layers=layers,
        input_shape=(8, 8, 1),
        output_shape=(3,),
        precision=8
    )

def create_mnist_cnn():
    """Create a CNN for MNIST-like classification"""
    layers = [
        LayerConfig(
            layer_type=LayerType.CONV2D,
            input_shape=(28, 28, 1),
            output_shape=(26, 26, 16),
            kernel_size=(3, 3),
            activation="relu",
            weights_shape=(3, 3, 1, 16)
        ),
        LayerConfig(
            layer_type=LayerType.POOLING,
            input_shape=(26, 26, 16),
            output_shape=(13, 13, 16),
            kernel_size=(2, 2)
        ),
        LayerConfig(
            layer_type=LayerType.CONV2D,
            input_shape=(13, 13, 16),
            output_shape=(11, 11, 32),
            kernel_size=(3, 3),
            activation="relu",
            weights_shape=(3, 3, 16, 32)
        ),
        LayerConfig(
            layer_type=LayerType.POOLING,
            input_shape=(11, 11, 32),
            output_shape=(5, 5, 32),
            kernel_size=(2, 2)
        ),
        LayerConfig(
            layer_type=LayerType.DENSE,
            input_shape=(800,),
            output_shape=(128,),
            activation="relu",
            weights_shape=(800, 128)
        ),
        LayerConfig(
            layer_type=LayerType.DENSE,
            input_shape=(128,),
            output_shape=(10,),
            activation="softmax",
            weights_shape=(128, 10)
        )
    ]
    
    return DNNConfig(
        model_name="MNIST_CNN",
        layers=layers,
        input_shape=(28, 28, 1),
        output_shape=(10,),
        precision=8
    )

def create_cifar_cnn():
    """Create a CNN for CIFAR-10 like classification"""
    layers = [
        LayerConfig(
            layer_type=LayerType.CONV2D,
            input_shape=(32, 32, 3),
            output_shape=(30, 30, 32),
            kernel_size=(3, 3),
            activation="relu",
            weights_shape=(3, 3, 3, 32)
        ),
        LayerConfig(
            layer_type=LayerType.CONV2D,
            input_shape=(30, 30, 32),
            output_shape=(28, 28, 32),
            kernel_size=(3, 3),
            activation="relu",
            weights_shape=(3, 3, 32, 32)
        ),
        LayerConfig(
            layer_type=LayerType.POOLING,
            input_shape=(28, 28, 32),
            output_shape=(14, 14, 32),
            kernel_size=(2, 2)
        ),
        LayerConfig(
            layer_type=LayerType.CONV2D,
            input_shape=(14, 14, 32),
            output_shape=(12, 12, 64),
            kernel_size=(3, 3),
            activation="relu",
            weights_shape=(3, 3, 32, 64)
        ),
        LayerConfig(
            layer_type=LayerType.CONV2D,
            input_shape=(12, 12, 64),
            output_shape=(10, 10, 64),
            kernel_size=(3, 3),
            activation="relu",
            weights_shape=(3, 3, 64, 64)
        ),
        LayerConfig(
            layer_type=LayerType.POOLING,
            input_shape=(10, 10, 64),
            output_shape=(5, 5, 64),
            kernel_size=(2, 2)
        ),
        LayerConfig(
            layer_type=LayerType.DENSE,
            input_shape=(1600,),
            output_shape=(512,),
            activation="relu",
            weights_shape=(1600, 512)
        ),
        LayerConfig(
            layer_type=LayerType.DENSE,
            input_shape=(512,),
            output_shape=(10,),
            activation="softmax",
            weights_shape=(512, 10)
        )
    ]
    
    return DNNConfig(
        model_name="CIFAR_CNN",
        layers=layers,
        input_shape=(32, 32, 3),
        output_shape=(10,),
        precision=8
    )

def create_adaptive_hardware(dnn_config):
    """Create hardware that adapts to DNN requirements"""
    hw_requirement = DNNManager(dnn_config).hw_requirement
    recommended = hw_requirement.recommend_hardware_config()
    
    crossbar_config = CrossbarConfig(
        rows=recommended['crossbar_size'][0],
        cols=recommended['crossbar_size'][1],
        device_variability=0.05
    )
    
    tile_config = TileConfig(
        crossbars_per_tile=recommended['crossbars_per_tile'],
        crossbar_config=crossbar_config,
        local_buffer_size=recommended['local_buffer_kb']
    )
    
    supertile_config = SuperTileConfig(
        tiles_per_supertile=recommended['tiles_per_supertile'],
        tile_config=tile_config,
        shared_buffer_size=recommended['shared_buffer_kb']
    )
    
    chip_config = ChipConfig(
        supertiles_per_chip=recommended['supertiles'],
        supertile_config=supertile_config,
        global_buffer_size=recommended['global_buffer_kb']
    )
    
    return ReRAMChip(chip_config)

def run_model_demo(model_name, dnn_config, execute_inference=True):
    """Run demo for a specific model"""
    print(f"\n{'='*60}")
    print(f"🧠 DEMO: {model_name}")
    print(f"{'='*60}")
    
    # Create adaptive hardware
    print("1. Creating adaptive hardware configuration...")
    chip = create_adaptive_hardware(dnn_config)
    config = chip.get_chip_configuration()
    print(f"   ✓ Hardware created: {config['compute_capacity']['total_crossbars']} crossbars")
    
    # Create DNN manager
    print("2. Setting up DNN manager...")
    dnn_manager = DNNManager(dnn_config, chip)
    print(f"   ✓ DNN: {dnn_config.total_parameters:,} parameters")
    
    # Validate capacity
    print("3. Validating hardware capacity...")
    validation = dnn_manager.validate_hardware_capacity()
    print(f"   ✓ Hardware sufficient: {validation['overall_sufficient']}")
    print(f"   ✓ Crossbar utilization: {validation['utilization']['crossbar_utilization']:.1%}")
    
    if not validation['overall_sufficient']:
        print("   ⚠️  Hardware insufficient - stopping demo")
        return False
    
    # Map to hardware
    print("4. Mapping DNN to hardware...")
    try:
        weight_data = {}
        layer_idx = 0
        for i, layer in enumerate(dnn_config.layers):
            if layer.weights_shape:
                weight_data[f"layer_{layer_idx}"] = np.random.randn(*layer.weights_shape) * 0.1
                layer_idx += 1
        
        layer_mappings = dnn_manager.map_dnn_to_hardware(weight_data)
        print(f"   ✓ Mapped {len(layer_mappings)} layers")
    except Exception as e:
        print(f"   ✗ Mapping failed: {e}")
        return False
    
    # Execute inference if requested
    if execute_inference:
        print("5. Running inference...")
        try:
            execution_config = ExecutionConfig(
                enable_cycle_accurate_simulation=False,  # Faster for demo
                enable_energy_modeling=True
            )
            execution_engine = ExecutionEngine(chip, dnn_manager, execution_config)
            
            # Generate test input
            test_input = np.random.randn(*dnn_config.input_shape)
            
            result = execution_engine.execute_inference(test_input)
            
            if result['success']:
                print("   ✓ Inference successful!")
                inference_result = result['inference_result']
                print(f"   ✓ Predicted class: {inference_result['predicted_class']}")
                print(f"   ✓ Confidence: {inference_result['confidence']:.3f}")
                
                # Show brief performance summary
                stats = result['system_statistics']
                if 'chip_statistics' in stats:
                    chip_stats = stats['chip_statistics']
                    if 'performance' in chip_stats:
                        total_ops = chip_stats['performance'].get('total_crossbar_operations', 0)
                        print(f"   ✓ Crossbar operations: {total_ops:,}")
                
                return result
            else:
                print(f"   ✗ Inference failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"   ✗ Inference failed: {e}")
            return False
    
    return True

def main():
    """Main demo function"""
    print("🎯 ReRAM Crossbar Simulator - Model Demos")
    print("="*60)
    
    models = [
        ("Tiny CNN", create_tiny_cnn()),
        ("MNIST CNN", create_mnist_cnn()),
        ("CIFAR CNN", create_cifar_cnn())
    ]
    
    results = {}
    
    for model_name, dnn_config in models:
        try:
            result = run_model_demo(model_name, dnn_config, execute_inference=True)
            results[model_name] = result
            
            if result and isinstance(result, dict) and result.get('success'):
                print(f"   🎉 {model_name} demo completed successfully!")
            else:
                print(f"   ⚠️  {model_name} demo completed with limitations")
                
        except Exception as e:
            print(f"   ❌ {model_name} demo failed: {e}")
            results[model_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 DEMO SUMMARY")
    print(f"{'='*60}")
    
    successful = 0
    for model_name, result in results.items():
        if result and isinstance(result, dict) and result.get('success'):
            status = "✅ Success"
            successful += 1
        elif result:
            status = "⚠️  Partial"
        else:
            status = "❌ Failed"
            
        print(f"{model_name:15s}: {status}")
    
    print(f"\nSuccessful demos: {successful}/{len(models)}")
    
    # Show detailed report for the first successful model
    for model_name, result in results.items():
        if result and isinstance(result, dict) and result.get('success'):
            print(f"\n🔍 DETAILED REPORT FOR: {model_name}")
            
            # Re-create the components for visualization
            if model_name == "Tiny CNN":
                dnn_config = create_tiny_cnn()
            elif model_name == "MNIST CNN":
                dnn_config = create_mnist_cnn()
            elif model_name == "CIFAR CNN":
                dnn_config = create_cifar_cnn()
                
            chip = create_adaptive_hardware(dnn_config)
            dnn_manager = DNNManager(dnn_config, chip)
            
            # Create dummy weight data for mapping
            weight_data = {}
            layer_idx = 0
            for i, layer in enumerate(dnn_config.layers):
                if layer.weights_shape:
                    weight_data[f"layer_{layer_idx}"] = np.random.randn(*layer.weights_shape) * 0.1
                    layer_idx += 1
            dnn_manager.map_dnn_to_hardware(weight_data)
            
            create_complete_text_report(chip, dnn_manager, result, detailed_endurance_report=False)
            break
    
    print(f"\n🏁 All demos completed!")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    main()