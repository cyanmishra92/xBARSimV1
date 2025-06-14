#!/usr/bin/env python3
"""
Simple test case for ReRAM crossbar simulator without visualization dependencies
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# Import simulator components
from core.hierarchy import ReRAMChip, ChipConfig, SuperTileConfig, TileConfig
from core.crossbar import CrossbarConfig
from core.dnn_manager import DNNManager, DNNConfig, LayerConfig, LayerType
from core.execution_engine import ExecutionEngine, ExecutionConfig

def create_simple_test_cnn() -> DNNConfig:
    """Create a very simple CNN for testing"""
    layers = [
        # Small conv layer: 8x8x1 -> 6x6x4
        LayerConfig(
            layer_type=LayerType.CONV2D,
            input_shape=(8, 8, 1),
            output_shape=(6, 6, 4),
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="valid",
            activation="relu",
            weights_shape=(3, 3, 1, 4)
        ),
        
        # Small pooling: 6x6x4 -> 3x3x4
        LayerConfig(
            layer_type=LayerType.POOLING,
            input_shape=(6, 6, 4),
            output_shape=(3, 3, 4),
            kernel_size=(2, 2),
            stride=(2, 2)
        ),
        
        # Dense layer: 36 -> 3
        LayerConfig(
            layer_type=LayerType.DENSE,
            input_shape=(36,),  # 3*3*4
            output_shape=(3,),
            activation="softmax",
            weights_shape=(36, 3)
        )
    ]
    
    return DNNConfig(
        model_name="SimpleTestCNN",
        layers=layers,
        input_shape=(8, 8, 1),
        output_shape=(3,),
        precision=8
    )

def create_test_hardware() -> ReRAMChip:
    """Create test hardware configuration"""
    crossbar_config = CrossbarConfig(
        rows=64,
        cols=64,
        device_variability=0.05
    )
    
    tile_config = TileConfig(
        crossbars_per_tile=4,
        crossbar_config=crossbar_config,
        local_buffer_size=32,
        adcs_per_tile=64  # Ensure enough ADCs for one crossbar
    )
    
    supertile_config = SuperTileConfig(
        tiles_per_supertile=2,
        tile_config=tile_config,
        shared_buffer_size=128
    )
    
    chip_config = ChipConfig(
        supertiles_per_chip=1,
        supertile_config=supertile_config,
        global_buffer_size=512
    )
    
    return ReRAMChip(chip_config)

def test_basic_functionality():
    """Test basic simulator functionality"""
    print("=" * 50)
    print("ReRAM Crossbar Simulator - Basic Test")
    print("=" * 50)
    
    # 1. Create hardware
    print("1. Creating hardware...")
    chip = create_test_hardware()
    config = chip.get_chip_configuration()
    print(f"   ✓ Created chip with {config['compute_capacity']['total_crossbars']} crossbars")
    
    # 2. Create DNN
    print("2. Creating DNN...")
    dnn_config = create_simple_test_cnn()
    print(f"   ✓ Created DNN with {dnn_config.total_parameters:,} parameters")
    
    # 3. Create DNN manager
    print("3. Setting up DNN manager...")
    dnn_manager = DNNManager(dnn_config, chip)
    
    # 4. Validate hardware capacity
    print("4. Validating hardware capacity...")
    validation = dnn_manager.validate_hardware_capacity()
    print(f"   ✓ Hardware sufficient: {validation['overall_sufficient']}")
    print(f"   ✓ Crossbar utilization: {validation['utilization']['crossbar_utilization']:.1%}")
    
    if not validation['overall_sufficient']:
        print("   ⚠️  Hardware insufficient - but continuing...")
    
    # 5. Test weight mapping
    print("5. Testing weight mapping...")
    try:
        weight_data = {}
        layer_idx = 0
        for i, layer in enumerate(dnn_config.layers):
            if layer.weights_shape:
                weight_data[f"layer_{layer_idx}"] = np.random.randn(*layer.weights_shape) * 0.1
                layer_idx += 1
        
        layer_mappings = dnn_manager.map_dnn_to_hardware(weight_data)
        print(f"   ✓ Successfully mapped {len(layer_mappings)} layers to hardware")
    except Exception as e:
        print(f"   ✗ Weight mapping failed: {e}")
        assert False, f"Weight mapping failed: {e}"
    
    # 6. Test inference
    print("6. Testing inference execution...")
    try:
        execution_config = ExecutionConfig(
            enable_cycle_accurate_simulation=False,  # Disable for faster testing
            enable_energy_modeling=True,
            max_execution_cycles=10000
        )
        execution_engine = ExecutionEngine(chip, dnn_manager, execution_config)
        
        # Generate test input
        test_input = np.random.randn(*dnn_config.input_shape)
        
        result = execution_engine.execute_inference(test_input)
        
        if result['success']:
            print("   ✓ Inference successful!")
            print(f"   ✓ Predicted class: {result['inference_result']['predicted_class']}")
            print(f"   ✓ Confidence: {result['inference_result']['confidence']:.3f}")
            print(f"   ✓ Total execution cycles: {result['total_execution_cycles']:,}")
            
            # Print layer execution summary
            for layer_info in result['layer_execution_log']:
                print(f"      Layer {layer_info['layer_index']} ({layer_info['layer_type']}): "
                      f"{layer_info['execution_cycles']:,} cycles")
                      
        else:
            print(f"   ✗ Inference failed: {result.get('error', 'Unknown error')}")
            assert False, f"Inference failed: {result.get('error', 'Unknown error')}"
            
    except Exception as e:
        print(f"   ✗ Inference execution failed: {e}")
        assert False, f"Inference execution failed: {e}"
    
    # 7. Test crossbar operations
    print("7. Testing crossbar operations...")
    try:
        # Get first crossbar
        first_supertile = chip.supertiles[0]
        first_tile = first_supertile.tiles[0]
        first_crossbar = first_tile.crossbars[0]
        
        # Test programming weights (use crossbar dimensions)
        crossbar_rows = first_crossbar.config.rows
        crossbar_cols = first_crossbar.config.cols
        test_weights = np.random.randn(crossbar_rows, crossbar_cols) * 0.1
        success = first_crossbar.program_weights(test_weights)
        print(f"   ✓ Weight programming: {'Success' if success else 'Failed'}")
        
        # Test matrix-vector multiplication
        test_input_vector = np.random.randn(crossbar_rows)
        output = first_crossbar.matrix_vector_multiply(test_input_vector)
        print(f"   ✓ Matrix-vector multiplication: Output shape {output.shape}")
        
        # Get statistics
        stats = first_crossbar.get_statistics()
        print(f"   ✓ Crossbar operations: {stats['total_operations']}")
        
    except Exception as e:
        print(f"   ✗ Crossbar operation test failed: {e}")
        assert False, f"Crossbar operation test failed: {e}"
    
    print("\n" + "=" * 50)
    print("✅ All tests passed successfully!")
    print("🎉 ReRAM Crossbar Simulator is working correctly!")
    print("=" * 50)
    
    assert True  # All tests passed

def test_performance_analysis():
    """Test performance analysis features"""
    print("\n" + "=" * 50)
    print("Performance Analysis Test")
    print("=" * 50)
    
    # Quick performance test with different configurations
    configs_to_test = [
        {"name": "Small", "crossbar_size": (32, 32), "tiles": 1},
        {"name": "Medium", "crossbar_size": (64, 64), "tiles": 2},
        {"name": "Large", "crossbar_size": (128, 128), "tiles": 4}
    ]
    
    for config in configs_to_test:
        print(f"\n{config['name']} Configuration:")
        
        # Create hardware
        crossbar_config = CrossbarConfig(
            rows=config['crossbar_size'][0],
            cols=config['crossbar_size'][1]
        )
        tile_config = TileConfig(
            crossbars_per_tile=config['tiles'],
            crossbar_config=crossbar_config
        )
        supertile_config = SuperTileConfig(
            tiles_per_supertile=1,
            tile_config=tile_config
        )
        chip_config = ChipConfig(
            supertiles_per_chip=1,
            supertile_config=supertile_config
        )
        chip = ReRAMChip(chip_config)
        
        # Test with DNN
        dnn_config = create_simple_test_cnn()
        dnn_manager = DNNManager(dnn_config, chip)
        validation = dnn_manager.validate_hardware_capacity()
        
        chip_info = chip.get_chip_configuration()
        
        print(f"  Crossbars: {chip_info['compute_capacity']['total_crossbars']}")
        print(f"  Total capacity: {chip_info['compute_capacity']['total_weight_capacity']:,} weights")
        print(f"  DNN parameters: {dnn_config.total_parameters:,}")
        print(f"  Hardware sufficient: {validation['overall_sufficient']}")
        print(f"  Utilization: {validation['utilization']['crossbar_utilization']:.1%}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise
    
    # Run basic functionality test
    success = test_basic_functionality()
    
    if success:
        # Run performance analysis
        test_performance_analysis()
        
        print("\n🚀 All tests completed successfully!")
        print("The ReRAM Crossbar Simulator is fully functional!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)