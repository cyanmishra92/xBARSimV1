#!/usr/bin/env python3
"""
Main entry point for the ReRAM Crossbar Simulator
"""

import sys
import os
import argparse
import json
import logging
import numpy as np
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import simulator components
from core.hierarchy import ReRAMChip, ChipConfig, SuperTileConfig, TileConfig
from core.crossbar import CrossbarConfig
from core.dnn_manager import DNNManager, DNNConfig, LayerConfig, LayerType
from core.execution_engine import ExecutionEngine, ExecutionConfig
from core.metrics import MetricsCollector
from visualization.text_viz import create_complete_text_report, print_architecture_diagram

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            stream=sys.stdout
        )

def load_config(config_file: str) -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config file {config_file}: {e}")
        return {}

def create_default_chip_config() -> ChipConfig:
    """Create a default chip configuration"""
    crossbar_config = CrossbarConfig(
        rows=128,
        cols=128,
        r_on=1e3,
        r_off=1e6,
        device_variability=0.1
    )
    
    tile_config = TileConfig(
        crossbars_per_tile=4,
        crossbar_config=crossbar_config,
        local_buffer_size=64
    )
    
    supertile_config = SuperTileConfig(
        tiles_per_supertile=4,
        tile_config=tile_config,
        shared_buffer_size=512
    )
    
    chip_config = ChipConfig(
        supertiles_per_chip=2,
        supertile_config=supertile_config,
        global_buffer_size=4096
    )
    
    return chip_config

def create_sample_dnn_config() -> DNNConfig:
    """Create a sample DNN configuration"""
    layers = [
        LayerConfig(
            layer_type=LayerType.CONV2D,
            input_shape=(16, 16, 1),
            output_shape=(14, 14, 8),
            kernel_size=(3, 3),
            activation="relu",
            weights_shape=(3, 3, 1, 8)
        ),
        LayerConfig(
            layer_type=LayerType.POOLING,
            input_shape=(14, 14, 8),
            output_shape=(7, 7, 8),
            kernel_size=(2, 2)
        ),
        LayerConfig(
            layer_type=LayerType.DENSE,
            input_shape=(392,),  # 7*7*8
            output_shape=(10,),
            activation="softmax",
            weights_shape=(392, 10)
        )
    ]
    
    return DNNConfig(
        model_name="SampleCNN",
        layers=layers,
        input_shape=(16, 16, 1),
        output_shape=(10,),
        precision=8
    )

def run_simulation(args):
    """Run the main simulation"""
    import numpy as np  # Import numpy in function scope
    print("=" * 60)
    print("ReRAM Crossbar Simulator v1.0")
    print("=" * 60)
    
    # 1. Load or create configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        # TODO: Parse config and create objects
    else:
        print("Using default configuration")
        
    # 2. Create chip configuration
    print("\n1. Creating chip configuration...")
    chip_config = create_default_chip_config()
    chip = ReRAMChip(chip_config)
    
    if args.verbose:
        chip.print_architecture_summary()
    
    # 3. Create DNN configuration
    print("\n2. Creating DNN configuration...")
    dnn_config = create_sample_dnn_config()
    dnn_manager = DNNManager(dnn_config, chip)
    
    if args.verbose:
        dnn_manager.print_dnn_summary()
    
    # 4. Map DNN to hardware
    print("\n3. Mapping DNN to hardware...")
    
    # Create dummy weights for demonstration
    weight_data = {}
    weight_idx = 0
    for i, layer in enumerate(dnn_config.layers):
        if layer.weights_shape:
            weight_data[f"layer_{weight_idx}"] = np.random.randn(*layer.weights_shape) * 0.1
            weight_idx += 1
    
    try:
        layer_mappings = dnn_manager.map_dnn_to_hardware(weight_data)
        if args.verbose:
            dnn_manager.print_mapping_summary()
    except Exception as e:
        logging.error(f"Failed to map DNN to hardware: {e}")
        return False
    
    # 5. Validate configuration
    print("\n4. Validating hardware capacity...")
    validation = dnn_manager.validate_hardware_capacity()
    print(f"   Hardware sufficient: {validation['overall_sufficient']}")
    
    if not validation['overall_sufficient']:
        print("   ⚠️  Hardware insufficient for this DNN!")
        if not args.force:
            print("   Use --force to continue anyway")
            return False
    
    # 6. Run execution if requested
    if args.execute:
        print("\n5. Running inference...")
        
        # Create execution engine
        execution_config = ExecutionConfig(
            enable_cycle_accurate_simulation=args.cycle_accurate,
            enable_energy_modeling=True,
            max_execution_cycles=args.max_cycles
        )
        execution_engine = ExecutionEngine(chip, dnn_manager, execution_config)
        
        # Generate test input
        import numpy as np
        test_input = np.random.randn(*dnn_config.input_shape)
        
        try:
            result = execution_engine.execute_inference(test_input)
            
            if result['success']:
                print("   ✓ Inference completed successfully!")
                final_result = result['inference_result']
                print(f"   Predicted class: {final_result.get('predicted_class', 'N/A')}")
                print(f"   Execution cycles: {result['total_execution_cycles']:,}")
                
                # Save results if requested
                if args.output:
                    output_file = Path(args.output)
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Convert numpy types to Python types for JSON serialization
                    def convert_numpy_types(obj):
                        if isinstance(obj, (np.integer, np.int32, np.int64)):
                            return int(obj)
                        elif isinstance(obj, (np.floating, np.float32, np.float64)):
                            return float(obj)
                        elif isinstance(obj, (np.bool_, bool)):
                            return bool(obj)
                        elif isinstance(obj, np.ndarray):
                            return obj.tolist()
                        elif isinstance(obj, dict):
                            return {k: convert_numpy_types(v) for k, v in obj.items()}
                        elif isinstance(obj, (list, tuple)):
                            return [convert_numpy_types(item) for item in obj]
                        else:
                            return obj
                    
                    results_data = {
                        'config': {
                            'chip': chip.get_chip_configuration(),
                            'dnn': {
                                'model_name': dnn_config.model_name,
                                'total_parameters': int(dnn_config.total_parameters),
                                'input_shape': list(dnn_config.input_shape),
                                'output_shape': list(dnn_config.output_shape)
                            }
                        },
                        'results': {
                            'inference_result': convert_numpy_types(result['inference_result']),
                            'execution_cycles': int(result['total_execution_cycles']),
                            'layer_execution_log': convert_numpy_types(result['layer_execution_log'])
                        },
                        'validation': convert_numpy_types(validation)
                    }
                    
                    with open(output_file, 'w') as f:
                        json.dump(results_data, f, indent=2)
                    
                    print(f"   Results saved to {output_file}")
                    
            else:
                print(f"   ✗ Inference failed: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logging.error(f"Execution failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    # 7. Generate visualizations if requested
    if args.visualize:
        print("\n6. Generating text-based visualization...")
        try:
            if args.execute and 'result' in locals():
                create_complete_text_report(chip, dnn_manager, result)
            else:
                print_architecture_diagram(chip)
            print("   ✓ Text visualization complete!")
        except Exception as e:
            logging.warning(f"Could not generate visualizations: {e}")
    
    print("\n" + "=" * 60)
    print("Simulation completed successfully!")
    print("=" * 60)
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="ReRAM Crossbar Simulator for Neural Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic simulation with default config
  python main.py --execute --visualize
  
  # Run with custom configuration
  python main.py --config my_config.json --execute --output results.json
  
  # Run cycle-accurate simulation
  python main.py --execute --cycle-accurate --max-cycles 100000
  
  # Generate only architecture visualization
  python main.py --visualize --output arch_viz
        """
    )
    
    parser.add_argument('--config', '-c', type=str,
                       help='Configuration file (JSON)')
    parser.add_argument('--execute', '-e', action='store_true',
                       help='Execute inference on the simulator')
    parser.add_argument('--cycle-accurate', action='store_true',
                       help='Enable cycle-accurate simulation (slower but more accurate)')
    parser.add_argument('--max-cycles', type=int, default=50000,
                       help='Maximum execution cycles (default: 50000)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for results')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Generate visualization diagrams')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--force', action='store_true',
                       help='Force execution even if hardware is insufficient')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', type=str,
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Numpy already imported at top
    
    # Run simulation
    try:
        success = run_simulation(args)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Simulation failed with unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()