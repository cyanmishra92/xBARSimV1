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
from dataclasses import fields
from enum import Enum

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
            stride=(1, 1),
            padding="valid",
            activation="relu",
            weights_shape=(3, 3, 1, 8)
        ),
        LayerConfig(
            layer_type=LayerType.POOLING,
            input_shape=(14, 14, 8),
            output_shape=(7, 7, 8),
            kernel_size=(2, 2),
            stride=(2, 2)
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

def create_tiny_cnn_config() -> DNNConfig:
    """Create a tiny CNN for quick testing (from demo_models.py)"""
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

def create_lenet_config() -> DNNConfig:
    """Create a LeNet-5 style DNN configuration"""
    layers = [
        # C1: Conv (kernel 5x5, 6 filters, stride 1, padding valid), output (24, 24, 6), activation 'relu'
        LayerConfig(
            layer_type=LayerType.CONV2D,
            input_shape=(28, 28, 1),
            output_shape=(24, 24, 6),
            kernel_size=(5, 5),
            stride=(1, 1),
            padding="valid",
            activation="relu",
            weights_shape=(5, 5, 1, 6)
        ),
        # S2: Pool (kernel 2x2, stride 2), output (12, 12, 6)
        LayerConfig(
            layer_type=LayerType.POOLING,
            input_shape=(24, 24, 6),
            output_shape=(12, 12, 6),
            kernel_size=(2, 2),
            stride=(2, 2)
        ),
        # C3: Conv (kernel 5x5, 16 filters, stride 1, padding valid), output (8, 8, 16), activation 'relu'
        LayerConfig(
            layer_type=LayerType.CONV2D,
            input_shape=(12, 12, 6),
            output_shape=(8, 8, 16),
            kernel_size=(5, 5),
            stride=(1, 1),
            padding="valid",
            activation="relu",
            weights_shape=(5, 5, 6, 16)
        ),
        # S4: Pool (kernel 2x2, stride 2), output (4, 4, 16)
        LayerConfig(
            layer_type=LayerType.POOLING,
            input_shape=(8, 8, 16),
            output_shape=(4, 4, 16),
            kernel_size=(2, 2),
            stride=(2, 2)
        ),
        # Flatten step is implicit in Dense layer input_shape
        # F5: Dense (input 4*4*16=256, output 120), activation 'relu'
        LayerConfig(
            layer_type=LayerType.DENSE,
            input_shape=(256,), # 4 * 4 * 16
            output_shape=(120,),
            activation="relu",
            weights_shape=(256, 120)
        ),
        # F6: Dense (input 120, output 84), activation 'relu'
        LayerConfig(
            layer_type=LayerType.DENSE,
            input_shape=(120,),
            output_shape=(84,),
            activation="relu",
            weights_shape=(120, 84)
        ),
        # Output: Dense (input 84, output 10), activation 'softmax'
        LayerConfig(
            layer_type=LayerType.DENSE,
            input_shape=(84,),
            output_shape=(10,),
            activation="softmax",
            weights_shape=(84, 10)
        )
    ]

    return DNNConfig(
        model_name="LeNet5",
        layers=layers,
        input_shape=(28, 28, 1),
        output_shape=(10,),
        precision=8 # Assuming 8-bit precision similar to other models
    )


def _dataclass_from_dict(cls, data: dict):
    """Helper to create dataclass instance from dictionary, ignoring unknown keys"""
    if data is None:
        data = {}
    field_info = cls.__dataclass_fields__
    kwargs = {}
    for name, field in field_info.items():
        if name in data:
            val = data[name]
            # Convert lists to tuples for fields expecting tuples
            if isinstance(val, list):
                val = tuple(val)
            try:
                if isinstance(field.type, type) and issubclass(field.type, Enum):
                    val = field.type(val)
            except Exception:
                pass
            kwargs[name] = val
    return cls(**kwargs)


def create_chip_config_from_dict(cfg: dict) -> ChipConfig:
    """Build ChipConfig hierarchy from dictionary"""
    supertile_dict = cfg.get('supertile_config', {})
    tile_dict = supertile_dict.get('tile_config', {})
    crossbar_dict = tile_dict.get('crossbar_config', {})

    crossbar_config = _dataclass_from_dict(CrossbarConfig, crossbar_dict)

    tile_dict = {k: v for k, v in tile_dict.items() if k != 'crossbar_config'}
    tile_config = _dataclass_from_dict(TileConfig, tile_dict)
    tile_config.crossbar_config = crossbar_config

    supertile_dict = {k: v for k, v in supertile_dict.items() if k != 'tile_config'}
    supertile_config = _dataclass_from_dict(SuperTileConfig, supertile_dict)
    supertile_config.tile_config = tile_config

    chip_dict = {k: v for k, v in cfg.items() if k != 'supertile_config'}
    chip_config = _dataclass_from_dict(ChipConfig, chip_dict)
    chip_config.supertile_config = supertile_config

    return chip_config


def create_dnn_config_from_dict(cfg: dict) -> DNNConfig:
    """Build DNNConfig from dictionary"""
    layer_dicts = cfg.get('layers', [])
    layers = []
    for ld in layer_dicts:
        lt = LayerType(ld.get('layer_type'))
        params = {k: (tuple(v) if isinstance(v, list) else v)
                  for k, v in ld.items() if k != 'layer_type'}
        layer = LayerConfig(layer_type=lt, **params)
        layers.append(layer)

    dnn_config = DNNConfig(
        model_name=cfg.get('model_name', 'CustomModel'),
        layers=layers,
        input_shape=tuple(cfg.get('input_shape', ())),
        output_shape=tuple(cfg.get('output_shape', ())),
        precision=cfg.get('precision', 8)
    )
    return dnn_config


def create_execution_config_from_dict(cfg: dict) -> ExecutionConfig:
    """Build ExecutionConfig from dictionary"""
    return _dataclass_from_dict(ExecutionConfig, cfg)

def run_simulation(args):
    """Run the main simulation"""
    import numpy as np  # Import numpy in function scope
    print("=" * 60)
    print("ReRAM Crossbar Simulator v1.0")
    print("=" * 60)
    
    # 1. Load or create configuration
    config = {}
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    else:
        print("Using default configuration")
        
    # 2. Create chip configuration
    print("\n1. Creating chip configuration...")
    if 'chip' in config:
        try:
            chip_config = create_chip_config_from_dict(config['chip'])
        except Exception as e:
            logging.error(f"Invalid chip configuration: {e}")
            return False
    else:
        chip_config = create_default_chip_config()
    chip = ReRAMChip(chip_config)
    
    if args.verbose:
        chip.print_architecture_summary()
    
    # 3. Create DNN configuration
    print("\n2. Creating DNN configuration...")
    if 'dnn' in config:
        try:
            dnn_config = create_dnn_config_from_dict(config['dnn'])
        except Exception as e:
            logging.error(f"Invalid DNN configuration: {e}")
            return False
    else:
        if args.model == 'sample_cnn':
            dnn_config = create_sample_dnn_config()
        elif args.model == 'tiny_cnn':
            dnn_config = create_tiny_cnn_config()
        elif args.model == 'lenet':
            dnn_config = create_lenet_config()
        else:  # Should not happen due to choices in argparse
            logging.error(f"Unknown model: {args.model}")
            return False
    print(f"\nSelected DNN Model: {dnn_config.model_name}")
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
    
    # 6. Setup web visualization if requested
    web_server = None
    if args.web_viz:
        try:
            from visualization.web_viz import create_web_visualization_server
            print(f"\n5. Starting web visualization server on port {args.web_port}...")
            web_server = create_web_visualization_server(port=args.web_port)
            
            # Start web server in a separate thread
            import threading
            web_thread = threading.Thread(target=lambda: web_server.run(debug=False))
            web_thread.daemon = True
            web_thread.start()
            
            # Give server a moment to start
            import time
            time.sleep(2)
            
            print("   ✅ Web visualization server started successfully!")
            print("")
            print("   🌐 OPEN WEB DASHBOARD:")
            print(f"   ➤ Real-time Monitoring: http://localhost:{args.web_port}/")
            print(f"   ➤ Educational Tool: http://localhost:{args.web_port}/educational")
            print("")
            print("   📋 Instructions:")
            print("   1. Click on the links above to open web visualization in your browser")
            print("   2. The real-time dashboard will show live monitoring during execution")
            print("   3. The educational tool provides step-by-step CNN mapping tutorial")
            print("")
            
            if args.execute and not args.auto_start:
                # Prompt user to continue only if executing and not auto-starting
                try:
                    input("   ⏸️  Press ENTER when ready to start simulation with web monitoring...")
                except (EOFError, KeyboardInterrupt):
                    print("\n   🛑 Simulation cancelled by user")
                    return False
                print("")
            elif args.execute and args.auto_start:
                print("   🚀 Auto-starting simulation with web monitoring...")
                print("")
            
        except ImportError as e:
            print(f"   ⚠️  Web visualization not available: {e}")
            print("   Install required packages: pip install flask flask-socketio")
            web_server = None
        except Exception as e:
            print(f"   ⚠️  Failed to start web server: {e}")
            web_server = None

    # 7. Run execution if requested
    if args.execute:
        step_num = "6" if args.web_viz else "5"
        print(f"\n{step_num}. Running inference...")

        # Create execution engine
        if 'execution' in config:
            execution_config = create_execution_config_from_dict(config['execution'])
        else:
            execution_config = ExecutionConfig(
                enable_cycle_accurate_simulation=args.cycle_accurate,
                enable_energy_modeling=True,
                max_execution_cycles=args.max_cycles
            )
        execution_engine = ExecutionEngine(chip, dnn_manager, execution_config)
        
        # Connect to web server if available
        if web_server:
            web_server.connect_execution_engine(execution_engine)
            if not args.live_viz:  # Start monitoring for web visualization
                web_server.start_monitoring()
        
        # Generate test input
        import numpy as np
        test_input = np.random.randn(*dnn_config.input_shape)
        
        try:
            # Use web-monitored execution if web server is available
            if web_server:
                result = web_server.execute_with_web_monitoring(test_input, enable_live_viz=args.live_viz)
            else:
                result = execution_engine.execute_inference(test_input, enable_live_viz=args.live_viz)
            
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
    
    # 8. Generate visualizations if requested
    if args.visualize:
        step_num = "7" if args.web_viz else "6"
        print(f"\n{step_num}. Generating text-based visualization...")
        try:
            if args.execute and 'result' in locals():
                create_complete_text_report(chip, dnn_manager, result)
            else:
                print_architecture_diagram(chip)
            print("   ✓ Text visualization complete!")
        except Exception as e:
            logging.warning(f"Could not generate visualizations: {e}")
    
    # Launch architecture explorer if requested
    if args.explore_arch:
        print("\n🏗️  Launching Interactive Architecture Explorer...")
        try:
            from visualization.architecture_viz import start_architecture_explorer
            start_architecture_explorer(chip)
        except ImportError:
            from visualization.live_viz import start_architecture_explorer
            start_architecture_explorer(chip)
        except Exception as e:
            logging.warning(f"Could not start architecture explorer: {e}")
    
    # Keep web server running if requested
    if args.web_viz and web_server:
        print(f"\n🌐 Web visualization server running at http://localhost:{args.web_port}/")
        print("   Press Ctrl+C to stop the server")
        try:
            # Keep the main thread alive to serve web requests
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Web server stopped")
            if web_server:
                web_server.stop_monitoring()
    
    print("\n" + "=" * 60)
    print("Simulation completed successfully!")
    if args.web_viz:
        print("Web visualization server may still be running...")
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
  
  # Web visualization (recommended) - prompts before starting
  python main.py --model lenet --execute --web-viz --cycle-accurate
  
  # Auto-start web visualization (no prompt)
  python main.py --model sample_cnn --execute --web-viz --auto-start
  
  # Run with live terminal visualization 
  python main.py --execute --live-viz
  
  # Explore chip architecture interactively
  python main.py --explore-arch
  
  # Run with custom configuration
  python main.py --config my_config.json --execute --output results.json
  
  # Generate comprehensive analysis with architecture exploration
  python main.py --execute --visualize --explore-arch

  # Run with a specific model (e.g., tiny_cnn, lenet)
  python main.py --model tiny_cnn --execute --visualize
  python main.py --model lenet --execute --web-viz
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
    parser.add_argument('--live-viz', action='store_true',
                       help='Enable live visualization during execution')
    parser.add_argument('--explore-arch', action='store_true',
                       help='Launch interactive architecture explorer')
    parser.add_argument(
        '--model',
        type=str,
        default='sample_cnn',
        choices=['sample_cnn', 'tiny_cnn', 'lenet'],
        help='Selects the DNN model to run (default: sample_cnn)'
    )
    parser.add_argument('--web-viz', action='store_true',
                       help='Enable web-based visualization dashboard')
    parser.add_argument('--web-port', type=int, default=8080,
                       help='Port for web visualization server (default: 8080)')
    parser.add_argument('--auto-start', action='store_true',
                       help='Skip web visualization prompt and start simulation automatically')
    
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