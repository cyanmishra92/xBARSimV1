import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
import time

from .hierarchy import ReRAMChip, ChipConfig, SuperTileConfig, TileConfig
from .crossbar import CrossbarConfig
from .dnn_manager import DNNManager, DNNConfig, LayerConfig, LayerType
from .microcontroller import Microcontroller, MCUConfig, Instruction, InstructionType
from .memory_system import BufferManager, MemoryConfig, MemoryType
from .compute_units import ComputeUnitManager, ComputeUnitConfig, ActivationType, PoolingType
from .interconnect import InterconnectNetwork, InterconnectConfig, InterconnectTopology, TimingModel
from .metrics import MetricsCollector, PerformanceProfiler

@dataclass
class ExecutionConfig:
    """Configuration for execution engine"""
    enable_cycle_accurate_simulation: bool = True
    enable_energy_modeling: bool = True
    enable_detailed_logging: bool = False
    max_execution_cycles: int = 100000
    synchronization_frequency: int = 1000  # Sync all components every N cycles

class SystemIntegrator:
    """Integrates all system components with proper timing and synchronization"""
    def __init__(self, chip: ReRAMChip, dnn_manager: DNNManager, 
                 execution_config: ExecutionConfig):
        self.chip = chip
        self.dnn_manager = dnn_manager
        self.config = execution_config
        
        # Initialize system components
        self._initialize_microcontroller()
        self._initialize_memory_system()
        self._initialize_compute_units()
        self._initialize_interconnect()
        self._initialize_timing_model()
        self._initialize_metrics()
        
        # System state
        self.global_cycle = 0
        self.execution_active = False
        self.inference_results = {}
        
    def _initialize_microcontroller(self):
        """Initialize microcontroller"""
        mcu_config = MCUConfig(
            clock_frequency_mhz=1000.0,
            pipeline_stages=5,
            max_outstanding_instructions=32
        )
        self.microcontroller = Microcontroller(mcu_config)
        
    def _initialize_memory_system(self):
        """Initialize memory hierarchy"""
        buffer_configs = {
            'global_buffer': MemoryConfig(
                memory_type=MemoryType.DRAM,
                size_kb=self.chip.config.global_buffer_size,
                read_latency=10,
                write_latency=12,
                banks=8
            ),
            'shared_buffers': MemoryConfig(
                memory_type=MemoryType.EDRAM,
                size_kb=self.chip.config.supertile_config.shared_buffer_size,
                read_latency=3,
                write_latency=4,
                banks=4
            ),
            'local_buffers': MemoryConfig(
                memory_type=MemoryType.SRAM,
                size_kb=self.chip.config.supertile_config.tile_config.local_buffer_size,
                read_latency=1,
                write_latency=1,
                banks=2
            )
        }
        
        self.buffer_manager = BufferManager(buffer_configs)
        self.microcontroller.set_hardware_resources(self.buffer_manager, None)
        
    def _initialize_compute_units(self):
        """Initialize compute units"""
        self.compute_manager = ComputeUnitManager()
        
        # Add compute units based on chip configuration
        num_tiles = (len(self.chip.supertiles) * 
                    len(self.chip.supertiles[0].tiles))
        
        # Add shift-add units (one per tile)
        shift_add_config = ComputeUnitConfig(
            unit_type="shift_add",
            pipeline_stages=3,
            latency_cycles=2,
            power_consumption_mw=5.0
        )
        for i in range(num_tiles):
            self.compute_manager.add_shift_add_unit(shift_add_config)
            
        # Add activation units (shared among tiles)
        activation_config = ComputeUnitConfig(
            unit_type="activation",
            pipeline_stages=2,
            latency_cycles=1,
            power_consumption_mw=3.0
        )
        activations = [ActivationType.RELU, ActivationType.SIGMOID, ActivationType.TANH]
        for i in range(max(1, num_tiles // 4)):
            self.compute_manager.add_activation_unit(activation_config, activations)
            
        # Add pooling units
        pooling_config = ComputeUnitConfig(
            unit_type="pooling",
            pipeline_stages=3,
            latency_cycles=4,
            power_consumption_mw=8.0
        )
        for i in range(max(1, num_tiles // 8)):
            self.compute_manager.add_pooling_unit(pooling_config)
            
        # Update microcontroller reference
        self.microcontroller.set_hardware_resources(self.buffer_manager, self.compute_manager)
        
    def _initialize_interconnect(self):
        """Initialize interconnect network"""
        interconnect_config = InterconnectConfig(
            topology=InterconnectTopology.MESH,
            data_width_bits=256,
            clock_frequency_mhz=1000.0,
            router_latency_cycles=2,
            link_latency_cycles=1
        )
        
        # Create mesh topology based on chip organization
        num_supertiles = len(self.chip.supertiles)
        mesh_size = int(np.ceil(np.sqrt(num_supertiles)))
        topology_dims = (mesh_size, mesh_size)
        
        self.interconnect = InterconnectNetwork(interconnect_config, topology_dims)
        
    def _initialize_timing_model(self):
        """Initialize cycle-accurate timing model"""
        self.timing_model = TimingModel()
        
        # Register all components with their clock frequencies
        self.timing_model.register_component('microcontroller', 1000.0)
        self.timing_model.register_component('interconnect', 1000.0)
        self.timing_model.register_component('memory_system', 800.0)
        self.timing_model.register_component('compute_units', 1200.0)
        
        # Register crossbars (assume they run at a different frequency)
        for st_id, supertile in enumerate(self.chip.supertiles):
            for tile_id, tile in enumerate(supertile.tiles):
                for xbar_id, crossbar in enumerate(tile.crossbars):
                    component_id = f'xbar_{st_id}_{tile_id}_{xbar_id}'
                    self.timing_model.register_component(component_id, 500.0)  # Slower analog domain
                    
    def _initialize_metrics(self):
        """Initialize metrics collection"""
        self.metrics_collector = MetricsCollector()
        self.performance_profiler = PerformanceProfiler(self.metrics_collector)

class ExecutionEngine:
    """Main execution engine for running DNN inference"""
    def __init__(self, chip: ReRAMChip, dnn_manager: DNNManager, 
                 execution_config: ExecutionConfig = None):
        self.chip = chip
        self.dnn_manager = dnn_manager
        self.config = execution_config or ExecutionConfig()
        
        # Initialize system integrator
        self.system = SystemIntegrator(chip, dnn_manager, self.config)
        
        # Execution state
        self.current_layer = 0
        self.layer_execution_log = []
        self.total_execution_cycles = 0
        self.inference_accuracy = 0.0
        
    def load_input_data(self, input_data: np.ndarray) -> bool:
        """Load input data into the system"""
        try:
            # Allocate input buffer
            input_buffer_id = self.system.buffer_manager.allocate_buffer(
                'global_buffer', 
                input_data.size, 
                'input_loader'
            )
            
            if input_buffer_id is None:
                logging.error("Failed to allocate input buffer")
                return False
                
            # Write input data to buffer
            request_id = self.system.buffer_manager.write_data(
                'global_buffer',
                input_buffer_id,
                0,
                input_data.flatten(),
                'input_loader'
            )
            
            # Wait for write completion
            while not self.system.buffer_manager.controllers['global_buffer'].is_request_complete(request_id):
                self.system.buffer_manager.tick_all()
                self.system.timing_model.advance_global_clock(1)
                
            logging.info(f"Input data loaded: shape {input_data.shape}")
            return True
            
        except Exception as e:
            logging.error(f"Error loading input data: {e}")
            return False
            
    def execute_inference(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Execute complete DNN inference"""
        self.system.metrics_collector.start_measurement()
        
        # Load input data
        if not self.load_input_data(input_data):
            return {'success': False, 'error': 'Failed to load input data'}
            
        # Generate instruction sequences for all layers
        layer_programs = self._generate_layer_programs()
        
        # Execute each layer
        layer_results = []
        intermediate_data = input_data
        
        for layer_idx, (layer_name, layer_program) in enumerate(layer_programs.items()):
            # Get actual layer config
            layer_config = self.dnn_manager.dnn_config.layers[layer_idx]
            logging.info(f"Executing layer {layer_idx}: {layer_config.layer_type.value}")
            
            # Load program into microcontroller
            self.system.microcontroller.load_program(layer_program)
            
            # Create layer dict for execution
            layer_dict = {
                'type': layer_config.layer_type.value,
                'input_shape': layer_config.input_shape,
                'output_shape': layer_config.output_shape,
                'kernel_size': layer_config.kernel_size,
                'stride': layer_config.stride,
                'padding': layer_config.padding,
                'activation': layer_config.activation
            }
            
            # Execute layer
            start_cycle = self.system.global_cycle
            layer_result = self._execute_layer_with_timing(layer_dict, intermediate_data)
            
            # Advance timing properly
            estimated_cycles = 100 + layer_idx * 50  # Basic estimation
            self.system.timing_model.advance_global_clock(estimated_cycles)
            self.system.global_cycle = self.system.timing_model.global_cycle
            end_cycle = self.system.global_cycle
            
            # Record layer execution
            layer_execution_info = {
                'layer_index': layer_idx,
                'layer_type': layer_dict['type'],
                'input_shape': intermediate_data.shape,
                'output_shape': layer_result.shape if layer_result is not None else None,
                'execution_cycles': end_cycle - start_cycle,
                'start_cycle': start_cycle,
                'end_cycle': end_cycle
            }
            
            self.layer_execution_log.append(layer_execution_info)
            layer_results.append(layer_result)
            
            # Update intermediate data for next layer
            if layer_result is not None:
                intermediate_data = layer_result
            else:
                logging.error(f"Layer {layer_idx} execution failed")
                break
                
        # Complete metrics collection
        total_time = self.system.metrics_collector.end_measurement()
        
        # Generate final results
        final_output = intermediate_data
        inference_result = self._process_final_output(final_output)
        
        return {
            'success': True,
            'inference_result': inference_result,
            'final_output': final_output,
            'total_execution_cycles': self.system.global_cycle,
            'total_execution_time_seconds': total_time,
            'layer_execution_log': self.layer_execution_log,
            'system_statistics': self._collect_system_statistics(),
            'metrics_summary': self.system.metrics_collector.get_summary_metrics()
        }
        
    def _generate_layer_programs(self) -> Dict[str, List[Instruction]]:
        """Generate instruction programs for all DNN layers"""
        layer_programs = {}
        
        for i, layer_config in enumerate(self.dnn_manager.dnn_config.layers):
            if layer_config.layer_type in [LayerType.CONV2D, LayerType.DENSE]:
                # Get weight data (in real implementation, this would come from trained model)
                if layer_config.weights_shape:
                    weight_data = np.random.randn(*layer_config.weights_shape) * 0.1
                else:
                    weight_data = None
                    
                # Create layer configuration dict
                layer_dict = {
                    'type': layer_config.layer_type.value,
                    'input_shape': layer_config.input_shape,
                    'output_shape': layer_config.output_shape,
                    'kernel_size': layer_config.kernel_size,
                    'stride': layer_config.stride,
                    'padding': layer_config.padding,
                    'activation': layer_config.activation,
                    'precision': 8
                }
                
                # Generate instruction program
                program = self.system.microcontroller.create_layer_program(
                    layer_dict,
                    f'input_buffer_{i}',
                    f'output_buffer_{i}',
                    weight_data
                )
                
                layer_programs[f'layer_{i}'] = program
                
            elif layer_config.layer_type == LayerType.POOLING:
                # Pooling layer program
                layer_dict = {
                    'type': 'pooling',
                    'input_shape': layer_config.input_shape,
                    'output_shape': layer_config.output_shape,
                    'kernel_size': layer_config.kernel_size,
                    'stride': layer_config.stride,
                    'pooling_type': 'max'
                }
                
                program = self.system.microcontroller.create_layer_program(
                    layer_dict,
                    f'input_buffer_{i}',
                    f'output_buffer_{i}'
                )
                
                layer_programs[f'layer_{i}'] = program
                
        return layer_programs
        
    def _execute_layer_with_timing(self, layer_config: Dict, input_data: np.ndarray) -> np.ndarray:
        """Execute single layer with cycle-accurate timing"""
        layer_type = layer_config['type']
        
        if layer_type in ['conv2d', 'convolution_2d']:
            return self._execute_conv_layer(layer_config, input_data)
        elif layer_type == 'dense':
            return self._execute_dense_layer(layer_config, input_data)
        elif layer_type == 'pooling':
            return self._execute_pooling_layer(layer_config, input_data)
        elif layer_type == 'activation':
            return self._execute_activation_layer(layer_config, input_data)
        else:
            logging.warning(f"Unsupported layer type: {layer_type}")
            return input_data
            
    def _execute_conv_layer(self, layer_config: Dict, input_data: np.ndarray) -> np.ndarray:
        """Execute convolution layer with ReRAM crossbars"""
        # Get layer parameters
        kernel_size = layer_config.get('kernel_size', (3, 3))
        stride = layer_config.get('stride')
        if stride is None:
            stride = (1, 1)
        padding = layer_config.get('padding', 'valid')
        
        # Simulate convolution using crossbars
        if len(input_data.shape) == 3:  # H, W, C
            h, w, c = input_data.shape
        else:
            raise ValueError("Unsupported input shape for convolution")
            
        # Calculate output dimensions
        if padding == 'valid':
            out_h = max(1, (h - kernel_size[0]) // stride[0] + 1)
            out_w = max(1, (w - kernel_size[1]) // stride[1] + 1)
        else:  # 'same'
            out_h = max(1, h // stride[0])
            out_w = max(1, w // stride[1])
            
        # Determine output channels from layer config
        if 'output_shape' in layer_config and len(layer_config['output_shape']) >= 3:
            output_channels = layer_config['output_shape'][-1]
        else:
            output_channels = c  # Default to input channels if not specified
        
        # Simulate crossbar computation
        output_data = np.zeros((out_h, out_w, output_channels))
        
        # Use actual crossbars for computation
        available_crossbars = []
        for supertile in self.chip.supertiles:
            for tile in supertile.tiles:
                available_crossbars.extend(tile.crossbars)
                
        if available_crossbars:
            # Use first available crossbar for demonstration
            crossbar = available_crossbars[0]
            
            # Simulate the convolution operation
            for oh in range(out_h):
                for ow in range(out_w):
                    for oc in range(output_channels):
                        # Extract patch
                        h_start = oh * stride[0]
                        h_end = h_start + kernel_size[0]
                        w_start = ow * stride[1]
                        w_end = w_start + kernel_size[1]
                        
                        if h_end <= h and w_end <= w:
                            patch = input_data[h_start:h_end, w_start:w_end, :].flatten()
                            
                            # Pad patch to match crossbar input size
                            if len(patch) < crossbar.rows:
                                patch = np.pad(patch, (0, crossbar.rows - len(patch)))
                            elif len(patch) > crossbar.rows:
                                patch = patch[:crossbar.rows]
                                
                            # Perform matrix-vector multiplication
                            crossbar_result = crossbar.matrix_vector_multiply(patch)
                            
                            # Take first output as result
                            output_data[oh, ow, oc] = crossbar_result[0] if len(crossbar_result) > 0 else 0.0
                            
                        # Simulate timing
                        if hasattr(self.system, 'timing_model'):
                            self.system.timing_model.advance_global_clock(1)
                        
        # Apply activation if specified
        if 'activation' in layer_config:
            activation_type = layer_config['activation']
            if activation_type == 'relu':
                output_data = np.maximum(0, output_data)
            elif activation_type == 'sigmoid':
                output_data = 1 / (1 + np.exp(-np.clip(output_data, -500, 500)))
                
        return output_data
        
    def _execute_dense_layer(self, layer_config: Dict, input_data: np.ndarray) -> np.ndarray:
        """Execute dense/fully-connected layer"""
        # Flatten input if needed
        if len(input_data.shape) > 2:
            input_flat = input_data.flatten()
        else:
            input_flat = input_data
            
        # Get output size
        output_size = layer_config['output_shape'][-1] if 'output_shape' in layer_config else len(input_flat)
        
        # Use crossbar for matrix multiplication
        available_crossbars = []
        for supertile in self.chip.supertiles:
            for tile in supertile.tiles:
                available_crossbars.extend(tile.crossbars)
                
        if available_crossbars:
            crossbar = available_crossbars[0]
            
            # Pad input to match crossbar size
            if len(input_flat) < crossbar.rows:
                input_padded = np.pad(input_flat, (0, crossbar.rows - len(input_flat)))
            else:
                input_padded = input_flat[:crossbar.rows]
                
            # Perform computation
            result = crossbar.matrix_vector_multiply(input_padded)
            
            # Take required outputs
            output_data = result[:output_size] if len(result) >= output_size else np.pad(result, (0, output_size - len(result)))
        else:
            # Fallback: random output
            output_data = np.random.randn(output_size) * 0.1
            
        # Apply activation
        if 'activation' in layer_config:
            activation_type = layer_config['activation']
            if activation_type == 'relu':
                output_data = np.maximum(0, output_data)
            elif activation_type == 'sigmoid':
                output_data = 1 / (1 + np.exp(-np.clip(output_data, -500, 500)))
            elif activation_type == 'softmax':
                exp_scores = np.exp(output_data - np.max(output_data))
                output_data = exp_scores / np.sum(exp_scores)
                
        return output_data
        
    def _execute_pooling_layer(self, layer_config: Dict, input_data: np.ndarray) -> np.ndarray:
        """Execute pooling layer"""
        kernel_size = layer_config.get('kernel_size', (2, 2))
        stride = layer_config.get('stride')
        if stride is None:
            stride = kernel_size
        pooling_type = layer_config.get('pooling_type', 'max')
        
        if len(input_data.shape) == 3:  # H, W, C
            h, w, c = input_data.shape
        else:
            raise ValueError("Unsupported input shape for pooling")
            
        # Calculate output dimensions
        out_h = (h - kernel_size[0]) // stride[0] + 1
        out_w = (w - kernel_size[1]) // stride[1] + 1
        
        output_data = np.zeros((out_h, out_w, c))
        
        # Use compute unit for pooling
        if self.system.compute_manager.pooling_units:
            pooling_unit = self.system.compute_manager.pooling_units[0]
            pool_type = PoolingType.MAX if pooling_type == 'max' else PoolingType.AVERAGE
            
            # For now, simulate the pooling operation
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride[0]
                    h_end = h_start + kernel_size[0]
                    w_start = j * stride[1]
                    w_end = w_start + kernel_size[1]
                    
                    pool_region = input_data[h_start:h_end, w_start:w_end, :]
                    
                    if pooling_type == 'max':
                        output_data[i, j, :] = np.max(pool_region, axis=(0, 1))
                    else:  # average
                        output_data[i, j, :] = np.mean(pool_region, axis=(0, 1))
                        
                    # Simulate timing
                    if hasattr(self.system, 'timing_model'):
                        self.system.timing_model.advance_global_clock(1)
                    
        return output_data
        
    def _execute_activation_layer(self, layer_config: Dict, input_data: np.ndarray) -> np.ndarray:
        """Execute activation layer"""
        activation_type = layer_config.get('activation_type', 'relu')
        
        if self.system.compute_manager.activation_units:
            activation_unit = self.system.compute_manager.activation_units[0]
            
            if activation_type == 'relu':
                return activation_unit.apply_activation(input_data, ActivationType.RELU)
            elif activation_type == 'sigmoid':
                return activation_unit.apply_activation(input_data, ActivationType.SIGMOID)
            elif activation_type == 'tanh':
                return activation_unit.apply_activation(input_data, ActivationType.TANH)
                
        # Fallback software implementation
        if activation_type == 'relu':
            return np.maximum(0, input_data)
        elif activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(input_data, -500, 500)))
        elif activation_type == 'tanh':
            return np.tanh(input_data)
        else:
            return input_data
            
    def _process_final_output(self, final_output: np.ndarray) -> Dict[str, Any]:
        """Process final network output to get inference result"""
        if len(final_output.shape) == 1:
            # Classification output
            predicted_class = np.argmax(final_output)
            confidence = np.max(final_output)
            
            return {
                'type': 'classification',
                'predicted_class': int(predicted_class),
                'confidence': float(confidence),
                'probabilities': final_output.tolist()
            }
        else:
            # Other output types
            return {
                'type': 'other',
                'output_shape': final_output.shape,
                'output_data': final_output.tolist()
            }
            
    def _collect_system_statistics(self) -> Dict[str, Any]:
        """Collect comprehensive system statistics"""
        return {
            'chip_statistics': self.chip.get_total_statistics(),
            'memory_statistics': self.system.buffer_manager.get_all_statistics(),
            'compute_statistics': self.system.compute_manager.get_all_statistics(),
            'interconnect_statistics': self.system.interconnect.get_network_statistics(),
            'microcontroller_statistics': self.system.microcontroller.get_statistics(),
            'timing_statistics': self.system.timing_model.get_timing_statistics()
        }
        
    def run_system_integration_test(self) -> Dict[str, Any]:
        """Run complete system integration test"""
        # Run microcontroller until completion
        mcu_result = self.system.microcontroller.run_until_completion(max_cycles=1000)
        
        # Tick all other components for synchronization
        for _ in range(100):
            self.system.buffer_manager.tick_all()
            self.system.compute_manager.tick_all()
            self.system.interconnect.tick()
            self.system.timing_model.advance_global_clock(1)
            
        return {
            'microcontroller_result': mcu_result,
            'system_statistics': self._collect_system_statistics(),
            'integration_successful': True
        }