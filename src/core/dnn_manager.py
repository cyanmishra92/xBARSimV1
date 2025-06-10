import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import math
import logging
from .hierarchy import ReRAMChip, ChipConfig, SuperTileConfig, TileConfig
from .crossbar import CrossbarConfig

class LayerType(Enum):
    CONV2D = "convolution_2d"
    DENSE = "dense"
    POOLING = "pooling"
    ACTIVATION = "activation"
    BATCH_NORM = "batch_normalization"

class DataflowPattern(Enum):
    WEIGHT_STATIONARY = "weight_stationary"
    INPUT_STATIONARY = "input_stationary"
    OUTPUT_STATIONARY = "output_stationary"
    ROW_STATIONARY = "row_stationary"

@dataclass
class LayerConfig:
    """Configuration for a DNN layer"""
    layer_type: LayerType
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    kernel_size: Optional[Tuple[int, int]] = None  # For conv layers
    stride: Optional[Tuple[int, int]] = None
    padding: Optional[str] = None
    activation: Optional[str] = None
    weights_shape: Optional[Tuple[int, ...]] = None
    bias_shape: Optional[Tuple[int, ...]] = None
    
@dataclass 
class QuantizationConfig:
    """Quantization configuration for edge computing"""
    weight_bits: int = 4  # 4-bit weights for edge inference
    input_bits: int = 8   # 8-bit input activations
    accumulation_bits: int = 16  # 16-bit accumulation
    enable_dynamic_quantization: bool = True
    quantization_scheme: str = "uniform"  # uniform, non-uniform
    clipping_strategy: str = "percentile"  # percentile, minmax

@dataclass
class DNNConfig:
    """Configuration for entire DNN model"""
    model_name: str
    layers: List[LayerConfig]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    total_parameters: int = 0
    total_macs: int = 0  # Multiply-accumulate operations
    precision: int = 8  # bits (legacy - use quantization_config for edge)
    quantization_config: Optional[QuantizationConfig] = None  # Edge quantization
    
    def __post_init__(self):
        # Calculate total parameters and MACs
        self.total_parameters = sum(
            np.prod(layer.weights_shape) if layer.weights_shape else 0 
            for layer in self.layers
        )
        
        self.total_macs = self._calculate_total_macs()
        
    def _calculate_total_macs(self) -> int:
        """Calculate total multiply-accumulate operations"""
        total_macs = 0
        for layer in self.layers:
            if layer.layer_type == LayerType.CONV2D:
                # For conv layer: output_h * output_w * output_channels * kernel_h * kernel_w * input_channels
                if layer.kernel_size and layer.weights_shape:
                    output_size = np.prod(layer.output_shape)
                    kernel_size = np.prod(layer.kernel_size)
                    input_channels = layer.weights_shape[-1] if len(layer.weights_shape) > 1 else 1
                    total_macs += output_size * kernel_size * input_channels
            elif layer.layer_type == LayerType.DENSE:
                # For dense layer: input_size * output_size
                if layer.weights_shape:
                    total_macs += np.prod(layer.weights_shape)
        return total_macs

class QuantizationUtils:
    """Utility functions for quantization operations"""
    
    @staticmethod
    def quantize_weights(weights: np.ndarray, bits: int, scheme: str = "uniform") -> Tuple[np.ndarray, Dict]:
        """Quantize weights to specified bit precision"""
        if scheme == "uniform":
            # Uniform quantization
            w_min, w_max = weights.min(), weights.max()
            scale = (w_max - w_min) / (2**bits - 1)
            zero_point = -round(w_min / scale)
            
            # Quantize
            q_weights = np.round(weights / scale + zero_point)
            q_weights = np.clip(q_weights, 0, 2**bits - 1)
            
            # Dequantize for validation
            dequant_weights = scale * (q_weights - zero_point)
            
            quantization_info = {
                'scale': scale,
                'zero_point': zero_point,
                'min_val': w_min,
                'max_val': w_max,
                'quantization_error': np.mean(np.abs(weights - dequant_weights))
            }
            
            return q_weights.astype(np.int8), quantization_info
        else:
            raise NotImplementedError(f"Quantization scheme {scheme} not implemented")
    
    @staticmethod
    def quantize_activations(activations: np.ndarray, bits: int, clipping_strategy: str = "percentile") -> Tuple[np.ndarray, Dict]:
        """Quantize activations with clipping"""
        if clipping_strategy == "percentile":
            # Use 99.9th percentile to avoid outliers
            clip_min = np.percentile(activations, 0.1)
            clip_max = np.percentile(activations, 99.9)
        else:  # minmax
            clip_min, clip_max = activations.min(), activations.max()
        
        # Clip activations
        clipped_activations = np.clip(activations, clip_min, clip_max)
        
        # Quantize
        scale = (clip_max - clip_min) / (2**bits - 1)
        zero_point = -round(clip_min / scale)
        
        q_activations = np.round(clipped_activations / scale + zero_point)
        q_activations = np.clip(q_activations, 0, 2**bits - 1)
        
        quantization_info = {
            'scale': scale,
            'zero_point': zero_point,
            'clip_min': clip_min,
            'clip_max': clip_max,
            'clipping_ratio': np.mean((activations < clip_min) | (activations > clip_max))
        }
        
        return q_activations.astype(np.uint8), quantization_info
    
    @staticmethod
    def simulate_quantized_inference(input_data: np.ndarray, weights: np.ndarray, 
                                   quant_config: QuantizationConfig) -> Tuple[np.ndarray, Dict]:
        """Simulate quantized matrix-vector multiplication"""
        # Quantize inputs
        q_inputs, input_info = QuantizationUtils.quantize_activations(
            input_data, quant_config.input_bits, quant_config.clipping_strategy)
        
        # Quantize weights  
        q_weights, weight_info = QuantizationUtils.quantize_weights(
            weights, quant_config.weight_bits, quant_config.quantization_scheme)
        
        # Perform quantized computation (simplified)
        # In reality, this would be done in ReRAM crossbars
        result = np.dot(q_inputs.astype(np.int32), q_weights.astype(np.int32))
        
        # Simulate accumulation with higher precision
        accumulated_result = result.astype(np.int32)
        
        # Convert back to float (this would be done by ADC in real system)
        input_scale = input_info['scale']
        weight_scale = weight_info['scale']
        output_scale = input_scale * weight_scale
        
        final_result = accumulated_result * output_scale
        
        simulation_info = {
            'input_quantization': input_info,
            'weight_quantization': weight_info,
            'output_scale': output_scale,
            'bit_precision_used': {
                'input_bits': quant_config.input_bits,
                'weight_bits': quant_config.weight_bits,
                'accumulation_bits': quant_config.accumulation_bits
            }
        }
        
        return final_result, simulation_info

class HardwareRequirement:
    """Calculate hardware requirements for a DNN"""
    def __init__(self, dnn_config: DNNConfig):
        self.dnn_config = dnn_config
        
    def calculate_minimum_crossbars(self, crossbar_size: Tuple[int, int]) -> int:
        """Calculate minimum number of crossbars needed"""
        crossbar_capacity = crossbar_size[0] * crossbar_size[1]
        total_weights = self.dnn_config.total_parameters
        
        # Account for precision - multiple crossbars may be needed for multi-bit weights
        precision_factor = max(1, self.dnn_config.precision // 1)  # Simplified
        
        min_crossbars = math.ceil(total_weights * precision_factor / crossbar_capacity)
        return min_crossbars
        
    def calculate_memory_requirements(self) -> Dict[str, int]:
        """Calculate memory requirements for intermediate data"""
        # Calculate activation memory requirements
        activation_memory = 0
        for layer in self.dnn_config.layers:
            layer_activation_size = np.prod(layer.output_shape) * (self.dnn_config.precision // 8)
            activation_memory += layer_activation_size
            
        # Weight memory
        weight_memory = self.dnn_config.total_parameters * (self.dnn_config.precision // 8)
        
        return {
            'weight_memory_bytes': weight_memory,
            'activation_memory_bytes': activation_memory,
            'total_memory_bytes': weight_memory + activation_memory
        }
        
    def recommend_hardware_config(self, crossbar_size: Tuple[int, int] = (128, 128)) -> Dict:
        """Recommend optimal hardware configuration"""
        min_crossbars = self.calculate_minimum_crossbars(crossbar_size)
        memory_req = self.calculate_memory_requirements()
        
        # Recommend hierarchy based on crossbar count
        if min_crossbars <= 4:
            # Small model - single tile
            recommended_config = {
                'supertiles': 1,
                'tiles_per_supertile': 1,
                'crossbars_per_tile': max(4, min_crossbars),
                'crossbar_size': crossbar_size
            }
        elif min_crossbars <= 16:
            # Medium model - single supertile, multiple tiles
            tiles_needed = math.ceil(min_crossbars / 4)
            recommended_config = {
                'supertiles': 1,
                'tiles_per_supertile': tiles_needed,
                'crossbars_per_tile': 4,
                'crossbar_size': crossbar_size
            }
        else:
            # Large model - multiple supertiles
            supertiles_needed = math.ceil(min_crossbars / 16)
            recommended_config = {
                'supertiles': supertiles_needed,
                'tiles_per_supertile': 4,
                'crossbars_per_tile': 4,
                'crossbar_size': crossbar_size
            }
            
        # Add memory recommendations
        recommended_config.update({
            'local_buffer_kb': max(64, memory_req['activation_memory_bytes'] // (1024 * recommended_config['supertiles'] * recommended_config['tiles_per_supertile'])),
            'shared_buffer_kb': max(512, memory_req['activation_memory_bytes'] // (1024 * recommended_config['supertiles'])),
            'global_buffer_kb': max(2048, memory_req['total_memory_bytes'] // 1024)
        })
        
        return recommended_config

class LayerMapper:
    """Maps DNN layers to hardware resources"""
    def __init__(self, chip: ReRAMChip):
        self.chip = chip
        
    def map_conv2d_layer(self, layer_config: LayerConfig, weight_data: np.ndarray) -> Dict:
        """Map convolution layer to crossbars"""
        if layer_config.layer_type != LayerType.CONV2D:
            raise ValueError("Layer must be CONV2D type")
            
        # Extract convolution parameters
        kernel_h, kernel_w = layer_config.kernel_size
        input_channels = layer_config.input_shape[-1]
        output_channels = layer_config.output_shape[-1]
        
        # Reshape weights for crossbar mapping
        # Weight shape: [kernel_h, kernel_w, input_channels, output_channels]
        weights_reshaped = weight_data.reshape(-1, output_channels)
        
        # Determine crossbar allocation
        mapping_info = self._allocate_crossbars_for_weights(weights_reshaped)
        
        return {
            'layer_type': layer_config.layer_type.value,
            'original_weight_shape': weight_data.shape,
            'reshaped_weight_shape': weights_reshaped.shape,
            'crossbar_allocation': mapping_info,
            'dataflow_pattern': DataflowPattern.WEIGHT_STATIONARY.value
        }
        
    def map_dense_layer(self, layer_config: LayerConfig, weight_data: np.ndarray) -> Dict:
        """Map dense layer to crossbars"""
        if layer_config.layer_type != LayerType.DENSE:
            raise ValueError("Layer must be DENSE type")
            
        # Dense layer weights are already in matrix form
        mapping_info = self._allocate_crossbars_for_weights(weight_data)
        
        return {
            'layer_type': layer_config.layer_type.value,
            'weight_shape': weight_data.shape,
            'crossbar_allocation': mapping_info,
            'dataflow_pattern': DataflowPattern.WEIGHT_STATIONARY.value
        }
        
    def _allocate_crossbars_for_weights(self, weight_matrix: np.ndarray) -> Dict:
        """Allocate crossbars for weight matrix"""
        weight_rows, weight_cols = weight_matrix.shape
        
        # Get crossbar configuration
        crossbar_config = self.chip.supertiles[0].tiles[0].config.crossbar_config
        crossbar_rows, crossbar_cols = crossbar_config.rows, crossbar_config.cols
        
        # Calculate how many crossbars needed
        crossbars_per_row = math.ceil(weight_cols / crossbar_cols)
        crossbars_per_col = math.ceil(weight_rows / crossbar_rows)
        total_crossbars_needed = crossbars_per_row * crossbars_per_col
        
        # Check if we have enough crossbars
        total_available_crossbars = (len(self.chip.supertiles) * 
                                   len(self.chip.supertiles[0].tiles) * 
                                   len(self.chip.supertiles[0].tiles[0].crossbars))
        
        if total_crossbars_needed > total_available_crossbars:
            raise ValueError(f"Not enough crossbars: need {total_crossbars_needed}, have {total_available_crossbars}")
            
        # Create allocation map
        allocation_map = []
        crossbar_idx = 0
        
        for row_block in range(crossbars_per_col):
            for col_block in range(crossbars_per_row):
                # Calculate supertile, tile, and crossbar indices
                supertile_idx = crossbar_idx // (len(self.chip.supertiles[0].tiles) * len(self.chip.supertiles[0].tiles[0].crossbars))
                remaining = crossbar_idx % (len(self.chip.supertiles[0].tiles) * len(self.chip.supertiles[0].tiles[0].crossbars))
                tile_idx = remaining // len(self.chip.supertiles[0].tiles[0].crossbars)
                xbar_idx = remaining % len(self.chip.supertiles[0].tiles[0].crossbars)
                
                # Calculate weight matrix slice
                row_start = row_block * crossbar_rows
                row_end = min((row_block + 1) * crossbar_rows, weight_rows)
                col_start = col_block * crossbar_cols
                col_end = min((col_block + 1) * crossbar_cols, weight_cols)
                
                allocation_map.append({
                    'supertile_id': supertile_idx,
                    'tile_id': tile_idx,
                    'crossbar_id': xbar_idx,
                    'weight_slice': {
                        'row_range': (row_start, row_end),
                        'col_range': (col_start, col_end)
                    }
                })
                
                crossbar_idx += 1
                
        return {
            'total_crossbars_used': total_crossbars_needed,
            'crossbars_per_row': crossbars_per_row,
            'crossbars_per_col': crossbars_per_col,
            'allocation_map': allocation_map
        }

class DNNManager:
    """Main manager for DNN configuration and execution"""
    def __init__(self, dnn_config: DNNConfig, chip: Optional[ReRAMChip] = None):
        self.dnn_config = dnn_config
        self.chip = chip
        self.layer_mappings = {}
        self.execution_stats = {}
        
        # Hardware requirement calculator
        self.hw_requirement = HardwareRequirement(dnn_config)
        
        # Auto-generate hardware if not provided
        if chip is None:
            self.chip = self._auto_generate_hardware()
            
        # Layer mapper
        self.layer_mapper = LayerMapper(self.chip)
        
    def _auto_generate_hardware(self) -> ReRAMChip:
        """Automatically generate hardware configuration based on DNN requirements"""
        recommended_config = self.hw_requirement.recommend_hardware_config()
        
        # Create crossbar configuration
        crossbar_config = CrossbarConfig(
            rows=recommended_config['crossbar_size'][0],
            cols=recommended_config['crossbar_size'][1]
        )
        
        # Create tile configuration
        tile_config = TileConfig(
            crossbars_per_tile=recommended_config['crossbars_per_tile'],
            crossbar_config=crossbar_config,
            local_buffer_size=recommended_config['local_buffer_kb']
        )
        
        # Create supertile configuration
        supertile_config = SuperTileConfig(
            tiles_per_supertile=recommended_config['tiles_per_supertile'],
            tile_config=tile_config,
            shared_buffer_size=recommended_config['shared_buffer_kb']
        )
        
        # Create chip configuration
        chip_config = ChipConfig(
            supertiles_per_chip=recommended_config['supertiles'],
            supertile_config=supertile_config,
            global_buffer_size=recommended_config['global_buffer_kb']
        )
        
        return ReRAMChip(chip_config)
        
    def map_dnn_to_hardware(self, weight_data: Dict[str, np.ndarray]) -> Dict:
        """Map entire DNN to hardware"""
        if len(weight_data) != len([l for l in self.dnn_config.layers if l.weights_shape]):
            raise ValueError("Weight data doesn't match number of layers with weights")
            
        layer_mappings = {}
        weight_layer_idx = 0
        
        for i, layer_config in enumerate(self.dnn_config.layers):
            if layer_config.weights_shape is not None:
                layer_name = f"layer_{i}_{layer_config.layer_type.value}"
                layer_weights = list(weight_data.values())[weight_layer_idx]
                
                if layer_config.layer_type == LayerType.CONV2D:
                    mapping = self.layer_mapper.map_conv2d_layer(layer_config, layer_weights)
                elif layer_config.layer_type == LayerType.DENSE:
                    mapping = self.layer_mapper.map_dense_layer(layer_config, layer_weights)
                else:
                    # Skip layers without weights
                    continue
                    
                layer_mappings[layer_name] = mapping
                weight_layer_idx += 1
                
        self.layer_mappings = layer_mappings
        return layer_mappings
        
    def validate_hardware_capacity(self) -> Dict[str, bool]:
        """Validate if hardware can accommodate the DNN"""
        chip_config = self.chip.get_chip_configuration()
        memory_req = self.hw_requirement.calculate_memory_requirements()
        
        # Check crossbar capacity
        total_crossbars = chip_config['compute_capacity']['total_crossbars']
        crossbar_size_parts = chip_config['compute_capacity']['crossbar_size'].split('x')
        min_crossbars = self.hw_requirement.calculate_minimum_crossbars(
            (int(crossbar_size_parts[0]), int(crossbar_size_parts[1]))
        )
        crossbar_sufficient = total_crossbars >= min_crossbars
        
        # Check memory capacity (simplified)
        global_buffer_kb = int(chip_config['memory_hierarchy']['global_buffer'].split()[0])
        memory_sufficient = (global_buffer_kb * 1024) >= memory_req['total_memory_bytes']
        
        return {
            'crossbar_capacity_sufficient': crossbar_sufficient,
            'memory_capacity_sufficient': memory_sufficient,
            'overall_sufficient': crossbar_sufficient and memory_sufficient,
            'utilization': {
                'crossbar_utilization': min_crossbars / total_crossbars,
                'memory_utilization': memory_req['total_memory_bytes'] / (global_buffer_kb * 1024)
            }
        }
        
    def get_hardware_recommendation(self) -> Dict:
        """Get hardware recommendations for current DNN"""
        return self.hw_requirement.recommend_hardware_config()
        
    def print_dnn_summary(self):
        """Print DNN configuration summary"""
        print("=" * 60)
        print(f"DNN Configuration Summary: {self.dnn_config.model_name}")
        print("=" * 60)
        
        print(f"Input Shape: {self.dnn_config.input_shape}")
        print(f"Output Shape: {self.dnn_config.output_shape}")
        print(f"Total Parameters: {self.dnn_config.total_parameters:,}")
        print(f"Total MACs: {self.dnn_config.total_macs:,}")
        print(f"Precision: {self.dnn_config.precision} bits")
        
        print(f"\nLayers ({len(self.dnn_config.layers)}):")
        for i, layer in enumerate(self.dnn_config.layers):
            print(f"  {i+1}. {layer.layer_type.value}: {layer.input_shape} -> {layer.output_shape}")
            if layer.weights_shape:
                print(f"     Weights: {layer.weights_shape} ({np.prod(layer.weights_shape):,} parameters)")
                
        # Hardware requirements
        hw_req = self.get_hardware_recommendation()
        print(f"\nRecommended Hardware:")
        print(f"  SuperTiles: {hw_req['supertiles']}")
        print(f"  Tiles per SuperTile: {hw_req['tiles_per_supertile']}")
        print(f"  Crossbars per Tile: {hw_req['crossbars_per_tile']}")
        print(f"  Total Crossbars: {hw_req['supertiles'] * hw_req['tiles_per_supertile'] * hw_req['crossbars_per_tile']}")
        
        print("=" * 60)
        
    def print_mapping_summary(self):
        """Print DNN to hardware mapping summary"""
        if not self.layer_mappings:
            print("No layer mappings available. Run map_dnn_to_hardware() first.")
            return
            
        print("=" * 60)
        print("DNN to Hardware Mapping Summary")
        print("=" * 60)
        
        total_crossbars_used = 0
        for layer_name, mapping in self.layer_mappings.items():
            print(f"\n{layer_name}:")
            print(f"  Type: {mapping['layer_type']}")
            print(f"  Crossbars used: {mapping['crossbar_allocation']['total_crossbars_used']}")
            print(f"  Mapping pattern: {mapping['crossbar_allocation']['crossbars_per_col']}x{mapping['crossbar_allocation']['crossbars_per_row']}")
            total_crossbars_used += mapping['crossbar_allocation']['total_crossbars_used']
            
        print(f"\nTotal crossbars used: {total_crossbars_used}")
        
        # Validation
        validation = self.validate_hardware_capacity()
        print(f"\nHardware Validation:")
        print(f"  Crossbar capacity sufficient: {validation['crossbar_capacity_sufficient']}")
        print(f"  Memory capacity sufficient: {validation['memory_capacity_sufficient']}")
        print(f"  Overall sufficient: {validation['overall_sufficient']}")
        print(f"  Crossbar utilization: {validation['utilization']['crossbar_utilization']:.2%}")
        print(f"  Memory utilization: {validation['utilization']['memory_utilization']:.2%}")
        
        print("=" * 60)