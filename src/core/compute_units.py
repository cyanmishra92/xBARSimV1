import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from .memory_system import PartialSumBuffer

class ActivationType(Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    SWISH = "swish"
    GELU = "gelu"

class PoolingType(Enum):
    MAX = "max"
    AVERAGE = "average"
    GLOBAL_AVERAGE = "global_average"

class NormalizationType(Enum):
    BATCH_NORM = "batch_normalization"
    LAYER_NORM = "layer_normalization"
    INSTANCE_NORM = "instance_normalization"

@dataclass
class ComputeUnitConfig:
    """Configuration for compute units"""
    unit_type: str
    pipeline_stages: int = 1
    throughput_ops_per_cycle: int = 1
    latency_cycles: int = 1
    power_consumption_mw: float = 1.0
    area_mm2: float = 0.1

class ShiftAddUnit:
    """Shift-and-add unit for handling multi-bit operations and partial sums"""
    def __init__(self, config: ComputeUnitConfig):
        self.config = config
        self.partial_sum_buffer = PartialSumBuffer(1024, 32)  # 1024 entries, 32-bit each
        self.pipeline = [[] for _ in range(config.pipeline_stages)]
        self.current_cycle = 0
        self.operation_count = 0
        self.total_energy = 0.0
        
    def shift_and_add(self, partial_sums: List[float], bit_positions: List[int]) -> float:
        """Perform shift-and-add operation for multi-bit computation"""
        if len(partial_sums) != len(bit_positions):
            raise ValueError("Partial sums and bit positions must have same length")
            
        # Schedule operation in pipeline
        operation = {
            'type': 'shift_add',
            'partial_sums': partial_sums.copy(),
            'bit_positions': bit_positions.copy(),
            'start_cycle': self.current_cycle,
            'completion_cycle': self.current_cycle + self.config.latency_cycles
        }
        
        self.pipeline[0].append(operation)
        self.operation_count += 1
        
        return self._execute_shift_add(partial_sums, bit_positions)
        
    def _execute_shift_add(self, partial_sums: List[float], bit_positions: List[int]) -> float:
        """Execute the actual shift-and-add computation"""
        result = 0.0
        for partial_sum, bit_pos in zip(partial_sums, bit_positions):
            # Shift partial sum by bit position and add
            result += partial_sum * (2 ** bit_pos)
            
        # Update energy consumption
        self.total_energy += len(partial_sums) * 0.1  # 0.1 pJ per operation
        
        return result
        
    def accumulate_partial_sum(self, output_index: int, value: float) -> bool:
        """Accumulate value into partial sum buffer"""
        return self.partial_sum_buffer.accumulate(output_index, value)
        
    def read_accumulated_sum(self, output_index: int) -> float:
        """Read and clear accumulated partial sum"""
        return self.partial_sum_buffer.read_and_clear(output_index)
        
    def tick(self) -> Dict[str, Any]:
        """Process one cycle of pipeline operation"""
        self.current_cycle += 1
        completed_operations = []
        
        # Move operations through pipeline
        for stage in range(len(self.pipeline) - 1, 0, -1):
            self.pipeline[stage] = self.pipeline[stage - 1]
            
        # Start new stage
        self.pipeline[0] = []
        
        # Check for completed operations
        if self.pipeline[-1]:
            completed_operations = self.pipeline[-1]
            self.pipeline[-1] = []
            
        return {
            'cycle': self.current_cycle,
            'completed_operations': len(completed_operations),
            'pipeline_utilization': sum(len(stage) for stage in self.pipeline) / len(self.pipeline)
        }
        
    def get_statistics(self) -> Dict:
        """Get shift-add unit statistics"""
        return {
            'operation_count': self.operation_count,
            'total_energy': self.total_energy,
            'partial_sum_stats': self.partial_sum_buffer.get_statistics(),
            'current_cycle': self.current_cycle,
            'average_power': self.total_energy / max(self.current_cycle, 1) * 1e12  # pW
        }

class ActivationUnit:
    """Activation function unit"""
    def __init__(self, config: ComputeUnitConfig, activation_types: List[ActivationType]):
        self.config = config
        self.supported_activations = activation_types
        self.pipeline = [[] for _ in range(config.pipeline_stages)]
        self.current_cycle = 0
        self.operation_count = 0
        self.total_energy = 0.0
        
        # Lookup tables for fast activation computation (optional)
        self.lut_size = 1024
        self._build_lookup_tables()
        
    def _build_lookup_tables(self):
        """Build lookup tables for activation functions"""
        self.luts = {}
        x_range = np.linspace(-8, 8, self.lut_size)
        
        for activation in self.supported_activations:
            if activation == ActivationType.RELU:
                self.luts[activation] = np.maximum(0, x_range)
            elif activation == ActivationType.SIGMOID:
                self.luts[activation] = 1 / (1 + np.exp(-x_range))
            elif activation == ActivationType.TANH:
                self.luts[activation] = np.tanh(x_range)
            elif activation == ActivationType.LEAKY_RELU:
                self.luts[activation] = np.where(x_range > 0, x_range, 0.01 * x_range)
            elif activation == ActivationType.SWISH:
                self.luts[activation] = x_range / (1 + np.exp(-x_range))
            elif activation == ActivationType.GELU:
                self.luts[activation] = 0.5 * x_range * (1 + np.tanh(np.sqrt(2/np.pi) * (x_range + 0.044715 * x_range**3)))
                
    def apply_activation(self, input_data: np.ndarray, 
                        activation_type: ActivationType) -> np.ndarray:
        """Apply activation function to input data"""
        if activation_type not in self.supported_activations:
            raise ValueError(f"Activation {activation_type} not supported")
            
        # Schedule operation in pipeline
        operation = {
            'type': 'activation',
            'activation': activation_type,
            'input_shape': input_data.shape,
            'start_cycle': self.current_cycle,
            'completion_cycle': self.current_cycle + self.config.latency_cycles
        }
        
        self.pipeline[0].append(operation)
        self.operation_count += 1
        
        return self._execute_activation(input_data, activation_type)
        
    def _execute_activation(self, input_data: np.ndarray, 
                           activation_type: ActivationType) -> np.ndarray:
        """Execute activation function"""
        if activation_type == ActivationType.RELU:
            result = np.maximum(0, input_data)
        elif activation_type == ActivationType.SIGMOID:
            result = 1 / (1 + np.exp(-np.clip(input_data, -500, 500)))
        elif activation_type == ActivationType.TANH:
            result = np.tanh(input_data)
        elif activation_type == ActivationType.LEAKY_RELU:
            result = np.where(input_data > 0, input_data, 0.01 * input_data)
        elif activation_type == ActivationType.SWISH:
            result = input_data / (1 + np.exp(-np.clip(input_data, -500, 500)))
        elif activation_type == ActivationType.GELU:
            result = 0.5 * input_data * (1 + np.tanh(np.sqrt(2/np.pi) * (input_data + 0.044715 * input_data**3)))
        else:
            result = input_data  # Identity
            
        # Update energy consumption
        self.total_energy += input_data.size * 0.05  # 0.05 pJ per element
        
        return result
        
    def tick(self) -> Dict[str, Any]:
        """Process one cycle of pipeline operation"""
        self.current_cycle += 1
        completed_operations = []
        
        # Move operations through pipeline
        for stage in range(len(self.pipeline) - 1, 0, -1):
            self.pipeline[stage] = self.pipeline[stage - 1]
            
        self.pipeline[0] = []
        
        # Check for completed operations
        if self.pipeline[-1]:
            completed_operations = self.pipeline[-1]
            self.pipeline[-1] = []
            
        return {
            'cycle': self.current_cycle,
            'completed_operations': len(completed_operations),
            'pipeline_utilization': sum(len(stage) for stage in self.pipeline) / len(self.pipeline)
        }
        
    def get_statistics(self) -> Dict:
        """Get activation unit statistics"""
        return {
            'operation_count': self.operation_count,
            'total_energy': self.total_energy,
            'supported_activations': [act.value for act in self.supported_activations],
            'current_cycle': self.current_cycle
        }

class PoolingUnit:
    """Pooling operation unit"""
    def __init__(self, config: ComputeUnitConfig):
        self.config = config
        self.pipeline = [[] for _ in range(config.pipeline_stages)]
        self.current_cycle = 0
        self.operation_count = 0
        self.total_energy = 0.0
        
    def apply_pooling(self, input_data: np.ndarray, pooling_type: PoolingType,
                     kernel_size: Tuple[int, int], stride: Tuple[int, int] = None,
                     padding: str = "valid") -> np.ndarray:
        """Apply pooling operation"""
        if stride is None:
            stride = kernel_size
            
        # Schedule operation
        operation = {
            'type': 'pooling',
            'pooling_type': pooling_type,
            'input_shape': input_data.shape,
            'kernel_size': kernel_size,
            'stride': stride,
            'start_cycle': self.current_cycle,
            'completion_cycle': self.current_cycle + self.config.latency_cycles
        }
        
        self.pipeline[0].append(operation)
        self.operation_count += 1
        
        return self._execute_pooling(input_data, pooling_type, kernel_size, stride, padding)
        
    def _execute_pooling(self, input_data: np.ndarray, pooling_type: PoolingType,
                        kernel_size: Tuple[int, int], stride: Tuple[int, int],
                        padding: str) -> np.ndarray:
        """Execute pooling operation"""
        if len(input_data.shape) == 3:  # H, W, C
            h, w, c = input_data.shape
        elif len(input_data.shape) == 4:  # N, H, W, C
            n, h, w, c = input_data.shape
        else:
            raise ValueError("Unsupported input shape for pooling")
            
        kh, kw = kernel_size
        sh, sw = stride
        
        # Calculate output dimensions
        if padding == "valid":
            oh = (h - kh) // sh + 1
            ow = (w - kw) // sw + 1
        else:  # "same"
            oh = (h + sh - 1) // sh
            ow = (w + sw - 1) // sw
            
        if len(input_data.shape) == 3:
            output = np.zeros((oh, ow, c))
            
            for i in range(oh):
                for j in range(ow):
                    h_start = i * sh
                    h_end = min(h_start + kh, h)
                    w_start = j * sw
                    w_end = min(w_start + kw, w)
                    
                    pool_region = input_data[h_start:h_end, w_start:w_end, :]
                    
                    if pooling_type == PoolingType.MAX:
                        output[i, j, :] = np.max(pool_region, axis=(0, 1))
                    elif pooling_type == PoolingType.AVERAGE:
                        output[i, j, :] = np.mean(pool_region, axis=(0, 1))
                        
        else:  # 4D input
            output = np.zeros((n, oh, ow, c))
            for batch in range(n):
                for i in range(oh):
                    for j in range(ow):
                        h_start = i * sh
                        h_end = min(h_start + kh, h)
                        w_start = j * sw
                        w_end = min(w_start + kw, w)
                        
                        pool_region = input_data[batch, h_start:h_end, w_start:w_end, :]
                        
                        if pooling_type == PoolingType.MAX:
                            output[batch, i, j, :] = np.max(pool_region, axis=(0, 1))
                        elif pooling_type == PoolingType.AVERAGE:
                            output[batch, i, j, :] = np.mean(pool_region, axis=(0, 1))
                            
        # Update energy consumption
        self.total_energy += output.size * 0.02  # 0.02 pJ per output element
        
        return output
        
    def tick(self) -> Dict[str, Any]:
        """Process one cycle of pipeline operation"""
        self.current_cycle += 1
        completed_operations = []
        
        # Move operations through pipeline
        for stage in range(len(self.pipeline) - 1, 0, -1):
            self.pipeline[stage] = self.pipeline[stage - 1]
            
        self.pipeline[0] = []
        
        # Check for completed operations
        if self.pipeline[-1]:
            completed_operations = self.pipeline[-1]
            self.pipeline[-1] = []
            
        return {
            'cycle': self.current_cycle,
            'completed_operations': len(completed_operations)
        }
        
    def get_statistics(self) -> Dict:
        """Get pooling unit statistics"""
        return {
            'operation_count': self.operation_count,
            'total_energy': self.total_energy,
            'current_cycle': self.current_cycle
        }

class NormalizationUnit:
    """Normalization operation unit (BatchNorm, LayerNorm, etc.)"""
    def __init__(self, config: ComputeUnitConfig):
        self.config = config
        self.pipeline = [[] for _ in range(config.pipeline_stages)]
        self.current_cycle = 0
        self.operation_count = 0
        self.total_energy = 0.0
        
    def apply_normalization(self, input_data: np.ndarray, norm_type: NormalizationType,
                           gamma: Optional[np.ndarray] = None, 
                           beta: Optional[np.ndarray] = None,
                           eps: float = 1e-5) -> np.ndarray:
        """Apply normalization operation"""
        # Schedule operation
        operation = {
            'type': 'normalization',
            'norm_type': norm_type,
            'input_shape': input_data.shape,
            'start_cycle': self.current_cycle,
            'completion_cycle': self.current_cycle + self.config.latency_cycles
        }
        
        self.pipeline[0].append(operation)
        self.operation_count += 1
        
        return self._execute_normalization(input_data, norm_type, gamma, beta, eps)
        
    def _execute_normalization(self, input_data: np.ndarray, norm_type: NormalizationType,
                              gamma: Optional[np.ndarray], beta: Optional[np.ndarray],
                              eps: float) -> np.ndarray:
        """Execute normalization operation"""
        if norm_type == NormalizationType.BATCH_NORM:
            # Batch normalization: normalize across batch dimension
            if len(input_data.shape) == 4:  # N, H, W, C
                mean = np.mean(input_data, axis=(0, 1, 2), keepdims=True)
                var = np.var(input_data, axis=(0, 1, 2), keepdims=True)
            elif len(input_data.shape) == 2:  # N, C
                mean = np.mean(input_data, axis=0, keepdims=True)
                var = np.var(input_data, axis=0, keepdims=True)
            else:
                raise ValueError("Unsupported input shape for batch norm")
                
        elif norm_type == NormalizationType.LAYER_NORM:
            # Layer normalization: normalize across feature dimension
            if len(input_data.shape) == 4:  # N, H, W, C
                mean = np.mean(input_data, axis=-1, keepdims=True)
                var = np.var(input_data, axis=-1, keepdims=True)
            elif len(input_data.shape) == 2:  # N, C
                mean = np.mean(input_data, axis=-1, keepdims=True)
                var = np.var(input_data, axis=-1, keepdims=True)
            else:
                raise ValueError("Unsupported input shape for layer norm")
                
        elif norm_type == NormalizationType.INSTANCE_NORM:
            # Instance normalization: normalize per instance
            if len(input_data.shape) == 4:  # N, H, W, C
                mean = np.mean(input_data, axis=(1, 2), keepdims=True)
                var = np.var(input_data, axis=(1, 2), keepdims=True)
            else:
                raise ValueError("Instance norm requires 4D input")
                
        # Normalize
        normalized = (input_data - mean) / np.sqrt(var + eps)
        
        # Apply scale and shift if provided
        if gamma is not None:
            normalized = normalized * gamma
        if beta is not None:
            normalized = normalized + beta
            
        # Update energy consumption
        self.total_energy += input_data.size * 0.1  # 0.1 pJ per element
        
        return normalized
        
    def tick(self) -> Dict[str, Any]:
        """Process one cycle of pipeline operation"""
        self.current_cycle += 1
        completed_operations = []
        
        # Move operations through pipeline
        for stage in range(len(self.pipeline) - 1, 0, -1):
            self.pipeline[stage] = self.pipeline[stage - 1]
            
        self.pipeline[0] = []
        
        # Check for completed operations
        if self.pipeline[-1]:
            completed_operations = self.pipeline[-1]
            self.pipeline[-1] = []
            
        return {
            'cycle': self.current_cycle,
            'completed_operations': len(completed_operations)
        }
        
    def get_statistics(self) -> Dict:
        """Get normalization unit statistics"""
        return {
            'operation_count': self.operation_count,
            'total_energy': self.total_energy,
            'current_cycle': self.current_cycle
        }

class ComputeUnitManager:
    """Manager for all compute units"""
    def __init__(self):
        self.shift_add_units = []
        self.activation_units = []
        self.pooling_units = []
        self.normalization_units = []
        self.current_cycle = 0
        
    def add_shift_add_unit(self, config: ComputeUnitConfig) -> int:
        """Add shift-add unit, returns unit ID"""
        unit = ShiftAddUnit(config)
        self.shift_add_units.append(unit)
        return len(self.shift_add_units) - 1
        
    def add_activation_unit(self, config: ComputeUnitConfig, 
                           activations: List[ActivationType]) -> int:
        """Add activation unit, returns unit ID"""
        unit = ActivationUnit(config, activations)
        self.activation_units.append(unit)
        return len(self.activation_units) - 1
        
    def add_pooling_unit(self, config: ComputeUnitConfig) -> int:
        """Add pooling unit, returns unit ID"""
        unit = PoolingUnit(config)
        self.pooling_units.append(unit)
        return len(self.pooling_units) - 1
        
    def add_normalization_unit(self, config: ComputeUnitConfig) -> int:
        """Add normalization unit, returns unit ID"""
        unit = NormalizationUnit(config)
        self.normalization_units.append(unit)
        return len(self.normalization_units) - 1
        
    def tick_all(self) -> Dict[str, Any]:
        """Tick all compute units"""
        self.current_cycle += 1
        
        events = {
            'cycle': self.current_cycle,
            'shift_add_events': [],
            'activation_events': [],
            'pooling_events': [],
            'normalization_events': []
        }
        
        # Tick all units
        for i, unit in enumerate(self.shift_add_units):
            unit_events = unit.tick()
            unit_events['unit_id'] = i
            events['shift_add_events'].append(unit_events)
            
        for i, unit in enumerate(self.activation_units):
            unit_events = unit.tick()
            unit_events['unit_id'] = i
            events['activation_events'].append(unit_events)
            
        for i, unit in enumerate(self.pooling_units):
            unit_events = unit.tick()
            unit_events['unit_id'] = i
            events['pooling_events'].append(unit_events)
            
        for i, unit in enumerate(self.normalization_units):
            unit_events = unit.tick()
            unit_events['unit_id'] = i
            events['normalization_events'].append(unit_events)
            
        return events
        
    def get_all_statistics(self) -> Dict:
        """Get statistics for all compute units"""
        return {
            'shift_add_units': [unit.get_statistics() for unit in self.shift_add_units],
            'activation_units': [unit.get_statistics() for unit in self.activation_units],
            'pooling_units': [unit.get_statistics() for unit in self.pooling_units],
            'normalization_units': [unit.get_statistics() for unit in self.normalization_units],
            'total_units': (len(self.shift_add_units) + len(self.activation_units) + 
                           len(self.pooling_units) + len(self.normalization_units))
        }