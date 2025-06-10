import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from .crossbar import CrossbarArray, CrossbarConfig
from .peripherals import PeripheralManager, ADCConfig, DACConfig, SenseAmplifierConfig, DriverConfig

class InterconnectType(Enum):
    BUS = "bus"
    MESH = "mesh"
    TORUS = "torus"
    CROSSBAR = "crossbar"

@dataclass
class TileConfig:
    """Configuration for a processing tile"""
    crossbars_per_tile: int = 4
    crossbar_config: CrossbarConfig = None
    local_buffer_size: int = 1024  # KB
    local_buffer_type: str = "SRAM"  # SRAM, eDRAM, ReRAM
    adc_sharing: bool = True  # Share ADCs among crossbars in tile
    adcs_per_tile: int = 32
    interconnect_type: InterconnectType = InterconnectType.BUS
    
@dataclass
class SuperTileConfig:
    """Configuration for a super tile (collection of tiles)"""
    tiles_per_supertile: int = 4
    tile_config: TileConfig = None
    shared_buffer_size: int = 4096  # KB
    shared_buffer_type: str = "eDRAM"
    interconnect_bandwidth: float = 1e9  # bits/second
    interconnect_type: InterconnectType = InterconnectType.MESH

@dataclass
class ChipConfig:
    """Configuration for the entire chip"""
    supertiles_per_chip: int = 4
    supertile_config: SuperTileConfig = None
    global_buffer_size: int = 16384  # KB
    global_buffer_type: str = "DRAM"
    memory_controller_bandwidth: float = 1e10  # bits/second
    on_chip_interconnect: InterconnectType = InterconnectType.MESH

class LocalBuffer:
    """Local buffer for tiles"""
    def __init__(self, size_kb: int, buffer_type: str = "SRAM"):
        self.size_kb = size_kb
        self.size_bytes = size_kb * 1024
        self.buffer_type = buffer_type
        self.data = {}  # Simplified storage
        self.access_count = 0
        self.total_energy = 0.0
        
        # Buffer characteristics based on type
        if buffer_type == "SRAM":
            self.read_energy_per_bit = 1e-15  # J/bit
            self.write_energy_per_bit = 2e-15  # J/bit
            self.access_time = 1e-9  # seconds
        elif buffer_type == "eDRAM":
            self.read_energy_per_bit = 0.5e-15  # J/bit
            self.write_energy_per_bit = 1e-15  # J/bit
            self.access_time = 2e-9  # seconds
        elif buffer_type == "ReRAM":
            self.read_energy_per_bit = 0.1e-15  # J/bit
            self.write_energy_per_bit = 10e-15  # J/bit
            self.access_time = 5e-9  # seconds
            
    def read(self, address: int, size_bits: int) -> Any:
        """Read data from buffer"""
        self.access_count += 1
        self.total_energy += size_bits * self.read_energy_per_bit
        return self.data.get(address, 0)
        
    def write(self, address: int, data: Any, size_bits: int):
        """Write data to buffer"""
        self.access_count += 1
        self.total_energy += size_bits * self.write_energy_per_bit
        self.data[address] = data
        
    def get_statistics(self) -> Dict:
        return {
            'access_count': self.access_count,
            'total_energy': self.total_energy,
            'utilization': len(self.data) / (self.size_bytes // 4)  # Assuming 4 bytes per entry
        }

class ProcessingTile:
    """Processing tile containing multiple crossbars"""
    def __init__(self, config: TileConfig, tile_id: int = 0):
        self.config = config
        self.tile_id = tile_id
        
        # Initialize crossbar configuration if not provided
        if config.crossbar_config is None:
            config.crossbar_config = CrossbarConfig()
            
        # Create crossbar arrays
        self.crossbars = [CrossbarArray(config.crossbar_config) 
                         for _ in range(config.crossbars_per_tile)]
        
        # Create local buffer
        self.local_buffer = LocalBuffer(config.local_buffer_size, config.local_buffer_type)
        
        # Create peripheral manager with edge-optimized configurations
        crossbar_size = (config.crossbar_config.rows, config.crossbar_config.cols)
        total_rows = crossbar_size[0] * config.crossbars_per_tile
        total_cols = crossbar_size[1] * config.crossbars_per_tile
        
        # Edge-optimized peripheral configurations
        edge_adc_config = ADCConfig(
            resolution=8,  # 8-bit for edge inference
            conversion_time=20e-9,  # 20ns
            power_consumption=250e-6,  # 250μW
            input_range=(0.0, 1.8)  # 1.8V supply
        )
        
        edge_dac_config = DACConfig(
            resolution=6,  # 6-bit for input quantization
            settling_time=5e-9,  # 5ns
            power_consumption=150e-6,  # 150μW
            output_range=(0.0, 1.8)  # 1.8V supply
        )
        
        edge_sense_amp_config = SenseAmplifierConfig(
            gain=1000.0,  # High gain for ReRAM currents
            power_consumption=50e-6,  # 50μW
            sensing_time=2e-9,  # 2ns
            current_range=(100e-9, 100e-6)  # 100nA to 100μA
        )
        
        edge_driver_config = DriverConfig(
            output_voltage_range=(0.0, 2.5),  # 2.5V for ReRAM
            power_consumption=300e-6,  # 300μW
            rise_time=2e-9,  # 2ns
            compliance_current=100e-6  # 100μA
        )
        
        if config.adc_sharing:
            # Shared ADCs among crossbars
            self.peripheral_manager = PeripheralManager(
                num_rows=total_rows,
                num_cols=config.adcs_per_tile,  # Reduced number of ADCs
                adc_config=edge_adc_config,
                dac_config=edge_dac_config,
                sense_amp_config=edge_sense_amp_config,
                driver_config=edge_driver_config
            )
        else:
            # Dedicated ADCs per crossbar
            self.peripheral_manager = PeripheralManager(
                num_rows=total_rows,
                num_cols=total_cols,
                adc_config=edge_adc_config,
                dac_config=edge_dac_config,
                sense_amp_config=edge_sense_amp_config,
                driver_config=edge_driver_config
            )
            
        # Statistics
        self.operation_count = 0
        self.total_energy = 0.0
        
    def execute_layer(self, input_data: np.ndarray, weight_matrices: List[np.ndarray]) -> np.ndarray:
        """Execute a neural network layer on this tile"""
        if len(weight_matrices) != len(self.crossbars):
            raise ValueError(f"Number of weight matrices ({len(weight_matrices)}) doesn't match crossbars ({len(self.crossbars)})")
            
        # Program weights into crossbars
        for i, weight_matrix in enumerate(weight_matrices):
            self.crossbars[i].program_weights(weight_matrix)
            
        # Perform matrix-vector multiplications
        partial_results = []
        for i, crossbar in enumerate(self.crossbars):
            result = crossbar.matrix_vector_multiply(input_data)
            partial_results.append(result)
            
        # Combine results (simplified - actual implementation would depend on layer type)
        final_result = np.concatenate(partial_results)
        
        self.operation_count += 1
        return final_result
        
    def get_statistics(self) -> Dict:
        """Get tile statistics"""
        crossbar_stats = [xbar.get_statistics() for xbar in self.crossbars]
        return {
            'tile_id': self.tile_id,
            'operation_count': self.operation_count,
            'total_energy': self.total_energy,
            'crossbar_statistics': crossbar_stats,
            'buffer_statistics': self.local_buffer.get_statistics(),
            'peripheral_statistics': self.peripheral_manager.get_statistics()
        }

class SuperTile:
    """Super tile containing multiple processing tiles"""
    def __init__(self, config: SuperTileConfig, supertile_id: int = 0):
        self.config = config
        self.supertile_id = supertile_id
        
        # Initialize tile configuration if not provided
        if config.tile_config is None:
            config.tile_config = TileConfig()
            
        # Create processing tiles
        self.tiles = [ProcessingTile(config.tile_config, i) 
                     for i in range(config.tiles_per_supertile)]
        
        # Create shared buffer
        self.shared_buffer = LocalBuffer(config.shared_buffer_size, config.shared_buffer_type)
        
        # Interconnect simulation (simplified)
        self.interconnect_energy_per_bit = 1e-12  # J/bit
        self.interconnect_latency = 1e-8  # seconds per hop
        
        # Statistics
        self.operation_count = 0
        self.total_energy = 0.0
        self.data_movement_count = 0
        
    def execute_distributed_layer(self, input_data: np.ndarray, 
                                  weight_matrices_per_tile: List[List[np.ndarray]]) -> np.ndarray:
        """Execute layer distributed across tiles"""
        if len(weight_matrices_per_tile) != len(self.tiles):
            raise ValueError("Weight matrices don't match number of tiles")
            
        # Execute on each tile
        tile_results = []
        for i, (tile, weight_matrices) in enumerate(zip(self.tiles, weight_matrices_per_tile)):
            result = tile.execute_layer(input_data, weight_matrices)
            tile_results.append(result)
            
        # Aggregate results
        final_result = np.concatenate(tile_results)
        
        # Account for data movement energy
        data_bits = input_data.size * 32  # Assuming 32-bit data
        self.total_energy += data_bits * self.interconnect_energy_per_bit * len(self.tiles)
        self.data_movement_count += len(self.tiles)
        
        self.operation_count += 1
        return final_result
        
    def get_statistics(self) -> Dict:
        """Get super tile statistics"""
        tile_stats = [tile.get_statistics() for tile in self.tiles]
        return {
            'supertile_id': self.supertile_id,
            'operation_count': self.operation_count,
            'total_energy': self.total_energy,
            'data_movement_count': self.data_movement_count,
            'tile_statistics': tile_stats,
            'shared_buffer_statistics': self.shared_buffer.get_statistics()
        }

class ReRAMChip:
    """Complete ReRAM chip containing multiple super tiles"""
    def __init__(self, config: ChipConfig):
        self.config = config
        
        # Initialize supertile configuration if not provided
        if config.supertile_config is None:
            config.supertile_config = SuperTileConfig()
            
        # Create super tiles
        self.supertiles = [SuperTile(config.supertile_config, i) 
                          for i in range(config.supertiles_per_chip)]
        
        # Create global memory/buffer
        self.global_buffer = LocalBuffer(config.global_buffer_size, config.global_buffer_type)
        
        # Memory controller simulation
        self.memory_controller_energy_per_bit = 10e-12  # J/bit
        self.memory_controller_latency = 100e-9  # seconds
        
        # Statistics
        self.total_operations = 0
        self.total_energy = 0.0
        self.global_data_movement = 0
        
    def get_chip_configuration(self) -> Dict:
        """Get comprehensive chip configuration"""
        # Calculate total resources
        total_crossbars = (self.config.supertiles_per_chip * 
                          self.config.supertile_config.tiles_per_supertile * 
                          self.config.supertile_config.tile_config.crossbars_per_tile)
        
        crossbar_config = self.config.supertile_config.tile_config.crossbar_config
        if crossbar_config is None:
            crossbar_config = CrossbarConfig()
            
        total_cells = total_crossbars * crossbar_config.rows * crossbar_config.cols
        total_weights = total_cells  # Each cell stores one weight
        
        return {
            'hierarchy': {
                'supertiles': self.config.supertiles_per_chip,
                'tiles_per_supertile': self.config.supertile_config.tiles_per_supertile,
                'crossbars_per_tile': self.config.supertile_config.tile_config.crossbars_per_tile,
                'total_crossbars': total_crossbars
            },
            'memory_hierarchy': {
                'global_buffer': f"{self.config.global_buffer_size} KB {self.config.global_buffer_type}",
                'shared_buffers': f"{self.config.supertile_config.shared_buffer_size} KB {self.config.supertile_config.shared_buffer_type}",
                'local_buffers': f"{self.config.supertile_config.tile_config.local_buffer_size} KB {self.config.supertile_config.tile_config.local_buffer_type}"
            },
            'compute_capacity': {
                'total_crossbars': total_crossbars,
                'crossbar_size': f"{crossbar_config.rows}x{crossbar_config.cols}",
                'total_ReRAM_cells': total_cells,
                'total_weight_capacity': total_weights
            },
            'interconnect': {
                'chip_level': self.config.on_chip_interconnect.value,
                'supertile_level': self.config.supertile_config.interconnect_type.value,
                'tile_level': self.config.supertile_config.tile_config.interconnect_type.value
            }
        }
        
    def get_total_statistics(self) -> Dict:
        """Get comprehensive chip statistics"""
        supertile_stats = [st.get_statistics() for st in self.supertiles]
        
        # Aggregate statistics
        total_crossbar_ops = 0
        total_tile_ops = 0
        total_compute_energy = 0.0
        
        for st_stat in supertile_stats:
            total_tile_ops += st_stat['operation_count']
            total_compute_energy += st_stat['total_energy']
            
            for tile_stat in st_stat['tile_statistics']:
                for xbar_stat in tile_stat['crossbar_statistics']:
                    total_crossbar_ops += xbar_stat['total_operations']
                    
        return {
            'chip_configuration': self.get_chip_configuration(),
            'performance': {
                'total_operations': self.total_operations,
                'total_crossbar_operations': total_crossbar_ops,
                'total_tile_operations': total_tile_ops
            },
            'energy': {
                'total_energy': self.total_energy,
                'compute_energy': total_compute_energy,
                'memory_energy': self.global_buffer.total_energy
            },
            'supertile_statistics': supertile_stats,
            'global_buffer_statistics': self.global_buffer.get_statistics()
        }
        
    def print_architecture_summary(self):
        """Print a summary of the chip architecture"""
        config = self.get_chip_configuration()
        
        print("=" * 60)
        print("ReRAM Crossbar Accelerator Architecture Summary")
        print("=" * 60)
        
        print(f"Hierarchy:")
        print(f"  └── Chip: {config['hierarchy']['supertiles']} SuperTiles")
        print(f"      └── SuperTile: {config['hierarchy']['tiles_per_supertile']} Tiles")
        print(f"          └── Tile: {config['hierarchy']['crossbars_per_tile']} Crossbars")
        print(f"              └── Crossbar: {config['compute_capacity']['crossbar_size']} ReRAM array")
        
        print(f"\nCompute Capacity:")
        print(f"  Total Crossbars: {config['compute_capacity']['total_crossbars']:,}")
        print(f"  Total ReRAM Cells: {config['compute_capacity']['total_ReRAM_cells']:,}")
        print(f"  Weight Capacity: {config['compute_capacity']['total_weight_capacity']:,}")
        
        print(f"\nMemory Hierarchy:")
        print(f"  Global Buffer: {config['memory_hierarchy']['global_buffer']}")
        print(f"  Shared Buffers: {config['memory_hierarchy']['shared_buffers']} per SuperTile")
        print(f"  Local Buffers: {config['memory_hierarchy']['local_buffers']} per Tile")
        
        print(f"\nInterconnect:")
        print(f"  Chip Level: {config['interconnect']['chip_level']}")
        print(f"  SuperTile Level: {config['interconnect']['supertile_level']}")
        print(f"  Tile Level: {config['interconnect']['tile_level']}")
        
        print("=" * 60)