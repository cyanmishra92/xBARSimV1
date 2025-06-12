import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

class OperationType(Enum):
    READ = "read"
    WRITE = "write"
    MVM = "matrix_vector_multiplication"

@dataclass
class CrossbarConfig:
    """Configuration for a single ReRAM crossbar - Edge Computing Optimized"""
    rows: int = 128
    cols: int = 128
    
    # ReRAM Device Parameters (based on recent research: Nature Electronics, IEEE JSSC)
    r_on: float = 1e3     # Low resistance state ~1kΩ (HfOx, TaOx ReRAM)
    r_off: float = 100e3  # High resistance state ~100kΩ (for higher on/off ratio)
    on_off_ratio: float = 100  # Typical for HfOx ReRAM
    
    # Operating Voltages (optimized for low power edge computing)
    v_read: float = 0.1       # Read voltage 100mV (low power)
    v_write: float = 2.0      # Write voltage 2V (reduced for endurance)
    v_set_threshold: float = 1.5   # SET threshold 1.5V
    v_reset_threshold: float = -1.2 # RESET threshold -1.2V
    
    # Edge Computing Specifications
    weight_bits: int = 4      # 4-bit weights for edge inference
    input_bits: int = 8       # 8-bit input quantization
    accumulation_bits: int = 16  # 16-bit accumulation precision
    
    # Device Reliability (edge deployment requirements)
    device_variability: float = 0.05  # 5% device variation (improved manufacturing)
    retention_time: float = 10 * 365 * 24 * 3600  # 10 years retention
    endurance: int = 1e9      # 1 billion cycles (improved for edge)
    
    # Timing Parameters (ns scale for edge performance)
    read_delay_ns: float = 10   # 10ns read access time
    write_delay_ns: float = 100 # 100ns write time
    set_delay_ns: float = 50    # 50ns SET operation
    reset_delay_ns: float = 30  # 30ns RESET operation
    
class ReRAMCell:
    """Individual ReRAM cell model"""
    def __init__(self, config: CrossbarConfig, row: int, col: int):
        self.config = config
        self.row = row
        self.col = col
        self.resistance = config.r_off  # Initialize to high resistance
        self.state = 0  # 0 = HRS (High Resistance State), 1 = LRS (Low Resistance State)
        self.write_count = 0
        self.last_access_time = 0.0
        
        # Add device variability
        self.r_on_actual = config.r_on * (1 + np.random.normal(0, config.device_variability))
        self.r_off_actual = config.r_off * (1 + np.random.normal(0, config.device_variability))
        
    def write(self, voltage: float, current_time: float = 0.0) -> bool:
        """Write operation on ReRAM cell"""
        if self.write_count >= self.config.endurance:
            return False  # Cell has reached endurance limit
            
        if voltage >= self.config.v_set_threshold:
            # SET operation: switch to LRS
            self.state = 1
            self.resistance = self.r_on_actual
        elif voltage <= self.config.v_reset_threshold:
            # RESET operation: switch to HRS
            self.state = 0
            self.resistance = self.r_off_actual
            
        self.write_count += 1
        self.last_access_time = current_time
        return True
        
    def read(self, voltage: float) -> float:
        """Read operation - returns current"""
        # Apply retention degradation if needed
        current = voltage / self.resistance
        return current
        
    def get_conductance(self) -> float:
        """Get conductance (1/resistance)"""
        return 1.0 / self.resistance

class CrossbarArray:
    """ReRAM Crossbar Array for matrix-vector multiplication"""
    def __init__(self, config: CrossbarConfig):
        self.config = config
        self.rows = config.rows
        self.cols = config.cols
        
        # Initialize ReRAM cells
        self.cells = [[ReRAMCell(config, i, j) for j in range(self.cols)] 
                      for i in range(self.rows)]
        
        # Peripheral circuits will be added later
        self.row_drivers = None
        self.col_sense_amps = None
        self.adc_units = None
        self.dac_units = None
        
        # Statistics
        self.total_operations = 0
        self.total_energy = 0.0
        self.operation_history = []
        
    def program_weights(self, weight_matrix: np.ndarray) -> bool:
        """Program weight matrix into crossbar"""
        if weight_matrix.shape != (self.rows, self.cols):
            raise ValueError(f"Weight matrix shape {weight_matrix.shape} doesn't match crossbar size {(self.rows, self.cols)}")
            
        # Normalize weights to conductance values
        # Map weights to conductance range
        min_weight, max_weight = weight_matrix.min(), weight_matrix.max()
        min_conductance = 1.0 / self.config.r_off
        max_conductance = 1.0 / self.config.r_on
        
        success = True
        for i in range(self.rows):
            for j in range(self.cols):
                # Map weight to conductance
                normalized_weight = (weight_matrix[i, j] - min_weight) / (max_weight - min_weight)
                target_conductance = min_conductance + normalized_weight * (max_conductance - min_conductance)
                target_resistance = 1.0 / target_conductance
                
                # Program cell
                if target_resistance < (self.config.r_on + self.config.r_off) / 2:
                    # SET operation
                    voltage = self.config.v_set_threshold
                else:
                    # RESET operation  
                    voltage = self.config.v_reset_threshold
                    
                if not self.cells[i][j].write(voltage):
                    success = False
                    
        return success
        
    def matrix_vector_multiply(self, input_vector: np.ndarray) -> np.ndarray:
        """Perform analog matrix-vector multiplication"""
        if len(input_vector) != self.rows:
            raise ValueError(f"Input vector size {len(input_vector)} doesn't match crossbar rows {self.rows}")
            
        # Convert input to voltages (assuming DAC conversion)
        input_voltages = input_vector * self.config.v_read
        
        # Perform analog MVM using Kirchhoff's current law
        output_currents = np.zeros(self.cols)
        
        for j in range(self.cols):
            column_current = 0.0
            for i in range(self.rows):
                # Current through each cell: I = V * G (where G = 1/R)
                cell_current = input_voltages[i] * self.cells[i][j].get_conductance()
                column_current += cell_current
            output_currents[j] = column_current
            
        # Update statistics
        self.total_operations += 1
        self.operation_history.append({
            'operation': OperationType.MVM,
            'input_size': len(input_vector),
            'output_size': len(output_currents)
        })
        
        return output_currents
        
    def get_resistance_matrix(self) -> np.ndarray:
        """Get current resistance matrix"""
        resistance_matrix = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                resistance_matrix[i, j] = self.cells[i][j].resistance
        return resistance_matrix
        
    def get_conductance_matrix(self) -> np.ndarray:
        """Get current conductance matrix"""
        conductance_matrix = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                conductance_matrix[i, j] = self.cells[i][j].get_conductance()
        return conductance_matrix
        
    def get_statistics(self, detailed_endurance: bool = True) -> Dict:
        """Get crossbar statistics"""
        return {
            'total_operations': self.total_operations,
            'total_energy': self.total_energy,
            'operation_history': self.operation_history,
            'endurance_status': self._get_endurance_status(detailed=detailed_endurance)
        }
        
    def _get_endurance_status(self, detailed: bool = True) -> Dict:
        """Check endurance status of all cells"""
        if not detailed:
            return {
                "avg_write_count": "N/A",
                "max_write_count": "N/A",
                "failed_cells": "N/A",
                "failure_rate": "N/A",
                "details_skipped": True
            }

        write_counts = []
        failed_cells = 0
        
        for i in range(self.rows):
            for j in range(self.cols):
                write_counts.append(self.cells[i][j].write_count)
                if self.cells[i][j].write_count >= self.config.endurance:
                    failed_cells += 1

        avg_wc = np.mean(write_counts) if write_counts else 0
        max_wc = np.max(write_counts) if write_counts else 0
        fail_rate = failed_cells / (self.rows * self.cols) if (self.rows * self.cols) > 0 else 0
                    
        return {
            'avg_write_count': avg_wc,
            'max_write_count': max_wc,
            'failed_cells': failed_cells,
            'failure_rate': fail_rate,
            "details_skipped": False
        }