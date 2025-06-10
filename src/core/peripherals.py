import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math

class ADCType(Enum):
    SAR = "successive_approximation"
    FLASH = "flash"
    VCO = "voltage_controlled_oscillator"
    SIGMA_DELTA = "sigma_delta"

class DACType(Enum):
    RESISTOR_STRING = "resistor_string"
    BINARY_WEIGHTED = "binary_weighted"
    CURRENT_STEERING = "current_steering"

@dataclass
class ADCConfig:
    """ADC configuration parameters - Edge Computing Optimized"""
    # Based on: IEEE JSSC 2023, Nature Electronics 2024, ISSCC 2024
    type: ADCType = ADCType.SAR  # SAR ADC optimal for edge applications
    resolution: int = 8  # 8-bit quantization for edge inference
    sampling_rate: float = 50e6  # 50 MHz (edge optimized speed)
    input_range: Tuple[float, float] = (0.0, 1.8)  # 1.8V supply for low power
    power_consumption: float = 250e-6  # 250 µW (ultra-low power)
    area: float = 0.025  # 0.025 mm² (compact design)
    conversion_time: float = 20e-9  # 20ns conversion time
    offset_voltage: float = 2e-3  # 2mV offset (improved calibration)
    gain_error: float = 0.5  # 0.5% gain error
    integral_nonlinearity: float = 0.4  # 0.4 LSB INL
    differential_nonlinearity: float = 0.3  # 0.3 LSB DNL
    
    # Edge-specific parameters
    enable_adaptive_resolution: bool = True  # Dynamic resolution scaling
    min_resolution: int = 4  # Minimum 4-bit for extreme low power
    calibration_cycles: int = 1000  # Calibration overhead

@dataclass
class DACConfig:
    """DAC configuration parameters - Edge Computing Optimized"""
    # Based on: IEEE TCAS-I 2024, VLSI 2024, ISSCC 2024
    type: DACType = DACType.CURRENT_STEERING  # Fastest settling for ReRAM
    resolution: int = 6  # 6-bit for input quantization (4-bit weight + headroom)
    settling_time: float = 5e-9  # 5ns settling time (high speed)
    output_range: Tuple[float, float] = (0.0, 1.8)  # 1.8V supply
    power_consumption: float = 150e-6  # 150 µW (low power)
    area: float = 0.015  # 0.015 mm² (compact)
    offset_voltage: float = 1.5e-3  # 1.5mV offset
    gain_error: float = 0.4  # 0.4% gain error
    integral_nonlinearity: float = 0.3  # 0.3 LSB INL
    differential_nonlinearity: float = 0.25  # 0.25 LSB DNL
    
    # Edge-specific parameters
    enable_segmentation: bool = True  # Segmented architecture
    thermometer_bits: int = 3  # 3 MSB thermometer coded
    binary_bits: int = 3  # 3 LSB binary weighted
    glitch_energy: float = 10e-15  # 10 fJ glitch energy

@dataclass
class SenseAmplifierConfig:
    """Sense amplifier configuration - Edge Optimized"""
    # Based on: IEEE JSSC 2024, TCAS-I 2024 (ReRAM sensing)
    gain: float = 1000.0  # 1000 V/V (high gain for small ReRAM currents)
    bandwidth: float = 100e6  # 100 MHz (fast sensing)
    input_offset: float = 500e-6  # 500 µV (precision sensing)
    power_consumption: float = 50e-6  # 50 µW (ultra-low power)
    area: float = 0.008  # 0.008 mm² (compact design)
    noise_voltage: float = 5e-7  # 0.5 µV/√Hz (low noise)
    common_mode_rejection: float = 100.0  # 100 dB (excellent CMRR)
    
    # ReRAM-specific parameters
    current_range: Tuple[float, float] = (100e-9, 100e-6)  # 100nA to 100µA
    sensing_time: float = 2e-9  # 2ns sensing time
    auto_zero_enabled: bool = True  # Auto-zero for offset cancellation
    chopper_frequency: float = 1e6  # 1 MHz chopper for 1/f noise

@dataclass
class DriverConfig:
    """Driver circuit configuration - Edge Optimized"""
    # Based on: ISSCC 2024, VLSI 2024 (ReRAM drivers)
    output_voltage_range: Tuple[float, float] = (0.0, 2.5)  # 2.5V for ReRAM SET/RESET
    output_current_capability: float = 1e-3  # 1mA (sufficient for ReRAM)
    rise_time: float = 2e-9  # 2ns rise time (fast switching)
    fall_time: float = 1.5e-9  # 1.5ns fall time
    power_consumption: float = 300e-6  # 300 µW (low power)
    area: float = 0.012  # 0.012 mm² (compact)
    
    # ReRAM-specific parameters
    compliance_current: float = 100e-6  # 100µA compliance current
    voltage_accuracy: float = 50e-3  # 50mV accuracy (±2%)
    slew_rate: float = 1e9  # 1 V/µs slew rate
    enable_current_limiting: bool = True  # Protect ReRAM devices

class ADC:
    """Analog-to-Digital Converter model"""
    def __init__(self, config: ADCConfig, instance_id: int = 0):
        self.config = config
        self.instance_id = instance_id
        self.conversion_count = 0
        self.total_energy = 0.0
        
        # Calculate quantization step
        self.v_min, self.v_max = config.input_range
        self.quantization_step = (self.v_max - self.v_min) / (2**config.resolution)
        
        # Non-idealities
        self.offset = config.offset_voltage
        self.gain_error = config.gain_error
        self.inl_error = config.integral_nonlinearity * self.quantization_step
        self.dnl_error = config.differential_nonlinearity * self.quantization_step
        
    def convert(self, analog_input: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
        """Convert analog input to digital output"""
        if isinstance(analog_input, np.ndarray):
            return np.array([self._convert_single(val) for val in analog_input])
        else:
            return self._convert_single(analog_input)
            
    def _convert_single(self, voltage: float) -> int:
        """Convert single analog value to digital"""
        # Apply offset and gain errors
        corrected_voltage = (voltage + self.offset) * (1 + self.gain_error)
        
        # Add nonlinearity errors (simplified model)
        corrected_voltage += np.random.normal(0, self.inl_error)
        
        # Clip to input range
        corrected_voltage = np.clip(corrected_voltage, self.v_min, self.v_max)
        
        # Quantize
        digital_code = int((corrected_voltage - self.v_min) / self.quantization_step)
        digital_code = np.clip(digital_code, 0, 2**self.config.resolution - 1)
        
        # Update statistics
        self.conversion_count += 1
        self.total_energy += self.config.power_consumption * self.config.conversion_time
        
        return digital_code
        
    def get_statistics(self) -> Dict:
        """Get ADC statistics"""
        return {
            'conversion_count': self.conversion_count,
            'total_energy': self.total_energy,
            'average_power': self.total_energy / max(self.conversion_count * self.config.conversion_time, 1e-9)
        }

class DAC:
    """Digital-to-Analog Converter model"""
    def __init__(self, config: DACConfig, instance_id: int = 0):
        self.config = config
        self.instance_id = instance_id
        self.conversion_count = 0
        self.total_energy = 0.0
        
        # Calculate step size
        self.v_min, self.v_max = config.output_range
        self.step_size = (self.v_max - self.v_min) / (2**config.resolution)
        
        # Non-idealities
        self.offset = config.offset_voltage
        self.gain_error = config.gain_error
        
    def convert(self, digital_input: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert digital input to analog output"""
        if isinstance(digital_input, np.ndarray):
            return np.array([self._convert_single(val) for val in digital_input])
        else:
            return self._convert_single(digital_input)
            
    def _convert_single(self, digital_code: int) -> float:
        """Convert single digital value to analog"""
        # Clip digital code
        digital_code = np.clip(digital_code, 0, 2**self.config.resolution - 1)
        
        # Convert to analog
        analog_voltage = self.v_min + digital_code * self.step_size
        
        # Apply offset and gain errors
        analog_voltage = (analog_voltage + self.offset) * (1 + self.gain_error)
        
        # Update statistics
        self.conversion_count += 1
        self.total_energy += self.config.power_consumption * self.config.settling_time
        
        return analog_voltage
        
    def get_statistics(self) -> Dict:
        """Get DAC statistics"""
        return {
            'conversion_count': self.conversion_count,
            'total_energy': self.total_energy,
            'average_power': self.total_energy / max(self.conversion_count * self.config.settling_time, 1e-9)
        }

class SenseAmplifier:
    """Sense amplifier for current sensing"""
    def __init__(self, config: SenseAmplifierConfig, instance_id: int = 0):
        self.config = config
        self.instance_id = instance_id
        self.operation_count = 0
        self.total_energy = 0.0
        
    def amplify(self, input_current: Union[float, np.ndarray], 
                load_resistance: float = 1e3) -> Union[float, np.ndarray]:
        """Amplify input current to voltage"""
        if isinstance(input_current, np.ndarray):
            return np.array([self._amplify_single(current, load_resistance) 
                           for current in input_current])
        else:
            return self._amplify_single(input_current, load_resistance)
            
    def _amplify_single(self, current: float, load_resistance: float) -> float:
        """Amplify single current value"""
        # Convert current to voltage (I*R)
        input_voltage = current * load_resistance
        
        # Apply amplification
        output_voltage = input_voltage * self.config.gain
        
        # Add offset and noise
        output_voltage += self.config.input_offset
        noise = np.random.normal(0, self.config.noise_voltage)
        output_voltage += noise
        
        # Update statistics
        self.operation_count += 1
        operation_time = 1.0 / self.config.bandwidth  # Simplified
        self.total_energy += self.config.power_consumption * operation_time
        
        return output_voltage
        
    def get_statistics(self) -> Dict:
        """Get sense amplifier statistics"""
        return {
            'operation_count': self.operation_count,
            'total_energy': self.total_energy,
            'average_power': self.config.power_consumption
        }

class Driver:
    """Driver circuit for applying voltages to crossbar"""
    def __init__(self, config: DriverConfig, instance_id: int = 0):
        self.config = config
        self.instance_id = instance_id
        self.drive_count = 0
        self.total_energy = 0.0
        
    def drive(self, target_voltages: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Drive target voltages"""
        if isinstance(target_voltages, np.ndarray):
            return np.array([self._drive_single(voltage) for voltage in target_voltages])
        else:
            return self._drive_single(target_voltages)
            
    def _drive_single(self, target_voltage: float) -> float:
        """Drive single voltage"""
        # Clip to output range
        v_min, v_max = self.config.output_voltage_range
        actual_voltage = np.clip(target_voltage, v_min, v_max)
        
        # Simulate rise/fall time effects (simplified)
        settling_time = max(self.config.rise_time, self.config.fall_time)
        
        # Update statistics
        self.drive_count += 1
        self.total_energy += self.config.power_consumption * settling_time
        
        return actual_voltage
        
    def get_statistics(self) -> Dict:
        """Get driver statistics"""
        return {
            'drive_count': self.drive_count,
            'total_energy': self.total_energy,
            'average_power': self.config.power_consumption
        }

class PeripheralManager:
    """Manager for all peripheral circuits"""
    def __init__(self, num_rows: int, num_cols: int,
                 adc_config: ADCConfig = None,
                 dac_config: DACConfig = None,
                 sense_amp_config: SenseAmplifierConfig = None,
                 driver_config: DriverConfig = None):
        
        # Use default configurations if not provided
        self.adc_config = adc_config or ADCConfig()
        self.dac_config = dac_config or DACConfig()
        self.sense_amp_config = sense_amp_config or SenseAmplifierConfig()
        self.driver_config = driver_config or DriverConfig()
        
        # Create peripheral circuits
        self.row_drivers = [Driver(self.driver_config, i) for i in range(num_rows)]
        self.input_dacs = [DAC(self.dac_config, i) for i in range(num_rows)]
        self.col_sense_amps = [SenseAmplifier(self.sense_amp_config, i) for i in range(num_cols)]
        self.output_adcs = [ADC(self.adc_config, i) for i in range(num_cols)]
        
    def get_total_area(self) -> float:
        """Calculate total peripheral area"""
        total_area = 0.0
        total_area += len(self.row_drivers) * self.driver_config.area
        total_area += len(self.input_dacs) * self.dac_config.area
        total_area += len(self.col_sense_amps) * self.sense_amp_config.area
        total_area += len(self.output_adcs) * self.adc_config.area
        return total_area
        
    def get_total_power(self) -> float:
        """Calculate total peripheral power"""
        total_power = 0.0
        total_power += len(self.row_drivers) * self.driver_config.power_consumption
        total_power += len(self.input_dacs) * self.dac_config.power_consumption
        total_power += len(self.col_sense_amps) * self.sense_amp_config.power_consumption
        total_power += len(self.output_adcs) * self.adc_config.power_consumption
        return total_power
        
    def get_statistics(self) -> Dict:
        """Get comprehensive peripheral statistics"""
        return {
            'total_area': self.get_total_area(),
            'total_power': self.get_total_power(),
            'component_counts': {
                'drivers': len(self.row_drivers),
                'dacs': len(self.input_dacs),
                'sense_amps': len(self.col_sense_amps),
                'adcs': len(self.output_adcs)
            },
            'individual_stats': {
                'drivers': [driver.get_statistics() for driver in self.row_drivers],
                'dacs': [dac.get_statistics() for dac in self.input_dacs],
                'sense_amps': [sa.get_statistics() for sa in self.col_sense_amps],
                'adcs': [adc.get_statistics() for adc in self.output_adcs]
            }
        }