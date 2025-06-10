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
    """ADC configuration parameters"""
    type: ADCType = ADCType.SAR
    resolution: int = 8  # bits
    sampling_rate: float = 1e6  # Hz
    input_range: Tuple[float, float] = (0.0, 3.3)  # V
    power_consumption: float = 1e-3  # W
    area: float = 0.1  # mm^2
    conversion_time: float = 1e-6  # seconds
    offset_voltage: float = 1e-3  # V
    gain_error: float = 0.01  # percentage
    integral_nonlinearity: float = 0.5  # LSB
    differential_nonlinearity: float = 0.5  # LSB

@dataclass
class DACConfig:
    """DAC configuration parameters"""
    type: DACType = DACType.CURRENT_STEERING
    resolution: int = 8  # bits
    settling_time: float = 1e-7  # seconds
    output_range: Tuple[float, float] = (0.0, 3.3)  # V
    power_consumption: float = 5e-4  # W
    area: float = 0.05  # mm^2
    offset_voltage: float = 1e-3  # V
    gain_error: float = 0.01  # percentage
    integral_nonlinearity: float = 0.5  # LSB
    differential_nonlinearity: float = 0.5  # LSB

@dataclass
class SenseAmplifierConfig:
    """Sense amplifier configuration"""
    gain: float = 100.0  # V/V
    bandwidth: float = 1e6  # Hz
    input_offset: float = 1e-3  # V
    power_consumption: float = 1e-4  # W
    area: float = 0.01  # mm^2
    noise_voltage: float = 1e-6  # V/sqrt(Hz)
    common_mode_rejection: float = 80.0  # dB

@dataclass
class DriverConfig:
    """Driver circuit configuration"""
    output_voltage_range: Tuple[float, float] = (0.0, 5.0)  # V
    output_current_capability: float = 10e-3  # A
    rise_time: float = 1e-8  # seconds
    fall_time: float = 1e-8  # seconds
    power_consumption: float = 2e-3  # W
    area: float = 0.02  # mm^2

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