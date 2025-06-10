import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from collections import defaultdict
import logging

class MetricType(Enum):
    LATENCY = "latency"
    POWER = "power"
    ENERGY = "energy"
    THROUGHPUT = "throughput"
    AREA = "area"
    UTILIZATION = "utilization"
    ACCURACY = "accuracy"

@dataclass
class LatencyMetrics:
    """Latency-related metrics"""
    compute_latency: float = 0.0  # seconds
    memory_access_latency: float = 0.0  # seconds
    interconnect_latency: float = 0.0  # seconds
    adc_conversion_latency: float = 0.0  # seconds
    dac_conversion_latency: float = 0.0  # seconds
    total_latency: float = 0.0  # seconds
    
    def __post_init__(self):
        self.total_latency = (self.compute_latency + self.memory_access_latency + 
                             self.interconnect_latency + self.adc_conversion_latency + 
                             self.dac_conversion_latency)

@dataclass
class PowerMetrics:
    """Power-related metrics"""
    crossbar_power: float = 0.0  # Watts
    peripheral_power: float = 0.0  # Watts (ADC, DAC, sense amps, drivers)
    memory_power: float = 0.0  # Watts (buffers, caches)
    interconnect_power: float = 0.0  # Watts
    static_power: float = 0.0  # Watts (leakage)
    dynamic_power: float = 0.0  # Watts (switching)
    total_power: float = 0.0  # Watts
    
    def __post_init__(self):
        self.dynamic_power = (self.crossbar_power + self.peripheral_power + 
                             self.memory_power + self.interconnect_power)
        self.total_power = self.dynamic_power + self.static_power

@dataclass
class EnergyMetrics:
    """Energy-related metrics"""
    compute_energy: float = 0.0  # Joules
    memory_energy: float = 0.0  # Joules
    data_movement_energy: float = 0.0  # Joules
    peripheral_energy: float = 0.0  # Joules
    total_energy: float = 0.0  # Joules
    energy_efficiency: float = 0.0  # TOPS/W (Tera operations per second per Watt)
    
    def __post_init__(self):
        self.total_energy = (self.compute_energy + self.memory_energy + 
                           self.data_movement_energy + self.peripheral_energy)

@dataclass
class ThroughputMetrics:
    """Throughput-related metrics"""
    operations_per_second: float = 0.0  # ops/s
    multiply_accumulates_per_second: float = 0.0  # MACs/s
    inferences_per_second: float = 0.0  # inferences/s
    bits_per_second: float = 0.0  # bps
    peak_throughput: float = 0.0  # theoretical maximum
    achieved_throughput: float = 0.0  # actual achieved
    efficiency: float = 0.0  # achieved/peak ratio
    
    def __post_init__(self):
        if self.peak_throughput > 0:
            self.efficiency = self.achieved_throughput / self.peak_throughput

@dataclass
class AreaMetrics:
    """Area-related metrics"""
    crossbar_area: float = 0.0  # mm^2
    peripheral_area: float = 0.0  # mm^2
    memory_area: float = 0.0  # mm^2
    interconnect_area: float = 0.0  # mm^2
    total_area: float = 0.0  # mm^2
    area_efficiency: float = 0.0  # TOPS/mm^2
    
    def __post_init__(self):
        self.total_area = (self.crossbar_area + self.peripheral_area + 
                          self.memory_area + self.interconnect_area)

@dataclass
class UtilizationMetrics:
    """Utilization-related metrics"""
    crossbar_utilization: float = 0.0  # 0.0 to 1.0
    memory_utilization: float = 0.0  # 0.0 to 1.0
    adc_utilization: float = 0.0  # 0.0 to 1.0
    dac_utilization: float = 0.0  # 0.0 to 1.0
    interconnect_utilization: float = 0.0  # 0.0 to 1.0
    overall_utilization: float = 0.0  # 0.0 to 1.0
    
    def __post_init__(self):
        utilizations = [self.crossbar_utilization, self.memory_utilization,
                       self.adc_utilization, self.dac_utilization, 
                       self.interconnect_utilization]
        self.overall_utilization = np.mean([u for u in utilizations if u > 0])

@dataclass
class AccuracyMetrics:
    """Accuracy-related metrics due to hardware non-idealities"""
    weight_precision_loss: float = 0.0  # Due to ReRAM quantization
    adc_quantization_error: float = 0.0  # Due to ADC resolution
    device_variation_error: float = 0.0  # Due to ReRAM device variation
    noise_error: float = 0.0  # Due to circuit noise
    retention_error: float = 0.0  # Due to data retention degradation
    endurance_error: float = 0.0  # Due to write/erase cycling
    total_error: float = 0.0  # Combined error
    snr_db: float = 0.0  # Signal-to-noise ratio in dB
    
    def __post_init__(self):
        # Simple error combination (actual would be more complex)
        errors = [self.weight_precision_loss, self.adc_quantization_error,
                 self.device_variation_error, self.noise_error,
                 self.retention_error, self.endurance_error]
        self.total_error = np.sqrt(sum(e**2 for e in errors))

class MetricsCollector:
    """Collects and manages all metrics"""
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.current_metrics = {}
        self.start_time = None
        self.end_time = None
        
    def start_measurement(self):
        """Start timing measurement"""
        self.start_time = time.time()
        
    def end_measurement(self):
        """End timing measurement"""
        self.end_time = time.time()
        return self.end_time - self.start_time if self.start_time else 0.0
        
    def record_latency(self, latency_metrics: LatencyMetrics, operation_name: str = "default"):
        """Record latency metrics"""
        self.current_metrics[f"{operation_name}_latency"] = latency_metrics
        self.metrics_history["latency"].append({
            "timestamp": time.time(),
            "operation": operation_name,
            "metrics": latency_metrics
        })
        
    def record_power(self, power_metrics: PowerMetrics, operation_name: str = "default"):
        """Record power metrics"""
        self.current_metrics[f"{operation_name}_power"] = power_metrics
        self.metrics_history["power"].append({
            "timestamp": time.time(),
            "operation": operation_name,
            "metrics": power_metrics
        })
        
    def record_energy(self, energy_metrics: EnergyMetrics, operation_name: str = "default"):
        """Record energy metrics"""
        self.current_metrics[f"{operation_name}_energy"] = energy_metrics
        self.metrics_history["energy"].append({
            "timestamp": time.time(),
            "operation": operation_name,
            "metrics": energy_metrics
        })
        
    def record_throughput(self, throughput_metrics: ThroughputMetrics, operation_name: str = "default"):
        """Record throughput metrics"""
        self.current_metrics[f"{operation_name}_throughput"] = throughput_metrics
        self.metrics_history["throughput"].append({
            "timestamp": time.time(),
            "operation": operation_name,
            "metrics": throughput_metrics
        })
        
    def record_area(self, area_metrics: AreaMetrics, component_name: str = "default"):
        """Record area metrics"""
        self.current_metrics[f"{component_name}_area"] = area_metrics
        self.metrics_history["area"].append({
            "timestamp": time.time(),
            "component": component_name,
            "metrics": area_metrics
        })
        
    def record_utilization(self, utilization_metrics: UtilizationMetrics, operation_name: str = "default"):
        """Record utilization metrics"""
        self.current_metrics[f"{operation_name}_utilization"] = utilization_metrics
        self.metrics_history["utilization"].append({
            "timestamp": time.time(),
            "operation": operation_name,
            "metrics": utilization_metrics
        })
        
    def record_accuracy(self, accuracy_metrics: AccuracyMetrics, operation_name: str = "default"):
        """Record accuracy metrics"""
        self.current_metrics[f"{operation_name}_accuracy"] = accuracy_metrics
        self.metrics_history["accuracy"].append({
            "timestamp": time.time(),
            "operation": operation_name,
            "metrics": accuracy_metrics
        })
        
    def get_summary_metrics(self) -> Dict:
        """Get summary of all collected metrics"""
        summary = {}
        
        # Aggregate latency metrics
        if "latency" in self.metrics_history:
            latencies = [entry["metrics"].total_latency for entry in self.metrics_history["latency"]]
            summary["latency"] = {
                "total_operations": len(latencies),
                "average_latency": np.mean(latencies),
                "min_latency": np.min(latencies),
                "max_latency": np.max(latencies),
                "std_latency": np.std(latencies)
            }
            
        # Aggregate power metrics
        if "power" in self.metrics_history:
            powers = [entry["metrics"].total_power for entry in self.metrics_history["power"]]
            summary["power"] = {
                "average_power": np.mean(powers),
                "peak_power": np.max(powers),
                "min_power": np.min(powers)
            }
            
        # Aggregate energy metrics
        if "energy" in self.metrics_history:
            energies = [entry["metrics"].total_energy for entry in self.metrics_history["energy"]]
            summary["energy"] = {
                "total_energy": np.sum(energies),
                "average_energy_per_op": np.mean(energies)
            }
            
        # Aggregate throughput metrics
        if "throughput" in self.metrics_history:
            throughputs = [entry["metrics"].achieved_throughput for entry in self.metrics_history["throughput"]]
            summary["throughput"] = {
                "average_throughput": np.mean(throughputs),
                "peak_throughput": np.max(throughputs),
                "min_throughput": np.min(throughputs)
            }
            
        # Aggregate utilization metrics
        if "utilization" in self.metrics_history:
            utilizations = [entry["metrics"].overall_utilization for entry in self.metrics_history["utilization"]]
            summary["utilization"] = {
                "average_utilization": np.mean(utilizations),
                "peak_utilization": np.max(utilizations),
                "min_utilization": np.min(utilizations)
            }
            
        return summary
        
    def export_metrics(self, filename: str, format: str = "json"):
        """Export metrics to file"""
        if format == "json":
            metrics_data = {
                "summary": self.get_summary_metrics(),
                "history": dict(self.metrics_history),
                "current": self.current_metrics
            }
            
            # Convert dataclass objects to dictionaries for JSON serialization
            def convert_dataclass(obj):
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                return obj
                
            # Custom JSON encoder for dataclasses
            class DataclassEncoder(json.JSONEncoder):
                def default(self, obj):
                    if hasattr(obj, '__dict__'):
                        return obj.__dict__
                    return super().default(obj)
            
            with open(filename, 'w') as f:
                json.dump(metrics_data, f, cls=DataclassEncoder, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def print_summary(self):
        """Print summary of all metrics"""
        summary = self.get_summary_metrics()
        
        print("=" * 60)
        print("Performance Metrics Summary")
        print("=" * 60)
        
        if "latency" in summary:
            print(f"Latency:")
            print(f"  Average: {summary['latency']['average_latency']*1e6:.2f} μs")
            print(f"  Min/Max: {summary['latency']['min_latency']*1e6:.2f} / {summary['latency']['max_latency']*1e6:.2f} μs")
            print(f"  Operations: {summary['latency']['total_operations']}")
            
        if "power" in summary:
            print(f"\nPower:")
            print(f"  Average: {summary['power']['average_power']*1e3:.2f} mW")
            print(f"  Peak: {summary['power']['peak_power']*1e3:.2f} mW")
            
        if "energy" in summary:
            print(f"\nEnergy:")
            print(f"  Total: {summary['energy']['total_energy']*1e6:.2f} μJ")
            print(f"  Per Operation: {summary['energy']['average_energy_per_op']*1e9:.2f} nJ")
            
        if "throughput" in summary:
            print(f"\nThroughput:")
            print(f"  Average: {summary['throughput']['average_throughput']/1e6:.2f} MOPS")
            print(f"  Peak: {summary['throughput']['peak_throughput']/1e6:.2f} MOPS")
            
        if "utilization" in summary:
            print(f"\nUtilization:")
            print(f"  Average: {summary['utilization']['average_utilization']*100:.1f}%")
            print(f"  Peak: {summary['utilization']['peak_utilization']*100:.1f}%")
            
        print("=" * 60)

class PerformanceProfiler:
    """Performance profiler for detailed analysis"""
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.profiling_data = defaultdict(list)
        
    def profile_crossbar_operation(self, crossbar, operation_type: str, input_data: np.ndarray):
        """Profile a crossbar operation"""
        start_time = time.time()
        
        # Simulate the operation (this would be replaced with actual crossbar execution)
        if operation_type == "matrix_vector_multiply":
            result = crossbar.matrix_vector_multiply(input_data)
        else:
            result = None
            
        end_time = time.time()
        operation_time = end_time - start_time
        
        # Calculate metrics
        latency_metrics = LatencyMetrics(compute_latency=operation_time)
        
        # Estimate power (simplified model)
        crossbar_power = len(input_data) * crossbar.cols * 1e-6  # Simplified power model
        power_metrics = PowerMetrics(crossbar_power=crossbar_power)
        
        # Calculate energy
        energy = crossbar_power * operation_time
        energy_metrics = EnergyMetrics(compute_energy=energy)
        
        # Calculate throughput
        ops_per_second = (len(input_data) * crossbar.cols) / operation_time
        throughput_metrics = ThroughputMetrics(
            operations_per_second=ops_per_second,
            achieved_throughput=ops_per_second
        )
        
        # Record metrics
        self.metrics_collector.record_latency(latency_metrics, operation_type)
        self.metrics_collector.record_power(power_metrics, operation_type)
        self.metrics_collector.record_energy(energy_metrics, operation_type)
        self.metrics_collector.record_throughput(throughput_metrics, operation_type)
        
        return result
        
    def profile_layer_execution(self, layer_name: str, execution_func, *args, **kwargs):
        """Profile execution of a neural network layer"""
        start_time = time.time()
        
        # Execute the layer
        result = execution_func(*args, **kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Record profiling data
        self.profiling_data[layer_name].append({
            "execution_time": execution_time,
            "timestamp": start_time,
            "args_size": sum(np.array(arg).size if isinstance(arg, np.ndarray) else 0 for arg in args)
        })
        
        return result
        
    def get_layer_profile_summary(self) -> Dict:
        """Get summary of layer profiling data"""
        summary = {}
        
        for layer_name, executions in self.profiling_data.items():
            execution_times = [e["execution_time"] for e in executions]
            summary[layer_name] = {
                "total_executions": len(executions),
                "total_time": sum(execution_times),
                "average_time": np.mean(execution_times),
                "min_time": np.min(execution_times),
                "max_time": np.max(execution_times),
                "std_time": np.std(execution_times)
            }
            
        return summary
        
    def print_profile_summary(self):
        """Print profiling summary"""
        summary = self.get_layer_profile_summary()
        
        print("=" * 60)
        print("Layer Execution Profile Summary")
        print("=" * 60)
        
        for layer_name, stats in summary.items():
            print(f"\n{layer_name}:")
            print(f"  Executions: {stats['total_executions']}")
            print(f"  Total Time: {stats['total_time']*1e3:.2f} ms")
            print(f"  Average Time: {stats['average_time']*1e6:.2f} μs")
            print(f"  Min/Max Time: {stats['min_time']*1e6:.2f} / {stats['max_time']*1e6:.2f} μs")
            
        print("=" * 60)