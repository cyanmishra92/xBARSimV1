#!/usr/bin/env python3
"""
Web-based visualization server for ReRAM Crossbar Simulator
Provides real-time monitoring and educational visualization capabilities
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import asdict
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import numpy as np

# Import simulator components
from ..core.execution_engine import ExecutionEngine
from ..core.hierarchy import ReRAMChip
from ..core.dnn_manager import DNNManager


class WebVisualizationServer:
    """
    Web-based visualization server for real-time monitoring and educational tools
    """
    
    def __init__(self, execution_engine: Optional[ExecutionEngine] = None, port: int = 8080):
        self.execution_engine = execution_engine
        self.port = port
        self.app = Flask(__name__, template_folder='web_templates', static_folder='web_static')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Data collection state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.last_stats = {}
        self.monitoring_interval = 0.5  # 500ms updates
        
        # Educational mode state
        self.educational_mode = False
        self.current_mapping_step = 0
        self.mapping_steps = []
        
        self._setup_routes()
        self._setup_websocket_handlers()
    
    def _setup_routes(self):
        """Setup Flask routes for the web interface"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/educational')
        def educational():
            """Educational CNN mapping visualization page"""
            return render_template('educational.html')
        
        @self.app.route('/api/chip/status')
        def get_chip_status():
            """Get current chip status and statistics"""
            if not self.execution_engine:
                return jsonify({'error': 'No execution engine connected'})
            
            try:
                stats = self.execution_engine.chip.get_total_statistics()
                return jsonify(self._serialize_stats(stats))
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/mapping/<model_name>')
        def get_mapping_visualization(model_name):
            """Get CNN-to-crossbar mapping visualization data"""
            if not self.execution_engine:
                return jsonify({'error': 'No execution engine connected'})
            
            try:
                mapping_data = self._generate_mapping_visualization(model_name)
                return jsonify(mapping_data)
            except Exception as e:
                return jsonify({'error': str(e)})
        
        @self.app.route('/api/architecture')
        def get_architecture():
            """Get chip architecture hierarchy"""
            if not self.execution_engine:
                return jsonify({'error': 'No execution engine connected'})
            
            return jsonify(self._get_architecture_hierarchy())
        
        @self.app.route('/api/educational/steps')
        def get_educational_steps():
            """Get step-by-step mapping tutorial data"""
            return jsonify(self._get_educational_steps())
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers for real-time communication"""
        
        @self.socketio.on('connect')
        def handle_connect():
            print(f"Client connected: {request.sid}")
            emit('connected', {'status': 'Connected to ReRAM Simulator'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('start_monitoring')
        def handle_start_monitoring():
            """Start real-time monitoring"""
            self.start_monitoring()
            emit('monitoring_started', {'status': 'Real-time monitoring started'})
        
        @self.socketio.on('stop_monitoring')
        def handle_stop_monitoring():
            """Stop real-time monitoring"""
            self.stop_monitoring()
            emit('monitoring_stopped', {'status': 'Real-time monitoring stopped'})
        
        @self.socketio.on('request_mapping_step')
        def handle_mapping_step(data):
            """Handle educational mapping step request"""
            step = data.get('step', 0)
            mapping_data = self._get_mapping_step_data(step)
            emit('mapping_step_data', mapping_data)
    
    def start_monitoring(self):
        """Start real-time monitoring in a separate thread"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
    
    def _monitoring_loop(self):
        """Main monitoring loop for real-time data collection"""
        while self.monitoring_active:
            try:
                if self.execution_engine:
                    # Collect current statistics
                    current_stats = self._collect_real_time_stats()
                    
                    # Check for changes and broadcast updates
                    if self._stats_changed(current_stats):
                        self.socketio.emit('stats_update', current_stats)
                        self.last_stats = current_stats
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def _collect_real_time_stats(self) -> Dict[str, Any]:
        """Collect real-time statistics from the simulator"""
        if not self.execution_engine:
            return {}
        
        try:
            # Chip-level statistics
            chip_stats = self.execution_engine.chip.get_total_statistics()
            
            # Memory system statistics
            memory_stats = {}
            if hasattr(self.execution_engine, 'buffer_manager'):
                memory_stats = self.execution_engine.buffer_manager.get_all_statistics()
            
            # Individual crossbar statistics
            crossbar_stats = self._collect_crossbar_stats()
            
            # Peripheral statistics
            peripheral_stats = self._collect_peripheral_stats()
            
            # Current execution state
            execution_state = self._get_execution_state()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'chip': self._serialize_stats(chip_stats),
                'memory': self._serialize_stats(memory_stats),
                'crossbars': crossbar_stats,
                'peripherals': peripheral_stats,
                'execution': execution_state
            }
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _collect_crossbar_stats(self) -> List[Dict[str, Any]]:
        """Collect statistics from individual crossbars"""
        crossbar_stats = []
        
        try:
            for st_idx, supertile in enumerate(self.execution_engine.chip.supertiles):
                for t_idx, tile in enumerate(supertile.tiles):
                    for xb_idx, crossbar in enumerate(tile.crossbars):
                        stats = crossbar.get_statistics()
                        crossbar_stats.append({
                            'id': f"ST{st_idx}_T{t_idx}_XB{xb_idx}",
                            'supertile': st_idx,
                            'tile': t_idx,
                            'crossbar': xb_idx,
                            'operations': stats.get('total_operations', 0),
                            'energy': stats.get('total_energy', 0.0),
                            'utilization': self._calculate_crossbar_utilization(crossbar),
                            'active': stats.get('total_operations', 0) > 0
                        })
        except Exception as e:
            print(f"Error collecting crossbar stats: {e}")
        
        return crossbar_stats
    
    def _collect_peripheral_stats(self) -> Dict[str, Any]:
        """Collect ADC/DAC peripheral statistics"""
        peripheral_stats = {
            'adc_conversions': 0,
            'dac_conversions': 0,
            'adc_utilization': 0.0,
            'dac_utilization': 0.0
        }
        
        try:
            total_adc_ops = 0
            total_dac_ops = 0
            total_adc_units = 0
            total_dac_units = 0
            
            for supertile in self.execution_engine.chip.supertiles:
                for tile in supertile.tiles:
                    if hasattr(tile, 'peripheral_manager') and tile.peripheral_manager:
                        pm = tile.peripheral_manager
                        
                        # Count ADC operations and units
                        if hasattr(pm, 'output_adcs'):
                            total_adc_units += len(pm.output_adcs)
                            total_adc_ops += sum(adc.conversion_count for adc in pm.output_adcs)
                        
                        # Count DAC operations and units  
                        if hasattr(pm, 'input_dacs'):
                            total_dac_units += len(pm.input_dacs)
                            total_dac_ops += sum(dac.conversion_count for dac in pm.input_dacs)
            
            peripheral_stats.update({
                'adc_conversions': total_adc_ops,
                'dac_conversions': total_dac_ops,
                'adc_utilization': min(1.0, total_adc_ops / max(total_adc_units, 1)),
                'dac_utilization': min(1.0, total_dac_ops / max(total_dac_units, 1))
            })
        except Exception as e:
            print(f"Error collecting peripheral stats: {e}")
        
        return peripheral_stats
    
    def _get_execution_state(self) -> Dict[str, Any]:
        """Get current execution state"""
        state = {
            'running': False,
            'current_layer': None,
            'total_layers': 0,
            'progress': 0.0,
            'execution_cycles': 0
        }
        
        try:
            if hasattr(self.execution_engine, 'current_layer_index'):
                state['current_layer'] = self.execution_engine.current_layer_index
            
            if hasattr(self.execution_engine, 'dnn_manager') and self.execution_engine.dnn_manager:
                state['total_layers'] = len(self.execution_engine.dnn_manager.dnn_config.layers)
                if state['current_layer'] is not None:
                    state['progress'] = state['current_layer'] / max(state['total_layers'], 1)
            
            if hasattr(self.execution_engine, 'total_execution_cycles'):
                state['execution_cycles'] = self.execution_engine.total_execution_cycles
        except Exception as e:
            print(f"Error getting execution state: {e}")
        
        return state
    
    def _calculate_crossbar_utilization(self, crossbar) -> float:
        """Calculate crossbar utilization based on operations"""
        try:
            stats = crossbar.get_statistics()
            operations = stats.get('total_operations', 0)
            # Simple utilization metric: operations / max_possible_operations
            return min(1.0, operations / 1000.0)  # Normalize to max 1000 operations
        except:
            return 0.0
    
    def _stats_changed(self, current_stats: Dict[str, Any]) -> bool:
        """Check if statistics have changed significantly"""
        if not self.last_stats:
            return True
        
        # Simple change detection - could be made more sophisticated
        try:
            current_ops = sum(cb.get('operations', 0) for cb in current_stats.get('crossbars', []))
            last_ops = sum(cb.get('operations', 0) for cb in self.last_stats.get('crossbars', []))
            return current_ops != last_ops
        except:
            return True
    
    def _generate_mapping_visualization(self, model_name: str) -> Dict[str, Any]:
        """Generate CNN-to-crossbar mapping visualization data"""
        if not self.execution_engine or not self.execution_engine.dnn_manager:
            return {'error': 'No DNN manager available'}
        
        dnn_manager = self.execution_engine.dnn_manager
        
        # Get layer mapping information
        layer_mappings = []
        for i, layer in enumerate(dnn_manager.dnn_config.layers):
            if hasattr(layer, 'weights_shape') and layer.weights_shape:
                mapping_info = self._analyze_layer_mapping(layer, i)
                layer_mappings.append(mapping_info)
        
        # Get crossbar allocation
        crossbar_allocation = self._get_crossbar_allocation()
        
        # Get architecture overview
        architecture = self._get_architecture_hierarchy()
        
        return {
            'model_name': model_name,
            'layer_mappings': layer_mappings,
            'crossbar_allocation': crossbar_allocation,
            'architecture': architecture,
            'mapping_strategy': 'weight_stationary',  # Default strategy
            'utilization_summary': self._calculate_utilization_summary()
        }
    
    def _analyze_layer_mapping(self, layer, layer_index: int) -> Dict[str, Any]:
        """Analyze how a specific layer maps to crossbars"""
        return {
            'layer_index': layer_index,
            'layer_type': layer.layer_type.value if hasattr(layer.layer_type, 'value') else str(layer.layer_type),
            'input_shape': list(layer.input_shape),
            'output_shape': list(layer.output_shape),
            'weights_shape': list(layer.weights_shape) if layer.weights_shape else None,
            'weight_matrix_size': self._calculate_weight_matrix_size(layer),
            'crossbars_needed': self._estimate_crossbars_needed(layer),
            'mapping_pattern': self._get_mapping_pattern(layer)
        }
    
    def _calculate_weight_matrix_size(self, layer) -> Dict[str, int]:
        """Calculate the 2D matrix size for crossbar mapping"""
        if not layer.weights_shape:
            return {'rows': 0, 'cols': 0}
        
        if str(layer.layer_type).lower() == 'conv2d':
            # Conv2D: reshape [H, W, C_in, C_out] -> [H*W*C_in, C_out]
            weights_shape = layer.weights_shape
            rows = weights_shape[0] * weights_shape[1] * weights_shape[2]
            cols = weights_shape[3]
        elif str(layer.layer_type).lower() == 'dense':
            # Dense: [input_size, output_size]
            rows, cols = layer.weights_shape
        else:
            return {'rows': 0, 'cols': 0}
        
        return {'rows': rows, 'cols': cols}
    
    def _estimate_crossbars_needed(self, layer) -> int:
        """Estimate number of crossbars needed for this layer"""
        matrix_size = self._calculate_weight_matrix_size(layer)
        if matrix_size['rows'] == 0 or matrix_size['cols'] == 0:
            return 0
        
        # Assume 128x128 crossbars
        crossbar_size = 128 * 128
        total_weights = matrix_size['rows'] * matrix_size['cols']
        return max(1, (total_weights + crossbar_size - 1) // crossbar_size)
    
    def _get_mapping_pattern(self, layer) -> str:
        """Get the mapping pattern description for this layer"""
        crossbars_needed = self._estimate_crossbars_needed(layer)
        if crossbars_needed == 1:
            return "1x1"
        elif crossbars_needed <= 4:
            return f"{crossbars_needed}x1"
        else:
            # Try to make it roughly square
            rows = int(np.sqrt(crossbars_needed))
            cols = (crossbars_needed + rows - 1) // rows
            return f"{rows}x{cols}"
    
    def _get_crossbar_allocation(self) -> List[Dict[str, Any]]:
        """Get current crossbar allocation status"""
        allocation = []
        
        try:
            for st_idx, supertile in enumerate(self.execution_engine.chip.supertiles):
                for t_idx, tile in enumerate(supertile.tiles):
                    for xb_idx, crossbar in enumerate(tile.crossbars):
                        allocation.append({
                            'id': f"ST{st_idx}_T{t_idx}_XB{xb_idx}",
                            'supertile': st_idx,
                            'tile': t_idx,
                            'crossbar': xb_idx,
                            'allocated': True,  # Simplified - could track actual allocation
                            'layer_assignment': None,  # Could track which layer uses this crossbar
                            'utilization': self._calculate_crossbar_utilization(crossbar)
                        })
        except Exception as e:
            print(f"Error getting crossbar allocation: {e}")
        
        return allocation
    
    def _get_architecture_hierarchy(self) -> Dict[str, Any]:
        """Get chip architecture hierarchy for visualization"""
        if not self.execution_engine:
            return {}
        
        chip = self.execution_engine.chip
        
        return {
            'chip': {
                'supertiles': len(chip.supertiles),
                'total_tiles': sum(len(st.tiles) for st in chip.supertiles),
                'total_crossbars': sum(sum(len(tile.crossbars) for tile in st.tiles) for st in chip.supertiles),
                'total_cells': sum(sum(sum(xb.rows * xb.cols for xb in tile.crossbars) for tile in st.tiles) for st in chip.supertiles)
            },
            'supertiles': [
                {
                    'id': i,
                    'tiles': len(st.tiles),
                    'shared_buffer_size': getattr(st.config, 'shared_buffer_size', 512) if hasattr(st, 'config') else 512
                } for i, st in enumerate(chip.supertiles)
            ],
            'memory_hierarchy': {
                'global_buffer': getattr(chip.config, 'global_buffer_size', 4096) if hasattr(chip, 'config') else 4096,
                'shared_buffer_size': 512,  # Default from config
                'local_buffer_size': 64    # Default from config
            }
        }
    
    def _calculate_utilization_summary(self) -> Dict[str, float]:
        """Calculate overall utilization summary"""
        try:
            # Get crossbar utilization
            crossbar_stats = self._collect_crossbar_stats()
            crossbar_util = np.mean([cb['utilization'] for cb in crossbar_stats]) if crossbar_stats else 0.0
            
            # Get memory utilization (simplified)
            memory_util = 0.1  # Placeholder
            
            # Get peripheral utilization
            peripheral_stats = self._collect_peripheral_stats()
            peripheral_util = (peripheral_stats['adc_utilization'] + peripheral_stats['dac_utilization']) / 2
            
            return {
                'crossbar_utilization': float(crossbar_util),
                'memory_utilization': float(memory_util),
                'peripheral_utilization': float(peripheral_util),
                'overall_utilization': float((crossbar_util + memory_util + peripheral_util) / 3)
            }
        except Exception as e:
            print(f"Error calculating utilization: {e}")
            return {'crossbar_utilization': 0.0, 'memory_utilization': 0.0, 'peripheral_utilization': 0.0, 'overall_utilization': 0.0}
    
    def _get_educational_steps(self) -> List[Dict[str, Any]]:
        """Get step-by-step educational content"""
        return [
            {
                'step': 0,
                'title': 'CNN Architecture Analysis',
                'description': 'Understand the CNN layer structure and weight organization',
                'focus': 'layer_analysis'
            },
            {
                'step': 1,
                'title': 'Weight Matrix Reshaping',
                'description': 'See how CNN weights are reshaped for crossbar arrays',
                'focus': 'weight_reshaping'
            },
            {
                'step': 2,
                'title': 'Crossbar Allocation',
                'description': 'Learn how layers are mapped to available crossbars',
                'focus': 'crossbar_mapping'
            },
            {
                'step': 3,
                'title': 'Data Flow Visualization',
                'description': 'Follow data as it flows through the ReRAM hierarchy',
                'focus': 'data_flow'
            },
            {
                'step': 4,
                'title': 'Execution Simulation',
                'description': 'Watch the actual execution with real-time monitoring',
                'focus': 'execution'
            }
        ]
    
    def _get_mapping_step_data(self, step: int) -> Dict[str, Any]:
        """Get data for a specific educational mapping step"""
        steps_data = self._get_educational_steps()
        if step >= len(steps_data):
            return {'error': 'Invalid step'}
        
        step_info = steps_data[step]
        
        # Generate step-specific data based on focus
        focus = step_info['focus']
        if focus == 'layer_analysis':
            return self._get_layer_analysis_data(step_info)
        elif focus == 'weight_reshaping':
            return self._get_weight_reshaping_data(step_info)
        elif focus == 'crossbar_mapping':
            return self._get_crossbar_mapping_data(step_info)
        elif focus == 'data_flow':
            return self._get_data_flow_data(step_info)
        elif focus == 'execution':
            return self._get_execution_data(step_info)
        
        return step_info
    
    def _get_layer_analysis_data(self, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get layer analysis data for educational visualization"""
        if not self.execution_engine or not self.execution_engine.dnn_manager:
            return step_info
        
        layers_info = []
        for i, layer in enumerate(self.execution_engine.dnn_manager.dnn_config.layers):
            layers_info.append({
                'index': i,
                'type': str(layer.layer_type),
                'input_shape': list(layer.input_shape),
                'output_shape': list(layer.output_shape),
                'parameters': np.prod(layer.weights_shape) if layer.weights_shape else 0,
                'weights_shape': list(layer.weights_shape) if layer.weights_shape else None
            })
        
        step_info['layers'] = layers_info
        return step_info
    
    def _get_weight_reshaping_data(self, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get weight reshaping visualization data"""
        # Implementation for weight reshaping step
        return step_info
    
    def _get_crossbar_mapping_data(self, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get crossbar mapping visualization data"""
        # Implementation for crossbar mapping step
        return step_info
    
    def _get_data_flow_data(self, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get data flow visualization data"""
        # Implementation for data flow step
        return step_info
    
    def _get_execution_data(self, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get execution visualization data"""
        # Implementation for execution step
        return step_info
    
    def _serialize_stats(self, stats: Any) -> Dict[str, Any]:
        """Serialize statistics for JSON transmission"""
        if isinstance(stats, dict):
            return {k: self._serialize_stats(v) for k, v in stats.items()}
        elif isinstance(stats, (list, tuple)):
            return [self._serialize_stats(item) for item in stats]
        elif isinstance(stats, np.ndarray):
            return stats.tolist()
        elif isinstance(stats, (np.integer, np.floating)):
            return float(stats)
        elif hasattr(stats, '__dict__'):
            return asdict(stats) if hasattr(stats, '__dataclass_fields__') else vars(stats)
        else:
            return stats
    
    def connect_execution_engine(self, execution_engine: ExecutionEngine):
        """Connect an execution engine to the web visualization server"""
        self.execution_engine = execution_engine
        print("Execution engine connected to web visualization server")
    
    def run(self, debug: bool = False, host: str = '0.0.0.0'):
        """Run the web visualization server"""
        print(f"Starting ReRAM Web Visualization Server on http://{host}:{self.port}")
        print("Dashboard: http://localhost:8080/")
        print("Educational Tool: http://localhost:8080/educational")
        
        self.socketio.run(
            self.app,
            host=host,
            port=self.port,
            debug=debug,
            allow_unsafe_werkzeug=True
        )


def create_web_visualization_server(execution_engine: Optional[ExecutionEngine] = None, 
                                   port: int = 8080) -> WebVisualizationServer:
    """Factory function to create a web visualization server"""
    return WebVisualizationServer(execution_engine, port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ReRAM Web Visualization Server")
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    
    args = parser.parse_args()
    
    # Create and run standalone server
    server = WebVisualizationServer(port=args.port)
    server.run(debug=args.debug, host=args.host)