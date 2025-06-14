# Web-Based Visualization Design for xBARSimV1

## 📋 Overview

This document outlines the design for implementing two key web-based visualization tools:
1. **Real-time Hardware Component Monitoring Dashboard**
2. **Interactive CNN-to-xBAR Mapping Visualization for Educational Use**

## 🏗️ Architecture Design

### Technology Stack Recommendation

#### Frontend
- **Framework**: React.js with D3.js for custom visualizations
- **Real-time Updates**: WebSocket connections for live data streaming
- **3D Graphics**: Three.js for 3D crossbar array visualization
- **Charts**: Plotly.js for performance metrics and graphs
- **UI Components**: Material-UI or Ant Design for professional interface

#### Backend Integration
- **Web Framework**: Flask or FastAPI for Python backend
- **Real-time Communication**: WebSocket support (Flask-SocketIO or FastAPI WebSockets)
- **Data Serialization**: JSON for communication between simulator and web frontend
- **Optional Dashboard Framework**: Dash (Plotly) for rapid prototyping

### Integration Approaches

#### Option 1: Embedded Web Server (Recommended)
```python
# Integrated into existing simulator
simulator = ExecutionEngine(chip, dnn_manager)
web_server = WebVisualizationServer(simulator)
web_server.start_async()  # Non-blocking web server
result = simulator.execute_inference(input_data, enable_web_viz=True)
```

#### Option 2: Standalone Dashboard
```python
# Separate dashboard process
from web_dashboard import ReRAMDashboard
dashboard = ReRAMDashboard()
dashboard.connect_to_simulator(port=8080)
dashboard.run(debug=True)
```

## 🖥️ Visualization Component 1: Real-time Hardware Monitoring Dashboard

### Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ ReRAM Crossbar Simulator - Live Hardware Monitor              │
├─────────────────────────────────────────────────────────────────┤
│ [Chip Overview] [Layer Progress] [Performance Metrics]        │
├─────────────────┬───────────────────┬───────────────────────────┤
│  3D Chip View   │   Memory System   │    ADC/DAC Activity      │
│                 │                   │                          │
│ ┌─SuperTile─┐   │ ┌──Global Buffer─┐│ ┌──ADC Conversions────┐  │
│ │ ┌─Tile──┐ │   │ │ [████░░░░] 45%││ │ [████████████] 92%  │  │
│ │ │XB XB  │ │   │ └────────────────┘│ └─────────────────────┘  │
│ │ │XB XB  │ │   │                   │                          │
│ │ └───────┘ │   │ ┌──Shared Buf───┐ │ ┌──DAC Conversions────┐  │
│ └───────────┘   │ │ [██████░░] 78%││ │ [███████████░] 85%  │  │
│                 │ └────────────────┘│ └─────────────────────┘  │
├─────────────────┼───────────────────┼───────────────────────────┤
│  Crossbar Heat  │  Interconnect     │    Energy Consumption   │
│      Map        │    Traffic        │                          │
│ ┌─────────────┐ │ ┌───────────────┐ │ ┌─────────────────────┐  │
│ │█▓▓░░░░░░░░░│ │ │ Mesh Network  │ │ │ Total: 66.2 mJ     │  │
│ │█▓▓░░░░░░░░░│ │ │ Congestion:   │ │ │ ┌─Crossbars────────┐│  │
│ │█▓▓░░░░░░░░░│ │ │ [███░░░░] 32% │ │ │ │ [██████░░░░] 60%││  │
│ │░░░░░░░░░░░░│ │ │               │ │ │ └───────────────────┘│  │
│ │░░░░░░░░░░░░│ │ │               │ │ │ ┌─Memory───────────┐ │  │
│ └─────────────┘ │ └───────────────┘ │ │ │ [████░░░░░░] 40%││  │
│                 │                   │ │ └───────────────────┘│  │
└─────────────────┴───────────────────┴───────────────────────────┘
```

### Real-time Data Streaming

#### WebSocket Event Types
```javascript
// Frontend receives real-time updates
socket.on('crossbar_activity', (data) => {
    updateCrossbarHeatmap(data.crossbar_id, data.operations, data.utilization);
});

socket.on('memory_operation', (data) => {
    updateMemoryBuffer(data.buffer_type, data.utilization, data.latency);
});

socket.on('adc_dac_activity', (data) => {
    updatePeripheralActivity(data.adc_conversions, data.dac_conversions);
});

socket.on('layer_progress', (data) => {
    updateLayerProgress(data.layer_index, data.progress_percent);
});
```

#### Backend Data Collection
```python
class WebVisualizationCollector:
    def __init__(self, simulator):
        self.simulator = simulator
        self.websocket_manager = WebSocketManager()
        
    def collect_and_broadcast(self):
        # Collect real-time statistics
        chip_stats = self.simulator.chip.get_total_statistics()
        memory_stats = self.simulator.buffer_manager.get_all_statistics()
        
        # Broadcast to all connected clients
        self.websocket_manager.broadcast('crossbar_activity', {
            'crossbar_operations': chip_stats['crossbar_operations'],
            'utilization': chip_stats['utilization']
        })
        
        self.websocket_manager.broadcast('memory_operation', {
            'global_buffer': memory_stats['global_buffer'],
            'local_buffers': memory_stats['local_buffers']
        })
```

### Interactive Features

#### 3D Chip Visualization (Three.js)
```javascript
// 3D interactive chip representation
class ChipVisualization3D {
    constructor(containerId) {
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer();
        this.initializeChipGeometry();
    }
    
    updateCrossbarActivity(crossbar_id, utilization) {
        const crossbar = this.crossbars[crossbar_id];
        const color = this.getHeatmapColor(utilization);
        crossbar.material.color = new THREE.Color(color);
    }
    
    getHeatmapColor(utilization) {
        // Blue (idle) -> Green (active) -> Red (high utilization)
        if (utilization < 0.3) return 0x0000ff;
        if (utilization < 0.7) return 0x00ff00;
        return 0xff0000;
    }
}
```

#### Crossbar Heatmap (D3.js)
```javascript
// Interactive crossbar array heatmap
class CrossbarHeatmap {
    constructor(svg, rows=128, cols=128) {
        this.svg = d3.select(svg);
        this.rows = rows;
        this.cols = cols;
        this.createGrid();
    }
    
    updateCellActivity(row, col, activity_level) {
        this.svg.select(`#cell-${row}-${col}`)
            .transition()
            .duration(100)
            .style("fill", this.getActivityColor(activity_level));
    }
    
    getActivityColor(level) {
        const colorScale = d3.scaleSequential(d3.interpolateYlOrRd)
            .domain([0, 1]);
        return colorScale(level);
    }
}
```

## 🎓 Visualization Component 2: CNN-to-xBAR Mapping Educational Tool

### Educational Dashboard Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ CNN to CrossBar Mapping - Educational Visualization           │
├─────────────────────────────────────────────────────────────────┤
│ [Model Selection] [Layer Selection] [Step-by-Step Mode]       │
├─────────────────┬───────────────────┬───────────────────────────┤
│   CNN Model     │  Mapping Process  │     CrossBar Layout      │
│                 │                   │                          │
│ ┌─Layer 0──────┐│ Step 1: Weight    │ ┌─SuperTile 0───────────┐ │
│ │ Conv2D       ││ Reshaping         │ │ ┌─Tile 0─────────────┐│ │
│ │ 16x16x1 →    ││ [3,3,1,8] →       │ │ │ XB0 XB1           ││ │
│ │ 14x14x8      ││ [9, 8]            │ │ │ ▓▓▓ ░░░ ← Layer 0 ││ │
│ └──────────────┘│                   │ │ │ ▓▓▓ ░░░           ││ │
│                 │ Step 2: Crossbar  │ │ │ XB2 XB3           ││ │
│ ┌─Layer 1──────┐│ Allocation        │ │ │ ███ ███ ← Layer 1 ││ │
│ │ Pool2D       ││ Layer 0: XB0      │ │ │ ███ ███           ││ │
│ │ 14x14x8 →    ││ Layer 1: XB2,XB3  │ │ └─────────────────────┘│ │
│ │ 7x7x8        ││                   │ └───────────────────────┘ │
│ └──────────────┘│ Step 3: Data Flow │                          │
│                 │ Input → Crossbar  │ Legend:                  │
│ ┌─Layer 2──────┐│ → ADC → Compute   │ ▓ = Conv Layer Weights   │
│ │ Dense        ││                   │ █ = Dense Layer Weights  │
│ │ 392 → 10     ││                   │ ░ = Unused               │
│ └──────────────┘│                   │                          │
└─────────────────┴───────────────────┴───────────────────────────┘
```

### Step-by-Step Mapping Visualization

#### Interactive Mapping Process
```javascript
class CNNMappingVisualizer {
    constructor() {
        this.currentStep = 0;
        this.mappingSteps = [
            'weight_analysis',
            'crossbar_allocation', 
            'data_flow',
            'execution_simulation'
        ];
    }
    
    showMappingStep(step) {
        switch(step) {
            case 'weight_analysis':
                this.highlightWeightTensors();
                this.showWeightReshaping();
                break;
            case 'crossbar_allocation':
                this.animateCrossbarAssignment();
                this.showUtilizationMetrics();
                break;
            case 'data_flow':
                this.animateDataFlow();
                this.highlightCriticalPath();
                break;
            case 'execution_simulation':
                this.runStepByStepExecution();
                break;
        }
    }
    
    animateCrossbarAssignment() {
        // Animate weight matrix assignment to crossbars
        d3.selectAll('.weight-matrix')
            .transition()
            .duration(1000)
            .style('transform', 'translate(...)');
    }
}
```

### Educational Features

#### Weight Matrix Decomposition Visualization
```python
class WeightMappingVisualizer:
    def visualize_conv_mapping(self, layer_config, crossbar_assignment):
        """Generate visualization data for CNN layer mapping"""
        weights_shape = layer_config.weights_shape  # e.g., (3, 3, 1, 8)
        
        # Show original weight tensor
        original_viz = {
            'type': 'conv_weights',
            'shape': weights_shape,
            'filters': weights_shape[-1],
            'kernel_size': weights_shape[:2]
        }
        
        # Show reshaped matrix for crossbar
        reshaped_viz = {
            'type': 'matrix',
            'rows': weights_shape[0] * weights_shape[1] * weights_shape[2],  # 9
            'cols': weights_shape[3],  # 8
            'crossbar_id': crossbar_assignment['crossbar_id'],
            'utilization': crossbar_assignment['utilization']
        }
        
        return {
            'original': original_viz,
            'reshaped': reshaped_viz,
            'mapping_explanation': self.generate_explanation(layer_config)
        }
```

#### Interactive Architecture Explorer
```javascript
class ArchitectureExplorer {
    constructor() {
        this.hierarchyLevels = ['chip', 'supertile', 'tile', 'crossbar'];
        this.currentLevel = 'chip';
    }
    
    drillDown(component_id) {
        const nextLevel = this.getNextLevel();
        this.renderLevel(nextLevel, component_id);
        this.updateBreadcrumb();
    }
    
    showComponentDetails(component) {
        return {
            capacity: component.capacity,
            utilization: component.current_utilization,
            energy_consumption: component.energy_stats,
            connected_components: component.connections,
            current_operations: component.active_operations
        };
    }
}
```

## 🔧 Implementation Plan

### Phase 1: Backend Integration (Week 1-2)
1. **Create Web Visualization Module**
   ```python
   # src/visualization/web_viz.py
   class WebVisualizationServer:
       def __init__(self, execution_engine):
           self.engine = execution_engine
           self.app = Flask(__name__)
           self.socketio = SocketIO(self.app)
   ```

2. **Integrate Real-time Data Collection**
   ```python
   # Modified execution_engine.py
   def execute_inference(self, input_data, enable_web_viz=False):
       if enable_web_viz:
           self.web_collector = WebVisualizationCollector(self)
           self.web_collector.start_collection()
   ```

3. **Create Data APIs**
   ```python
   @app.route('/api/chip/status')
   def get_chip_status():
       return jsonify(self.engine.chip.get_total_statistics())
   
   @app.route('/api/mapping/<model_name>')
   def get_mapping_visualization(model_name):
       return jsonify(self.engine.dnn_manager.get_mapping_visualization())
   ```

### Phase 2: Frontend Development (Week 3-4)
1. **Setup React Application**
   ```bash
   npx create-react-app reram-visualizer
   cd reram-visualizer
   npm install d3 three plotly.js socket.io-client
   ```

2. **Create Component Architecture**
   ```
   src/
   ├── components/
   │   ├── RealTimeMonitor/
   │   │   ├── ChipOverview.jsx
   │   │   ├── CrossbarHeatmap.jsx
   │   │   ├── MemoryMonitor.jsx
   │   │   └── PeripheralActivity.jsx
   │   ├── EducationalTool/
   │   │   ├── CNNModelViewer.jsx
   │   │   ├── MappingVisualizer.jsx
   │   │   ├── StepByStepGuide.jsx
   │   │   └── ArchitectureExplorer.jsx
   │   └── Common/
   │       ├── WebSocketManager.js
   │       └── DataFormatters.js
   ```

3. **Implement Core Visualizations**
   - Real-time dashboard with WebSocket integration
   - Interactive 3D chip visualization
   - Educational mapping tool with step-by-step guides

### Phase 3: Educational Content (Week 5)
1. **Create Tutorial Modes**
   - Guided tours of hardware architecture
   - Step-by-step CNN mapping explanations
   - Interactive exercises for students

2. **Add Explanatory Content**
   - Tooltips explaining each component
   - Mathematical formulations for operations
   - Performance implications of design choices

### Phase 4: Testing & Polish (Week 6)
1. **Performance Optimization**
   - Efficient WebSocket data streaming
   - Smooth animations and transitions
   - Mobile-responsive design

2. **Educational Validation**
   - User testing with students/researchers
   - Documentation and tutorials
   - Example scenarios and use cases

## 📊 Expected Features

### Real-time Monitoring Dashboard
✅ Live crossbar activity heatmaps
✅ Memory system utilization tracking  
✅ ADC/DAC conversion monitoring
✅ Energy consumption analysis
✅ Interconnect traffic visualization
✅ Performance bottleneck identification

### Educational CNN Mapping Tool
✅ Interactive layer-by-layer mapping visualization
✅ Weight matrix decomposition animations
✅ Crossbar allocation optimization explanations
✅ Data flow path highlighting
✅ Comparative analysis of different mappings
✅ Hardware utilization impact demonstrations

### Student Learning Features
✅ Step-by-step guided tutorials
✅ Interactive quizzes on mapping concepts
✅ "What-if" scenario exploration
✅ Performance prediction exercises
✅ Architecture design challenges

## 🚀 Getting Started

### Quick Setup
```bash
# Install additional dependencies
pip install flask flask-socketio

# Start web visualization server
python -m src.visualization.web_viz --port 8080

# In another terminal, run simulation with web viz enabled
python main.py --model sample_cnn --execute --web-viz
```

### Development Mode
```bash
# Start backend development server
python -m src.visualization.web_viz --debug

# Start frontend development server  
cd web-frontend
npm start
```

This design provides a comprehensive foundation for both real-time hardware monitoring and educational CNN mapping visualization, making the ReRAM crossbar simulator accessible to researchers, engineers, and students alike.