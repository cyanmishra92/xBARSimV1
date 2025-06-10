import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import seaborn as sns
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

try:
    from ..core.hierarchy import ReRAMChip, SuperTile, ProcessingTile
    from ..core.dnn_manager import DNNManager
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.hierarchy import ReRAMChip, SuperTile, ProcessingTile
    from core.dnn_manager import DNNManager

class ArchitectureVisualizer:
    """Visualizes ReRAM crossbar architecture"""
    def __init__(self, chip: ReRAMChip):
        self.chip = chip
        self.fig = None
        self.axes = None
        
    def draw_crossbar(self, ax, position: Tuple[float, float], size: Tuple[float, float], 
                      crossbar_data: Optional[np.ndarray] = None, title: str = ""):
        """Draw a single crossbar"""
        x, y = position
        width, height = size
        
        # Draw crossbar outline
        rect = Rectangle((x, y), width, height, linewidth=1, 
                        edgecolor='black', facecolor='lightgray', alpha=0.7)
        ax.add_patch(rect)
        
        if crossbar_data is not None:
            # Visualize weight/conductance data as heatmap
            im = ax.imshow(crossbar_data, extent=[x, x+width, y, y+height],
                          aspect='auto', cmap='viridis', alpha=0.8)
        
        # Add title
        if title:
            ax.text(x + width/2, y + height + 0.05, title, 
                   ha='center', va='bottom', fontsize=8)
        
        # Add grid to show individual cells (for small crossbars)
        if crossbar_data is not None and crossbar_data.shape[0] <= 32:
            rows, cols = crossbar_data.shape
            for i in range(rows + 1):
                ax.plot([x, x + width], [y + i*height/rows, y + i*height/rows], 
                       'k-', linewidth=0.1, alpha=0.3)
            for j in range(cols + 1):
                ax.plot([x + j*width/cols, x + j*width/cols], [y, y + height], 
                       'k-', linewidth=0.1, alpha=0.3)
        
        return rect
    
    def draw_tile(self, ax, position: Tuple[float, float], tile: ProcessingTile, 
                  tile_size: Tuple[float, float] = (2.0, 2.0)):
        """Draw a processing tile"""
        x, y = position
        tile_width, tile_height = tile_size
        
        # Draw tile outline
        tile_rect = FancyBboxPatch((x, y), tile_width, tile_height,
                                  boxstyle="round,pad=0.05", linewidth=2,
                                  edgecolor='blue', facecolor='lightblue', alpha=0.3)
        ax.add_patch(tile_rect)
        
        # Calculate crossbar positions within tile
        num_crossbars = len(tile.crossbars)
        crossbars_per_row = int(np.ceil(np.sqrt(num_crossbars)))
        crossbar_width = tile_width * 0.8 / crossbars_per_row
        crossbar_height = tile_height * 0.6 / crossbars_per_row
        
        # Draw crossbars
        for i, crossbar in enumerate(tile.crossbars):
            row = i // crossbars_per_row
            col = i % crossbars_per_row
            
            xbar_x = x + 0.1 * tile_width + col * crossbar_width
            xbar_y = y + 0.2 * tile_height + row * crossbar_height
            
            # Get conductance matrix for visualization
            conductance_data = crossbar.get_conductance_matrix()
            
            self.draw_crossbar(ax, (xbar_x, xbar_y), 
                             (crossbar_width * 0.9, crossbar_height * 0.9),
                             conductance_data, f"XBar{i}")
        
        # Draw local buffer
        buffer_x = x + 0.1 * tile_width
        buffer_y = y + 0.05 * tile_height
        buffer_rect = Rectangle((buffer_x, buffer_y), tile_width * 0.8, tile_height * 0.1,
                               linewidth=1, edgecolor='green', facecolor='lightgreen', alpha=0.5)
        ax.add_patch(buffer_rect)
        ax.text(buffer_x + tile_width * 0.4, buffer_y + tile_height * 0.05, 
               f"Local Buffer\n{tile.config.local_buffer_size}KB", 
               ha='center', va='center', fontsize=6)
        
        # Add tile label
        ax.text(x + tile_width/2, y + tile_height + 0.1, f"Tile {tile.tile_id}", 
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        return tile_rect
    
    def draw_supertile(self, ax, position: Tuple[float, float], supertile: SuperTile,
                      supertile_size: Tuple[float, float] = (5.0, 5.0)):
        """Draw a super tile"""
        x, y = position
        st_width, st_height = supertile_size
        
        # Draw supertile outline
        st_rect = FancyBboxPatch((x, y), st_width, st_height,
                                boxstyle="round,pad=0.1", linewidth=3,
                                edgecolor='red', facecolor='mistyrose', alpha=0.2)
        ax.add_patch(st_rect)
        
        # Calculate tile positions within supertile
        num_tiles = len(supertile.tiles)
        tiles_per_row = int(np.ceil(np.sqrt(num_tiles)))
        tile_width = st_width * 0.8 / tiles_per_row
        tile_height = st_height * 0.7 / tiles_per_row
        
        # Draw tiles
        for i, tile in enumerate(supertile.tiles):
            row = i // tiles_per_row
            col = i % tiles_per_row
            
            tile_x = x + 0.1 * st_width + col * tile_width
            tile_y = y + 0.2 * st_height + row * tile_height
            
            self.draw_tile(ax, (tile_x, tile_y), tile, 
                          (tile_width * 0.9, tile_height * 0.9))
        
        # Draw shared buffer
        shared_buffer_x = x + 0.1 * st_width
        shared_buffer_y = y + 0.05 * st_height
        shared_buffer_rect = Rectangle((shared_buffer_x, shared_buffer_y), 
                                     st_width * 0.8, st_height * 0.1,
                                     linewidth=1, edgecolor='purple', 
                                     facecolor='plum', alpha=0.5)
        ax.add_patch(shared_buffer_rect)
        ax.text(shared_buffer_x + st_width * 0.4, shared_buffer_y + st_height * 0.05, 
               f"Shared Buffer\n{supertile.config.shared_buffer_size}KB", 
               ha='center', va='center', fontsize=8)
        
        # Add supertile label
        ax.text(x + st_width/2, y + st_height + 0.2, f"SuperTile {supertile.supertile_id}", 
               ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        return st_rect
    
    def draw_chip_architecture(self, figsize: Tuple[int, int] = (16, 12)):
        """Draw the complete chip architecture"""
        self.fig, self.axes = plt.subplots(1, 1, figsize=figsize)
        ax = self.axes
        
        # Calculate chip dimensions
        num_supertiles = len(self.chip.supertiles)
        supertiles_per_row = int(np.ceil(np.sqrt(num_supertiles)))
        chip_width = supertiles_per_row * 6.0
        chip_height = supertiles_per_row * 6.0
        
        # Draw chip outline
        chip_rect = Rectangle((0, 0), chip_width, chip_height,
                             linewidth=4, edgecolor='black', 
                             facecolor='whitesmoke', alpha=0.3)
        ax.add_patch(chip_rect)
        
        # Draw supertiles
        for i, supertile in enumerate(self.chip.supertiles):
            row = i // supertiles_per_row
            col = i % supertiles_per_row
            
            st_x = 0.5 + col * 6.0
            st_y = chip_height - 5.5 - row * 6.0
            
            self.draw_supertile(ax, (st_x, st_y), supertile)
        
        # Draw global buffer
        global_buffer_x = 0.2
        global_buffer_y = 0.1
        global_buffer_rect = Rectangle((global_buffer_x, global_buffer_y), 
                                     chip_width * 0.6, 0.3,
                                     linewidth=2, edgecolor='orange', 
                                     facecolor='moccasin', alpha=0.7)
        ax.add_patch(global_buffer_rect)
        ax.text(global_buffer_x + chip_width * 0.3, global_buffer_y + 0.15, 
               f"Global Buffer\n{self.chip.config.global_buffer_size}KB", 
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Set axis properties
        ax.set_xlim(-0.5, chip_width + 0.5)
        ax.set_ylim(-0.5, chip_height + 0.5)
        ax.set_aspect('equal')
        ax.set_title(f"ReRAM Crossbar Chip Architecture\n"
                    f"({len(self.chip.supertiles)} SuperTiles, "
                    f"{sum(len(st.tiles) for st in self.chip.supertiles)} Tiles, "
                    f"{sum(len(tile.crossbars) for st in self.chip.supertiles for tile in st.tiles)} Crossbars)",
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            patches.Patch(color='mistyrose', label='SuperTile'),
            patches.Patch(color='lightblue', label='Processing Tile'),
            patches.Patch(color='lightgray', label='ReRAM Crossbar'),
            patches.Patch(color='lightgreen', label='Local Buffer'),
            patches.Patch(color='plum', label='Shared Buffer'),
            patches.Patch(color='moccasin', label='Global Buffer')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        return self.fig, self.axes
    
    def save_architecture_diagram(self, filename: str, dpi: int = 300):
        """Save architecture diagram to file"""
        if self.fig is None:
            self.draw_chip_architecture()
        self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Architecture diagram saved to {filename}")

class DataflowVisualizer:
    """Visualizes data flow through the ReRAM crossbar architecture"""
    def __init__(self, chip: ReRAMChip, dnn_manager: DNNManager):
        self.chip = chip
        self.dnn_manager = dnn_manager
        self.fig = None
        self.axes = None
        
    def visualize_weight_mapping(self, layer_name: str, figsize: Tuple[int, int] = (12, 8)):
        """Visualize how weights are mapped to crossbars"""
        if layer_name not in self.dnn_manager.layer_mappings:
            raise ValueError(f"Layer {layer_name} not found in mappings")
            
        mapping = self.dnn_manager.layer_mappings[layer_name]
        allocation_map = mapping['crossbar_allocation']['allocation_map']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Crossbar allocation overview
        ax1 = axes[0, 0]
        crossbars_used = []
        supertiles_used = []
        tiles_used = []
        
        for alloc in allocation_map:
            crossbars_used.append(alloc['crossbar_id'])
            supertiles_used.append(alloc['supertile_id'])
            tiles_used.append(alloc['tile_id'])
            
        ax1.scatter(supertiles_used, tiles_used, c=crossbars_used, 
                   s=100, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('SuperTile ID')
        ax1.set_ylabel('Tile ID')
        ax1.set_title(f'Crossbar Allocation for {layer_name}')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Weight distribution heatmap
        ax2 = axes[0, 1]
        crossbars_per_row = mapping['crossbar_allocation']['crossbars_per_row']
        crossbars_per_col = mapping['crossbar_allocation']['crossbars_per_col']
        
        weight_distribution = np.zeros((crossbars_per_col, crossbars_per_row))
        for i, alloc in enumerate(allocation_map):
            row_idx = i // crossbars_per_row
            col_idx = i % crossbars_per_row
            if row_idx < crossbars_per_col and col_idx < crossbars_per_row:
                weight_distribution[row_idx, col_idx] = len(allocation_map) - i
                
        im = ax2.imshow(weight_distribution, cmap='plasma', aspect='auto')
        ax2.set_title('Weight Block Distribution')
        ax2.set_xlabel('Crossbar Column')
        ax2.set_ylabel('Crossbar Row')
        plt.colorbar(im, ax=ax2, label='Allocation Order')
        
        # Plot 3: Utilization by component
        ax3 = axes[1, 0]
        component_counts = {'SuperTiles': len(set(supertiles_used)),
                           'Tiles': len(set(zip(supertiles_used, tiles_used))),
                           'Crossbars': len(crossbars_used)}
        
        total_components = {'SuperTiles': len(self.chip.supertiles),
                           'Tiles': sum(len(st.tiles) for st in self.chip.supertiles),
                           'Crossbars': sum(len(tile.crossbars) for st in self.chip.supertiles for tile in st.tiles)}
        
        utilization = [component_counts[k] / total_components[k] * 100 for k in component_counts.keys()]
        
        bars = ax3.bar(component_counts.keys(), utilization, color=['red', 'blue', 'green'], alpha=0.7)
        ax3.set_ylabel('Utilization (%)')
        ax3.set_title('Component Utilization')
        ax3.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, util in zip(bars, utilization):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{util:.1f}%', ha='center', va='bottom')
        
        # Plot 4: Memory hierarchy usage
        ax4 = axes[1, 1]
        # This would show memory usage across the hierarchy
        memory_levels = ['Global', 'Shared', 'Local']
        memory_usage = [30, 60, 80]  # Placeholder values
        
        ax4.barh(memory_levels, memory_usage, color=['orange', 'purple', 'green'], alpha=0.7)
        ax4.set_xlabel('Memory Usage (%)')
        ax4.set_title('Memory Hierarchy Usage')
        ax4.set_xlim(0, 100)
        
        plt.suptitle(f'Weight Mapping Analysis: {layer_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, axes
    
    def animate_dataflow(self, input_data: np.ndarray, layer_sequence: List[str],
                        figsize: Tuple[int, int] = (14, 10), interval: int = 1000):
        """Create animated visualization of data flow"""
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Initialize architecture visualization
        arch_viz = ArchitectureVisualizer(self.chip)
        arch_viz.fig = fig
        arch_viz.axes = ax
        
        # Draw static architecture
        arch_viz.draw_chip_architecture()
        
        # Initialize data flow indicators
        flow_indicators = []
        
        def animate_frame(frame):
            # Clear previous flow indicators
            for indicator in flow_indicators:
                indicator.remove()
            flow_indicators.clear()
            
            if frame < len(layer_sequence):
                layer_name = layer_sequence[frame]
                
                # Highlight active components for current layer
                if layer_name in self.dnn_manager.layer_mappings:
                    mapping = self.dnn_manager.layer_mappings[layer_name]
                    allocation_map = mapping['crossbar_allocation']['allocation_map']
                    
                    # Add flowing data indicators
                    for alloc in allocation_map:
                        st_id = alloc['supertile_id']
                        tile_id = alloc['tile_id']
                        
                        # Calculate approximate position
                        st_x = 0.5 + (st_id % 2) * 6.0 + 2.5
                        st_y = 10 - 5.5 - (st_id // 2) * 6.0 + 2.5
                        
                        # Add pulsing circle to indicate active processing
                        circle = plt.Circle((st_x, st_y), 0.3, 
                                          color='yellow', alpha=0.8, 
                                          linewidth=3, fill=False)
                        ax.add_patch(circle)
                        flow_indicators.append(circle)
                
                # Update title with current layer
                ax.set_title(f"Data Flow Animation - Current Layer: {layer_name}\n"
                           f"Frame {frame + 1}/{len(layer_sequence)}", 
                           fontsize=14, fontweight='bold')
            
        # Create animation
        anim = animation.FuncAnimation(fig, animate_frame, frames=len(layer_sequence),
                                     interval=interval, repeat=True, blit=False)
        
        return anim
    
    def create_performance_heatmap(self, metrics_data: Dict, figsize: Tuple[int, int] = (12, 8)):
        """Create heatmap showing performance metrics across the chip"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Extract component-wise metrics
        num_supertiles = len(self.chip.supertiles)
        num_tiles_per_st = len(self.chip.supertiles[0].tiles)
        
        # Create synthetic performance data (in real implementation, this would come from actual metrics)
        latency_data = np.random.uniform(0.5, 2.0, (num_supertiles, num_tiles_per_st))
        power_data = np.random.uniform(10, 50, (num_supertiles, num_tiles_per_st))
        utilization_data = np.random.uniform(0.3, 0.9, (num_supertiles, num_tiles_per_st))
        throughput_data = np.random.uniform(100, 500, (num_supertiles, num_tiles_per_st))
        
        # Plot heatmaps
        heatmaps = [
            (axes[0, 0], latency_data, 'Latency (ms)', 'Reds'),
            (axes[0, 1], power_data, 'Power (mW)', 'Oranges'),
            (axes[1, 0], utilization_data, 'Utilization', 'Greens'),
            (axes[1, 1], throughput_data, 'Throughput (MOPS)', 'Blues')
        ]
        
        for ax, data, title, cmap in heatmaps:
            im = ax.imshow(data, cmap=cmap, aspect='auto')
            ax.set_title(title)
            ax.set_xlabel('Tile ID')
            ax.set_ylabel('SuperTile ID')
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            # Add value annotations
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, f'{data[i, j]:.1f}', 
                           ha='center', va='center', fontsize=8)
        
        plt.suptitle('Performance Metrics Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, axes

def create_comprehensive_visualization(chip: ReRAMChip, dnn_manager: DNNManager, 
                                     metrics_data: Optional[Dict] = None,
                                     save_path: Optional[str] = None):
    """Create comprehensive visualization suite"""
    
    # 1. Architecture diagram
    arch_viz = ArchitectureVisualizer(chip)
    arch_fig, arch_ax = arch_viz.draw_chip_architecture()
    
    if save_path:
        arch_fig.savefig(f"{save_path}_architecture.png", dpi=300, bbox_inches='tight')
    
    # 2. Dataflow visualization (if DNN is mapped)
    if dnn_manager.layer_mappings:
        dataflow_viz = DataflowVisualizer(chip, dnn_manager)
        
        # Create weight mapping visualization for first layer
        first_layer = list(dnn_manager.layer_mappings.keys())[0]
        mapping_fig, mapping_axes = dataflow_viz.visualize_weight_mapping(first_layer)
        
        if save_path:
            mapping_fig.savefig(f"{save_path}_weight_mapping.png", dpi=300, bbox_inches='tight')
    
    # 3. Performance heatmap (if metrics available)
    if metrics_data:
        dataflow_viz = DataflowVisualizer(chip, dnn_manager)
        perf_fig, perf_axes = dataflow_viz.create_performance_heatmap(metrics_data)
        
        if save_path:
            perf_fig.savefig(f"{save_path}_performance.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return {
        'architecture': (arch_fig, arch_ax),
        'weight_mapping': (mapping_fig, mapping_axes) if dnn_manager.layer_mappings else None,
        'performance': (perf_fig, perf_axes) if metrics_data else None
    }