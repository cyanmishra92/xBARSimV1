"""
Live visualization for real-time monitoring of ReRAM crossbar simulator
"""

import time
import sys
import threading
from typing import Dict, List, Any, Optional
import numpy as np

class LiveVisualization:
    """Real-time visualization of simulator execution"""
    
    def __init__(self, chip, dnn_manager):
        self.chip = chip
        self.dnn_manager = dnn_manager
        self.running = False
        self.current_layer = 0
        self.layer_progress = {}
        self.crossbar_activity = {}
        self.memory_activity = {}
        self.start_time = None
        
    def start_live_monitoring(self):
        """Start live monitoring in a separate thread"""
        self.running = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_live_monitoring(self):
        """Stop live monitoring"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
            
    def update_layer_progress(self, layer_idx: int, progress: float):
        """Update progress for a specific layer"""
        self.layer_progress[layer_idx] = progress
        self.current_layer = layer_idx
        
    def update_crossbar_activity(self, st_idx: int, tile_idx: int, xbar_idx: int, 
                                operations: int, utilization: float):
        """Update crossbar activity"""
        key = f"ST{st_idx}_T{tile_idx}_XB{xbar_idx}"
        self.crossbar_activity[key] = {
            'operations': operations,
            'utilization': utilization,
            'last_update': time.time()
        }
        
    def update_memory_activity(self, buffer_name: str, operations: int, 
                              latency: float, utilization: float):
        """Update memory activity"""
        # If this is a new operation, increment our counter
        if buffer_name not in self.memory_activity:
            self.memory_activity[buffer_name] = {
                'operations': 0,
                'latency': latency,
                'utilization': utilization,
                'last_update': time.time()
            }
        
        # Update operations (accumulate if we get more)
        if operations > self.memory_activity[buffer_name]['operations']:
            self.memory_activity[buffer_name]['operations'] = operations
        
        # Always update other fields
        self.memory_activity[buffer_name]['latency'] = latency
        self.memory_activity[buffer_name]['utilization'] = utilization
        self.memory_activity[buffer_name]['last_update'] = time.time()
    
    def increment_memory_operation(self, buffer_name: str):
        """Increment memory operation count for a buffer"""
        if buffer_name not in self.memory_activity:
            self.memory_activity[buffer_name] = {
                'operations': 0,
                'latency': 1.0,
                'utilization': 0.1,
                'last_update': time.time()
            }
        
        self.memory_activity[buffer_name]['operations'] += 1
        self.memory_activity[buffer_name]['last_update'] = time.time()
        self.memory_activity[buffer_name]['utilization'] = min(1.0, 
                                                               self.memory_activity[buffer_name]['utilization'] + 0.1)
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            self._refresh_display()
            time.sleep(0.5)  # Update every 500ms
            
    def _refresh_display(self):
        """Refresh the live display"""
        # Clear screen (ANSI escape sequences)
        print("\033[2J\033[H", end="")
        
        # Print header
        elapsed = time.time() - self.start_time if self.start_time else 0
        print("🔄 LIVE RERAM SIMULATOR MONITORING")
        print("=" * 80)
        print(f"⏱️  Elapsed Time: {elapsed:.1f}s | Current Layer: {self.current_layer}")
        print("=" * 80)
        
        # Layer execution progress
        self._print_layer_progress()
        
        # Hardware utilization
        self._print_hardware_activity()
        
        # Memory activity
        self._print_memory_activity()
        
        # Live metrics
        self._print_live_metrics()
        
        print("\n📊 Press Ctrl+C to stop monitoring...")
        
    def _print_layer_progress(self):
        """Print current layer execution progress"""
        print("\n🧠 LAYER EXECUTION PROGRESS")
        print("-" * 40)
        
        if not self.dnn_manager.dnn_config.layers:
            print("   No layers configured")
            return
            
        for i, layer_config in enumerate(self.dnn_manager.dnn_config.layers):
            progress = self.layer_progress.get(i, 0.0)
            status = "🔄" if i == self.current_layer else "✅" if progress >= 1.0 else "⏸️"
            
            # Create progress bar
            bar_length = 20
            filled = int(bar_length * min(progress, 1.0))
            bar = "█" * filled + "░" * (bar_length - filled)
            
            print(f"   {status} Layer {i} ({layer_config.layer_type.value[:8]:8s}): [{bar}] {progress*100:5.1f}%")
            
    def _print_hardware_activity(self):
        """Print hardware component activity"""
        print("\n🔧 HARDWARE ACTIVITY")
        print("-" * 40)
        
        # Crossbar activity
        active_crossbars = 0
        total_operations = 0
        
        for key, activity in self.crossbar_activity.items():
            if activity['operations'] > 0:
                active_crossbars += 1
                total_operations += activity['operations']
                
                # Show activity indicator
                age = time.time() - activity['last_update']
                indicator = "🔥" if age < 1.0 else "🔸" if age < 5.0 else "⚪"
                
                print(f"   {indicator} {key}: {activity['operations']:4d} ops ({activity['utilization']*100:5.1f}%)")
                
        print(f"   📊 Active Crossbars: {active_crossbars} | Total Ops: {total_operations:,}")
        
    def _print_memory_activity(self):
        """Print memory system activity"""
        print("\n💾 MEMORY ACTIVITY")
        print("-" * 40)
        
        if not self.memory_activity:
            print("   No memory activity detected")
            return
            
        for buffer_name, activity in self.memory_activity.items():
            age = time.time() - activity['last_update']
            indicator = "💥" if age < 1.0 else "📦" if age < 5.0 else "💤"
            
            print(f"   {indicator} {buffer_name:15s}: {activity['operations']:4d} ops | "
                  f"{activity['latency']:5.1f} cyc | {activity['utilization']*100:5.1f}%")
                  
    def _print_live_metrics(self):
        """Print live performance metrics"""
        print("\n📈 LIVE METRICS")
        print("-" * 40)
        
        # Calculate some basic metrics
        total_xbar_ops = sum(act['operations'] for act in self.crossbar_activity.values())
        total_mem_ops = sum(act['operations'] for act in self.memory_activity.values())
        
        elapsed = time.time() - self.start_time if self.start_time else 1
        ops_per_second = total_xbar_ops / elapsed
        
        print(f"   🚀 Crossbar Ops/sec: {ops_per_second:.1f}")
        print(f"   💾 Memory Operations: {total_mem_ops:,}")
        print(f"   ⚡ Total Crossbar Ops: {total_xbar_ops:,}")


class InteractiveArchitectureExplorer:
    """Interactive exploration of chip architecture"""
    
    def __init__(self, chip):
        self.chip = chip
        
    def explore_architecture(self):
        """Start interactive architecture exploration"""
        print("🏗️  INTERACTIVE ARCHITECTURE EXPLORER")
        print("=" * 60)
        
        while True:
            self._print_main_menu()
            choice = input("\nEnter your choice (or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == '1':
                self._explore_chip_overview()
            elif choice == '2':
                self._explore_supertiles()
            elif choice == '3':
                self._explore_tiles()
            elif choice == '4':
                self._explore_crossbars()
            elif choice == '5':
                self._explore_memory_hierarchy()
            elif choice == '6':
                self._show_detailed_stats()
            else:
                print("❌ Invalid choice. Please try again.")
                
    def _print_main_menu(self):
        """Print the main exploration menu"""
        print("\n" + "=" * 60)
        print("🔍 ARCHITECTURE EXPLORATION MENU")
        print("=" * 60)
        print("1. 🎯 Chip Overview")
        print("2. 🏢 SuperTile Details")
        print("3. 🏠 Tile Details") 
        print("4. ⚡ Crossbar Arrays")
        print("5. 💾 Memory Hierarchy")
        print("6. 📊 Detailed Statistics")
        print("q. 🚪 Quit Explorer")
        
    def _explore_chip_overview(self):
        """Show chip overview"""
        config = self.chip.get_chip_configuration()
        
        print("\n🎯 CHIP OVERVIEW")
        print("-" * 40)
        print(f"Total SuperTiles: {config['hierarchy']['supertiles']}")
        print(f"Total Tiles: {config['hierarchy']['supertiles'] * config['hierarchy']['tiles_per_supertile']}")
        print(f"Total Crossbars: {config['hierarchy']['total_crossbars']}")
        print(f"Total ReRAM Cells: {config['compute_capacity']['total_ReRAM_cells']:,}")
        print(f"Crossbar Size: {config['compute_capacity']['crossbar_size']}")
        
        # Calculate utilization
        stats = self.chip.get_total_statistics()
        total_ops = stats['performance']['total_crossbar_operations']
        print(f"\nCurrent Activity:")
        print(f"Total Operations: {total_ops:,}")
        
        input("\nPress Enter to continue...")
        
    def _explore_supertiles(self):
        """Explore SuperTile details"""
        print("\n🏢 SUPERTILE EXPLORATION")
        print("-" * 40)
        
        for st_idx, supertile in enumerate(self.chip.supertiles):
            print(f"\n📍 SuperTile {st_idx}:")
            print(f"   Tiles: {len(supertile.tiles)}")
            print(f"   Shared Buffer: {supertile.config.shared_buffer_size} KB {supertile.config.shared_buffer_type}")
            
            # Get supertile statistics
            st_stats = supertile.get_statistics()
            print(f"   Operations: {st_stats['operation_count']:,}")
            print(f"   Energy: {st_stats['total_energy']:.2e} J")
            
        st_choice = input(f"\nEnter SuperTile ID to explore (0-{len(self.chip.supertiles)-1}) or 'b' for back: ")
        
        if st_choice.isdigit() and 0 <= int(st_choice) < len(self.chip.supertiles):
            self._explore_supertile_details(int(st_choice))
            
    def _explore_supertile_details(self, st_idx: int):
        """Explore specific SuperTile details"""
        supertile = self.chip.supertiles[st_idx]
        
        print(f"\n🔍 SUPERTILE {st_idx} DETAILED VIEW")
        print("-" * 50)
        
        for tile_idx, tile in enumerate(supertile.tiles):
            tile_stats = tile.get_statistics()
            print(f"   Tile {tile_idx}: {tile_stats['operation_count']:,} operations")
            
            for xbar_idx, crossbar in enumerate(tile.crossbars):
                xbar_stats = crossbar.get_statistics()
                ops = xbar_stats['total_operations']
                if ops > 0:
                    print(f"     └─ Crossbar {xbar_idx}: {ops:,} operations")
                    
        input("\nPress Enter to continue...")
        
    def _explore_tiles(self):
        """Explore Tile details"""
        print("\n🏠 TILE EXPLORATION")
        print("-" * 40)
        
        tile_count = 0
        for st_idx, supertile in enumerate(self.chip.supertiles):
            for tile_idx, tile in enumerate(supertile.tiles):
                global_tile_id = tile_count
                tile_stats = tile.get_statistics()
                
                print(f"Tile ST{st_idx}_T{tile_idx} (Global ID: {global_tile_id}):")
                print(f"   Crossbars: {len(tile.crossbars)}")
                print(f"   Operations: {tile_stats['operation_count']:,}")
                print(f"   Local Buffer: {tile.config.local_buffer_size} KB")
                tile_count += 1
                
        input("\nPress Enter to continue...")
        
    def _explore_crossbars(self):
        """Explore individual crossbar details"""
        print("\n⚡ CROSSBAR EXPLORATION")
        print("-" * 40)
        
        for st_idx, supertile in enumerate(self.chip.supertiles):
            for tile_idx, tile in enumerate(supertile.tiles):
                for xbar_idx, crossbar in enumerate(tile.crossbars):
                    xbar_stats = crossbar.get_statistics()
                    ops = xbar_stats['total_operations']
                    
                    if ops > 0:  # Only show active crossbars
                        print(f"🔥 ST{st_idx}_T{tile_idx}_XB{xbar_idx}: {ops:,} operations")
                        
                        # Show resistance/conductance info
                        resistance_matrix = crossbar.get_resistance_matrix()
                        conductance_matrix = crossbar.get_conductance_matrix()
                        
                        print(f"   Size: {resistance_matrix.shape}")
                        print(f"   Resistance Range: {resistance_matrix.min():.2e} - {resistance_matrix.max():.2e} Ω")
                        print(f"   Conductance Range: {conductance_matrix.min():.2e} - {conductance_matrix.max():.2e} S")
                        
                        # Endurance status
                        endurance = xbar_stats['endurance_status']
                        print(f"   Write Cycles: avg={endurance['avg_write_count']:.1f}, max={endurance['max_write_count']}")
                        print(f"   Failed Cells: {endurance['failed_cells']}")
                        
        input("\nPress Enter to continue...")
        
    def _explore_memory_hierarchy(self):
        """Explore memory hierarchy details"""
        print("\n💾 MEMORY HIERARCHY EXPLORATION")
        print("-" * 50)
        
        config = self.chip.get_chip_configuration()
        memory_config = config['memory_hierarchy']
        
        print(f"Global Buffer: {memory_config['global_buffer']}")
        print(f"Shared Buffers: {memory_config['shared_buffers']} (per SuperTile)")
        print(f"Local Buffers: {memory_config['local_buffers']} (per Tile)")
        
        # Show buffer statistics if available
        global_stats = self.chip.global_buffer.get_statistics()
        print(f"\nGlobal Buffer Activity:")
        print(f"   Access Count: {global_stats['access_count']:,}")
        print(f"   Total Energy: {global_stats['total_energy']:.2e} J")
        print(f"   Utilization: {global_stats['utilization']:.1%}")
        
        input("\nPress Enter to continue...")
        
    def _show_detailed_stats(self):
        """Show comprehensive statistics"""
        print("\n📊 DETAILED STATISTICS")
        print("-" * 50)
        
        stats = self.chip.get_total_statistics()
        
        # Performance stats
        perf = stats['performance']
        print(f"Performance Metrics:")
        print(f"   Total Operations: {perf['total_operations']:,}")
        print(f"   Crossbar Operations: {perf['total_crossbar_operations']:,}")
        print(f"   Tile Operations: {perf['total_tile_operations']:,}")
        
        # Energy stats
        energy = stats['energy']
        print(f"\nEnergy Metrics:")
        print(f"   Total Energy: {energy['total_energy']:.2e} J")
        print(f"   Compute Energy: {energy['compute_energy']:.2e} J")
        print(f"   Memory Energy: {energy['memory_energy']:.2e} J")
        
        input("\nPress Enter to continue...")


def start_live_visualization(chip, dnn_manager):
    """Start live visualization and return the visualizer object"""
    visualizer = LiveVisualization(chip, dnn_manager)
    visualizer.start_live_monitoring()
    return visualizer


def start_architecture_explorer(chip):
    """Start interactive architecture explorer"""
    explorer = InteractiveArchitectureExplorer(chip)
    explorer.explore_architecture()